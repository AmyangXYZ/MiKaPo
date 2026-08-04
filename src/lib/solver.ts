import { Landmark } from "@mediapipe/tasks-vision"
import { Quat, Vec3 } from "reze-engine"
import { QuaternionOneEuroFilter, Vec3OneEuroFilter } from "./filters"
import { HandIndexTable, PoseLandmarksTable } from "./landmarks"

/** One of the model's rigid bodies, flattened for the solver's clearance pass. */
export interface BodyCollider {
  bone: string
  /** 0 sphere, 1 box, 2 capsule (PMX order). */
  shape: number
  size: XYZ
  /** Rest-pose world position from the PMX. */
  position: XYZ
}

export interface BoneState {
  name: string
  rotation: Quat
  /** Only bones that MOVE carry this: センター (the body's height over the
   *  ground) and the leg IK bones (where each foot actually is). Everything
   *  else keeps its rest translation, as MMD expects. */
  translation?: Vec3
}

/** The landmark arrays the solver reads — HolisticLandmarkerResult and the
 * worker's trimmed payload both satisfy this structurally. */
export interface SolverInput {
  poseWorldLandmarks: Landmark[][]
  leftHandWorldLandmarks: Landmark[][]
  rightHandWorldLandmarks: Landmark[][]
}

// ---------------------------------------------------------------------------
// Bone definitions
//
// The solver is a single generic pipeline over this table. Each entry solves one
// MMD bone in its parent's local frame; parent world rotations accumulate in
// solve order, so every chain product is computed exactly once per frame.
// ---------------------------------------------------------------------------

type LandmarkSource = "pose" | "leftHand" | "rightHand"
/** A landmark name, or a pair whose midpoint is used. */
type Point = string | [string, string]

interface BasisDef {
  kind: "basis"
  name: "上半身" | "下半身" | "頭"
  parent: string | null
}

interface BendLimit {
  /** Flexion axis in the bone's parent-local frame; positive twist = curl toward palm. */
  axis: Vec3
  /** Flexion range in radians (min < 0 allows slight hyperextension). */
  min: number
  max: number
  /** Max out-of-plane swing (spread/abduction) in radians. */
  spreadMax: number
}

interface DirectionDef {
  kind: "direction"
  name: string
  parent: string | null
  source: LandmarkSource
  from: Point
  to: Point
  /**
   * Roll witness: name of the child-segment def whose live direction pins the
   * rotation around this bone's axis (e.g. the forearm orients upper-arm roll).
   * Direction-only solving (shortest-arc) leaves that degree of freedom to
   * chance; the witness resolves it whenever the child joint is bent enough
   * for the roll to be observable, blending back to shortest-arc when straight.
   */
  witness?: string
  /**
   * Second witness, used to the extent the first one has faded out.
   *
   * The knee witness dies exactly when the leg straightens, and a straight leg
   * is most of standing, walking and most of any dance — so for the thigh the
   * primary witness is absent precisely when it is needed, and hip rotation
   * falls back to shortest-arc, which carries no twist at all. That is what
   * makes mocap legs read as stiff: knees and feet locked forward relative to
   * the pelvis, the leg swinging like a pendulum.
   *
   * A straight knee cannot report roll, but the FOOT can. With the knee
   * extended the shin cannot twist independently, so where the toes point is
   * where the femur is rotated. Ankle flexion changes only how far the toe
   * direction leans off the leg axis — its bearing around the axis, which is
   * all the witness basis uses, still reads femoral rotation.
   */
  rollFallback?: string
  /**
   * Anatomical clamp (fingers): shortest-arc solving happily bends a joint the
   * wrong way when a noisy landmark frame lands on the extension side — clamp
   * flexion and spread to human ranges so glitches can't produce backward curls.
   */
  bend?: BendLimit
}

interface TwistDef {
  kind: "twist"
  name: string
  parent: string
  source: LandmarkSource
  from: string
  to: string
  /** Ref key of the bone whose rest direction is the twist axis (the forearm). */
  axisRef: string
}

interface FingerRatioDef {
  kind: "fingerRatio"
  name: string
  /** Base joint (proximal phalanx) whose bend this joint follows at a fixed ratio. */
  base: string
  bendAxis: Vec3
  ratio: number
}

type BoneDef = BasisDef | DirectionDef | TwistDef | FingerRatioDef

const fingerCurl = (side: "左" | "右", finger: string, axis: Vec3, ratios: [number, number]): FingerRatioDef[] => [
  { kind: "fingerRatio", name: `${side}${finger}２`, base: `${side}${finger}１`, bendAxis: axis, ratio: ratios[0] },
  { kind: "fingerRatio", name: `${side}${finger}３`, base: `${side}${finger}１`, bendAxis: axis, ratio: ratios[1] },
]

const DEG = Math.PI / 180
/** Finger flexion axes: fingers point ±X at rest, palms face inward/down, so
 * curl-toward-palm is a rotation about ∓Z (mirrored between hands). */
const FINGER_BEND: Record<"左" | "右", BendLimit> = {
  左: { axis: new Vec3(0, 0, -1), min: -15 * DEG, max: 110 * DEG, spreadMax: 22 * DEG },
  右: { axis: new Vec3(0, 0, 1), min: -15 * DEG, max: 110 * DEG, spreadMax: 22 * DEG },
}
const THUMB_BEND: Record<"左" | "右", BendLimit> = {
  左: { axis: new Vec3(-1, -1, 0).normalize(), min: -25 * DEG, max: 80 * DEG, spreadMax: 40 * DEG },
  右: { axis: new Vec3(-1, 1, 0).normalize(), min: -25 * DEG, max: 80 * DEG, spreadMax: 40 * DEG },
}

const fingerBase = (side: "左" | "右", source: LandmarkSource, finger: string, mcp: string, pip: string): DirectionDef => ({
  kind: "direction",
  name: `${side}${finger}１`,
  parent: `${side}手首`,
  source,
  from: mcp,
  to: pip,
  bend: FINGER_BEND[side],
})

const BONE_DEFS: BoneDef[] = [
  { kind: "basis", name: "上半身", parent: null },
  {
    kind: "direction",
    name: "首",
    parent: "上半身",
    source: "pose",
    from: ["left_shoulder", "right_shoulder"],
    to: ["left_ear", "right_ear"],
  },
  { kind: "basis", name: "頭", parent: "首" },
  { kind: "basis", name: "下半身", parent: null },

  { kind: "direction", name: "左足", parent: "下半身", source: "pose", from: "left_hip", to: "left_knee", witness: "左ひざ", rollFallback: "左足首" },
  { kind: "direction", name: "右足", parent: "下半身", source: "pose", from: "right_hip", to: "right_knee", witness: "右ひざ", rollFallback: "右足首" },
  { kind: "direction", name: "左ひざ", parent: "左足", source: "pose", from: "left_knee", to: "left_ankle" },
  { kind: "direction", name: "右ひざ", parent: "右足", source: "pose", from: "right_knee", to: "right_ankle" },
  // ankle→foot_index matches the calibrated 足首→つま先 bone reference (ankle is
  // above heel; a heel baseline tilts the rest direction ~30° off the bone ref)
  { kind: "direction", name: "左足首", parent: "左ひざ", source: "pose", from: "left_ankle", to: "left_foot_index" },
  { kind: "direction", name: "右足首", parent: "右ひざ", source: "pose", from: "right_ankle", to: "right_foot_index" },

  { kind: "direction", name: "左腕", parent: "上半身", source: "pose", from: "left_shoulder", to: "left_elbow", witness: "左ひじ" },
  { kind: "direction", name: "右腕", parent: "上半身", source: "pose", from: "right_shoulder", to: "right_elbow", witness: "右ひじ" },
  { kind: "direction", name: "左ひじ", parent: "左腕", source: "pose", from: "left_elbow", to: "left_wrist" },
  { kind: "direction", name: "右ひじ", parent: "右腕", source: "pose", from: "right_elbow", to: "right_wrist" },

  // Wrist twist: rotation of the hand's index−ring axis about the forearm.
  // Swing residue is absorbed by 手首, whose parent chain includes 手捩.
  { kind: "twist", name: "左手捩", parent: "左ひじ", source: "leftHand", from: "ring_mcp", to: "index_mcp", axisRef: "左ひじ" },
  { kind: "twist", name: "右手捩", parent: "右ひじ", source: "rightHand", from: "ring_mcp", to: "index_mcp", axisRef: "右ひじ" },
  { kind: "direction", name: "左手首", parent: "左手捩", source: "leftHand", from: "wrist", to: "middle_mcp" },
  { kind: "direction", name: "右手首", parent: "右手捩", source: "rightHand", from: "wrist", to: "middle_mcp" },

  { kind: "direction", name: "左親指１", parent: "左手首", source: "leftHand", from: "thumb_mcp", to: "thumb_ip", bend: THUMB_BEND["左"] },
  fingerBase("左", "leftHand", "人指", "index_mcp", "index_pip"),
  fingerBase("左", "leftHand", "中指", "middle_mcp", "middle_pip"),
  fingerBase("左", "leftHand", "薬指", "ring_mcp", "ring_pip"),
  fingerBase("左", "leftHand", "小指", "pinky_mcp", "pinky_pip"),
  { kind: "direction", name: "右親指１", parent: "右手首", source: "rightHand", from: "thumb_mcp", to: "thumb_ip", bend: THUMB_BEND["右"] },
  fingerBase("右", "rightHand", "人指", "index_mcp", "index_pip"),
  fingerBase("右", "rightHand", "中指", "middle_mcp", "middle_pip"),
  fingerBase("右", "rightHand", "薬指", "ring_mcp", "ring_pip"),
  fingerBase("右", "rightHand", "小指", "pinky_mcp", "pinky_pip"),

  // Distal joints follow the base joint's bend at a fixed ratio (kept simple on
  // purpose — works well in practice and is robust to noisy fingertip landmarks).
  { kind: "fingerRatio", name: "左親指２", base: "左親指１", bendAxis: new Vec3(-1, -1, 0).normalize(), ratio: 0.85 },
  ...fingerCurl("左", "人指", new Vec3(-0.031, 0, -0.993).normalize(), [0.9, 0.65]),
  ...fingerCurl("左", "中指", new Vec3(0.03, 0, -0.996).normalize(), [0.9, 0.65]),
  ...fingerCurl("左", "薬指", new Vec3(0.048, 0, 0.997).normalize(), [0.88, 0.6]),
  ...fingerCurl("左", "小指", new Vec3(0.088, 0, -0.997).normalize(), [0.85, 0.55]),
  { kind: "fingerRatio", name: "右親指２", base: "右親指１", bendAxis: new Vec3(-1, 1, 0).normalize(), ratio: 0.85 },
  ...fingerCurl("右", "人指", new Vec3(-0.031, 0, 0.993).normalize(), [0.9, 0.65]),
  ...fingerCurl("右", "中指", new Vec3(0.03, 0, 0.996).normalize(), [0.9, 0.65]),
  ...fingerCurl("右", "薬指", new Vec3(0.048, 0, 0.997).normalize(), [0.88, 0.6]),
  ...fingerCurl("右", "小指", new Vec3(0.088, 0, 0.997).normalize(), [0.85, 0.55]),
]

const DEF_BY_NAME: Record<string, BoneDef> = Object.fromEntries(BONE_DEFS.map((d) => [d.name, d]))

/** Solved geometrically rather than from landmark directions. */
const GROUNDING_BONES = ["センター", "左足ＩＫ", "右足ＩＫ"] as const
/** Derived from the arm rather than from landmarks: MediaPipe's shoulder point
 *  is the joint itself, which says nothing about clavicle elevation. */
const SHOULDER_BONES = ["左肩", "右肩"] as const

// Scratch for the clearance pass — it runs per arm, per frame.
const _clearA = Quat.identity()
const _clearB = Quat.identity()
const _clearC = Quat.identity()
const _clearV = new Vec3(0, 0, 0)
const _clearFrom = new Vec3(0, 0, 0)
const _gA = new Vec3(0, 0, 0)
const _gB = new Vec3(0, 0, 0)
const _clearTo = new Vec3(0, 0, 0)

/** Pose landmarks each basis bone reads, for visibility gating. */
const BASIS_LANDMARKS: Record<string, string[]> = {
  上半身: ["left_shoulder", "right_shoulder"],
  下半身: ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
  頭: ["left_ear", "right_ear", "left_eye", "right_eye"],
}

// Below this average visibility the measurement is noise (limb off-frame or
// occluded) — hold the last solved rotation instead of chasing garbage.
const MIN_VISIBILITY = 0.35

// Witness blend: sine of the child-joint bend angle below which roll is
// unobservable (straight limb) and we fall back to shortest-arc.
const WITNESS_FADE_LO = 0.15
const WITNESS_FADE_HI = 0.35

// Canonical rest bend planes in parent-local frame. MMD rest poses have straight
// elbows/knees, so the rest child direction can't serve as the roll reference —
// instead anchor to anatomy: elbows flex forward (−Z), knees flex backward (+Z).
const WITNESS_REST: Record<string, Vec3> = {
  左腕: new Vec3(0, 0, -1),
  右腕: new Vec3(0, 0, -1),
  左足: new Vec3(0, 0, 1),
  右足: new Vec3(0, 0, 1),
}

// ---------------------------------------------------------------------------
// Rest-pose calibration
// ---------------------------------------------------------------------------

// Bones whose rest world positions calibrate() reads. Caller queries each
// from the loaded MMD model and passes them as `restWorldPos`.
export const SOLVER_REST_BONES: readonly string[] = [
  "左足", "右足", "左ひざ", "右ひざ", "左足首", "右足首",
  "左つま先", "右つま先",
  "首", "頭", "左肩", "右肩", "左目", "右目",
  "上半身", "上半身2", "下半身",
  "センター", "左足ＩＫ", "右足ＩＫ",
  "左腕", "右腕", "左ひじ", "右ひじ", "左手首", "右手首",
  "左中指１", "右中指１",
  "左親指１", "左親指２", "右親指１", "右親指２",
  "左人指１", "左人指２", "右人指１", "右人指２",
  "左中指２", "右中指２",
  "左薬指１", "左薬指２", "右薬指１", "右薬指２",
  "左小指１", "左小指２", "右小指１", "右小指２",
]

// Fallback reference directions in each bone's parent-local frame at rest.
// `Solver.calibrate()` overrides any of these from the loaded model's rest pose.
// 左手捩/右手捩 use a canonical hand-local axis that calibrate() can't derive
// from bones, so they always come from here.
const DEFAULT_REFS: Record<string, Vec3> = {
  左腕: new Vec3(0.80917156, -0.58753001, -0.00706277).normalize(),
  右腕: new Vec3(-0.80917129, -0.58753035, -0.00706463).normalize(),
  左ひじ: new Vec3(0.80886214, -0.58772615, -0.01788871).normalize(),
  右ひじ: new Vec3(-0.80886264, -0.58772542, -0.01789011).normalize(),
  左足: new Vec3(-0.01338665, -0.99819434, 0.05855645).normalize(),
  右足: new Vec3(0.01338609, -0.99819433, 0.05855677).normalize(),
  左ひざ: new Vec3(-0.01333798, -0.98954426, 0.14361147).normalize(),
  右ひざ: new Vec3(0.01333724, -0.98954425, 0.14361163).normalize(),
  左足首: new Vec3(0.00000064, -0.80765191, -0.58965955).normalize(),
  右足首: new Vec3(0.00000054, -0.80765185, -0.58965964).normalize(),
  首: new Vec3(0.00000258, 0.97346054, -0.22885491).normalize(),
  左手首: new Vec3(0.81635913, -0.57754444, -0.00043314).normalize(),
  右手首: new Vec3(-0.81635927, -0.57754425, -0.00043491).normalize(),
  左親指１: new Vec3(0.62716533, -0.72577692, -0.28268623).normalize(),
  右親指１: new Vec3(-0.62716428, -0.72578107, -0.28267792).normalize(),
  左人指１: new Vec3(0.84121176, -0.54001806, 0.02726296).normalize(),
  右人指１: new Vec3(-0.84121092, -0.54001943, 0.02726177).normalize(),
  左中指１: new Vec3(0.82851523, -0.55942638, 0.0245895).normalize(),
  右中指１: new Vec3(-0.82851643, -0.55942465, 0.02458833).normalize(),
  左薬指１: new Vec3(0.80448878, -0.59258445, 0.04051516).normalize(),
  右薬指１: new Vec3(-0.8044868, -0.59258726, 0.04051333).normalize(),
  左小指１: new Vec3(0.86110206, -0.49661517, 0.10897986).normalize(),
  右小指１: new Vec3(-0.86110169, -0.49661597, 0.10897917).normalize(),
  // 左手捩/右手捩: canonical hand-local axis used for wrist twist roll extraction.
  左手捩: new Vec3(0, 0, -1),
  右手捩: new Vec3(0, 0, -1),
}

interface XYZ {
  x: number
  y: number
  z: number
}

// Scratch registers — the entire per-frame solve allocates nothing.
const sFrom = Vec3.zeros()
const sTo = Vec3.zeros()
const sDir = Vec3.zeros()
const sWit = Vec3.zeros()
const sA = Vec3.zeros()
const sB = Vec3.zeros()
const sC = Vec3.zeros()
const sQ = Quat.identity()
const sQ2 = Quat.identity()
// The two witness solutions, held apart so the second can be built while the
// first is still waiting to be blended.
const sQ3 = Quat.identity()
const sQ4 = Quat.identity()

export class Solver {
  private pose: Landmark[] | null = null
  private leftHand: Landmark[] | null = null
  private rightHand: Landmark[] | null = null

  /** Unfiltered parent-local rotation per bone; doubles as the held value on dropout. */
  private locals: Record<string, Quat> = {}
  /** Accumulated world rotation per bone (parent chain product), rebuilt each frame. */
  private worlds: Record<string, Quat> = {}
  /** World rotations recomposed from the FILTERED locals — what grounding reads. */
  private filteredWorlds: Record<string, Quat> = {}
  private moveFilters: Record<string, Vec3OneEuroFilter> = {}
  private filters: Record<string, QuaternionOneEuroFilter> = {}
  // One-Euro tuning: minCutoff governs rest-pose jitter suppression (lower =
  // calmer, laggier at rest); beta governs how fast the cutoff opens with
  // speed (higher = fast/dramatic moves track tighter with less lag and less
  // amplitude loss). Rest stability and motion tracking are tuned independently.
  //
  // dCutoff is the low-pass on the SPEED estimate, and it decides how quickly
  // the adaptive term reacts — the term that is supposed to open the filter up
  // and let a fast move through. One-Euro's published default of 1.0 Hz is
  // tuned for a mouse pointer. On a limb it is far too slow: at 30 fps it
  // reaches only 63% of a step in about six frames, so a strike that is over in
  // three is filtered at its RESTING cutoff throughout and lands short. 4 Hz
  // gets there in roughly two frames, inside the action rather than after it.
  // This is the single parameter behind 动作没做到位.
  private smoothing = { minCutoff: 1.5, beta: 1.5, dCutoff: 4.0 }

  /**
   * Retune smoothing. Existing filters are dropped so the new values take effect
   * on the next frame rather than only on bones that appear later — a filter
   * carries its cutoffs from construction.
   */
  setSmoothing(minCutoff: number, beta: number, dCutoff?: number): void {
    this.smoothing = { ...this.smoothing, minCutoff, beta, dCutoff: dCutoff ?? this.smoothing.dCutoff }
    this.filters = {}
    // Position filters carry the same cutoffs and must be rebuilt too, or the
    // IK targets keep smoothing at the old settings while the rotations change.
    this.moveFilters = {}
  }

  /** The live smoothing settings, so an offline pass can match them. */
  getSmoothing(): { minCutoff: number; beta: number; dCutoff: number } {
    return { ...this.smoothing }
  }
  // Calibrated reference directions in each bone's parent-local frame at rest.
  // Populated by calibrate() from the loaded model. Falls through to DEFAULT_REFS.
  private refs: Record<string, Vec3> = {}
  /** Stable output array: one BoneState per def, quats mutated in place each frame. */
  private outputs: BoneState[]
  private outputByName: Record<string, BoneState> = {}
  /** Roll-witness solving for arms/legs; disable to fall back to shortest-arc only. */
  witnessEnabled = true
  /** Anatomical finger clamps; disable to reproduce unclamped shortest-arc output. */
  bendClampEnabled = true

  constructor() {
    this.outputs = BONE_DEFS.map((def) => {
      const state: BoneState = { name: def.name, rotation: Quat.identity() }
      this.outputByName[def.name] = state
      this.locals[def.name] = Quat.identity()
      this.worlds[def.name] = Quat.identity()
      this.filteredWorlds[def.name] = Quat.identity()
      return state
    })
    // The clavicles: no landmark pair drives them, they take a share of the
    // arm's own rotation (see applyShoulderRhythm).
    for (const name of SHOULDER_BONES) {
      const state: BoneState = { name, rotation: Quat.identity() }
      this.outputByName[name] = state
      this.outputs.push(state)
      this.locals[name] = Quat.identity()
      this.worlds[name] = Quat.identity()
      this.filteredWorlds[name] = Quat.identity()
    }
    // Bones the definition table does not solve, because they are not driven by
    // a landmark pair: the root's height and the two leg IK targets. They are
    // computed geometrically after the chain resolves.
    for (const name of GROUNDING_BONES) {
      const state: BoneState = { name, rotation: Quat.identity(), translation: new Vec3(0, 0, 0) }
      this.outputByName[name] = state
      this.outputs.push(state)
    }
  }

  reset(): void {
    this.heldDy = 0
    for (const key of Object.keys(this.filters)) this.filters[key].reset()
    // Position filters too, or a second still eases out of the first one's
    // body placement instead of simply being that pose.
    for (const key of Object.keys(this.moveFilters)) this.moveFilters[key].reset()
    for (const key of Object.keys(this.locals)) this.locals[key].setIdentity()
    for (const name of GROUNDING_BONES) this.outputByName[name]?.translation?.setXYZ(0, 0, 0)
  }

  /** Body collision volumes taken from the MODEL'S OWN rigid bodies — the
   *  author's approximation of their character, already shaped and sized to fit
   *  it. MMD's physics never tests these against each other (they are all
   *  bone-following statics, so the broadphase filters the pairs), which is
   *  exactly why an arm can end up inside a chest. Positions are stored in the
   *  chest's rest frame; capsules keep their half-height axis. */
  private restPos: Record<string, XYZ> = {}
  /** Overlap (model units) tolerated before the arm is pushed — a little contact
   *  reads as touching; a lot reads as 穿模. */
  clearanceSlack = 0.05
  /** 0 disables grounding entirely (rotation-only output, as before 3.3);
   *  1 follows the measured hip height exactly. */
  groundingGain = 1
  /** Share of arm elevation the clavicle carries (anatomy is roughly 1:2
   *  scapula:humerus). 0 restores the old frozen-shoulder behaviour. */
  shoulderRatio = 0.33
  /** Elevation below which the clavicle does not move at all (radians). */
  shoulderStart = 30 * DEG
  /** Ramp width above the threshold, so the shoulder eases in (radians). */
  shoulderRamp = 30 * DEG
  /** Hard cap on clavicle rotation (radians). */
  shoulderMax = 35 * DEG
  /** How high the hips rest when the body is lying on them, as a fraction of
   *  leg length — roughly half a torso's thickness. */
  hipClearance = 0.22
  /** Last committed body height, held through non-upright poses. */
  private heldDy = 0
  private bodyVolumes: { x: number; y: number; z: number; r: number; half: number }[] = []
  /** Arm colliders in their own bone's rest frame, plus the chain geometry the
   *  clearance FK needs. */
  private armVolumes: Record<string, { ox: number; oy: number; oz: number; r: number; half: number; upper: number }> = {}
  private chestRest: { x: number; y: number; z: number } | null = null

  /**
   * Feed the model's rigid bodies (see `Model.getRigidbodies()`). Torso and head
   * shapes become the volume an arm may not enter; arm shapes give the limb its
   * thickness. Without this the solver still runs — it simply has no idea the
   * character occupies space, which is the 穿模 everyone reports.
   */
  calibrateColliders(
    colliders: BodyCollider[],
    rest: Record<string, XYZ>,
  ): void {
    const chest = rest["上半身2"] ?? rest["上半身"]
    this.bodyVolumes = []
    this.armVolumes = {}
    this.chestRest = chest ? { x: chest.x, y: chest.y, z: chest.z } : null
    if (!chest) return

    const TORSO_BONES = new Set(["上半身", "上半身2", "下半身", "首", "頭"])
    for (const c of colliders) {
      // Sphere: size.x is the radius. Capsule: size.x radius, size.y height.
      // Box: take the smallest half-extent as a conservative radius so a boxy
      // torso does not over-claim space.
      const r = c.shape === 1 ? Math.min(c.size.x, c.size.z) : c.size.x
      const half = c.shape === 2 ? c.size.y * 0.5 : 0
      if (TORSO_BONES.has(c.bone)) {
        this.bodyVolumes.push({ x: c.position.x - chest.x, y: c.position.y - chest.y, z: c.position.z - chest.z, r, half })
      } else if (c.bone.endsWith("腕") || c.bone.endsWith("ひじ")) {
        const bone = rest[c.bone]
        const child = rest[c.bone.endsWith("腕") ? c.bone.replace("腕", "ひじ") : c.bone.replace("ひじ", "手首")]
        if (!bone) continue
        this.armVolumes[c.bone] = {
          ox: c.position.x - bone.x,
          oy: c.position.y - bone.y,
          oz: c.position.z - bone.z,
          r,
          half,
          upper: child ? Math.hypot(child.x - bone.x, child.y - bone.y, child.z - bone.z) : 0,
        }
      }
    }
  }

  /**
   * Push a solved arm out of the body when the pose puts it inside.
   *
   * Rotation-only by design: the correction is a swing at the shoulder, so the
   * elbow and hand ride along and the pose keeps its shape — no IK, no
   * retargeting of the chain. Depth is MediaPipe's weakest axis and the error
   * is largest exactly when a limb crosses the torso in frame, so this catches
   * the failure the landmarks cannot avoid.
   */
  private enforceBodyClearance(): void {
    if (this.bodyVolumes.length === 0 || !this.chestRest) return
    const chest = this.worlds["上半身2"] ?? this.worlds["上半身"]
    if (!chest) return

    for (const side of ["左", "右"] as const) {
      const armName = side + "腕"
      const armWorld = this.worlds[armName]
      const arm = this.armVolumes[armName]
      const fore = this.armVolumes[side + "ひじ"]
      if (!armWorld || !arm) continue
      const shoulderRest = this.refsPos(side + "腕")
      if (!shoulderRest) continue

      // Chest-local shoulder joint, then the two limb capsule centres by FK.
      const sx = shoulderRest.x - this.chestRest.x
      const sy = shoulderRest.y - this.chestRest.y
      const sz = shoulderRest.z - this.chestRest.z
      const probes: { x: number; y: number; z: number; r: number }[] = []
      const local = _clearV
      const push = (ox: number, oy: number, oz: number, r: number, base: { x: number; y: number; z: number }) => {
        local.setXYZ(ox, oy, oz)
        Quat.rotateVecInto(armWorld, local, local)
        Quat.rotateVecInvInto(chest, local, local)
        probes.push({ x: base.x + local.x, y: base.y + local.y, z: base.z + local.z, r })
      }
      const shoulder = { x: sx, y: sy, z: sz }
      push(arm.ox, arm.oy, arm.oz, arm.r, shoulder)
      if (fore) {
        local.setXYZ(0, -arm.upper, 0)
        Quat.rotateVecInto(armWorld, local, local)
        Quat.rotateVecInvInto(chest, local, local)
        const elbow = { x: sx + local.x, y: sy + local.y, z: sz + local.z }
        const foreWorld = this.worlds[side + "ひじ"] ?? armWorld
        local.setXYZ(fore.ox, fore.oy, fore.oz)
        Quat.rotateVecInto(foreWorld, local, local)
        Quat.rotateVecInvInto(chest, local, local)
        probes.push({ x: elbow.x + local.x, y: elbow.y + local.y, z: elbow.z + local.z, r: fore.r })
      }

      // Deepest overlap against any body volume wins the correction.
      let worst: { x: number; y: number; z: number } | null = null
      let worstDepth = this.clearanceSlack
      for (const p of probes) {
        for (const b of this.bodyVolumes) {
          // Closest point on the body capsule's axis (a sphere has half = 0).
          const ay = Math.min(b.y + b.half, Math.max(b.y - b.half, p.y))
          const dx = p.x - b.x
          const dy = p.y - ay
          const dz = p.z - b.z
          const d = Math.hypot(dx, dy, dz)
          const depth = b.r + p.r - d
          if (depth > worstDepth) {
            worstDepth = depth
            worst = { x: p.x, y: p.y, z: p.z }
          }
        }
      }
      if (!worst) continue

      // Swing the shoulder outward along the radial direction, just past contact.
      const fromX = worst.x - sx
      const fromY = worst.y - sy
      const fromZ = worst.z - sz
      const fromLen = Math.hypot(fromX, fromY, fromZ)
      if (fromLen < 1e-6) continue
      const radial = Math.hypot(worst.x, worst.z)
      if (radial < 1e-6) continue
      const scale = (radial + worstDepth) / radial
      const tX = worst.x * scale - sx
      const tY = fromY
      const tZ = worst.z * scale - sz
      const tLen = Math.hypot(tX, tY, tZ)
      if (tLen < 1e-6) continue
      _clearFrom.setXYZ(fromX / fromLen, fromY / fromLen, fromZ / fromLen)
      _clearTo.setXYZ(tX / tLen, tY / tLen, tZ / tLen)
      Quat.fromUnitVectorsInto(_clearFrom, _clearTo, _clearA)

      // Chest-local swing → world → back into the arm's parent-local frame.
      Quat.conjugateInto(chest, _clearB)
      Quat.multiplyInto(_clearA, _clearB, _clearC)
      Quat.multiplyInto(chest, _clearC, _clearA)
      Quat.multiplyInto(_clearA, armWorld, _clearB)
      armWorld.set(_clearB)
      const def = DEF_BY_NAME[armName]
      const parentName = def && def.kind !== "fingerRatio" ? def.parent : undefined
      const parentWorld = parentName ? this.worlds[parentName] : null
      if (parentWorld) {
        Quat.conjugateInto(parentWorld, _clearA)
        Quat.multiplyInto(_clearA, armWorld, _clearB)
        this.locals[armName].set(_clearB)
      } else {
        this.locals[armName].set(armWorld)
      }
    }
  }

  /**
   * Let the shoulder carry its share of the raise (scapulohumeral rhythm).
   *
   * The bone table hangs 腕 straight off 上半身, so the clavicle never moves and
   * the humerus performs the whole lift alone — which is anatomically
   * impossible past about 30° and looks it: arms overhead fold at a joint that
   * cannot fold that way. Anatomy splits elevation roughly 2:1 between humerus
   * and scapula once past that threshold, so hand the shoulder a fraction of
   * the arm's own rotation and take the same amount back out of the arm, which
   * leaves the arm pointing exactly where the landmarks put it.
   *
   * Both rotations live in 上半身-local space, so this is a straight factoring —
   * no frame conversions, nothing to drift.
   */
  private applyShoulderRhythm(): void {
    if (this.shoulderRatio <= 0) return
    for (const side of ["左", "右"] as const) {
      const armLocal = this.locals[side + "腕"]
      const shoulderLocal = this.locals[side + "肩"]
      if (!armLocal || !shoulderLocal) continue

      const w = Math.min(1, Math.abs(armLocal.w))
      const angle = 2 * Math.acos(w)
      if (angle <= this.shoulderStart) {
        shoulderLocal.setIdentity()
        continue
      }
      const sin = Math.sqrt(Math.max(0, 1 - w * w))
      if (sin < 1e-6) {
        shoulderLocal.setIdentity()
        continue
      }
      // armLocal's sign convention follows w; keep the axis on the same side.
      const flip = armLocal.w < 0 ? -1 : 1
      const ax = (armLocal.x * flip) / sin
      const ay = (armLocal.y * flip) / sin
      const az = (armLocal.z * flip) / sin

      // Ramp in over the first 30° past the threshold rather than stepping.
      const ramp = Math.min(1, (angle - this.shoulderStart) / this.shoulderRamp)
      let take = (angle - this.shoulderStart) * this.shoulderRatio * ramp
      if (take > this.shoulderMax) take = this.shoulderMax
      Quat.fromAxisAngleInto(ax, ay, az, take, shoulderLocal)
      // Remove exactly what the shoulder took, so the arm still points where the
      // landmarks say it does.
      Quat.conjugateInto(shoulderLocal, _clearA)
      Quat.multiplyInto(_clearA, armLocal, _clearB)
      armLocal.set(_clearB)
    }
  }

  /**
   * Place the body in space: how high the hips ride, and where each foot lands.
   *
   * MediaPipe's world landmarks are hip-centred, so the hip's own height is not
   * in them — but the distance from hip down to the LOWER foot is, and that is
   * the same number as long as one foot is on the ground. Taking the lower foot
   * is what makes this survive a raised leg: a standing split measures against
   * the standing foot and the free leg goes wherever the pose sends it. No
   * contact detection, no floor assumption, nothing to misfire.
   *
   * Scale comes from leg length — the model's hip→knee→ankle against the
   * landmarks' — so no calibration pose is needed.
   *
   * Then the leg chain is walked forward from the placed hip to find each ankle,
   * which becomes that leg's IK target. Feet land on the floor by construction:
   * if the hips sit exactly one bent-leg's height above it, the bent leg reaches
   * it. What this cannot do is leave the ground — both feet airborne is
   * indistinguishable from standing, in hip-centred coordinates.
   */
  private solveGrounding(timestampMs: number, unfiltered = false): void {
    const center = this.outputByName["センター"]
    if (!center?.translation) return
    const rest = this.restPos
    if (!this.pose || !rest["左足"] || !rest["右足"]) return

    // Walk both legs from the leg root FIRST, with no body offset, to find
    // where each ankle lands. Placing the body is then exact: drop it until the
    // LOWER foot rests at the height it rests at in the model's own bind pose.
    //
    // Deriving the offset from measured hip height instead needs the landmark
    // scale and the model's rest stance to agree, and they do not — MMD models
    // commonly stand with a slight bend, which leaves a constant float. This
    // formulation cannot drift: the supporting foot is on the floor because the
    // body was placed to put it there.
    const ankleY: Record<string, number> = {}
    const ankleWorld: Record<string, { x: number; y: number; z: number }> = {}
    // 足 hangs off 下半身, which the solver rotates every frame — a leaning or
    // twisting lower body carries the hip joints with it. Starting the chain at
    // 足's REST position ignores that and leaves the body floating (or sunk) by
    // whatever the lean displaced.
    const lowerRest = rest["下半身"]
    const lowerWorld = this.filteredWorlds["下半身"]
    for (const side of ["左", "右"] as const) {
      const hipRest = rest[side + "足"]
      const knee = rest[side + "ひざ"]
      const ankle = rest[side + "足首"]
      const thighWorld = this.filteredWorlds[side + "足"]
      const shinWorld = this.filteredWorlds[side + "ひざ"]
      if (!hipRest || !knee || !ankle || !thighWorld || !shinWorld) continue
      let hip = hipRest
      if (lowerRest && lowerWorld) {
        _gA.setXYZ(hipRest.x - lowerRest.x, hipRest.y - lowerRest.y, hipRest.z - lowerRest.z)
        Quat.rotateVecInto(lowerWorld, _gA, _gA)
        hip = { x: lowerRest.x + _gA.x, y: lowerRest.y + _gA.y, z: lowerRest.z + _gA.z }
      }
      _gA.setXYZ(knee.x - hipRest.x, knee.y - hipRest.y, knee.z - hipRest.z)
      Quat.rotateVecInto(thighWorld, _gA, _gA)
      _gB.setXYZ(ankle.x - knee.x, ankle.y - knee.y, ankle.z - knee.z)
      Quat.rotateVecInto(shinWorld, _gB, _gB)
      const p = { x: hip.x + _gA.x + _gB.x, y: hip.y + _gA.y + _gB.y, z: hip.z + _gA.z + _gB.z }
      ankleWorld[side] = p
      ankleY[side] = p.y
    }
    const ys = Object.values(ankleY)
    if (ys.length === 0 || !ys.every(Number.isFinite)) return
    const restAnkleY = Math.min(rest["左足首"]?.y ?? 0, rest["右足首"]?.y ?? 0)
    const legRootY = ((rest["左足"]?.y ?? 0) + (rest["右足"]?.y ?? 0)) / 2
    const legSpan = Math.max(1e-3, legRootY - restAnkleY)

    // The body rests on its LOWEST PART, not on its feet. Grounding purely off
    // ankles assumes someone is standing on them — so a person rolling on the
    // floor, whose weight is on their back with legs in the air, had the body
    // hauled up and down by whatever the legs were doing. That is the bouncing.
    //
    // Each candidate says how far the body must move for IT to sit on the floor
    // (its own resting height minus where the pose put it), and the binding one
    // is simply the largest: satisfy that and nothing else is underground. The
    // ankles rest at their bind height; the hips rest a body's thickness up.
    // Continuous by construction — at a handover the two agree, so support
    // passes from feet to hip without a step.
    let rawDy = -Infinity
    for (const y of ys) rawDy = Math.max(rawDy, restAnkleY - y)
    const hipRestY = rest["下半身"]?.y ?? legRootY
    rawDy = Math.max(rawDy, legSpan * this.hipClearance - hipRestY)
    rawDy *= this.groundingGain
    // A body cannot climb above its own bind height, whatever the landmarks say.
    if (rawDy > legSpan * 0.15) rawDy = legSpan * 0.15

    // Grounding is a standing-pose idea. Once the torso leaves vertical — rolling,
    // lying, a floor move — the hips are on the ground and the legs are in the
    // air, and nothing in hip-centred landmarks says how far the body dropped.
    // Tracking anyway means the legs drive the height and the body bounces.
    // So: hold the last height instead of guessing, fading over rather than
    // switching. A pose we place imperfectly is forgivable; one that jitters
    // is not.
    const spineWorld = this.filteredWorlds["上半身"]
    if (spineWorld) {
      _gA.setXYZ(0, 1, 0)
      Quat.rotateVecInto(spineWorld, _gA, _gA)
      const upright = Math.min(1, Math.max(0, (_gA.y - 0.35) / 0.3))
      rawDy = this.heldDy + (rawDy - this.heldDy) * upright
    }
    this.heldDy = rawDy

    let mf = this.moveFilters["センター"]
    if (!mf) {
      mf = new Vec3OneEuroFilter(this.smoothing.minCutoff, this.smoothing.beta, this.smoothing.dCutoff)
      this.moveFilters["センター"] = mf
    }
    if (unfiltered) {
      mf.reset()
      center.translation.setXYZ(0, rawDy, 0)
    } else {
      mf.filterInto(0, rawDy, 0, timestampMs, center.translation)
    }
    const dy = center.translation.y

    // Each ankle, lifted by the same offset, is that leg's IK target.
    for (const side of ["左", "右"] as const) {
      const ikOut = this.outputByName[side + "足ＩＫ"]
      const ikRest = rest[side + "足ＩＫ"] ?? rest[side + "足首"]
      const p = ankleWorld[side]
      if (!ikOut?.translation || !ikRest || !p) continue
      let f = this.moveFilters[side + "足ＩＫ"]
      if (!f) {
        f = new Vec3OneEuroFilter(this.smoothing.minCutoff, this.smoothing.beta, this.smoothing.dCutoff)
        this.moveFilters[side + "足ＩＫ"] = f
      }
      if (unfiltered) {
        f.reset()
        ikOut.translation.setXYZ(p.x - ikRest.x, p.y + dy - ikRest.y, p.z - ikRest.z)
      } else {
        f.filterInto(p.x - ikRest.x, p.y + dy - ikRest.y, p.z - ikRest.z, timestampMs, ikOut.translation)
      }
      // With IK driving the chain the foot takes its orientation from the IK
      // bone, so hand it the ankle's (filtered) world rotation.
      const footWorld = this.filteredWorlds[side + "足首"]
      if (footWorld) ikOut.rotation.set(footWorld)
    }
  }

  /** Rest world position of a bone, as captured by calibrate(). */
  private refsPos(name: string): XYZ | null {
    return this.restPos[name] ?? null
  }

  // Calibrate reference directions from the model's rest-pose world bone positions.
  // Parent chains are identity at rest, so world-space (child − parent) IS the
  // parent-local reference direction.
  calibrate(restWorldPos: Record<string, XYZ>): void {
    const dir = (parent: string, child: string): Vec3 | null => {
      const p = restWorldPos[parent]
      const c = restWorldPos[child]
      if (!p || !c) return null
      const v = new Vec3(c.x - p.x, c.y - p.y, c.z - p.z)
      if (v.length() < 1e-6) return null
      return v.normalizeInPlace()
    }
    const set = (key: string, v: Vec3 | null): void => {
      if (v) this.refs[key] = v
    }

    // Limbs
    set("左腕", dir("左腕", "左ひじ"))
    set("右腕", dir("右腕", "右ひじ"))
    set("左ひじ", dir("左ひじ", "左手首"))
    set("右ひじ", dir("右ひじ", "右手首"))
    set("左足", dir("左足", "左ひざ"))
    set("右足", dir("右足", "右ひざ"))
    set("左ひざ", dir("左ひざ", "左足首"))
    set("右ひざ", dir("右ひざ", "右足首"))

    this.restPos = restWorldPos

    // Ankle: pose runtime uses ankle→foot_index, so calibrate the same shape.
    set("左足首", dir("左足首", "左つま先"))
    set("右足首", dir("右足首", "右つま先"))

    // Neck: bone-direct (首→頭) doesn't match the pose runtime measurement
    // (ear_center − shoulder_center), so even at rest the rotation isn't identity.
    // Use eye/shoulder bone proxies — eye height ≈ ear height, shoulder bone ≈
    // shoulder landmark. Falls through to 首→頭 if any of the four bones is missing.
    set("首", dir("首", "頭"))
    const ls = restWorldPos["左肩"]
    const rs = restWorldPos["右肩"]
    const le = restWorldPos["左目"]
    const re = restWorldPos["右目"]
    if (ls && rs && le && re) {
      const v = new Vec3(
        (le.x + re.x - ls.x - rs.x) / 2,
        (le.y + re.y - ls.y - rs.y) / 2,
        (le.z + re.z - ls.z - rs.z) / 2,
      )
      if (v.length() > 1e-6) this.refs["首"] = v.normalizeInPlace()
    }

    // Wrists — middle finger root is the natural "forward" axis of the hand
    set("左手首", dir("左手首", "左中指１"))
    set("右手首", dir("右手首", "右中指１"))

    // Wrist-twist witness axis: index_mcp − ring_mcp at rest. The twist solve
    // compares the live hand axis to this reference and projects onto the
    // forearm to extract twist. Without calibration, the (0, 0, -1) fallback
    // bakes in a 90°-ish baseline twist for every frame including rest.
    set("左手捩", dir("左薬指１", "左人指１"))
    set("右手捩", dir("右薬指１", "右人指１"))

    // Finger base joints (proximal phalanges)
    set("左親指１", dir("左親指１", "左親指２"))
    set("右親指１", dir("右親指１", "右親指２"))
    set("左人指１", dir("左人指１", "左人指２"))
    set("右人指１", dir("右人指１", "右人指２"))
    set("左中指１", dir("左中指１", "左中指２"))
    set("右中指１", dir("右中指１", "右中指２"))
    set("左薬指１", dir("左薬指１", "左薬指２"))
    set("右薬指１", dir("右薬指１", "右薬指２"))
    set("左小指１", dir("左小指１", "左小指２"))
    set("右小指１", dir("右小指１", "右小指２"))
  }

  /**
   * Solve all bone rotations from one MediaPipe result.
   * `timestampMs`: media time for video (so seeks reset smoothing correctly),
   * wall time for live camera. Defaults to wall time.
   */
  /**
   * @param unfiltered  A still has no time axis: there is nothing to smooth
   * between, and passing one pose through a temporal filter can only soften it.
   * Set for image input — the solved pose is handed back exactly, and the
   * filters are reseeded so switching back to video starts clean.
   */
  solve(landmarks: SolverInput, timestampMs: number = performance.now(), unfiltered = false): BoneState[] {
    this.pose =
      landmarks.poseWorldLandmarks.length > 0 && landmarks.poseWorldLandmarks[0].length === 33
        ? landmarks.poseWorldLandmarks[0]
        : null
    this.leftHand =
      landmarks.leftHandWorldLandmarks.length > 0 && landmarks.leftHandWorldLandmarks[0].length === 21
        ? landmarks.leftHandWorldLandmarks[0]
        : null
    this.rightHand =
      landmarks.rightHandWorldLandmarks.length > 0 && landmarks.rightHandWorldLandmarks[0].length === 21
        ? landmarks.rightHandWorldLandmarks[0]
        : null

    for (const def of BONE_DEFS) {
      const local = this.locals[def.name]
      // Each solve writes into `local`, or leaves it untouched (hold) when its
      // landmarks are missing or below the visibility gate.
      switch (def.kind) {
        case "basis":
          this.solveBasis(def, local)
          break
        case "direction":
          this.solveDirection(def, local)
          break
        case "twist":
          this.solveTwist(def, local)
          break
        case "fingerRatio":
          this.solveFingerRatio(def, local)
          break
      }
      if (def.kind !== "fingerRatio") {
        const world = this.worlds[def.name]
        const parent = def.parent ? this.worlds[def.parent] : null
        if (parent) Quat.multiplyInto(parent, local, world)
        else world.set(local)
      }
    }

    // Shoulder rhythm rewrites 肩 and 腕 locals, so the world chain is rebuilt
    // before anything downstream reads it.
    this.applyShoulderRhythm()
    for (const def of BONE_DEFS) {
      if (def.kind === "fingerRatio") continue
      const world = this.worlds[def.name]
      const parent = def.parent ? this.worlds[def.parent] : null
      if (parent) Quat.multiplyInto(parent, this.locals[def.name], world)
      else world.set(this.locals[def.name])
    }

    this.enforceBodyClearance()

    // One-Euro post-pass on the outputs only — the hierarchy above always
    // composes unfiltered locals, so parent-chain math stays exact.
    for (const def of BONE_DEFS) {
      if (unfiltered) {
        this.outputByName[def.name].rotation.set(this.locals[def.name])
        this.filters[def.name]?.reset()
        continue
      }
      let f = this.filters[def.name]
      if (!f) {
        f = new QuaternionOneEuroFilter(this.smoothing.minCutoff, this.smoothing.beta, this.smoothing.dCutoff)
        this.filters[def.name] = f
      }
      f.filterInto(this.locals[def.name], timestampMs, this.outputByName[def.name].rotation)
    }

    for (const name of SHOULDER_BONES) {
      if (unfiltered) {
        this.outputByName[name].rotation.set(this.locals[name])
        this.filters[name]?.reset()
        continue
      }
      let f = this.filters[name]
      if (!f) {
        f = new QuaternionOneEuroFilter(this.smoothing.minCutoff, this.smoothing.beta, this.smoothing.dCutoff)
        this.filters[name] = f
      }
      f.filterInto(this.locals[name], timestampMs, this.outputByName[name].rotation)
    }

    // Grounding walks the leg chain, so it must read the rotations that will
    // actually be shown — otherwise the body is placed against an unfiltered
    // skeleton and the feet inherit every landmark twitch. Recompose world
    // rotations from the filtered locals (same order, so parents are ready).
    for (const def of BONE_DEFS) {
      if (def.kind === "fingerRatio") continue
      const world = this.filteredWorlds[def.name]
      const parent = def.parent ? this.filteredWorlds[def.parent] : null
      const local = this.outputByName[def.name].rotation
      if (parent) Quat.multiplyInto(parent, local, world)
      else world.set(local)
    }
    this.solveGrounding(timestampMs, unfiltered)

    return this.outputs
  }

  // -------------------------------------------------------------------------

  private getRef(key: string): Vec3 {
    return this.refs[key] ?? DEFAULT_REFS[key]
  }

  private sourceLandmarks(source: LandmarkSource): Landmark[] | null {
    return source === "pose" ? this.pose : source === "leftHand" ? this.leftHand : this.rightHand
  }

  private landmarkIndex(source: LandmarkSource, name: string): number {
    return source === "pose" ? PoseLandmarksTable[name] : HandIndexTable[name]
  }

  /** Writes the landmark (or midpoint) into `out` in MMD coords (y flipped). */
  private point(source: LandmarkSource, p: Point, out: Vec3): Vec3 | null {
    const lms = this.sourceLandmarks(source)
    if (!lms) return null
    if (typeof p === "string") {
      const lm = lms[this.landmarkIndex(source, p)]
      if (!lm) return null
      return out.setXYZ(lm.x, -lm.y, lm.z)
    }
    const a = lms[this.landmarkIndex(source, p[0])]
    const b = lms[this.landmarkIndex(source, p[1])]
    if (!a || !b) return null
    return out.setXYZ((a.x + b.x) / 2, -(a.y + b.y) / 2, (a.z + b.z) / 2)
  }

  /** Average MediaPipe visibility across the pose landmarks a bone reads (1 for hands). */
  /**
   * Confidence for a hand-sourced bone. MediaPipe's hand landmarks carry no
   * visibility field of their own, and — worse — when tracking degrades they do
   * not disappear: they collapse toward the origin, still 21 points, still
   * structurally valid. Solving from that is what snaps a wrist into an
   * impossible angle and folds fingers backwards.
   *
   * Two honest signals exist. The hand's own SPAN (a real hand is never a
   * point), and the POSE's wrist landmark — the same joint, tracked by the
   * other model, and it does report visibility.
   */
  private handConfidence(source: LandmarkSource): number {
    const hand = source === "leftHand" ? this.leftHand : this.rightHand
    if (!hand || hand.length < 21) return 0
    const w = hand[HandIndexTable.wrist]
    const mid = hand[HandIndexTable.middle_mcp]
    if (!w || !mid) return 0
    // World landmarks are metres; a palm is ~8 cm. Anything under a centimetre
    // is a collapsed cloud, not a hand.
    if (Math.hypot(mid.x - w.x, mid.y - w.y, mid.z - w.z) < 0.01) return 0
    const wristName = source === "leftHand" ? "left_wrist" : "right_wrist"
    return this.pose?.[PoseLandmarksTable[wristName]]?.visibility ?? 1
  }

  private visibility(source: LandmarkSource, points: Point[]): number {
    if (source !== "pose") return this.handConfidence(source)
    if (!this.pose) return 1
    let sum = 0
    let n = 0
    for (const p of points) {
      for (const name of typeof p === "string" ? [p] : p) {
        sum += this.pose[PoseLandmarksTable[name]]?.visibility ?? 1
        n++
      }
    }
    return n > 0 ? sum / n : 1
  }

  private solveDirection(def: DirectionDef, out: Quat): void {
    const from = this.point(def.source, def.from, sFrom)
    const to = this.point(def.source, def.to, sTo)
    if (!from || !to) return
    if (this.visibility(def.source, [def.from, def.to]) < MIN_VISIBILITY) return

    Vec3.subtractInto(to, from, sDir)
    const parentWorld = def.parent ? this.worlds[def.parent] : null
    if (parentWorld) Quat.rotateVecInvInto(parentWorld, sDir, sDir)
    if (sDir.length() < 1e-6) return
    sDir.normalizeInPlace()

    Quat.fromUnitVectorsInto(this.getRef(def.name), sDir, out)

    if (def.witness && this.witnessEnabled) this.applyWitness(def, parentWorld, out)
    if (def.bend && this.bendClampEnabled) Solver.clampBend(def.bend, out)
  }

  /**
   * Clamp a joint rotation to anatomical range: decompose q = swing ∘ twist
   * about the flexion axis, clamp the signed twist (flexion) angle and the
   * swing (spread) magnitude, and recompose.
   */
  private static clampBend(bend: BendLimit, q: Quat): void {
    Quat.twistAroundAxisInto(q, bend.axis, sQ) // twist
    // Signed flexion angle about the axis, wrapped to [-π, π]
    const k = sQ.x * bend.axis.x + sQ.y * bend.axis.y + sQ.z * bend.axis.z
    let angle = 2 * Math.atan2(k, sQ.w)
    if (angle > Math.PI) angle -= 2 * Math.PI
    else if (angle < -Math.PI) angle += 2 * Math.PI
    const clamped = Math.min(bend.max, Math.max(bend.min, angle))

    // swing = q ∘ twist⁻¹
    sQ.conjugate()
    Quat.multiplyInto(q, sQ, sQ2)
    // Clamp swing magnitude by nlerp toward identity
    const swingAngle = 2 * Math.acos(Math.min(1, Math.abs(sQ2.w)))
    if (swingAngle > bend.spreadMax) {
      const t = bend.spreadMax / swingAngle
      const sign = sQ2.w < 0 ? -1 : 1
      sQ2.setXYZW(sQ2.x * t * sign, sQ2.y * t * sign, sQ2.z * t * sign, sign * sQ2.w * t + (1 - t))
      sQ2.normalize()
    }

    Quat.fromAxisAngleInto(bend.axis.x, bend.axis.y, bend.axis.z, clamped, sQ)
    Quat.multiplyInto(sQ2, sQ, q) // q = swing ∘ twist
  }

  /**
   * Pin the roll (rotation about the bone axis) using the live direction of the
   * child segment. Builds full rest/live orthonormal bases and replaces the
   * shortest-arc rotation with basisLive ∘ basisRest⁻¹, faded by how observable
   * the roll actually is (≈ sine of the child bend angle).
   */
  private applyWitness(def: DirectionDef, parentWorld: Quat | null, out: Quat): void {
    const primary = this.witnessSolution(def, def.witness!, WITNESS_REST[def.name] ?? null, parentWorld, sQ3)
    // How much of the roll the primary witness could actually see. Whatever it
    // leaves is the room the fallback may claim.
    const t = Solver.witnessFade(primary)
    if (t > 0) {
      if (Quat.dot(out, sQ3) < 0) sQ3.setXYZW(-sQ3.x, -sQ3.y, -sQ3.z, -sQ3.w)
      Quat.nlerpInto(out, sQ3, t, out)
    }

    // Straight limb. The knee has nothing left to say about roll, so ask the
    // foot — see `rollFallback`. Both are absolute orientations, so handing
    // over is a weighted average and stays continuous across the crossover.
    const room = 1 - t
    if (room <= 1e-3 || !def.rollFallback) return
    const fallback = this.witnessSolution(def, def.rollFallback, this.getRef(def.rollFallback), parentWorld, sQ4)
    const t2 = Solver.witnessFade(fallback) * room
    if (t2 <= 0) return
    if (Quat.dot(out, sQ4) < 0) sQ4.setXYZW(-sQ4.x, -sQ4.y, -sQ4.z, -sQ4.w)
    Quat.nlerpInto(out, sQ4, t2, out)
  }

  /** Smoothstep from "roll unobservable" to "witness fully trusted". */
  private static witnessFade(perpLen: number): number {
    if (!(perpLen > WITNESS_FADE_LO)) return 0
    const t = Math.min(1, (perpLen - WITNESS_FADE_LO) / (WITNESS_FADE_HI - WITNESS_FADE_LO))
    return t * t * (3 - 2 * t)
  }

  /**
   * The rest→live basis rotation that pins this bone's roll using a witness
   * segment, written into `outQ`. Returns how observable that roll is — the
   * witness's component perpendicular to the bone axis, which is the sine of
   * the joint's bend — or -1 when the witness cannot be read at all.
   *
   * `restWit` is the witness direction in THIS bone's parent-local frame at
   * rest. Parent chains are identity at rest, so a child def's own reference
   * direction serves directly.
   */
  private witnessSolution(
    def: DirectionDef,
    witnessName: string,
    restWit: Vec3 | null,
    parentWorld: Quat | null,
    outQ: Quat,
  ): number {
    if (!restWit) return -1
    const wdef = DEF_BY_NAME[witnessName] as DirectionDef | undefined
    if (!wdef) return -1
    const wFrom = this.point(wdef.source, wdef.from, sA)
    const wTo = this.point(wdef.source, wdef.to, sB)
    if (!wFrom || !wTo) return -1
    if (this.visibility(wdef.source, [wdef.from, wdef.to]) < MIN_VISIBILITY) return -1

    Vec3.subtractInto(wTo, wFrom, sWit)
    if (parentWorld) Quat.rotateVecInvInto(parentWorld, sWit, sWit)
    if (sWit.length() < 1e-6) return -1
    sWit.normalizeInPlace()

    // Live witness component perpendicular to the live bone direction (sDir
    // still holds it). Its magnitude is the observability of the roll; its
    // bearing around the axis is the roll itself.
    const dLive = sWit.dot(sDir)
    sA.setXYZ(sWit.x - sDir.x * dLive, sWit.y - sDir.y * dLive, sWit.z - sDir.z * dLive)
    const perpLen = sA.length()
    if (perpLen < WITNESS_FADE_LO) return perpLen
    sA.normalizeInPlace()

    // Rest witness component perpendicular to the rest bone direction.
    const ref = this.getRef(def.name)
    const dRest = restWit.dot(ref)
    sB.setXYZ(restWit.x - ref.x * dRest, restWit.y - ref.y * dRest, restWit.z - ref.z * dRest)
    if (sB.length() < 1e-3) return -1
    sB.normalizeInPlace()

    // q = basisLive ∘ basisRest⁻¹ maps (ref → dir, restWitness⊥ → liveWitness⊥).
    Vec3.crossInto(ref, sB, sC)
    Quat.fromBasisInto(ref, sB, sC, sQ) // basisRest
    sQ.conjugate()
    Vec3.crossInto(sDir, sA, sC)
    Quat.fromBasisInto(sDir, sA, sC, sQ2) // basisLive
    Quat.multiplyInto(sQ2, sQ, outQ) // apply rest⁻¹ first, then live
    return perpLen
  }

  private solveTwist(def: TwistDef, out: Quat): void {
    const from = this.point(def.source, def.from, sFrom)
    const to = this.point(def.source, def.to, sTo)
    if (!from || !to) return

    Vec3.subtractInto(to, from, sDir)
    Quat.rotateVecInvInto(this.worlds[def.parent], sDir, sDir)
    if (sDir.length() < 1e-6) return
    sDir.normalizeInPlace()

    // Total rotation aligning the rest hand axis to the live one includes the
    // wrist swing; project onto the forearm axis to keep only the twist.
    Quat.fromUnitVectorsInto(this.getRef(def.name), sDir, sQ)
    Quat.twistAroundAxisInto(sQ, this.getRef(def.axisRef), out)
  }

  private solveFingerRatio(def: FingerRatioDef, out: Quat): void {
    const base = this.locals[def.base]
    const bendDegrees = Solver.extractBendDegrees(base, def.bendAxis)
    const radians = (bendDegrees * def.ratio * Math.PI) / 180
    Quat.fromAxisAngleInto(def.bendAxis.x, def.bendAxis.y, def.bendAxis.z, radians, out)
  }

  /** Signed rotation ABOUT the bend axis. The previous form took the total
   *  rotation angle and merely borrowed the axis for its sign, so any spread or
   *  twist in the base joint inflated the curl that the derived joints copy —
   *  a sideways-splayed finger drove its own knuckles into a fist. This is the
   *  twist component and nothing else. */
  private static extractBendDegrees(quat: Quat, bendAxis: Vec3): number {
    const axisComponent = quat.x * bendAxis.x + quat.y * bendAxis.y + quat.z * bendAxis.z
    return 2 * Math.atan2(axisComponent, quat.w) * (180 / Math.PI)
  }

  private solveBasis(def: BasisDef, out: Quat): void {
    if (!this.pose) return
    if (this.visibility("pose", BASIS_LANDMARKS[def.name]) < MIN_VISIBILITY) return

    switch (def.name) {
      case "上半身": {
        if (!this.point("pose", "left_shoulder", sA) || !this.point("pose", "right_shoulder", sB)) return
        // spineY = shoulder center (pose world origin is the hip center)
        sDir.setXYZ((sA.x + sB.x) / 2, (sA.y + sB.y) / 2, (sA.z + sB.z) / 2).normalizeInPlace()
        Vec3.subtractInto(sA, sB, sC).normalizeInPlace()
        Solver.basisFromYAndX(sDir, sC, out)
        return
      }
      case "下半身": {
        if (!this.point("pose", "left_shoulder", sA) || !this.point("pose", "right_shoulder", sB)) return
        sFrom.setXYZ((sA.x + sB.x) / 2, (sA.y + sB.y) / 2, (sA.z + sB.z) / 2)
        if (!this.point("pose", "left_hip", sA) || !this.point("pose", "right_hip", sB)) return
        sTo.setXYZ((sA.x + sB.x) / 2, (sA.y + sB.y) / 2, (sA.z + sB.z) / 2)
        // Pelvis basis shares the trunk Y with 上半身 (no separate pelvis-tilt
        // landmark exists); lower/upper differ in X (hip vs shoulder line),
        // which captures twist.
        Vec3.subtractInto(sFrom, sTo, sDir).normalizeInPlace()
        Vec3.subtractInto(sA, sB, sC).normalizeInPlace()
        Solver.basisFromYAndX(sDir, sC, out)
        return
      }
      case "頭": {
        if (!this.point("pose", "left_ear", sA) || !this.point("pose", "right_ear", sB)) return
        if (!this.point("pose", "left_eye", sFrom) || !this.point("pose", "right_eye", sTo)) return
        const parentWorld = this.worlds[def.parent!]
        // X = ear axis, Z = back (ear center − eye center; eyes sit forward of
        // ears), Y = cross — one basis, one decomposition, no gimbal compounding.
        Vec3.subtractInto(sA, sB, sC)
        Quat.rotateVecInvInto(parentWorld, sC, sC).normalizeInPlace() // earX in parent frame
        sDir.setXYZ(
          (sA.x + sB.x - sFrom.x - sTo.x) / 2,
          (sA.y + sB.y - sFrom.y - sTo.y) / 2,
          (sA.z + sB.z - sFrom.z - sTo.z) / 2,
        )
        Quat.rotateVecInvInto(parentWorld, sDir, sDir).normalizeInPlace() // back in parent frame
        // Gram-Schmidt earX ⊥ back, then Y = back × X
        const d = sC.dot(sDir)
        sC.setXYZ(sC.x - sDir.x * d, sC.y - sDir.y * d, sC.z - sDir.z * d).normalizeInPlace()
        Vec3.crossInto(sDir, sC, sA)
        Quat.fromBasisInto(sC, sA, sDir, out)
        return
      }
    }
  }

  /** Basis from a trunk Y axis and a raw (non-orthogonal) X axis: X ⊥ Y, Z = X×Y. */
  private static basisFromYAndX(y: Vec3, rawX: Vec3, out: Quat): void {
    const d = rawX.dot(y)
    sA.setXYZ(rawX.x - y.x * d, rawX.y - y.y * d, rawX.z - y.z * d).normalizeInPlace()
    Vec3.crossInto(sA, y, sB)
    Quat.fromBasisInto(sA, y, sB, out)
  }
}
