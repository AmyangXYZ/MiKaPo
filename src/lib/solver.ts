import { Landmark } from "@mediapipe/tasks-vision"
import { Quat, Vec3 } from "reze-engine"
import { QuaternionOneEuroFilter } from "./filters"
import { HandIndexTable, PoseLandmarksTable } from "./landmarks"
import { quatFromBasis, quatFromUnitVectors, quatNlerp, quatDot, quatTwistAroundAxis, rotateVecInv } from "./math-utils"

export interface BoneState {
  name: string
  rotation: Quat
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

  { kind: "direction", name: "左足", parent: "下半身", source: "pose", from: "left_hip", to: "left_knee", witness: "左ひざ" },
  { kind: "direction", name: "右足", parent: "下半身", source: "pose", from: "right_hip", to: "right_knee", witness: "右ひざ" },
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

export class Solver {
  private pose: Landmark[] | null = null
  private leftHand: Landmark[] | null = null
  private rightHand: Landmark[] | null = null

  /** Unfiltered parent-local rotation per bone; doubles as the held value on dropout. */
  private locals: Record<string, Quat> = {}
  /** Accumulated world rotation per bone (parent chain product), rebuilt each frame. */
  private worlds: Record<string, Quat> = {}
  private filters: Record<string, QuaternionOneEuroFilter> = {}
  // One-Euro tuning: minCutoff governs rest-pose jitter suppression (lower =
  // calmer, laggier at rest); beta governs how fast the cutoff opens with
  // speed (higher = fast/dramatic moves track tighter with less lag and less
  // amplitude loss). Rest stability and motion tracking are tuned independently.
  private smoothing = { minCutoff: 1.5, beta: 1.5, dCutoff: 1.0 }
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
      return state
    })
  }

  reset(): void {
    for (const key of Object.keys(this.filters)) this.filters[key].reset()
    for (const key of Object.keys(this.locals)) this.locals[key].setIdentity()
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
  solve(landmarks: SolverInput, timestampMs: number = performance.now()): BoneState[] {
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

    // One-Euro post-pass on the outputs only — the hierarchy above always
    // composes unfiltered locals, so parent-chain math stays exact.
    for (const def of BONE_DEFS) {
      let f = this.filters[def.name]
      if (!f) {
        f = new QuaternionOneEuroFilter(this.smoothing.minCutoff, this.smoothing.beta, this.smoothing.dCutoff)
        this.filters[def.name] = f
      }
      f.filterInto(this.locals[def.name], timestampMs, this.outputByName[def.name].rotation)
    }

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
  private visibility(source: LandmarkSource, points: Point[]): number {
    if (source !== "pose" || !this.pose) return 1
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
    if (parentWorld) rotateVecInv(parentWorld, sDir, sDir)
    if (sDir.length() < 1e-6) return
    sDir.normalizeInPlace()

    quatFromUnitVectors(this.getRef(def.name), sDir, out)

    if (def.witness && this.witnessEnabled) this.applyWitness(def, parentWorld, out)
    if (def.bend && this.bendClampEnabled) Solver.clampBend(def.bend, out)
  }

  /**
   * Clamp a joint rotation to anatomical range: decompose q = swing ∘ twist
   * about the flexion axis, clamp the signed twist (flexion) angle and the
   * swing (spread) magnitude, and recompose.
   */
  private static clampBend(bend: BendLimit, q: Quat): void {
    quatTwistAroundAxis(q, bend.axis, sQ) // twist
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
    const wdef = DEF_BY_NAME[def.witness!] as DirectionDef
    const wFrom = this.point(wdef.source, wdef.from, sA)
    const wTo = this.point(wdef.source, wdef.to, sB)
    if (!wFrom || !wTo) return
    if (this.visibility(wdef.source, [wdef.from, wdef.to]) < MIN_VISIBILITY) return

    Vec3.subtractInto(wTo, wFrom, sWit)
    if (parentWorld) rotateVecInv(parentWorld, sWit, sWit)
    if (sWit.length() < 1e-6) return
    sWit.normalizeInPlace()

    // Live witness component perpendicular to the live bone direction (sDir
    // still holds it). Its magnitude is the observability of the roll.
    const dLive = sWit.dot(sDir)
    sA.setXYZ(sWit.x - sDir.x * dLive, sWit.y - sDir.y * dLive, sWit.z - sDir.z * dLive)
    const perpLen = sA.length()
    if (perpLen < WITNESS_FADE_LO) return
    sA.normalizeInPlace()

    // Rest witness component perpendicular to the rest bone direction.
    const ref = this.getRef(def.name)
    const restWit = WITNESS_REST[def.name]
    if (!restWit) return
    const dRest = restWit.dot(ref)
    sB.setXYZ(restWit.x - ref.x * dRest, restWit.y - ref.y * dRest, restWit.z - ref.z * dRest)
    if (sB.length() < 1e-3) return
    sB.normalizeInPlace()

    // q = basisLive ∘ basisRest⁻¹ maps (ref → dir, restWitness⊥ → liveWitness⊥).
    Vec3.crossInto(ref, sB, sC)
    quatFromBasis(ref, sB, sC, sQ) // basisRest
    sQ.conjugate()
    Vec3.crossInto(sDir, sA, sC)
    quatFromBasis(sDir, sA, sC, sQ2) // basisLive
    Quat.multiplyInto(sQ2, sQ, sQ) // apply rest⁻¹ first, then live

    // Fade between shortest-arc (straight limb, roll unobservable) and the
    // witness solution, hemisphere-aligned so the blend is short-path.
    const t = Math.min(1, Math.max(0, (perpLen - WITNESS_FADE_LO) / (WITNESS_FADE_HI - WITNESS_FADE_LO)))
    if (quatDot(out, sQ) < 0) sQ.setXYZW(-sQ.x, -sQ.y, -sQ.z, -sQ.w)
    quatNlerp(out, sQ, t * t * (3 - 2 * t), out)
  }

  private solveTwist(def: TwistDef, out: Quat): void {
    const from = this.point(def.source, def.from, sFrom)
    const to = this.point(def.source, def.to, sTo)
    if (!from || !to) return

    Vec3.subtractInto(to, from, sDir)
    rotateVecInv(this.worlds[def.parent], sDir, sDir)
    if (sDir.length() < 1e-6) return
    sDir.normalizeInPlace()

    // Total rotation aligning the rest hand axis to the live one includes the
    // wrist swing; project onto the forearm axis to keep only the twist.
    quatFromUnitVectors(this.getRef(def.name), sDir, sQ)
    quatTwistAroundAxis(sQ, this.getRef(def.axisRef), out)
  }

  private solveFingerRatio(def: FingerRatioDef, out: Quat): void {
    const base = this.locals[def.base]
    const bendDegrees = Solver.extractBendDegrees(base, def.bendAxis)
    const radians = (bendDegrees * def.ratio * Math.PI) / 180
    Quat.fromAxisAngleInto(def.bendAxis.x, def.bendAxis.y, def.bendAxis.z, radians, out)
  }

  private static extractBendDegrees(quat: Quat, bendAxis: Vec3): number {
    const totalAngle = 2 * Math.acos(Math.min(1, Math.abs(quat.w))) * (180 / Math.PI)
    const axisComponent = quat.x * bendAxis.x + quat.y * bendAxis.y + quat.z * bendAxis.z
    return axisComponent < 0 ? -totalAngle : totalAngle
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
        rotateVecInv(parentWorld, sC, sC).normalizeInPlace() // earX in parent frame
        sDir.setXYZ(
          (sA.x + sB.x - sFrom.x - sTo.x) / 2,
          (sA.y + sB.y - sFrom.y - sTo.y) / 2,
          (sA.z + sB.z - sFrom.z - sTo.z) / 2,
        )
        rotateVecInv(parentWorld, sDir, sDir).normalizeInPlace() // back in parent frame
        // Gram-Schmidt earX ⊥ back, then Y = back × X
        const d = sC.dot(sDir)
        sC.setXYZ(sC.x - sDir.x * d, sC.y - sDir.y * d, sC.z - sDir.z * d).normalizeInPlace()
        Vec3.crossInto(sDir, sC, sA)
        quatFromBasis(sC, sA, sDir, out)
        return
      }
    }
  }

  /** Basis from a trunk Y axis and a raw (non-orthogonal) X axis: X ⊥ Y, Z = X×Y. */
  private static basisFromYAndX(y: Vec3, rawX: Vec3, out: Quat): void {
    const d = rawX.dot(y)
    sA.setXYZ(rawX.x - y.x * d, rawX.y - y.y * d, rawX.z - y.z * d).normalizeInPlace()
    Vec3.crossInto(sA, y, sB)
    quatFromBasis(sA, y, sB, out)
  }
}
