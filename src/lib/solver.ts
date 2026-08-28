import { Landmark, NormalizedLandmark } from "@mediapipe/tasks-vision"
import { Quat, Vec3 } from "reze-engine"
import { OneEuroFilter, QuaternionOneEuroFilter, Vec3OneEuroFilter } from "./filters"
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
  /** Only センター carries this — the body's height over the ground (and its
   *  camera distance when depth travel is on). Everything else keeps its rest
   *  translation, as MMD expects. */
  translation?: Vec3
}

/** The landmark arrays the solver reads — HolisticLandmarkerResult and the
 * worker's trimmed payload both satisfy this structurally. */
export interface SolverInput {
  poseWorldLandmarks: Landmark[][]
  leftHandWorldLandmarks: Landmark[][]
  rightHandWorldLandmarks: Landmark[][]
  /** Image-space landmarks: the projective depth rebuild reads the 2D spine.
   * Optional — without them センター keeps its depth. */
  poseLandmarks?: NormalizedLandmark[][]
  /** Frame width/height. 2D landmark x is width-normalized, so a length
   * measured across both axes needs this to be comparable. */
  imageAspect?: number
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
  name: "上半身" | "上半身2" | "下半身" | "頭"
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
  // The measured chest rotation is split evenly across 上半身∘上半身2, so the
  // spine curves instead of hinging at one joint. Everything that sits under
  // 上半身2 in a PMX (neck, clavicles, arms) is parented to it here too, so
  // the solver's chain math matches the model's.
  { kind: "basis", name: "上半身2", parent: "上半身" },
  {
    kind: "direction",
    name: "首",
    parent: "上半身2",
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

  { kind: "direction", name: "左腕", parent: "上半身2", source: "pose", from: "left_shoulder", to: "left_elbow", witness: "左ひじ" },
  { kind: "direction", name: "右腕", parent: "上半身2", source: "pose", from: "right_shoulder", to: "right_elbow", witness: "右ひじ" },
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

/** Solved geometrically rather than from landmark directions. The legs are
 *  pure FK — no IK targets, matching how the exported VMD plays back. */
const GROUNDING_BONES = ["センター"] as const
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
  上半身2: ["left_shoulder", "right_shoulder"],
  下半身: ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
  頭: ["left_ear", "right_ear", "left_eye", "right_eye"],
}

// Below this average visibility the measurement is noise (limb off-frame or
// occluded) — hold the last solved rotation instead of chasing garbage.
const MIN_VISIBILITY = 0.35

// ---------------------------------------------------------------------------
// Anatomical plausibility
//
// The detector's own confidence does not report this. Measured on footage
// where it swings a body 165° between two frames, it rates those frames 0.999
// — the same as every other frame in the take. Confidence answers "can I see
// this point", and a body reconstructed inside out is perfectly visible.
//
// What does report it is the skeleton's own geometry. The distance across the
// shoulders is a bone: it cannot change length whatever the body does, and on
// the frames where the detection collapses it drops to a quarter of what it
// was. A frame that fails that test is refused outright rather than repaired
// — its geometry is exactly what cannot be trusted to decide anything, and
// the solver already knows how to hold a pose through frames it has no
// measurement for.
// ---------------------------------------------------------------------------

/** Segments that hold their length whatever the body does. */
const RIGID_SEGMENTS: [string, string][] = [
  ["left_shoulder", "right_shoulder"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
]
/** How far a rigid segment may stray from its own running length. Measured
 *  across three real takes: never tripped on 1267 frames of sound detection,
 *  and caught the collapsed frames in the take that flips. */
const RIGID_TOLERANCE = 0.6
/** Frames of sound detection before the check has a length to judge against. */
const RIGID_WARMUP = 20

// ---------------------------------------------------------------------------
// Z-depth policy (docs/solver-improvements.md #1)
//
// MediaPipe's x/y come off the image and arrive already smoothed; its z is an
// inference, and by far the noisiest channel — front-back torso wobble is z
// noise on the shoulder/hip lines, nothing else. So z gets its own low-pass
// per landmark, and x/y pass through untouched.
// ---------------------------------------------------------------------------

const Z_SMOOTHING = { minCutoff: 1.0, beta: 1.0, dCutoff: 2.0 }

/** Focal length in image-height units (vertical FOV ≈ 43°, a typical webcam).
 *  Only scales the amplitude of toward/away motion — the direction and the
 *  proportions survive a wrong guess. */
const DEPTH_FOCAL = 1.25
/** Valid measurements averaged before the standing distance freezes (~2 s):
 *  センター depth is reported relative to where the subject started. */
const DEPTH_BASE_FRAMES = 60
/** |cos(torso pitch)| below this, the projected spine is too foreshortened to
 *  carry distance — a deep bow toward the camera holds the last depth. */
const DEPTH_MIN_COS = 0.35

// ---------------------------------------------------------------------------
// Dropout crossfades (docs/solver-improvements.md #2)
//
// A bone whose landmarks disappear used to hold its last rotation forever —
// right for one dropped frame, wrong for two seconds: limbs froze mid-air.
// Instead each bone crossfades: into tracking over 250ms, out to REST over
// 500ms, cosine-eased. A single missed frame costs ~7% of the blend and
// recovers in two, so brief dropouts still behave like a hold.
//
// Hands get hysteresis on top. MediaPipe hands flicker in and out, and a
// three-frame hand is usually garbage — so hand-sourced bones engage only
// after the hand has been continuously present for ~1s, and disengage after
// it has been gone 400ms. Within the grace window a flicker costs nothing.
// ---------------------------------------------------------------------------

const FADE_IN_MS = 250
const FADE_OUT_MS = 500
const HAND_WARMUP_MS = 1000
const HAND_GRACE_MS = 400
/** How long a bone holds its last measurement before the fade to rest even
 *  begins. Detection drops a frame here and there — a limb crossing the body,
 *  a motion blur, a frame the model simply missed — and fading from the first
 *  miss turns those into a visible dip toward bind and back. Holding first is
 *  what the fade was meant to replace only for LONG absences. */
const LOST_GRACE_MS = 250

// Gate hysteresis: a bone that is already tracking stays in until visibility
// drops WELL below the entry threshold. Without the gap, a landmark hovering
// at the threshold makes the crossfade oscillate — a visible breathing.
const VISIBILITY_EXIT = 0.25

// Roll stabilizing. Roll — rotation about the bone's own axis — is the
// noisiest channel the witness solve produces: near a straight limb the
// perpendicular lever is short, so centimetre landmark noise becomes several
// degrees of spin, and no bone DIRECTION changes — the debug skeleton looks
// clean while the model's flesh visibly shimmers. Roll is also slow in real
// motion, so its twist component tolerates a much lower cutoff than swing.
// These scale the main smoothing settings for the roll-only filter.
const ROLL_STABILIZED = new Set(["左腕", "右腕", "左足", "右足"])
const ROLL_CUTOFF_SCALE = 0.3
const ROLL_BETA_SCALE = 0.25

// ---------------------------------------------------------------------------
// Trunk decomposition (docs/solver-improvements.md #3)
// ---------------------------------------------------------------------------

/** Share of thigh pitch fed into 下半身 as posterior pelvic tilt, so sitting
 *  and crouching read as the pelvis tucking rather than a hinge at the hip. */
const PELVIS_THIGH_SHARE = 0.25
/** Share of the measured shoulder-line tilt carried by each 肩 bone. The
 *  trunk basis orthogonalizes the tilt away entirely, so without this a shrug
 *  or a one-shoulder-up move never reaches the model. */
const SHOULDER_TILT_SHARE = 0.5
const SHOULDER_TILT_MAX = 15 * DEG
/** Arm elevation over which the tilt share fades out — a raised arm's
 *  clavicle is already rotated by the rhythm; stacking the tilt on top would
 *  double-count it. */
const SHOULDER_TILT_DAMP_ANGLE = 90 * DEG

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
  "センター",
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
// Scratch for the current frame's measurement, before the dropout crossfade
// decides how much of it reaches the bone.
const sMeas = Quat.identity()

const landmarkBuf = (n: number): Landmark[] =>
  Array.from({ length: n }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }))
const zBank = (n: number): OneEuroFilter[] =>
  Array.from({ length: n }, () => new OneEuroFilter(Z_SMOOTHING.minCutoff, Z_SMOOTHING.beta, Z_SMOOTHING.dCutoff))

export class Solver {
  private pose: Landmark[] | null = null
  private leftHand: Landmark[] | null = null
  private rightHand: Landmark[] | null = null
  private pose2d: NormalizedLandmark[] | null = null
  private imageAspect = 16 / 9

  // Landmarks are copied here with z filtered, so the caller's arrays (which
  // also feed the debug preview) keep showing what MediaPipe actually said.
  private poseBuf = landmarkBuf(33)
  private leftHandBuf = landmarkBuf(21)
  private rightHandBuf = landmarkBuf(21)
  private zFilters: Record<LandmarkSource, OneEuroFilter[]> = {
    pose: zBank(33),
    leftHand: zBank(21),
    rightHand: zBank(21),
  }
  /** Running length of each rigid segment, and how many sound frames have
   *  contributed to it. */
  private rigidLength: number[] = RIGID_SEGMENTS.map(() => 0)
  private rigidSeen = 0
  /** Anatomical plausibility gate; disable to solve every frame as given. */
  plausibilityEnabled = true
  /** Whether each bank carries state — a source that dropped out re-seeds its
   *  filters on return instead of easing z across the gap. */
  private zActive: Record<LandmarkSource, boolean> = { pose: false, leftHand: false, rightHand: false }
  /** Per-landmark z low-pass; disable to hand the solver MediaPipe's raw z. */
  zFilterEnabled = true
  /** Projective センター depth. Off — mocap is in-place: the 2D spine estimate
   *  drifts over a session and slowly walks the model off its mark, so the
   *  rebuild is opt-in for whoever wants toward/away travel despite that. */
  depthEnabled = false
  /** Lateral センター travel from the 2D hip midpoint. Off — mocap is fully
   *  in-place: センター moves only vertically (squats, crouches, via
   *  grounding). Captures are an editing base for artists, and root travel
   *  that wanders is worse to clean up than root travel that is absent. */
  swayEnabled = false
  /** Model spine length (shoulder-line centre to hip-line centre), the known
   *  length the projective depth rebuild scales against. From calibrate(). */
  private spineLen = 0
  /** Running means over the first valid measurements — the "standing spot"
   *  both axes are reported relative to. */
  private depthBase = 0
  private depthBaseX = 0
  private depthBaseFrames = 0
  /** Last committed depth/lateral offsets, held through unmeasurable frames
   *  (deep bows, off-frame torso). Model units, before groundingGain. */
  private heldDz = 0
  private heldDx = 0
  /** This frame's センター displacement, written by measureRootShift(). */
  private rootDx = 0
  private rootDz = 0

  /** Unfiltered parent-local rotation per bone, after the dropout crossfade. */
  private locals: Record<string, Quat> = {}
  /** Last MEASURED local per bone (hemisphere w≥0) — what the crossfade blends
   *  between rest and, held through dropouts while the fade runs out. */
  private heldMeasured: Record<string, Quat> = {}
  /** Tracking blend per bone: 0 = at rest, 1 = fully on the measurement. */
  private fades: Record<string, number> = {}
  /** How long each bone has gone unmeasured, for the grace window. */
  private lostMs: Record<string, number> = {}
  private fadePrevTs: number | null = null
  /** Hand hysteresis state (see HAND_WARMUP_MS / HAND_GRACE_MS). */
  private handEngagement = {
    leftHand: { seen: 0, gone: 0, engaged: false },
    rightHand: { seen: 0, gone: 0, engaged: false },
  }
  /** Full measured chest rotation and its half, staged by the 上半身 solve for
   *  the 上半身2 def that runs right after it. */
  private chestHalf = Quat.identity()
  private chestMeasuredFrame = false
  /** Whether the calibrated model has an 上半身2 to take its half; without one
   *  the whole chest rotation stays on 上半身. */
  private hasUpperBody2 = true
  /** Shoulder-line tilt (radians, + = left shoulder up), for the 肩 share. */
  private shoulderTilt = 0
  /** Extra-calm scalar filters on the roll channel (see ROLL_STABILIZED). */
  private rollFilters: Record<string, OneEuroFilter> = {}
  /** Last filtered roll angle per bone, for ±π unwrapping. */
  private prevRoll: Record<string, number> = {}
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
  //
  // beta 1.5: a trial at 3 measured ~8ms less filter lag but read as jitter by
  // eye — and the eye is the instrument this number was tuned on.
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
    // Roll filters scale off the same settings.
    this.rollFilters = {}
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
      this.heldMeasured[def.name] = Quat.identity()
      this.fades[def.name] = 0
      this.lostMs[def.name] = 0
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
    this.heldDz = 0
    this.heldDx = 0
    this.depthBase = 0
    this.depthBaseX = 0
    this.depthBaseFrames = 0
    this.fadePrevTs = null
    this.rigidLength = RIGID_SEGMENTS.map(() => 0)
    this.rigidSeen = 0
    this.chestMeasuredFrame = false
    this.shoulderTilt = 0
    for (const key of Object.keys(this.rollFilters)) this.rollFilters[key].reset()
    this.prevRoll = {}
    for (const key of Object.keys(this.fades)) {
      this.fades[key] = 0
      this.lostMs[key] = 0
      this.heldMeasured[key].setIdentity()
    }
    for (const side of ["leftHand", "rightHand"] as const) {
      const h = this.handEngagement[side]
      h.seen = 0
      h.gone = 0
      h.engaged = false
    }
    for (const source of ["pose", "leftHand", "rightHand"] as const) {
      for (const f of this.zFilters[source]) f.reset()
      this.zActive[source] = false
    }
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
    // Shoulder-line tilt share: the trunk basis orthogonalizes the tilt away,
    // so the clavicles are the only place a shrug can live. Faded with the
    // chest's own tracking blend.
    const tilt = this.shoulderTilt * (this.fades["上半身"] ?? 0)
    for (const side of ["左", "右"] as const) {
      const armLocal = this.locals[side + "腕"]
      const shoulderLocal = this.locals[side + "肩"]
      if (!armLocal || !shoulderLocal) continue

      const w = Math.min(1, Math.abs(armLocal.w))
      const angle = 2 * Math.acos(w)

      // Elevation share (scapulohumeral rhythm), ramped past the threshold.
      shoulderLocal.setIdentity()
      if (this.shoulderRatio > 0 && angle > this.shoulderStart) {
        const sin = Math.sqrt(Math.max(0, 1 - w * w))
        if (sin >= 1e-6) {
          // armLocal's sign convention follows w; keep the axis on the same side.
          const flip = armLocal.w < 0 ? -1 : 1
          const ax = (armLocal.x * flip) / sin
          const ay = (armLocal.y * flip) / sin
          const az = (armLocal.z * flip) / sin
          const ramp = Math.min(1, (angle - this.shoulderStart) / this.shoulderRamp)
          let take = (angle - this.shoulderStart) * this.shoulderRatio * ramp
          if (take > this.shoulderMax) take = this.shoulderMax
          Quat.fromAxisAngleInto(ax, ay, az, take, shoulderLocal)
        }
      }

      // Tilt share, damped as the arm rises — a raised arm's clavicle is
      // already rotated by the elevation share above.
      let tiltSide = tilt * SHOULDER_TILT_SHARE * Math.max(0, 1 - angle / SHOULDER_TILT_DAMP_ANGLE)
      if (tiltSide > SHOULDER_TILT_MAX) tiltSide = SHOULDER_TILT_MAX
      else if (tiltSide < -SHOULDER_TILT_MAX) tiltSide = -SHOULDER_TILT_MAX
      if (Math.abs(tiltSide) > 1e-4) {
        Quat.fromAxisAngleInto(0, 0, 1, tiltSide, _clearC)
        Quat.multiplyInto(_clearC, shoulderLocal, _clearA)
        shoulderLocal.set(_clearA)
      }

      if (shoulderLocal.w > 1 - 1e-9) continue
      // Remove exactly what the shoulder took, so the arm still points where the
      // landmarks say it does.
      Quat.conjugateInto(shoulderLocal, _clearA)
      Quat.multiplyInto(_clearA, armLocal, _clearB)
      armLocal.set(_clearB)
    }
  }

  /**
   * Place the body in space: how high the hips ride.
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
   * The legs themselves stay pure FK: the chain walk below only asks where the
   * solved rotations put each ankle, so the body can be dropped until the lower
   * one rests on the floor. What this cannot do is leave the ground — both feet
   * airborne is indistinguishable from standing, in hip-centred coordinates.
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
      ankleY[side] = hip.y + _gA.y + _gB.y
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

    this.measureRootShift()
    const rawDx = this.rootDx * this.groundingGain
    const rawDz = this.rootDz * this.groundingGain

    let mf = this.moveFilters["センター"]
    if (!mf) {
      mf = new Vec3OneEuroFilter(this.smoothing.minCutoff, this.smoothing.beta, this.smoothing.dCutoff)
      this.moveFilters["センター"] = mf
    }
    if (unfiltered) {
      mf.reset()
      center.translation.setXYZ(rawDx, rawDy, rawDz)
    } else {
      mf.filterInto(rawDx, rawDy, rawDz, timestampMs, center.translation)
    }
  }

  /**
   * Whether this frame's skeleton could belong to a body, and fold it into the
   * running lengths if so.
   *
   * Returns false for a frame the detector got wrong in a way its confidence
   * does not admit to.
   */
  private plausible(pose: Landmark[]): boolean {
    const lengths: number[] = []
    for (const [a, b] of RIGID_SEGMENTS) {
      const p = pose[PoseLandmarksTable[a]]
      const q = pose[PoseLandmarksTable[b]]
      if (!p || !q) return true
      lengths.push(Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z))
    }
    if (this.rigidSeen >= RIGID_WARMUP) {
      for (let i = 0; i < lengths.length; i++) {
        const ref = this.rigidLength[i]
        if (ref <= 1e-6) continue
        const ratio = lengths[i] / ref
        if (ratio < RIGID_TOLERANCE || ratio > 1 / RIGID_TOLERANCE) return false
      }
    }
    // Only sound frames shape the reference, or a run of bad ones would teach
    // the check to accept them.
    this.rigidSeen++
    const w = this.rigidSeen <= RIGID_WARMUP ? 1 / this.rigidSeen : 0.02
    for (let i = 0; i < lengths.length; i++) {
      this.rigidLength[i] += (lengths[i] - this.rigidLength[i]) * w
    }
    return true
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
    this.hasUpperBody2 = !!restWorldPos["上半身2"]

    // Spine length for the projective depth rebuild: shoulder-line centre to
    // hip-line centre, the same segment the 2D measurement spans (the arm and
    // leg roots sit at the joints MediaPipe's landmarks mark).
    const la = restWorldPos["左腕"]
    const ra = restWorldPos["右腕"]
    const ll = restWorldPos["左足"]
    const rl = restWorldPos["右足"]
    if (la && ra && ll && rl) {
      this.spineLen = Math.hypot(
        (la.x + ra.x - ll.x - rl.x) / 2,
        (la.y + ra.y - ll.y - rl.y) / 2,
        (la.z + ra.z - ll.z - rl.z) / 2,
      )
    }

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
    const filterZ = this.zFilterEnabled && !unfiltered
    // Optional chaining throughout: a set can be present, absent, or present
    // and empty, and the solver is not the place to find that out the hard way.
    this.pose = this.intake(
      landmarks.poseWorldLandmarks?.[0]?.length === 33 ? landmarks.poseWorldLandmarks[0] : null,
      "pose", this.poseBuf, timestampMs, filterZ,
    )
    this.leftHand = this.intake(
      landmarks.leftHandWorldLandmarks?.[0]?.length === 21 ? landmarks.leftHandWorldLandmarks[0] : null,
      "leftHand", this.leftHandBuf, timestampMs, filterZ,
    )
    this.rightHand = this.intake(
      landmarks.rightHandWorldLandmarks?.[0]?.length === 21 ? landmarks.rightHandWorldLandmarks[0] : null,
      "rightHand", this.rightHandBuf, timestampMs, filterZ,
    )
    // A frame that fails the shape test is treated as one the detector did not
    // produce: every pose-driven bone holds, and the crossfades take it from
    // there.
    if (this.pose && this.plausibilityEnabled && !this.plausible(this.pose)) this.pose = null
    this.pose2d = landmarks.poseLandmarks?.[0]?.length === 33 ? landmarks.poseLandmarks[0] : null
    if (landmarks.imageAspect) this.imageAspect = landmarks.imageAspect

    // Crossfade clock (media time, so conversions pace identically to live).
    // Capped per tick: a detection stall must not hand one frame a huge step —
    // MediaPipe often whiffs its first frame after a gap, and an uncapped step
    // let that single miss slam a tracked limb to near-rest before the very
    // next frame pulled it back (the "snaps to bind for an instant" bug).
    let fadeDt = 33.3
    if (this.fadePrevTs !== null) {
      const d = timestampMs - this.fadePrevTs
      if (d > 0) fadeDt = Math.min(d, 100)
    }
    this.fadePrevTs = timestampMs
    for (const side of ["leftHand", "rightHand"] as const) {
      const h = this.handEngagement[side]
      if (this.handConfidence(side) >= MIN_VISIBILITY) {
        h.seen += fadeDt
        h.gone = 0
        if (h.seen >= HAND_WARMUP_MS) h.engaged = true
      } else {
        h.gone += fadeDt
        if (h.gone >= HAND_GRACE_MS) {
          h.engaged = false
          h.seen = 0
        }
      }
    }

    this.chestMeasuredFrame = false
    for (const def of BONE_DEFS) {
      const local = this.locals[def.name]
      if (def.kind === "fingerRatio") {
        // Derived joints read their base's already-faded local, so they follow
        // its crossfade for free.
        this.solveFingerRatio(def, local)
        continue
      }
      let measured: boolean
      switch (def.kind) {
        case "basis":
          measured = this.solveBasis(def, sMeas)
          break
        case "direction":
          measured = this.solveDirection(def, sMeas)
          break
        default:
          measured = this.solveTwist(def, sMeas)
          break
      }
      // A present-but-unengaged hand is treated as absent (see hysteresis note).
      if (measured && def.kind !== "basis" && def.source !== "pose" && !this.handEngagement[def.source].engaged) {
        measured = false
      }
      const held = this.heldMeasured[def.name]
      if (measured) {
        if (sMeas.w < 0) sMeas.setXYZW(-sMeas.x, -sMeas.y, -sMeas.z, -sMeas.w)
        held.set(sMeas)
      }
      let fade = this.fades[def.name]
      let lost = this.lostMs[def.name]
      lost = measured ? 0 : lost + fadeDt
      this.lostMs[def.name] = lost
      // An engaged hand inside its own grace window holds too — a flicker
      // costs nothing.
      const graceHold =
        lost < LOST_GRACE_MS ||
        (def.kind !== "basis" && def.source !== "pose" && this.handEngagement[def.source].engaged)
      if (unfiltered) fade = measured ? 1 : 0
      else if (measured) fade = Math.min(1, fade + fadeDt / FADE_IN_MS)
      else if (!graceHold) fade = Math.max(0, fade - fadeDt / FADE_OUT_MS)
      this.fades[def.name] = fade
      const w = 0.5 - 0.5 * Math.cos(Math.PI * fade)
      if (w >= 1) local.set(held)
      else if (w <= 0) local.setIdentity()
      else {
        // nlerp(identity, held, w); held is hemisphere-aligned to identity.
        local.setXYZW(held.x * w, held.y * w, held.z * w, held.w * w + (1 - w))
        local.normalize()
      }
      if (ROLL_STABILIZED.has(def.name)) this.stabilizeRoll(def.name, local, timestampMs, unfiltered)
      const world = this.worlds[def.name]
      const parent = def.parent ? this.worlds[def.parent] : null
      if (parent) Quat.multiplyInto(parent, local, world)
      else world.set(local)
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

  /** Copy a landmark array into its buffer with the z channel low-passed. */
  private intake(
    src: Landmark[] | null,
    source: LandmarkSource,
    buf: Landmark[],
    ts: number,
    filterZ: boolean,
  ): Landmark[] | null {
    const bank = this.zFilters[source]
    const active = filterZ && src !== null
    if (!active && this.zActive[source]) for (const f of bank) f.reset()
    this.zActive[source] = active
    if (!src) return null
    for (let i = 0; i < buf.length; i++) {
      const s = src[i]
      const d = buf[i]
      d.x = s.x
      d.y = s.y
      d.visibility = s.visibility
      d.z = active ? bank[i].filter(s.z, ts) : s.z
    }
    return buf
  }

  /**
   * センター displacement, rebuilt from 2D projective geometry.
   *
   * MediaPipe's world landmarks are hip-centred: the pelvis is the ORIGIN, so
   * pelvis translation — hip sway, a side-step, walking toward the camera — is
   * exactly the signal that coordinate system deletes. The 2D projection still
   * carries all of it.
   *
   * Lateral: the hip midpoint's position in the frame, converted to model
   * units by spineLen / projected-spine — the focal length cancels, so
   * side-to-side motion needs no camera guess and doesn't drift.
   *
   * Depth: the projected spine shrinks as the subject steps back, by the
   * pinhole ratio; dividing by |cos(torso pitch)| first undoes foreshortening
   * so a bow doesn't read as walking away. (Off by default — the length
   * estimate drifts over a session.)
   *
   * Both are offsets from the standing baseline (running mean of the first
   * ~2s), in model units, written to rootDx / rootDz. Negative z is toward
   * the viewer.
   */
  private measureRootShift(): void {
    this.rootDx = this.swayEnabled ? this.heldDx : 0
    this.rootDz = this.depthEnabled ? this.heldDz : 0
    if ((!this.swayEnabled && !this.depthEnabled) || this.spineLen <= 0) return
    const lm = this.pose2d
    const pose = this.pose
    if (!lm || !pose) return
    const ls = lm[PoseLandmarksTable.left_shoulder]
    const rs = lm[PoseLandmarksTable.right_shoulder]
    const lh = lm[PoseLandmarksTable.left_hip]
    const rh = lm[PoseLandmarksTable.right_hip]
    if (!ls || !rs || !lh || !rh) return
    if (Math.min(ls.visibility ?? 1, rs.visibility ?? 1, lh.visibility ?? 1, rh.visibility ?? 1) < MIN_VISIBILITY)
      return

    // Projected spine, in image-height units (x is width-normalized).
    const px = ((ls.x + rs.x - lh.x - rh.x) / 2) * this.imageAspect
    const py = (ls.y + rs.y - lh.y - rh.y) / 2
    const len2d = Math.hypot(px, py)
    if (len2d < 1e-4) return

    // Torso pitch out of the image plane, from the world landmarks (their z
    // has already been through the z low-pass, so this is a calm signal).
    const wls = pose[PoseLandmarksTable.left_shoulder]
    const wrs = pose[PoseLandmarksTable.right_shoulder]
    const wlh = pose[PoseLandmarksTable.left_hip]
    const wrh = pose[PoseLandmarksTable.right_hip]
    if (!wls || !wrs || !wlh || !wrh) return
    const sx = (wls.x + wrs.x - wlh.x - wrh.x) / 2
    const sy = (wls.y + wrs.y - wlh.y - wrh.y) / 2
    const sz = (wls.z + wrs.z - wlh.z - wrh.z) / 2
    const s3d = Math.hypot(sx, sy, sz)
    if (s3d < 1e-6) return
    const cos = Math.sqrt(Math.max(0, 1 - (sz / s3d) ** 2))
    if (cos < DEPTH_MIN_COS) return

    // Model units per image-height unit at the subject's distance (= D/f).
    const scale = (this.spineLen * cos) / len2d
    const hipX = (((lh.x + rh.x) / 2) * this.imageAspect) * scale
    const dist = scale * DEPTH_FOCAL
    if (this.depthBaseFrames < DEPTH_BASE_FRAMES) {
      this.depthBaseFrames++
      this.depthBase += (dist - this.depthBase) / this.depthBaseFrames
      this.depthBaseX += (hipX - this.depthBaseX) / this.depthBaseFrames
    }
    // One bad 2D frame must not send the model across the stage.
    const lim = this.spineLen * 3
    const clamp = (v: number) => (v > lim ? lim : v < -lim ? -lim : v)
    this.heldDx = clamp(hipX - this.depthBaseX)
    this.heldDz = clamp(dist - this.depthBase)
    this.rootDx = this.swayEnabled ? this.heldDx : 0
    this.rootDz = this.depthEnabled ? this.heldDz : 0
  }

  private getRef(key: string): Vec3 {
    return this.refs[key] ?? DEFAULT_REFS[key]
  }

  /** Tracking-gate threshold with hysteresis: a bone already tracking stays
   *  in down to VISIBILITY_EXIT, so a landmark hovering at the entry
   *  threshold cannot make the crossfade oscillate. */
  private visGate(name: string): number {
    return (this.fades[name] ?? 0) >= 0.5 ? VISIBILITY_EXIT : MIN_VISIBILITY
  }

  /**
   * Swing-twist decompose the local about the bone's own rest axis, pass the
   * twist angle through its extra-calm scalar filter, and recompose on the
   * right — the swing (bone direction) is untouched, only the spin about the
   * axis is steadied.
   */
  private stabilizeRoll(name: string, local: Quat, ts: number, unfiltered: boolean): void {
    const axis = this.getRef(name)
    Quat.twistAroundAxisInto(local, axis, sQ)
    const k = sQ.x * axis.x + sQ.y * axis.y + sQ.z * axis.z
    let angle = 2 * Math.atan2(k, sQ.w)
    if (angle > Math.PI) angle -= 2 * Math.PI
    else if (angle < -Math.PI) angle += 2 * Math.PI
    // Unwrap against the last output so a roll crossing ±π doesn't feed the
    // filter a fake 2π step.
    const prev = this.prevRoll[name]
    if (prev !== undefined) {
      while (angle - prev > Math.PI) angle -= 2 * Math.PI
      while (angle - prev < -Math.PI) angle += 2 * Math.PI
    }
    let f = this.rollFilters[name]
    if (!f) {
      f = new OneEuroFilter(
        this.smoothing.minCutoff * ROLL_CUTOFF_SCALE,
        this.smoothing.beta * ROLL_BETA_SCALE,
        this.smoothing.dCutoff,
      )
      this.rollFilters[name] = f
    }
    let filtered = angle
    if (unfiltered) f.reset()
    else filtered = f.filter(angle, ts)
    this.prevRoll[name] = filtered
    // swing = local ∘ twist⁻¹, then recompose with the steadied twist.
    sQ.conjugate()
    Quat.multiplyInto(local, sQ, sQ2)
    Quat.fromAxisAngleInto(axis.x, axis.y, axis.z, filtered, sQ)
    Quat.multiplyInto(sQ2, sQ, local)
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

  /** Returns whether a measurement was written into `out`. */
  private solveDirection(def: DirectionDef, out: Quat): boolean {
    const from = this.point(def.source, def.from, sFrom)
    const to = this.point(def.source, def.to, sTo)
    if (!from || !to) return false
    if (this.visibility(def.source, [def.from, def.to]) < this.visGate(def.name)) return false

    Vec3.subtractInto(to, from, sDir)
    const parentWorld = def.parent ? this.worlds[def.parent] : null
    if (parentWorld) Quat.rotateVecInvInto(parentWorld, sDir, sDir)
    if (sDir.length() < 1e-6) return false
    sDir.normalizeInPlace()

    Quat.fromUnitVectorsInto(this.getRef(def.name), sDir, out)

    if (def.witness && this.witnessEnabled) this.applyWitness(def, parentWorld, out)
    if (def.bend && this.bendClampEnabled) Solver.clampBend(def.bend, out)
    return true
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

  /** Returns whether a measurement was written into `out`. */
  private solveTwist(def: TwistDef, out: Quat): boolean {
    const from = this.point(def.source, def.from, sFrom)
    const to = this.point(def.source, def.to, sTo)
    if (!from || !to) return false
    if (this.visibility(def.source, [def.from, def.to]) < this.visGate(def.name)) return false

    Vec3.subtractInto(to, from, sDir)
    Quat.rotateVecInvInto(this.worlds[def.parent], sDir, sDir)
    if (sDir.length() < 1e-6) return false
    sDir.normalizeInPlace()

    // Total rotation aligning the rest hand axis to the live one includes the
    // wrist swing; project onto the forearm axis to keep only the twist.
    Quat.fromUnitVectorsInto(this.getRef(def.name), sDir, sQ)
    Quat.twistAroundAxisInto(sQ, this.getRef(def.axisRef), out)
    return true
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

  /** Returns whether a measurement was written into `out`. */
  private solveBasis(def: BasisDef, out: Quat): boolean {
    if (!this.pose) return false
    if (this.visibility("pose", BASIS_LANDMARKS[def.name]) < this.visGate(def.name)) return false

    switch (def.name) {
      case "上半身": {
        if (!this.point("pose", "left_shoulder", sA) || !this.point("pose", "right_shoulder", sB)) return false
        // spineY = shoulder center (pose world origin is the hip center)
        sDir.setXYZ((sA.x + sB.x) / 2, (sA.y + sB.y) / 2, (sA.z + sB.z) / 2).normalizeInPlace()
        Vec3.subtractInto(sA, sB, sC).normalizeInPlace()
        // Shoulder-line tilt, read BEFORE the basis orthogonalizes it away —
        // the clavicles carry it (applyShoulderRhythm), the trunk cannot.
        const lean = sC.dot(sDir)
        this.shoulderTilt = Math.asin(Math.max(-1, Math.min(1, lean)))
        Solver.basisFromYAndX(sDir, sC, out)
        if (out.w < 0) out.setXYZW(-out.x, -out.y, -out.z, -out.w)
        this.chestMeasuredFrame = true
        if (this.hasUpperBody2) {
          // Split the chest rotation evenly across 上半身∘上半身2 so the spine
          // curves. normalize(I + R) is exactly R^½, and R^½∘R^½ = R.
          this.chestHalf.setXYZW(out.x, out.y, out.z, out.w + 1)
          this.chestHalf.normalize()
          out.set(this.chestHalf)
        }
        return true
      }
      case "上半身2": {
        if (!this.hasUpperBody2 || !this.chestMeasuredFrame) return false
        out.set(this.chestHalf)
        return true
      }
      case "下半身": {
        if (!this.point("pose", "left_shoulder", sA) || !this.point("pose", "right_shoulder", sB)) return false
        sFrom.setXYZ((sA.x + sB.x) / 2, (sA.y + sB.y) / 2, (sA.z + sB.z) / 2)
        if (!this.point("pose", "left_hip", sA) || !this.point("pose", "right_hip", sB)) return false
        sTo.setXYZ((sA.x + sB.x) / 2, (sA.y + sB.y) / 2, (sA.z + sB.z) / 2)
        // Pelvis basis shares the trunk Y with 上半身 (no separate pelvis-tilt
        // landmark exists); lower/upper differ in X (hip vs shoulder line),
        // which captures twist.
        Vec3.subtractInto(sFrom, sTo, sDir).normalizeInPlace()
        Vec3.subtractInto(sA, sB, sC).normalizeInPlace()
        Solver.basisFromYAndX(sDir, sC, out)
        this.applyPelvisTuck(out)
        return true
      }
      case "頭": {
        if (!this.point("pose", "left_ear", sA) || !this.point("pose", "right_ear", sB)) return false
        if (!this.point("pose", "left_eye", sFrom) || !this.point("pose", "right_eye", sTo)) return false
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
        return true
      }
    }
    return false
  }

  /**
   * Sitting and crouching read as the pelvis tucking under, not a 90° hinge
   * at the hip: feed a share of the mean thigh pitch into 下半身 as posterior
   * tilt. Measured in the pelvis' own frame, so a body lying flat (thighs in
   * line with the trunk) adds nothing.
   */
  private applyPelvisTuck(out: Quat): void {
    let sum = 0
    let n = 0
    for (const side of ["left", "right"] as const) {
      if (this.visibility("pose", [`${side}_hip`, `${side}_knee`]) < MIN_VISIBILITY) continue
      const hip = this.point("pose", `${side}_hip`, sFrom)
      if (!hip) continue
      const knee = this.point("pose", `${side}_knee`, sTo)
      if (!knee) continue
      Vec3.subtractInto(sTo, sFrom, sWit)
      Quat.rotateVecInvInto(out, sWit, sWit)
      if (sWit.length() < 1e-6) continue
      sWit.normalizeInPlace()
      // 0 = thigh straight down, +π/2 = thigh forward (model −Z).
      sum += Math.atan2(-sWit.z, -sWit.y)
      n++
    }
    if (n === 0) return
    let extra = (sum / n) * PELVIS_THIGH_SHARE
    if (extra > 30 * DEG) extra = 30 * DEG
    else if (extra < -10 * DEG) extra = -10 * DEG
    if (Math.abs(extra) < 1e-4) return
    // Post-multiply = rotation about the pelvis' own X: posterior tilt.
    Quat.fromAxisAngleInto(1, 0, 0, extra, sQ)
    Quat.multiplyInto(out, sQ, sQ2)
    out.set(sQ2)
  }

  /** Basis from a trunk Y axis and a raw (non-orthogonal) X axis: X ⊥ Y, Z = X×Y. */
  private static basisFromYAndX(y: Vec3, rawX: Vec3, out: Quat): void {
    const d = rawX.dot(y)
    sA.setXYZ(rawX.x - y.x * d, rawX.y - y.y * d, rawX.z - y.z * d).normalizeInPlace()
    Vec3.crossInto(sA, y, sB)
    Quat.fromBasisInto(sA, y, sB, out)
  }
}
