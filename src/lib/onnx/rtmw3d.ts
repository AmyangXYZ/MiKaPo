// RTMW3D (mmpose projects/rtmpose3d) decode & geometry — the math only.
//
// The model is top-down: one person crop, affine-warped to 288x384, goes in;
// three SimCC classification vectors per keypoint come out (x/y over crop
// pixels at 2 bins per pixel, z over the same bin count but spanning a metric
// depth range). No canvas and no ONNX session in here, so a Node harness can
// drive the exact code the worker runs.
//
// Ported from rtmlib (Tau-J/rtmlib, Apache-2.0):
//   tools/pose_estimation/{rtmpose3d,pre_processings,post_processings}.py

export const MODEL_W = 288
export const MODEL_H = 384
export const SIMCC_RATIO = 2
/** Depth half-range in meters: z bins span [-Z_RANGE, +Z_RANGE], root-relative. */
export const Z_RANGE = 2.1744869
export const NUM_KEYPOINTS = 133

/** ImageNet RGB mean/std — mmpose data_preprocessor values, applied channel-wise. */
export const MEAN = [123.675, 116.28, 103.53] as const
export const STD = [58.395, 57.12, 57.375] as const

// COCO-WholeBody layout: 17 body, 6 feet, 68 face, 21 left hand, 21 right hand.
export const COCO = {
  nose: 0,
  left_eye: 1,
  right_eye: 2,
  left_ear: 3,
  right_ear: 4,
  left_shoulder: 5,
  right_shoulder: 6,
  left_elbow: 7,
  right_elbow: 8,
  left_wrist: 9,
  right_wrist: 10,
  left_hip: 11,
  right_hip: 12,
  left_knee: 13,
  right_knee: 14,
  left_ankle: 15,
  right_ankle: 16,
  left_big_toe: 17,
  left_small_toe: 18,
  left_heel: 19,
  right_big_toe: 20,
  right_small_toe: 21,
  right_heel: 22,
  face: 23, // ..90 (68-point face)
  left_hand: 91, // ..111
  right_hand: 112, // ..132
} as const

/** Structurally identical to MediaPipe's Landmark — what the solver reads. */
export interface WorldLandmark {
  x: number
  y: number
  z: number
  visibility: number
}

/** The worker result shape the app consumes (matches PoseWorkerResult). */
export interface Rtmw3dResult {
  poseWorldLandmarks: WorldLandmark[][]
  leftHandWorldLandmarks: WorldLandmark[][]
  rightHandWorldLandmarks: WorldLandmark[][]
  faceLandmarks: WorldLandmark[][]
}

export interface CenterScale {
  cx: number
  cy: number
  /** Padded, aspect-fixed crop size in source pixels. */
  w: number
  h: number
}

/**
 * bbox (x1,y1,x2,y2) → padded center/scale with the crop aspect forced to the
 * model's. Combines rtmlib's bbox_xyxy2cs(padding=1.25) and the aspect reshape
 * inside top_down_affine.
 */
export function bboxCenterScale(bbox: readonly number[], padding = 1.25): CenterScale {
  const cx = (bbox[0] + bbox[2]) * 0.5
  const cy = (bbox[1] + bbox[3]) * 0.5
  let w = (bbox[2] - bbox[0]) * padding
  let h = (bbox[3] - bbox[1]) * padding
  const aspect = MODEL_W / MODEL_H
  if (w > h * aspect) h = w / aspect
  else w = h * aspect
  return { cx, cy, w, h }
}

/**
 * The affine placing the crop into the model input: dst = k·src + t.
 * mmpose's get_warp_matrix with rot=0 reduces to this uniform scale +
 * translation (the three-point construction maps center→center and the
 * half-width vector→half-width vector, both axes sharing dst_w/src_w).
 */
export function warpTransform(cs: CenterScale): { k: number; tx: number; ty: number } {
  const k = MODEL_W / cs.w
  return { k, tx: MODEL_W / 2 - k * cs.cx, ty: MODEL_H / 2 - k * cs.cy }
}

/**
 * RGBA crop pixels → NCHW float tensor, (v − mean) / std per channel.
 * `bgr` swaps R and B for parity testing against rtmlib's cv2 pipeline.
 */
export function imageDataToTensor(rgba: Uint8ClampedArray | Uint8Array, out: Float32Array, bgr = false): void {
  const px = MODEL_W * MODEL_H
  const rOff = bgr ? 2 : 0
  const bOff = bgr ? 0 : 2
  for (let i = 0; i < px; i++) {
    const j = i * 4
    out[i] = (rgba[j + rOff] - MEAN[0]) / STD[0]
    out[px + i] = (rgba[j + 1] - MEAN[1]) / STD[1]
    out[2 * px + i] = (rgba[j + bOff] - MEAN[2]) / STD[2]
  }
}

export interface Decoded {
  /** K×3: x,y in SOURCE-image pixels, z in meters (root-relative). */
  kpts: Float32Array
  /** K: min of the x/y SimCC max responses (rtmlib's 3D scoring). */
  scores: Float32Array
}

/**
 * SimCC decode. Bin argmax per axis, half-bin to pixels, z bins to meters,
 * then crop→image through the inverse warp so everything downstream lives in
 * one space.
 */
export function decodeSimCC3D(
  simccX: Float32Array,
  simccY: Float32Array,
  simccZ: Float32Array,
  warp: { k: number; tx: number; ty: number },
): Decoded {
  const K = NUM_KEYPOINTS
  const wx = simccX.length / K
  const wy = simccY.length / K
  const wz = simccZ.length / K
  const kpts = new Float32Array(K * 3)
  const scores = new Float32Array(K)
  for (let i = 0; i < K; i++) {
    let xLoc = 0
    let xMax = -Infinity
    for (let j = 0; j < wx; j++) {
      const v = simccX[i * wx + j]
      if (v > xMax) {
        xMax = v
        xLoc = j
      }
    }
    let yLoc = 0
    let yMax = -Infinity
    for (let j = 0; j < wy; j++) {
      const v = simccY[i * wy + j]
      if (v > yMax) {
        yMax = v
        yLoc = j
      }
    }
    let zLoc = 0
    let zMax = -Infinity
    for (let j = 0; j < wz; j++) {
      const v = simccZ[i * wz + j]
      if (v > zMax) {
        zMax = v
        zLoc = j
      }
    }
    const cropX = xLoc / SIMCC_RATIO
    const cropY = yLoc / SIMCC_RATIO
    kpts[i * 3] = (cropX - warp.tx) / warp.k
    kpts[i * 3 + 1] = (cropY - warp.ty) / warp.k
    // z bins span the codec's depth dimension (wz/SIMCC_RATIO, 288 for this
    // model — NOT the input height; mmpose SimCC3DLabel input_size is 3D),
    // mapped symmetrically to [-Z_RANGE, +Z_RANGE].
    kpts[i * 3 + 2] = ((zLoc / SIMCC_RATIO) / (wz / SIMCC_RATIO / 2) - 1) * Z_RANGE
    scores[i] = Math.min(xMax, yMax)
  }
  return { kpts, scores }
}

/** Score below which a keypoint is noise for tracking/visibility purposes. */
export const KPT_THR = 0.3

/**
 * Next frame's crop box from this frame's keypoints — the single-person
 * replacement for a detector model. Body, feet and hand extents (confident
 * points only) plus a 10% margin; bboxCenterScale adds its 1.25 padding on
 * top. Null when too little of the body is confident, which sends the caller
 * back to a full-frame crop.
 */
export function bboxFromKeypoints(
  kpts: Float32Array,
  scores: Float32Array,
  imgW: number,
  imgH: number,
): number[] | null {
  let bodyCount = 0
  for (let i = 0; i <= COCO.right_ankle; i++) if (scores[i] >= KPT_THR) bodyCount++
  if (bodyCount < 8) return null

  let x1 = Infinity
  let y1 = Infinity
  let x2 = -Infinity
  let y2 = -Infinity
  for (let i = 0; i < NUM_KEYPOINTS; i++) {
    if (i >= COCO.face && i < COCO.left_hand) continue // face can't move the box
    if (scores[i] < KPT_THR) continue
    const x = kpts[i * 3]
    const y = kpts[i * 3 + 1]
    if (x < x1) x1 = x
    if (x > x2) x2 = x
    if (y < y1) y1 = y
    if (y > y2) y2 = y
  }
  if (!(x2 > x1) || !(y2 > y1)) return null
  const margin = 0.1 * Math.max(x2 - x1, y2 - y1)
  return [
    Math.max(0, x1 - margin),
    Math.max(0, y1 - margin),
    Math.min(imgW, x2 + margin),
    Math.min(imgH, y2 + margin),
  ]
}

// Canonical adult segment lengths in meters, used to recover the image's
// meters-per-pixel. Individual deviation is a few percent — tolerable, since
// the solver consumes directions, and per-axis scale only skews them when it
// is badly wrong.
const SCALE_SEGMENTS: [number, number, number][] = [
  [COCO.left_shoulder, COCO.right_shoulder, 0.37],
  [COCO.left_shoulder, COCO.left_elbow, 0.28],
  [COCO.right_shoulder, COCO.right_elbow, 0.28],
  [COCO.left_elbow, COCO.left_wrist, 0.26],
  [COCO.right_elbow, COCO.right_wrist, 0.26],
  [COCO.left_hip, COCO.left_knee, 0.44],
  [COCO.right_hip, COCO.right_knee, 0.44],
  [COCO.left_knee, COCO.left_ankle, 0.43],
  [COCO.right_knee, COCO.right_ankle, 0.43],
]

/**
 * Meters per image pixel, from known segment lengths. The model hands us
 * metric z but pixel x/y; for a segment of true length L with pixel extent p
 * and depth extent dz, L² = (s·p)² + dz² solves for s. Median across limbs
 * shrugs off a bent or foreshortened one. Null when nothing is measurable —
 * hold the previous estimate.
 */
export function estimateMetersPerPixel(kpts: Float32Array, scores: Float32Array): number | null {
  const estimates: number[] = []
  for (const [a, b, len] of SCALE_SEGMENTS) {
    if (scores[a] < KPT_THR || scores[b] < KPT_THR) continue
    const dx = kpts[a * 3] - kpts[b * 3]
    const dy = kpts[a * 3 + 1] - kpts[b * 3 + 1]
    const dz = kpts[a * 3 + 2] - kpts[b * 3 + 2]
    const p2 = dx * dx + dy * dy
    const rem = len * len - dz * dz
    // A segment pointing nearly at the camera says nothing about pixel scale.
    if (p2 < 25 || rem < 0.25 * len * len) continue
    estimates.push(Math.sqrt(rem / p2))
  }
  if (estimates.length < 3) return null
  estimates.sort((a, b) => a - b)
  return estimates[estimates.length >> 1]
}

// MediaPipe pose index ← COCO-WholeBody index, for everything the app reads.
// The debug preview draws all POSE_CONNECTIONS, so MediaPipe-only points are
// filled from close equivalents rather than left at the origin: inner/outer
// eye corners collapse to the eye, mouth corners come from the 68-point face,
// pose-level fingers from the hand sets.
const MP_FROM_COCO: [number, number][] = [
  [0, COCO.nose],
  [1, COCO.left_eye],
  [2, COCO.left_eye],
  [3, COCO.left_eye],
  [4, COCO.right_eye],
  [5, COCO.right_eye],
  [6, COCO.right_eye],
  [7, COCO.left_ear],
  [8, COCO.right_ear],
  [9, COCO.face + 54], // mouth corners in the 68-point face layout
  [10, COCO.face + 48],
  [11, COCO.left_shoulder],
  [12, COCO.right_shoulder],
  [13, COCO.left_elbow],
  [14, COCO.right_elbow],
  [15, COCO.left_wrist],
  [16, COCO.right_wrist],
  [17, COCO.left_hand + 20], // pinky tip
  [18, COCO.right_hand + 20],
  [19, COCO.left_hand + 8], // index tip
  [20, COCO.right_hand + 8],
  [21, COCO.left_hand + 4], // thumb tip
  [22, COCO.right_hand + 4],
  [23, COCO.left_hip],
  [24, COCO.right_hip],
  [25, COCO.left_knee],
  [26, COCO.right_knee],
  [27, COCO.left_ankle],
  [28, COCO.right_ankle],
  [29, COCO.left_heel],
  [30, COCO.right_heel],
  [31, COCO.left_big_toe],
  [32, COCO.right_big_toe],
]

/**
 * Decoded keypoints → the MediaPipe-shaped world-landmark payload the solver
 * and debug preview already consume. Hip-center origin, meters on all axes
 * (x/y scaled by `mpp`, z already metric), y down and z away from the camera —
 * the same conventions as MediaPipe pose world landmarks.
 *
 * Face ships empty on this engine (no mesh, and morphs are out of scope).
 */
export function toWorkerResult(kpts: Float32Array, scores: Float32Array, mpp: number): Rtmw3dResult {
  const bodyVisible =
    (scores[COCO.left_hip] >= KPT_THR || scores[COCO.right_hip] >= KPT_THR) &&
    (scores[COCO.left_shoulder] >= KPT_THR || scores[COCO.right_shoulder] >= KPT_THR)
  if (!bodyVisible) {
    return {
      poseWorldLandmarks: [],
      leftHandWorldLandmarks: [],
      rightHandWorldLandmarks: [],
      faceLandmarks: [],
    }
  }

  const rx = (kpts[COCO.left_hip * 3] + kpts[COCO.right_hip * 3]) / 2
  const ry = (kpts[COCO.left_hip * 3 + 1] + kpts[COCO.right_hip * 3 + 1]) / 2
  const rz = (kpts[COCO.left_hip * 3 + 2] + kpts[COCO.right_hip * 3 + 2]) / 2
  const world = (i: number): WorldLandmark => ({
    x: (kpts[i * 3] - rx) * mpp,
    y: (kpts[i * 3 + 1] - ry) * mpp,
    z: kpts[i * 3 + 2] - rz,
    visibility: Math.min(1, Math.max(0, scores[i])),
  })

  const pose: WorldLandmark[] = new Array(33)
  for (const [mp, coco] of MP_FROM_COCO) pose[mp] = world(coco)

  // Head landmarks from the 68-point face when it is confident. The body's
  // four lone eye/ear points are exactly what the head basis solve reads, and
  // under a profile turn the OCCLUDED far ear has nothing to see — its z-bin
  // argmax can land meters away while x/y stay plausible, which rolls the head
  // basis catastrophically (and reads fine in a front-view debug draw, where z
  // barely shows). The face set is dense, mutually consistent, and scored much
  // higher; jaw endpoints stand in for ears, eyelid centroids for eyes.
  const avg = (from: number, to: number): [number, number, number, number] => {
    let x = 0
    let y = 0
    let z = 0
    let s = 0
    const n = to - from + 1
    for (let i = from; i <= to; i++) {
      x += kpts[i * 3]
      y += kpts[i * 3 + 1]
      z += kpts[i * 3 + 2]
      s += scores[i]
    }
    return [x / n, y / n, z / n, s / n]
  }
  const F = COCO.face
  const rightJaw = avg(F, F) // iBUG 0 — contour start at the subject's right ear
  const leftJaw = avg(F + 16, F + 16)
  const rightEye = avg(F + 36, F + 41)
  const leftEye = avg(F + 42, F + 47)
  const noseTip = avg(F + 30, F + 30)
  const faceScore = (rightJaw[3] + leftJaw[3] + rightEye[3] + leftEye[3]) / 4
  if (faceScore >= 0.5) {
    const put = (mp: number, [x, y, z, s]: [number, number, number, number]) => {
      pose[mp] = { x: (x - rx) * mpp, y: (y - ry) * mpp, z: z - rz, visibility: Math.min(1, Math.max(0, s)) }
    }
    put(0, noseTip)
    put(1, leftEye)
    put(2, leftEye)
    put(3, leftEye)
    put(4, rightEye)
    put(5, rightEye)
    put(6, rightEye)
    put(7, leftJaw)
    put(8, rightJaw)
  }

  // Depth sanity for the whole head: no point of a skull sits more than ~25cm
  // from the rest of it. Clamp each head z toward their median so one bad
  // depth bin cannot fold the neck or roll the head, whatever its source.
  const headZ = [] as number[]
  for (let i = 0; i <= 10; i++) headZ.push(pose[i].z)
  headZ.sort((a, b) => a - b)
  const medZ = headZ[headZ.length >> 1]
  for (let i = 0; i <= 10; i++) {
    if (pose[i].z > medZ + 0.25) pose[i].z = medZ + 0.25
    else if (pose[i].z < medZ - 0.25) pose[i].z = medZ - 0.25
  }

  const hand = (start: number): WorldLandmark[][] => {
    let sum = 0
    for (let i = 0; i < 21; i++) sum += scores[start + i]
    if (sum / 21 < KPT_THR) return []
    const out: WorldLandmark[] = new Array(21)
    for (let i = 0; i < 21; i++) out[i] = world(start + i)
    return [out]
  }

  return {
    poseWorldLandmarks: [pose],
    leftHandWorldLandmarks: hand(COCO.left_hand),
    rightHandWorldLandmarks: hand(COCO.right_hand),
    faceLandmarks: [],
  }
}
