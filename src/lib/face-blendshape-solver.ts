import { NormalizedLandmark } from "@mediapipe/tasks-vision"
import { Quat } from "reze-engine"
import { BoneState } from "./solver"
import { OneEuroFilter } from "./filters"

/** Morph weights keyed by the loaded model's actual morph names. */
export type FaceMorphWeights = Record<string, number>

export interface FaceSolverResult {
  boneStates: BoneState[]
  morphWeights: FaceMorphWeights
}

// Geometric face solver: eye/mouth ratios measured directly on the MediaPipe
// face mesh. (MediaPipe's ARKit blendshape output would be the cleaner source,
// but the blendshape subgraph doesn't run on the holistic GPU delegate —
// "No support of const" — so the landmark geometry stays the driver.)

/**
 * Face landmark indices from MediaPipe 478-point face mesh
 */
const FaceIndex = {
  // Left eye (from camera's perspective, so appears on right side of image)
  LeftEyeUpper: 159,
  LeftEyeLower: 145,
  LeftEyeLeft: 33,
  LeftEyeRight: 133,
  LeftEyeIris: 468,

  // Right eye (from camera's perspective, so appears on left side of image)
  RightEyeUpper: 386,
  RightEyeLower: 374,
  RightEyeLeft: 362,
  RightEyeRight: 263,
  RightEyeIris: 473,

  // Mouth
  UpperLipTop: 13,
  LowerLipBottom: 14,
  MouthLeft: 61,
  MouthRight: 291,
} as const

/** Canonical MMD morph names driven by this solver, with per-model aliases. */
const MORPH_ALIASES: Record<string, string[]> = {
  まばたき: ["瞬き"],
  ウィンク: ["ウィンク２"],
  ウィンク右: ["ウィンク右２", "ウインク右"],
  あ: ["あ２"],
  ワ: ["にっこり", "にやり"],
}

export class FaceBlendshapeSolver {
  /** canonical → actual morph name on the loaded model. */
  private morphNames: Record<string, string>

  // One-Euro per channel: snappier than the old EMA for fast events (blinks)
  // while still suppressing landmark flutter at rest.
  private leftOpenFilter = new OneEuroFilter(2.0, 15, 1.0)
  private rightOpenFilter = new OneEuroFilter(2.0, 15, 1.0)
  private mouthFilter = new OneEuroFilter(2.0, 15, 1.0)
  private smileFilter = new OneEuroFilter(2.0, 15, 1.0)
  private gazeXFilter = new OneEuroFilter(2.0, 10, 1.0)
  private gazeYFilter = new OneEuroFilter(2.0, 10, 1.0)

  constructor() {
    this.morphNames = Object.fromEntries(Object.keys(MORPH_ALIASES).map((n) => [n, n]))
  }

  /**
   * Resolve canonical morph names against the loaded model's actual morph list
   * (names vary across models). Unresolved names stay canonical — the engine
   * ignores unknown morphs, same graceful degradation as before.
   */
  configure(availableMorphs: string[]): void {
    const available = new Set(availableMorphs)
    for (const canonical of Object.keys(MORPH_ALIASES)) {
      this.morphNames[canonical] =
        [canonical, ...MORPH_ALIASES[canonical]].find((n) => available.has(n)) ?? canonical
    }
  }

  reset(): void {
    this.leftOpenFilter.reset()
    this.rightOpenFilter.reset()
    this.mouthFilter.reset()
    this.smileFilter.reset()
    this.gazeXFilter.reset()
    this.gazeYFilter.reset()
  }

  solve(faceLandmarks: NormalizedLandmark[], timestampMs: number = performance.now()): FaceSolverResult {
    const names = this.morphNames
    const defaultResult: FaceSolverResult = {
      boneStates: [],
      morphWeights: {
        [names["まばたき"]]: 0,
        [names["ウィンク"]]: 0,
        [names["ウィンク右"]]: 0,
        [names["あ"]]: 0,
        [names["ワ"]]: 0,
      },
    }

    // Highest index we use is 473 (RightEyeIris)
    if (!faceLandmarks || faceLandmarks.length < 474) {
      return defaultResult
    }

    // Eye gaze from iris position relative to eye corners
    const leftEyeGaze = this.calculateEyeGaze(
      faceLandmarks[FaceIndex.LeftEyeLeft],
      faceLandmarks[FaceIndex.LeftEyeRight],
      faceLandmarks[FaceIndex.LeftEyeIris],
    )
    const rightEyeGaze = this.calculateEyeGaze(
      faceLandmarks[FaceIndex.RightEyeLeft],
      faceLandmarks[FaceIndex.RightEyeRight],
      faceLandmarks[FaceIndex.RightEyeIris],
    )

    const gazeX = this.gazeXFilter.filter((leftEyeGaze.x + rightEyeGaze.x) / 2, timestampMs)
    const gazeY = this.gazeYFilter.filter((leftEyeGaze.y + rightEyeGaze.y) / 2, timestampMs)
    const eyeRotation = this.calculateEyeRotation(gazeX, gazeY)

    // Eye openness — left/right swapped for mirror UX (user's right eye drives
    // the model eye on screen-left).
    const leftEyeOpenness = this.leftOpenFilter.filter(
      this.calculateEyeOpenness(
        faceLandmarks[FaceIndex.RightEyeLeft],
        faceLandmarks[FaceIndex.RightEyeRight],
        faceLandmarks[FaceIndex.RightEyeUpper],
        faceLandmarks[FaceIndex.RightEyeLower],
      ),
      timestampMs,
    )
    const rightEyeOpenness = this.rightOpenFilter.filter(
      this.calculateEyeOpenness(
        faceLandmarks[FaceIndex.LeftEyeLeft],
        faceLandmarks[FaceIndex.LeftEyeRight],
        faceLandmarks[FaceIndex.LeftEyeUpper],
        faceLandmarks[FaceIndex.LeftEyeLower],
      ),
      timestampMs,
    )

    const mouthOpenness = this.mouthFilter.filter(
      this.calculateMouthOpenness(
        faceLandmarks[FaceIndex.UpperLipTop],
        faceLandmarks[FaceIndex.LowerLipBottom],
        faceLandmarks[FaceIndex.MouthLeft],
        faceLandmarks[FaceIndex.MouthRight],
      ),
      timestampMs,
    )
    const smile = this.smileFilter.filter(
      this.calculateSmile(
        faceLandmarks[FaceIndex.UpperLipTop],
        faceLandmarks[FaceIndex.LowerLipBottom],
        faceLandmarks[FaceIndex.MouthLeft],
        faceLandmarks[FaceIndex.MouthRight],
      ),
      timestampMs,
    )

    // Convert openness to blink (0 = open, 1 = closed)
    const leftBlink = 1 - leftEyeOpenness
    const rightBlink = 1 - rightEyeOpenness

    const boneStates: BoneState[] = [
      { name: "左目", rotation: eyeRotation },
      { name: "右目", rotation: eyeRotation.clone() },
    ]

    const morphWeights: FaceMorphWeights = {
      [names["まばたき"]]: (leftBlink + rightBlink) / 2,
      [names["ウィンク"]]: leftBlink > 0.5 && rightBlink < 0.3 ? leftBlink : 0,
      [names["ウィンク右"]]: rightBlink > 0.5 && leftBlink < 0.3 ? rightBlink : 0,
      [names["あ"]]: mouthOpenness,
      [names["ワ"]]: smile,
    }

    return { boneStates, morphWeights }
  }

  /**
   * Eye gaze direction from iris position relative to eye corners,
   * normalized x,y in [-1, 1]
   */
  private calculateEyeGaze(
    eyeLeft: NormalizedLandmark,
    eyeRight: NormalizedLandmark,
    iris: NormalizedLandmark,
  ): { x: number; y: number } {
    const scale = 10.0

    const eyeCenterX = (eyeLeft.x * scale + eyeRight.x * scale) / 2
    const eyeCenterY = (eyeLeft.y * scale + eyeRight.y * scale) / 2
    const eyeWidth = Math.abs(eyeLeft.x * scale - eyeRight.x * scale)
    const eyeHeight = eyeWidth * 0.5

    const irisX = iris.x * scale
    const irisY = iris.y * scale

    const x = (irisX - eyeCenterX) / (eyeWidth * 0.5)
    const y = (irisY - eyeCenterY) / (eyeHeight * 0.5)

    return {
      x: this.clamp(x, -1, 1),
      y: this.clamp(y, -0.5, 0.5),
    }
  }

  private calculateEyeRotation(gazeX: number, gazeY: number): Quat {
    const maxHorizontalRotation = Math.PI / 6 // 30 degrees
    const maxVerticalRotation = Math.PI / 12 // 15 degrees

    const xRotation = gazeY * maxVerticalRotation
    const yRotation = -gazeX * maxHorizontalRotation

    return Quat.fromEuler(xRotation, yRotation, 0)
  }

  /**
   * Eye openness from aspect ratio: 0 (closed) to 1 (fully open)
   */
  private calculateEyeOpenness(
    eyeLeft: NormalizedLandmark,
    eyeRight: NormalizedLandmark,
    eyeUpper: NormalizedLandmark,
    eyeLower: NormalizedLandmark,
  ): number {
    const eyeHeight = this.distance(eyeUpper, eyeLower)
    const eyeWidth = this.distance(eyeLeft, eyeRight)

    if (eyeWidth === 0) return 1

    const aspectRatio = eyeHeight / eyeWidth

    // Less sensitive blink: low closedRatio so eyes need clear closure to trigger
    const openRatio = 0.3
    const closedRatio = 0.1

    if (aspectRatio <= closedRatio) {
      return 0
    }
    if (aspectRatio >= openRatio) {
      return 1
    }

    return (aspectRatio - closedRatio) / (openRatio - closedRatio)
  }

  /**
   * Mouth openness: 0 (closed) to 1 (max open)
   */
  private calculateMouthOpenness(
    upperLipTop: NormalizedLandmark,
    lowerLipBottom: NormalizedLandmark,
    mouthLeft: NormalizedLandmark,
    mouthRight: NormalizedLandmark,
  ): number {
    const mouthHeight = this.distance(upperLipTop, lowerLipBottom)
    const mouthWidth = this.distance(mouthLeft, mouthRight)

    if (mouthWidth === 0) return 0

    // High threshold (closed mouth won't trigger), fast ramp-up once open
    const threshold = 0.18
    const ratio = mouthHeight / mouthWidth

    if (ratio <= threshold) {
      return 0
    }

    const openness = (ratio - threshold) / 0.2
    return this.clamp(openness, 0, 1)
  }

  /**
   * Smile from mouth corner height: 0 (no smile) to 1 (full smile)
   */
  private calculateSmile(
    upperLipTop: NormalizedLandmark,
    lowerLipBottom: NormalizedLandmark,
    mouthLeft: NormalizedLandmark,
    mouthRight: NormalizedLandmark,
  ): number {
    // Mouth corners are higher than center when smiling
    const mouthCenterY = (upperLipTop.y + lowerLipBottom.y) / 2
    const cornerY = (mouthLeft.y + mouthRight.y) / 2

    const rawSmile = mouthCenterY - cornerY

    // High threshold before triggering, then ramp up fast
    const threshold = 0.008
    if (rawSmile <= threshold) {
      return 0
    }

    const smileAmount = (rawSmile - threshold) * 120
    return this.clamp(smileAmount, 0, 1)
  }

  private distance(a: NormalizedLandmark, b: NormalizedLandmark): number {
    const dx = a.x - b.x
    const dy = a.y - b.y
    const dz = (a.z || 0) - (b.z || 0)
    return Math.sqrt(dx * dx + dy * dy + dz * dz)
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
  }
}
