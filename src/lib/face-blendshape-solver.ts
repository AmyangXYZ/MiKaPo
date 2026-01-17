import { NormalizedLandmark } from "@mediapipe/tasks-vision"
import { Quaternion } from "@babylonjs/core"
import { BoneState } from "./solver"

/**
 * Face morph weights for MMD models
 */
export interface FaceMorphWeights {
  // Eye morphs (0 = open, 1 = closed)
  まばたき: number // blink both
  ウィンク: number // wink left
  ウィンク右: number // wink right

  // Mouth morphs
  あ: number // mouth open
  ワ: number // big smile/laugh (wide mouth with teeth)
}

/**
 * Result from face solver - includes both bone rotations and morph weights
 */
export interface FaceSolverResult {
  boneStates: BoneState[]
  morphWeights: FaceMorphWeights
}

/**
 * Face landmark indices from MediaPipe 478-point face mesh
 * Matching the Rust FaceIndex enum
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

  // Mouth (matching original Rust FaceIndex)
  UpperLipTop: 13,
  LowerLipBottom: 14,
  MouthLeft: 61,
  MouthRight: 291,

  // Reference
  LeftEar: 234,
  RightEar: 454,
} as const

/**
 * FaceBlendshapeSolver - computes eye rotations (bones) and face morphs from landmarks
 * Following the style of the Rust PoseSolver
 */
export class FaceBlendshapeSolver {
  // Smoothing
  private smoothingFactor: number
  private prevLeftEyeOpenness = 1.0
  private prevRightEyeOpenness = 1.0
  private prevMouthOpenness = 0.0
  private prevLeftEyeGaze = { x: 0, y: 0 }
  private prevRightEyeGaze = { x: 0, y: 0 }

  constructor(options?: { smoothingFactor?: number }) {
    this.smoothingFactor = options?.smoothingFactor ?? 0.3
  }

  // Additional smoothing for smile
  private prevSmile = 0.0

  /**
   * Main solve function
   */
  solve(faceLandmarks: NormalizedLandmark[]): FaceSolverResult {
    const defaultResult: FaceSolverResult = {
      boneStates: [],
      morphWeights: {
        まばたき: 0,
        ウィンク: 0,
        ウィンク右: 0,
        あ: 0,
        ワ: 0,
      },
    }

    // HolisticLandmarker returns 478 face landmarks, but check for at least the ones we need
    // Highest index we use is 473 (RightEyeIris)
    if (!faceLandmarks || faceLandmarks.length < 474) {
      return defaultResult
    }

    // Calculate eye gaze
    const leftEyeGaze = this.calculateEyeGaze(
      faceLandmarks[FaceIndex.LeftEyeLeft],
      faceLandmarks[FaceIndex.LeftEyeRight],
      faceLandmarks[FaceIndex.LeftEyeIris]
    )
    const rightEyeGaze = this.calculateEyeGaze(
      faceLandmarks[FaceIndex.RightEyeLeft],
      faceLandmarks[FaceIndex.RightEyeRight],
      faceLandmarks[FaceIndex.RightEyeIris]
    )

    // Smooth gaze
    const smoothedLeftGaze = {
      x: this.lerp(this.prevLeftEyeGaze.x, leftEyeGaze.x, 1 - this.smoothingFactor),
      y: this.lerp(this.prevLeftEyeGaze.y, leftEyeGaze.y, 1 - this.smoothingFactor),
    }
    const smoothedRightGaze = {
      x: this.lerp(this.prevRightEyeGaze.x, rightEyeGaze.x, 1 - this.smoothingFactor),
      y: this.lerp(this.prevRightEyeGaze.y, rightEyeGaze.y, 1 - this.smoothingFactor),
    }
    this.prevLeftEyeGaze = smoothedLeftGaze
    this.prevRightEyeGaze = smoothedRightGaze

    // Average gaze for both eyes (like the Rust version)
    const averageGaze = {
      x: (smoothedLeftGaze.x + smoothedRightGaze.x) / 2,
      y: (smoothedLeftGaze.y + smoothedRightGaze.y) / 2,
    }

    // Calculate eye rotations from gaze
    const leftEyeRotation = this.calculateEyeRotation(averageGaze.x, averageGaze.y)
    const rightEyeRotation = this.calculateEyeRotation(averageGaze.x, averageGaze.y)

    // Calculate eye openness
    // Note: In Rust version, left/right are swapped due to mirroring
    let leftEyeOpenness = this.calculateEyeOpenness(
      faceLandmarks[FaceIndex.RightEyeLeft],
      faceLandmarks[FaceIndex.RightEyeRight],
      faceLandmarks[FaceIndex.RightEyeUpper],
      faceLandmarks[FaceIndex.RightEyeLower]
    )
    let rightEyeOpenness = this.calculateEyeOpenness(
      faceLandmarks[FaceIndex.LeftEyeLeft],
      faceLandmarks[FaceIndex.LeftEyeRight],
      faceLandmarks[FaceIndex.LeftEyeUpper],
      faceLandmarks[FaceIndex.LeftEyeLower]
    )

    // Smooth eye openness
    leftEyeOpenness = this.lerp(this.prevLeftEyeOpenness, leftEyeOpenness, 1 - this.smoothingFactor)
    rightEyeOpenness = this.lerp(this.prevRightEyeOpenness, rightEyeOpenness, 1 - this.smoothingFactor)
    this.prevLeftEyeOpenness = leftEyeOpenness
    this.prevRightEyeOpenness = rightEyeOpenness

    // Calculate mouth openness (original simple formula from Rust)
    let mouthOpenness = this.calculateMouthOpenness(
      faceLandmarks[FaceIndex.UpperLipTop],
      faceLandmarks[FaceIndex.LowerLipBottom],
      faceLandmarks[FaceIndex.MouthLeft],
      faceLandmarks[FaceIndex.MouthRight]
    )

    // Calculate smile
    let smile = this.calculateSmile(
      faceLandmarks[FaceIndex.UpperLipTop],
      faceLandmarks[FaceIndex.LowerLipBottom],
      faceLandmarks[FaceIndex.MouthLeft],
      faceLandmarks[FaceIndex.MouthRight]
    )

    // Smooth mouth openness and smile
    mouthOpenness = this.lerp(this.prevMouthOpenness, mouthOpenness, 1 - this.smoothingFactor)
    smile = this.lerp(this.prevSmile, smile, 1 - this.smoothingFactor)
    this.prevMouthOpenness = mouthOpenness
    this.prevSmile = smile

    // Convert openness to blink (0 = open, 1 = closed -> blink morph)
    const leftBlink = 1 - leftEyeOpenness
    const rightBlink = 1 - rightEyeOpenness

    // Build result
    const boneStates: BoneState[] = [
      { name: "左目", rotation: leftEyeRotation },
      { name: "右目", rotation: rightEyeRotation },
    ]

    const morphWeights: FaceMorphWeights = {
      まばたき: (leftBlink + rightBlink) / 2,
      ウィンク: leftBlink > 0.5 && rightBlink < 0.3 ? leftBlink : 0,
      ウィンク右: rightBlink > 0.5 && leftBlink < 0.3 ? rightBlink : 0,
      あ: mouthOpenness,
      ワ: smile,
    }

    return { boneStates, morphWeights }
  }

  /**
   * Calculate eye gaze direction from iris position relative to eye corners
   * Returns normalized x,y in range [-1, 1]
   */
  private calculateEyeGaze(
    eyeLeft: NormalizedLandmark,
    eyeRight: NormalizedLandmark,
    iris: NormalizedLandmark
  ): { x: number; y: number } {
    // Scale up for better precision (like Rust version's scale(10.0))
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

  /**
   * Calculate eye rotation quaternion from gaze direction
   * Following the Rust calculate_eye_rotation
   */
  private calculateEyeRotation(gazeX: number, gazeY: number): Quaternion {
    const maxHorizontalRotation = Math.PI / 6 // 30 degrees
    const maxVerticalRotation = Math.PI / 12 // 15 degrees

    const xRotation = gazeY * maxVerticalRotation
    const yRotation = -gazeX * maxHorizontalRotation

    // Create quaternion from euler angles (pitch, yaw, roll)
    return Quaternion.FromEulerAngles(xRotation, yRotation, 0)
  }

  /**
   * Calculate eye openness using aspect ratio
   * Returns 0 (closed) to 1 (fully open)
   */
  private calculateEyeOpenness(
    eyeLeft: NormalizedLandmark,
    eyeRight: NormalizedLandmark,
    eyeUpper: NormalizedLandmark,
    eyeLower: NormalizedLandmark
  ): number {
    const eyeHeight = this.distance(eyeUpper, eyeLower)
    const eyeWidth = this.distance(eyeLeft, eyeRight)

    if (eyeWidth === 0) return 1

    const aspectRatio = eyeHeight / eyeWidth

    // Less sensitive blink: lower closedRatio so eyes need to be more closed
    const openRatio = 0.3
    const closedRatio = 0.1  // Was 0.15, now requires more closure to trigger blink

    if (aspectRatio <= closedRatio) {
      return 0
    }
    if (aspectRatio >= openRatio) {
      return 1
    }

    return (aspectRatio - closedRatio) / (openRatio - closedRatio)
  }

  /**
   * Calculate mouth openness
   * Returns 0 (closed) to 1 (max open)
   */
  private calculateMouthOpenness(
    upperLipTop: NormalizedLandmark,
    lowerLipBottom: NormalizedLandmark,
    mouthLeft: NormalizedLandmark,
    mouthRight: NormalizedLandmark
  ): number {
    const mouthHeight = this.distance(upperLipTop, lowerLipBottom)
    const mouthWidth = this.distance(mouthLeft, mouthRight)

    if (mouthWidth === 0) return 0

    // High threshold (closed mouth won't trigger), but fast ramp-up once open
    const threshold = 0.18  // Mouth needs to be clearly open
    const ratio = mouthHeight / mouthWidth
    
    if (ratio <= threshold) {
      return 0  // No morph when mouth appears closed
    }
    
    // Once past threshold, ramp up quickly
    const openness = (ratio - threshold) / 0.2
    return this.clamp(openness, 0, 1)
  }

  /**
   * Calculate smile from mouth corner positions
   * Returns 0 (no smile) to 1 (full smile)
   */
  private calculateSmile(
    upperLipTop: NormalizedLandmark,
    lowerLipBottom: NormalizedLandmark,
    mouthLeft: NormalizedLandmark,
    mouthRight: NormalizedLandmark
  ): number {
    // Smile detection: mouth corners are higher than center when smiling
    const mouthCenterY = (upperLipTop.y + lowerLipBottom.y) / 2
    const cornerY = (mouthLeft.y + mouthRight.y) / 2
    
    // Raw smile amount (positive when corners are up)
    const rawSmile = (mouthCenterY - cornerY)
    
    // High threshold before triggering, then ramp up fast
    const threshold = 0.008  // Needs clear smile to trigger
    if (rawSmile <= threshold) {
      return 0
    }
    
    // Once past threshold, ramp up quickly
    const smileAmount = (rawSmile - threshold) * 120
    return this.clamp(smileAmount, 0, 1)
  }

  // Utility functions
  private distance(a: NormalizedLandmark, b: NormalizedLandmark): number {
    const dx = a.x - b.x
    const dy = a.y - b.y
    const dz = (a.z || 0) - (b.z || 0)
    return Math.sqrt(dx * dx + dy * dy + dz * dz)
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
  }

  private lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t
  }

  /**
   * Set smoothing factor (0 = no smoothing, 1 = max smoothing)
   */
  setSmoothingFactor(factor: number) {
    this.smoothingFactor = this.clamp(factor, 0, 0.95)
  }
}
