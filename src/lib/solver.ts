import { Matrix, Quaternion, Vector3 } from "@babylonjs/core"
import { HolisticLandmarkerResult, Landmark } from "@mediapipe/tasks-vision"

export interface BoneState {
  name: string
  rotation: Quaternion
}

export const KeyBones: string[] = [
  "首",
  "頭",
  "上半身",
  "下半身",
  "左足",
  "右足",
  "左ひざ",
  "右ひざ",
  "左足首",
  "右足首",
  "左腕",
  "右腕",
  "左ひじ",
  "右ひじ",
  "左足ＩＫ",
  "右足ＩＫ",
  "右つま先ＩＫ",
  "左つま先ＩＫ",
  "左目",
  "右目",
  "左手首",
  "右手首",
  "左手捩",
  "右手捩",
  "右親指１",
  "右親指２",
  "右人指１",
  "右人指２",
  "右人指３",
  "右中指１",
  "右中指２",
  "右中指３",
  "右薬指１",
  "右薬指２",
  "右薬指３",
  "右小指１",
  "右小指２",
  "右小指３",
  "左親指１",
  "左親指２",
  "左人指１",
  "左人指２",
  "左人指３",
  "左中指１",
  "左中指２",
  "左中指３",
  "左薬指１",
  "左薬指２",
  "左薬指３",
  "左小指１",
  "左小指２",
  "左小指３",
]

const PoseLandmarksTable: Record<string, number> = {
  nose: 0,
  left_eye_inner: 1,
  left_eye: 2,
  left_eye_outer: 3,
  right_eye_inner: 4,
  right_eye: 5,
  right_eye_outer: 6,
  left_ear: 7,
  right_ear: 8,
  mouth_left: 9,
  mouth_right: 10,
  left_shoulder: 11,
  right_shoulder: 12,
  left_elbow: 13,
  right_elbow: 14,
  left_wrist: 15,
  right_wrist: 16,
  left_pinky: 17,
  right_pinky: 18,
  left_index: 19,
  right_index: 20,
  left_thumb: 21,
  right_thumb: 22,
  left_hip: 23,
  right_hip: 24,
  left_knee: 25,
  right_knee: 26,
  left_ankle: 27,
  right_ankle: 28,
  left_heel: 29,
  right_heel: 30,
  left_foot_index: 31,
  right_foot_index: 32,
}

const HandIndexTable: Record<string, number> = {
  wrist: 0,
  thumb_cmc: 1,
  thumb_mcp: 2,
  thumb_ip: 3,
  thumb_tip: 4,
  index_mcp: 5,
  index_pip: 6,
  index_dip: 7,
  index_tip: 8,
  middle_mcp: 9,
  middle_pip: 10,
  middle_dip: 11,
  middle_tip: 12,
  ring_mcp: 13,
  ring_pip: 14,
  ring_dip: 15,
  ring_tip: 16,
  pinky_mcp: 17,
  pinky_pip: 18,
  pinky_dip: 19,
  pinky_tip: 20,
}

export class Solver {
  private poseWorldLandmarks: Landmark[] | null = null
  private leftHandWorldLandmarks: Landmark[] | null = null
  private rightHandWorldLandmarks: Landmark[] | null = null
  private boneStates: Record<string, BoneState> = {}

  constructor() {}

  solve(landmarks: HolisticLandmarkerResult): BoneState[] | null {
    this.boneStates = {}
    this.poseWorldLandmarks = null
    this.leftHandWorldLandmarks = null
    this.rightHandWorldLandmarks = null

    if (landmarks.poseWorldLandmarks.length > 0 && landmarks.poseWorldLandmarks[0].length === 33) {
      this.poseWorldLandmarks = landmarks.poseWorldLandmarks[0]
    }

    if (landmarks.leftHandWorldLandmarks.length > 0 && landmarks.leftHandWorldLandmarks[0].length === 21) {
      this.leftHandWorldLandmarks = landmarks.leftHandWorldLandmarks[0]
    }
    if (landmarks.rightHandWorldLandmarks.length > 0 && landmarks.rightHandWorldLandmarks[0].length === 21) {
      this.rightHandWorldLandmarks = landmarks.rightHandWorldLandmarks[0]
    }

    this.boneStates["upper_body"] = this.solveUpperBody()
    this.boneStates["neck"] = this.solveNeck()
    this.boneStates["head"] = this.solveHead()
    this.boneStates["lower_body"] = this.solveLowerBody()
    this.boneStates["left_leg"] = this.solveLeftLeg()
    this.boneStates["right_leg"] = this.solveRightLeg()
    this.boneStates["left_knee"] = this.solveLeftKnee()
    this.boneStates["right_knee"] = this.solveRightKnee()
    this.boneStates["left_ankle"] = this.solveLeftAnkle()
    this.boneStates["right_ankle"] = this.solveRightAnkle()
    this.boneStates["left_arm"] = this.solveLeftArm()
    this.boneStates["right_arm"] = this.solveRightArm()
    this.boneStates["left_elbow"] = this.solveLeftElbow()
    this.boneStates["right_elbow"] = this.solveRightElbow()
    this.boneStates["left_wrist_twist"] = this.solveLeftWristTwist()
    this.boneStates["right_wrist_twist"] = this.solveRightWristTwist()
    this.boneStates["left_wrist"] = this.solveLeftWrist()
    this.boneStates["right_wrist"] = this.solveRightWrist()

    // Left Hand Fingers
    this.boneStates["left_thumb_1"] = this.solveLeftThumb1()
    this.boneStates["left_thumb_2"] = this.solveLeftThumb2()
    this.boneStates["left_index_1"] = this.solveLeftIndex1()
    this.boneStates["left_index_2"] = this.solveLeftIndex2()
    this.boneStates["left_index_3"] = this.solveLeftIndex3()
    this.boneStates["left_middle_1"] = this.solveLeftMiddle1()
    this.boneStates["left_middle_2"] = this.solveLeftMiddle2()
    this.boneStates["left_middle_3"] = this.solveLeftMiddle3()
    this.boneStates["left_ring_1"] = this.solveLeftRing1()
    this.boneStates["left_ring_2"] = this.solveLeftRing2()
    this.boneStates["left_ring_3"] = this.solveLeftRing3()
    this.boneStates["left_pinky_1"] = this.solveLeftPinky1()
    this.boneStates["left_pinky_2"] = this.solveLeftPinky2()
    this.boneStates["left_pinky_3"] = this.solveLeftPinky3()

    // Right Hand Fingers
    this.boneStates["right_thumb_1"] = this.solveRightThumb1()
    this.boneStates["right_thumb_2"] = this.solveRightThumb2()
    this.boneStates["right_index_1"] = this.solveRightIndex1()
    this.boneStates["right_index_2"] = this.solveRightIndex2()
    this.boneStates["right_index_3"] = this.solveRightIndex3()
    this.boneStates["right_middle_1"] = this.solveRightMiddle1()
    this.boneStates["right_middle_2"] = this.solveRightMiddle2()
    this.boneStates["right_middle_3"] = this.solveRightMiddle3()
    this.boneStates["right_ring_1"] = this.solveRightRing1()
    this.boneStates["right_ring_2"] = this.solveRightRing2()
    this.boneStates["right_ring_3"] = this.solveRightRing3()
    this.boneStates["right_pinky_1"] = this.solveRightPinky1()
    this.boneStates["right_pinky_2"] = this.solveRightPinky2()
    this.boneStates["right_pinky_3"] = this.solveRightPinky3()

    return Object.values(this.boneStates)
  }

  private getPoseLandmark(name: string): Vector3 | null {
    if (!this.poseWorldLandmarks) return null
    return this.landmarkToVector3(this.poseWorldLandmarks[PoseLandmarksTable[name]])
  }

  private getLeftHandLandmark(name: string): Vector3 | null {
    if (!this.leftHandWorldLandmarks) return null
    return this.landmarkToVector3(this.leftHandWorldLandmarks[HandIndexTable[name]])
  }

  private getRightHandLandmark(name: string): Vector3 | null {
    if (!this.rightHandWorldLandmarks) return null
    return this.landmarkToVector3(this.rightHandWorldLandmarks[HandIndexTable[name]])
  }

  private landmarkToVector3(landmark: Landmark): Vector3 {
    if (!landmark) return new Vector3(0, 0, 0)
    return new Vector3(landmark.x, -landmark.y, landmark.z)
  }

  private solveLowerBody(): BoneState {
    const leftHip = this.getPoseLandmark("left_hip")
    const rightHip = this.getPoseLandmark("right_hip")

    if (!leftHip || !rightHip) return { name: "下半身", rotation: Quaternion.Identity() }

    const hipDirection = leftHip.subtract(rightHip).normalize()
    const referenceDirection = new Vector3(1, 0, 0)

    return {
      name: "下半身",
      rotation: Quaternion.FromUnitVectorsToRef(referenceDirection, hipDirection, new Quaternion()),
    }
  }

  private solveUpperBody(): BoneState {
    const leftShoulder = this.getPoseLandmark("left_shoulder")
    const rightShoulder = this.getPoseLandmark("right_shoulder")

    if (!leftShoulder || !rightShoulder) return { name: "上半身", rotation: Quaternion.Identity() }

    const shoulderCenter = leftShoulder.add(rightShoulder).scale(0.5)

    const shoulderX = leftShoulder.subtract(rightShoulder).normalize()

    const spineY = shoulderCenter.normalize()

    const upperBodyZ = Vector3.Cross(shoulderX, spineY).normalize()

    const upperBodyMatrix = Matrix.FromValues(
      shoulderX.x,
      shoulderX.y,
      shoulderX.z,
      0,
      spineY.x,
      spineY.y,
      spineY.z,
      0,
      upperBodyZ.x,
      upperBodyZ.y,
      upperBodyZ.z,
      0,
      0,
      0,
      0,
      1
    )

    const scaling = new Vector3()
    const rotation = new Quaternion()
    const translation = new Vector3()
    upperBodyMatrix.decompose(scaling, rotation, translation)

    return {
      name: "上半身",
      rotation: rotation,
    }
  }

  private solveNeck(): BoneState {
    const worldLeftEar = this.getPoseLandmark("left_ear")
    const worldRightEar = this.getPoseLandmark("right_ear")
    const worldLeftShoulder = this.getPoseLandmark("left_shoulder")
    const worldRightShoulder = this.getPoseLandmark("right_shoulder")

    if (!worldLeftEar || !worldRightEar || !worldLeftShoulder || !worldRightShoulder)
      return { name: "首", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const upperBodyMatrix = new Matrix()
    Matrix.FromQuaternionToRef(upperBodyQuat, upperBodyMatrix)
    const worldToUpperBody = upperBodyMatrix.invert()

    const localLeftEar = Vector3.TransformCoordinates(worldLeftEar, worldToUpperBody)
    const localRightEar = Vector3.TransformCoordinates(worldRightEar, worldToUpperBody)
    const localLeftShoulder = Vector3.TransformCoordinates(worldLeftShoulder, worldToUpperBody)
    const localRightShoulder = Vector3.TransformCoordinates(worldRightShoulder, worldToUpperBody)

    // Calculate neck direction in upper body space
    const localEarCenter = localLeftEar.add(localRightEar).scale(0.5)
    const localShoulderCenter = localLeftShoulder.add(localRightShoulder).scale(0.5)
    const neckDirection = localEarCenter.subtract(localShoulderCenter).normalize()
    const reference = new Vector3(0, 0.9758578206707508, -0.21840676233975218).normalize()

    return {
      name: "首",
      rotation: Quaternion.FromUnitVectorsToRef(reference, neckDirection, new Quaternion()),
    }
  }

  private solveHead(): BoneState {
    const worldLeftEar = this.getPoseLandmark("left_ear")
    const worldRightEar = this.getPoseLandmark("right_ear")
    const worldLeftEye = this.getPoseLandmark("left_eye")
    const worldRightEye = this.getPoseLandmark("right_eye")

    if (!worldLeftEar || !worldRightEar || !worldLeftEye || !worldRightEye)
      return { name: "頭", rotation: Quaternion.Identity() }

    // Use full parent chain: upper_body * neck
    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const neckQuat = this.boneStates["neck"].rotation

    const fullParentQuat = upperBodyQuat.multiply(neckQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localLeftEar = Vector3.TransformCoordinates(worldLeftEar, worldToFullParent)
    const localRightEar = Vector3.TransformCoordinates(worldRightEar, worldToFullParent)
    const localLeftEye = Vector3.TransformCoordinates(worldLeftEye, worldToFullParent)
    const localRightEye = Vector3.TransformCoordinates(worldRightEye, worldToFullParent)

    const localEarCenter = localLeftEar.add(localRightEar).scale(0.5)
    const localEyeCenter = localLeftEye.add(localRightEye).scale(0.5)

    const earDirection = localLeftEar.subtract(localRightEar).normalize()
    const horizontalRef = new Vector3(1, 0, 0).normalize()
    const horizontalRotation = Quaternion.FromUnitVectorsToRef(horizontalRef, earDirection, new Quaternion())

    const bendDirection = localEyeCenter.subtract(localEarCenter).normalize()
    const verticalRef = new Vector3(0, 0, -1).normalize()
    const verticalRotation = Quaternion.FromUnitVectorsToRef(verticalRef, bendDirection, new Quaternion())

    const combinedRotation = horizontalRotation.multiply(verticalRotation)

    return {
      name: "頭",
      rotation: combinedRotation,
    }
  }

  private solveLeftLeg(): BoneState {
    const worldLeftHip = this.getPoseLandmark("left_hip")
    const worldLeftKnee = this.getPoseLandmark("left_knee")

    if (!worldLeftHip || !worldLeftKnee) return { name: "左足", rotation: Quaternion.Identity() }

    const lowerBodyQuat = this.boneStates["lower_body"].rotation
    const lowerBodyMatrix = new Matrix()
    Matrix.FromQuaternionToRef(lowerBodyQuat, lowerBodyMatrix)
    const worldToLowerBody = lowerBodyMatrix.invert()
    const localLeftHip = Vector3.TransformCoordinates(worldLeftHip, worldToLowerBody)
    const localLeftKnee = Vector3.TransformCoordinates(worldLeftKnee, worldToLowerBody)

    const leftLegDirection = localLeftKnee.subtract(localLeftHip).normalize()

    const reference = new Vector3(-0.009540689177369048, -0.998440855265296, 0.05499848895310636).normalize()

    return {
      name: "左足",
      rotation: Quaternion.FromUnitVectorsToRef(reference, leftLegDirection, new Quaternion()),
    }
  }

  private solveRightLeg(): BoneState {
    const worldRightHip = this.getPoseLandmark("right_hip")
    const worldRightKnee = this.getPoseLandmark("right_knee")

    if (!worldRightHip || !worldRightKnee) return { name: "右足", rotation: Quaternion.Identity() }

    const lowerBodyQuat = this.boneStates["lower_body"].rotation
    const lowerBodyMatrix = new Matrix()
    Matrix.FromQuaternionToRef(lowerBodyQuat, lowerBodyMatrix)
    const worldToLowerBody = lowerBodyMatrix.invert()

    const localRightHip = Vector3.TransformCoordinates(worldRightHip, worldToLowerBody)
    const localRightKnee = Vector3.TransformCoordinates(worldRightKnee, worldToLowerBody)

    const rightLegDirection = localRightKnee.subtract(localRightHip).normalize()

    const reference = new Vector3(-0.009540689177369048, -0.998440855265296, 0.05499848895310636).normalize()

    return {
      name: "右足",
      rotation: Quaternion.FromUnitVectorsToRef(reference, rightLegDirection, new Quaternion()),
    }
  }

  private solveLeftKnee(): BoneState {
    const worldLeftKnee = this.getPoseLandmark("left_knee")
    const worldLeftAnkle = this.getPoseLandmark("left_ankle")

    if (!worldLeftKnee || !worldLeftAnkle) return { name: "左ひざ", rotation: Quaternion.Identity() }

    const leftLegQuat = this.boneStates["left_leg"].rotation
    const lowerBodyQuat = this.boneStates["lower_body"].rotation

    const fullParentQuat = lowerBodyQuat.multiply(leftLegQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localLeftKnee = Vector3.TransformCoordinates(worldLeftKnee, worldToFullParent)
    const localLeftAnkle = Vector3.TransformCoordinates(worldLeftAnkle, worldToFullParent)

    const kneeDirection = localLeftAnkle.subtract(localLeftKnee).normalize()
    const reference = new Vector3(-0.0007085292291306043, -0.9908517790187175, 0.1349527695224302).normalize()

    return {
      name: "左ひざ",
      rotation: Quaternion.FromUnitVectorsToRef(reference, kneeDirection, new Quaternion()),
    }
  }

  private solveRightKnee(): BoneState {
    const worldRightKnee = this.getPoseLandmark("right_knee")
    const worldRightAnkle = this.getPoseLandmark("right_ankle")

    if (!worldRightKnee || !worldRightAnkle) return { name: "右ひざ", rotation: Quaternion.Identity() }

    const rightLegQuat = this.boneStates["right_leg"].rotation
    const lowerBodyQuat = this.boneStates["lower_body"].rotation

    const fullParentQuat = lowerBodyQuat.multiply(rightLegQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localRightKnee = Vector3.TransformCoordinates(worldRightKnee, worldToFullParent)
    const localRightAnkle = Vector3.TransformCoordinates(worldRightAnkle, worldToFullParent)

    const kneeDirection = localRightAnkle.subtract(localRightKnee).normalize()
    const reference = new Vector3(0.0007079817891811808, -0.9908517794028981, 0.13495276957475513).normalize()

    return {
      name: "右ひざ",
      rotation: Quaternion.FromUnitVectorsToRef(reference, kneeDirection, new Quaternion()),
    }
  }

  private solveLeftAnkle(): BoneState {
    const worldLeftHeel = this.getPoseLandmark("left_heel")
    const worldLeftFootIndex = this.getPoseLandmark("left_foot_index")

    if (!worldLeftHeel || !worldLeftFootIndex) return { name: "左足首", rotation: Quaternion.Identity() }

    const lowerBodyQuat = this.boneStates["lower_body"].rotation
    const leftLegQuat = this.boneStates["left_leg"].rotation
    const leftKneeQuat = this.boneStates["left_knee"].rotation

    const fullParentQuat = lowerBodyQuat.multiply(leftLegQuat).multiply(leftKneeQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localLeftHeel = Vector3.TransformCoordinates(worldLeftHeel, worldToFullParent)
    const localLeftFootIndex = Vector3.TransformCoordinates(worldLeftFootIndex, worldToFullParent)

    const ankleDirection = localLeftFootIndex.subtract(localLeftHeel).normalize()
    const reference = new Vector3(0, -0.65728916525082, -0.7536384764884819).normalize()

    return {
      name: "左足首",
      rotation: Quaternion.FromUnitVectorsToRef(reference, ankleDirection, new Quaternion()),
    }
  }

  private solveRightAnkle(): BoneState {
    const worldRightHeel = this.getPoseLandmark("right_heel")
    const worldRightFootIndex = this.getPoseLandmark("right_foot_index")

    if (!worldRightHeel || !worldRightFootIndex) return { name: "右足首", rotation: Quaternion.Identity() }

    // Full parent chain: lower_body * right_leg * right_knee
    const lowerBodyQuat = this.boneStates["lower_body"].rotation
    const rightLegQuat = this.boneStates["right_leg"].rotation
    const rightKneeQuat = this.boneStates["right_knee"].rotation

    const fullParentQuat = lowerBodyQuat.multiply(rightLegQuat).multiply(rightKneeQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localRightHeel = Vector3.TransformCoordinates(worldRightHeel, worldToFullParent)
    const localRightFootIndex = Vector3.TransformCoordinates(worldRightFootIndex, worldToFullParent)

    const ankleDirection = localRightFootIndex.subtract(localRightHeel).normalize()
    const reference = new Vector3(0, -0.65728916525082, -0.7536384764884819).normalize()

    return {
      name: "右足首",
      rotation: Quaternion.FromUnitVectorsToRef(reference, ankleDirection, new Quaternion()),
    }
  }

  private solveLeftArm(): BoneState {
    const worldLeftShoulder = this.getPoseLandmark("left_shoulder")
    const worldLeftElbow = this.getPoseLandmark("left_elbow")

    if (!worldLeftShoulder || !worldLeftElbow) return { name: "左腕", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const upperBodyMatrix = new Matrix()
    Matrix.FromQuaternionToRef(upperBodyQuat, upperBodyMatrix)
    const worldToUpperBody = upperBodyMatrix.invert()

    const localLeftShoulder = Vector3.TransformCoordinates(worldLeftShoulder, worldToUpperBody)
    const localLeftElbow = Vector3.TransformCoordinates(worldLeftElbow, worldToUpperBody)

    const leftArmDirection = localLeftElbow.subtract(localLeftShoulder).normalize()
    const reference = new Vector3(0.8012514930735141, -0.5966378711527615, -0.04493657256361681).normalize()

    return {
      name: "左腕",
      rotation: Quaternion.FromUnitVectorsToRef(reference, leftArmDirection, new Quaternion()),
    }
  }

  private solveRightArm(): BoneState {
    const worldRightShoulder = this.getPoseLandmark("right_shoulder")
    const worldRightElbow = this.getPoseLandmark("right_elbow")

    if (!worldRightShoulder || !worldRightElbow) return { name: "右腕", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const upperBodyMatrix = new Matrix()
    Matrix.FromQuaternionToRef(upperBodyQuat, upperBodyMatrix)
    const worldToUpperBody = upperBodyMatrix.invert()

    const localRightShoulder = Vector3.TransformCoordinates(worldRightShoulder, worldToUpperBody)
    const localRightElbow = Vector3.TransformCoordinates(worldRightElbow, worldToUpperBody)

    const rightArmDirection = localRightElbow.subtract(localRightShoulder).normalize()
    const reference = new Vector3(-0.8020376176381924, -0.5972232450219962, -0.007749548286792409).normalize()

    return {
      name: "右腕",
      rotation: Quaternion.FromUnitVectorsToRef(reference, rightArmDirection, new Quaternion()),
    }
  }

  private solveLeftElbow(): BoneState {
    const worldLeftElbow = this.getPoseLandmark("left_elbow")
    const worldLeftWrist = this.getPoseLandmark("left_wrist")

    if (!worldLeftElbow || !worldLeftWrist) return { name: "左ひじ", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const leftArmQuat = this.boneStates["left_arm"].rotation

    const fullParentQuat = upperBodyQuat.multiply(leftArmQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localLeftElbow = Vector3.TransformCoordinates(worldLeftElbow, worldToFullParent)
    const localLeftWrist = Vector3.TransformCoordinates(worldLeftWrist, worldToFullParent)

    const leftElbowDirection = localLeftWrist.subtract(localLeftElbow).normalize()
    const reference = new Vector3(0.7991214493734219, -0.600241324846603, -0.03339552511514752).normalize()

    return {
      name: "左ひじ",
      rotation: Quaternion.FromUnitVectorsToRef(reference, leftElbowDirection, new Quaternion()),
    }
  }

  private solveRightElbow(): BoneState {
    const worldRightElbow = this.getPoseLandmark("right_elbow")
    const worldRightWrist = this.getPoseLandmark("right_wrist")

    if (!worldRightElbow || !worldRightWrist) return { name: "右ひじ", rotation: Quaternion.Identity() }

    const rightArmQuat = this.boneStates["right_arm"].rotation
    const upperBodyQuat = this.boneStates["upper_body"].rotation

    const fullParentQuat = upperBodyQuat.multiply(rightArmQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localRightElbow = Vector3.TransformCoordinates(worldRightElbow, worldToFullParent)
    const localRightWrist = Vector3.TransformCoordinates(worldRightWrist, worldToFullParent)

    const rightElbowDirection = localRightWrist.subtract(localRightElbow).normalize()
    const reference = new Vector3(-0.7991213083626819, -0.6002415122251716, -0.03339553147285845).normalize()

    return {
      name: "右ひじ",
      rotation: Quaternion.FromUnitVectorsToRef(reference, rightElbowDirection, new Quaternion()),
    }
  }

  private solveLeftWristTwist(): BoneState {
    const worldLeftWrist = this.getLeftHandLandmark("wrist")
    const worldLeftIndex = this.getLeftHandLandmark("index_mcp")
    const worldLeftRing = this.getLeftHandLandmark("ring_mcp")

    if (!worldLeftWrist || !worldLeftIndex || !worldLeftRing) return { name: "左手捩", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const leftArmQuat = this.boneStates["left_arm"].rotation
    const leftElbowQuat = this.boneStates["left_elbow"].rotation

    const fullParentQuat = upperBodyQuat.multiply(leftArmQuat).multiply(leftElbowQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localLeftIndex = Vector3.TransformCoordinates(worldLeftIndex, worldToFullParent)
    const localLeftRing = Vector3.TransformCoordinates(worldLeftRing, worldToFullParent)

    const handDirection = localLeftIndex.subtract(localLeftRing).normalize()
    const reference = new Vector3(0, 0, -1).normalize()

    const fullRotation = Quaternion.FromUnitVectorsToRef(reference, handDirection, new Quaternion())
    const eulerAngles = fullRotation.toEulerAngles()
    const rollOnly = Quaternion.RotationYawPitchRoll(eulerAngles.z, 0, 0)

    return {
      name: "左手捩",
      rotation: rollOnly,
    }
  }

  private solveRightWristTwist(): BoneState {
    const worldRightWrist = this.getRightHandLandmark("wrist")
    const worldRightIndex = this.getRightHandLandmark("index_mcp")
    const worldRightRing = this.getRightHandLandmark("ring_mcp")
    if (!worldRightWrist || !worldRightIndex || !worldRightRing)
      return { name: "右手捩", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const rightArmQuat = this.boneStates["right_arm"].rotation
    const rightElbowQuat = this.boneStates["right_elbow"].rotation

    const fullParentQuat = upperBodyQuat.multiply(rightArmQuat).multiply(rightElbowQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localRightIndex = Vector3.TransformCoordinates(worldRightIndex, worldToFullParent)
    const localRightRing = Vector3.TransformCoordinates(worldRightRing, worldToFullParent)

    const handDirection = localRightIndex.subtract(localRightRing).normalize()
    const reference = new Vector3(0, 0, -1).normalize()

    const fullRotation = Quaternion.FromUnitVectorsToRef(reference, handDirection, new Quaternion())

    const eulerAngles = fullRotation.toEulerAngles()
    const rollOnly = Quaternion.RotationYawPitchRoll(eulerAngles.z, 0, 0)

    return {
      name: "右手捩",
      rotation: rollOnly,
    }
  }

  private solveLeftWrist(): BoneState {
    // Full parent chain: upper_body * left_arm * left_elbows
    const worldLeftWrist = this.getLeftHandLandmark("wrist")
    const worldLeftMiddleMcp = this.getLeftHandLandmark("middle_mcp")

    if (!worldLeftWrist || !worldLeftMiddleMcp) return { name: "左手首", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const leftArmQuat = this.boneStates["left_arm"].rotation
    const leftElbowQuat = this.boneStates["left_elbow"].rotation
    const leftWristTwistQuat = this.boneStates["left_wrist_twist"].rotation

    const fullParentQuat = upperBodyQuat.multiply(leftArmQuat).multiply(leftElbowQuat).multiply(leftWristTwistQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localLeftWrist = Vector3.TransformCoordinates(worldLeftWrist, worldToFullParent)
    const localLeftMiddleMcp = Vector3.TransformCoordinates(worldLeftMiddleMcp, worldToFullParent)

    const wristDirection = localLeftMiddleMcp.subtract(localLeftWrist).normalize()
    const reference = new Vector3(0.72573996, -0.40247154, -0.01692206).normalize()

    return {
      name: "左手首",
      rotation: Quaternion.FromUnitVectorsToRef(reference, wristDirection, new Quaternion()),
    }
  }

  private solveRightWrist(): BoneState {
    const worldRightWrist = this.getRightHandLandmark("wrist")
    const worldRightMiddleMcp = this.getRightHandLandmark("index_mcp")

    if (!worldRightWrist || !worldRightMiddleMcp) return { name: "右手首", rotation: Quaternion.Identity() }

    // Full parent chain: upper_body * right_arm * right_elbow
    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const rightArmQuat = this.boneStates["right_arm"].rotation
    const rightElbowQuat = this.boneStates["right_elbow"].rotation
    const rightWristTwistQuat = this.boneStates["right_wrist_twist"].rotation

    const fullParentQuat = upperBodyQuat.multiply(rightArmQuat).multiply(rightElbowQuat).multiply(rightWristTwistQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localRightWrist = Vector3.TransformCoordinates(worldRightWrist, worldToFullParent)
    const localRightMiddleMcp = Vector3.TransformCoordinates(worldRightMiddleMcp, worldToFullParent)

    const wristDirection = localRightMiddleMcp.subtract(localRightWrist).normalize()
    const reference = new Vector3(-0.72573996, -0.40247154, 0.01692206).normalize()

    return {
      name: "右手首",
      rotation: Quaternion.FromUnitVectorsToRef(reference, wristDirection, new Quaternion()),
    }
  }

  private solveLeftThumb1(): BoneState {
    const thumbMCP = this.getLeftHandLandmark("thumb_mcp")
    const thumbIP = this.getLeftHandLandmark("thumb_ip")
    if (!thumbMCP || !thumbIP) return { name: "左親指１", rotation: Quaternion.Identity() }

    // Get full parent chain including thumb base
    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const leftArmQuat = this.boneStates["left_arm"].rotation
    const leftElbowQuat = this.boneStates["left_elbow"].rotation
    const leftWristTwistQuat = this.boneStates["left_wrist_twist"].rotation
    const leftWristQuat = this.boneStates["left_wrist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(leftArmQuat)
      .multiply(leftElbowQuat)
      .multiply(leftWristTwistQuat)
      .multiply(leftWristQuat)

    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localThumbMCP = Vector3.TransformCoordinates(thumbMCP, worldToFullParent)
    const localThumbIP = Vector3.TransformCoordinates(thumbIP, worldToFullParent)

    const thumbDirection = localThumbIP.subtract(localThumbMCP).normalize()
    const reference = new Vector3(0.6236582921350833, -0.7035050354478427, -0.34077998730952624).normalize()
    return {
      name: "左親指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, thumbDirection, new Quaternion()),
    }
  }

  private solveLeftThumb2(): BoneState {
    return this.solveFingerJoint("left_thumb_1", "左親指２", new Vector3(-1.0, -1.0, 0.0).normalize(), 0.85)
  }

  private solveLeftIndex1(): BoneState {
    const wrist = this.getLeftHandLandmark("wrist")
    const indexMCP = this.getLeftHandLandmark("index_mcp")
    const indexPIP = this.getLeftHandLandmark("index_pip")
    if (!wrist || !indexMCP || !indexPIP) return { name: "左人指１", rotation: Quaternion.Identity() }

    // Get full parent chain including index base
    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const leftArmQuat = this.boneStates["left_arm"].rotation
    const leftElbowQuat = this.boneStates["left_elbow"].rotation
    const leftWristTwistQuat = this.boneStates["left_wrist_twist"].rotation
    const leftWristQuat = this.boneStates["left_wrist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(leftArmQuat)
      .multiply(leftElbowQuat)
      .multiply(leftWristTwistQuat)
      .multiply(leftWristQuat)

    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localIndexMCP = Vector3.TransformCoordinates(indexMCP, worldToFullParent)
    const localIndexPIP = Vector3.TransformCoordinates(indexPIP, worldToFullParent)

    const indexDirection = localIndexPIP.subtract(localIndexMCP).normalize()
    const reference = new Vector3(0.8432431728071625, -0.5368768421934949, 0.026536914486258466).normalize()

    return {
      name: "左人指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, indexDirection, new Quaternion()),
    }
  }

  private solveLeftIndex2(): BoneState {
    return this.solveFingerJoint("left_index_1", "左人指２", new Vector3(-0.031, 0.0, -0.993).normalize(), 0.9)
  }

  private solveLeftIndex3(): BoneState {
    return this.solveFingerJoint("left_index_1", "左人指３", new Vector3(-0.031, 0.0, -0.993).normalize(), 0.65)
  }

  // Left Middle Finger
  private solveLeftMiddle1(): BoneState {
    const middleMCP = this.getLeftHandLandmark("middle_mcp")
    const middlePIP = this.getLeftHandLandmark("middle_pip")
    if (!middleMCP || !middlePIP) return { name: "左中指１", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const leftArmQuat = this.boneStates["left_arm"].rotation
    const leftElbowQuat = this.boneStates["left_elbow"].rotation
    const leftWristTwistQuat = this.boneStates["left_wrist_twist"].rotation
    const leftWristQuat = this.boneStates["left_wrist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(leftArmQuat)
      .multiply(leftElbowQuat)
      .multiply(leftWristQuat)
      .multiply(leftWristTwistQuat)

    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localMiddleMCP = Vector3.TransformCoordinates(middleMCP, worldToFullParent)
    const localMiddlePIP = Vector3.TransformCoordinates(middlePIP, worldToFullParent)

    const middleDirection = localMiddlePIP.subtract(localMiddleMCP).normalize()
    const reference = new Vector3(0.8303922987881693, -0.5566343204926274, 0.02463459687127938).normalize()

    return {
      name: "左中指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, middleDirection, new Quaternion()),
    }
  }

  private solveLeftMiddle2(): BoneState {
    return this.solveFingerJoint("left_middle_1", "左中指２", new Vector3(0.03, 0.0, -0.996).normalize(), 0.9)
  }

  private solveLeftMiddle3(): BoneState {
    return this.solveFingerJoint("left_middle_1", "左中指３", new Vector3(0.03, 0.0, -0.996).normalize(), 0.65)
  }

  // Left Ring Finger
  private solveLeftRing1(): BoneState {
    const ringMCP = this.getLeftHandLandmark("ring_mcp")
    const ringPIP = this.getLeftHandLandmark("ring_pip")
    if (!ringMCP || !ringPIP) return { name: "左薬指１", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const leftArmQuat = this.boneStates["left_arm"].rotation
    const leftElbowQuat = this.boneStates["left_elbow"].rotation
    const leftWristTwistQuat = this.boneStates["left_wrist_twist"].rotation
    const leftWristQuat = this.boneStates["left_wrist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(leftArmQuat)
      .multiply(leftElbowQuat)
      .multiply(leftWristTwistQuat)
      .multiply(leftWristQuat)
    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localRingMCP = Vector3.TransformCoordinates(ringMCP, worldToFullParent)
    const localRingPIP = Vector3.TransformCoordinates(ringPIP, worldToFullParent)

    const ringDirection = localRingPIP.subtract(localRingMCP).normalize()
    const reference = new Vector3(0.8076445279586488, -0.5883930602992032, 0.038780446750771254).normalize()

    return {
      name: "左薬指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, ringDirection, new Quaternion()),
    }
  }

  private solveLeftRing2(): BoneState {
    return this.solveFingerJoint("left_ring_1", "左薬指２", new Vector3(0.048, 0.0, 0.997).normalize(), 0.88)
  }

  private solveLeftRing3(): BoneState {
    return this.solveFingerJoint("left_ring_1", "左薬指３", new Vector3(0.048, 0.0, 0.997).normalize(), 0.6)
  }

  // Left Pinky Finger
  private solveLeftPinky1(): BoneState {
    const pinkyMCP = this.getLeftHandLandmark("pinky_mcp")
    const pinkyPIP = this.getLeftHandLandmark("pinky_pip")
    if (!pinkyMCP || !pinkyPIP) return { name: "左小指１", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const leftArmQuat = this.boneStates["left_arm"].rotation
    const leftElbowQuat = this.boneStates["left_elbow"].rotation
    const leftWristTwistQuat = this.boneStates["left_wrist_twist"].rotation
    const leftWristQuat = this.boneStates["left_wrist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(leftArmQuat)
      .multiply(leftElbowQuat)
      .multiply(leftWristTwistQuat)
      .multiply(leftWristQuat)

    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localPinkyMCP = Vector3.TransformCoordinates(pinkyMCP, worldToFullParent)
    const localPinkyPIP = Vector3.TransformCoordinates(pinkyPIP, worldToFullParent)

    const pinkyDirection = localPinkyPIP.subtract(localPinkyMCP).normalize()
    const reference = new Vector3(0.8462256262210587, -0.5275922475926769, 0.07448899117913084).normalize()

    return {
      name: "左小指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, pinkyDirection, new Quaternion()),
    }
  }

  private solveLeftPinky2(): BoneState {
    return this.solveFingerJoint("left_pinky_1", "左小指２", new Vector3(0.088, 0.0, -0.997).normalize(), 0.85)
  }

  private solveLeftPinky3(): BoneState {
    return this.solveFingerJoint("left_pinky_1", "左小指３", new Vector3(0.088, 0.0, -0.997).normalize(), 0.55)
  }

  private solveRightThumb1(): BoneState {
    const thumbMCP = this.getRightHandLandmark("thumb_mcp")
    const thumbIP = this.getRightHandLandmark("thumb_ip")

    if (!thumbMCP || !thumbIP) return { name: "右親指１", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const rightArmQuat = this.boneStates["right_arm"].rotation
    const rightElbowQuat = this.boneStates["right_elbow"].rotation
    const rightWristTwistQuat = this.boneStates["right_wrist_twist"].rotation
    const rightWristQuat = this.boneStates["right_wrist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(rightArmQuat)
      .multiply(rightElbowQuat)
      .multiply(rightWristTwistQuat)
      .multiply(rightWristQuat)

    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localThumbMCP = Vector3.TransformCoordinates(thumbMCP, worldToFullParent)
    const localThumbIP = Vector3.TransformCoordinates(thumbIP, worldToFullParent)

    const thumbDirection = localThumbIP.subtract(localThumbMCP).normalize()
    const reference = new Vector3(-0.6236753178947897, -0.7034896546159694, -0.34078057998826367).normalize()

    return {
      name: "右親指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, thumbDirection, new Quaternion()),
    }
  }

  private solveRightThumb2(): BoneState {
    return this.solveFingerJoint("right_thumb_1", "右親指２", new Vector3(-1.0, 1.0, 0.0).normalize(), 0.85)
  }

  private solveRightIndex1(): BoneState {
    const indexMCP = this.getRightHandLandmark("index_mcp")
    const indexPIP = this.getRightHandLandmark("index_pip")
    if (!indexMCP || !indexPIP) return { name: "右人指１", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const rightArmQuat = this.boneStates["right_arm"].rotation
    const rightElbowQuat = this.boneStates["right_elbow"].rotation
    const rightWristQuat = this.boneStates["right_wrist"].rotation
    const rightWristTwistQuat = this.boneStates["right_wrist_twist"].rotation

    // Transform to wrist local space (NOT including wrist twist)
    const wristSpaceQuat = upperBodyQuat
      .multiply(rightArmQuat)
      .multiply(rightElbowQuat)
      .multiply(rightWristTwistQuat)
      .multiply(rightWristQuat)

    const wristSpaceMatrix = new Matrix()
    Matrix.FromQuaternionToRef(wristSpaceQuat, wristSpaceMatrix)
    const worldToWristSpace = wristSpaceMatrix.invert()

    const localIndexMCP = Vector3.TransformCoordinates(indexMCP, worldToWristSpace)
    const localIndexPIP = Vector3.TransformCoordinates(indexPIP, worldToWristSpace)

    const indexDirection = localIndexPIP.subtract(localIndexMCP).normalize()
    const reference = new Vector3(-0.8432487044803304, -0.5368678957658006, 0.026542134206687714).normalize()

    return {
      name: "右人指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, indexDirection, new Quaternion()),
    }
  }

  private solveRightIndex2(): BoneState {
    return this.solveFingerJoint("right_index_1", "右人指２", new Vector3(-0.031, 0.0, 0.993).normalize(), 0.9)
  }

  private solveRightIndex3(): BoneState {
    return this.solveFingerJoint("right_index_1", "右人指３", new Vector3(-0.031, 0.0, 0.993).normalize(), 0.65)
  }

  private solveRightMiddle1(): BoneState {
    const middleMCP = this.getRightHandLandmark("middle_mcp")
    const middlePIP = this.getRightHandLandmark("middle_pip")
    if (!middleMCP || !middlePIP) return { name: "右中指１", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const rightArmQuat = this.boneStates["right_arm"].rotation
    const rightElbowQuat = this.boneStates["right_elbow"].rotation
    const rightWristTwistQuat = this.boneStates["right_wrist_twist"].rotation
    const rightWristQuat = this.boneStates["right_wrist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(rightArmQuat)
      .multiply(rightElbowQuat)
      .multiply(rightWristTwistQuat)
      .multiply(rightWristQuat)

    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localMiddleMCP = Vector3.TransformCoordinates(middleMCP, worldToFullParent)
    const localMiddlePIP = Vector3.TransformCoordinates(middlePIP, worldToFullParent)

    const middleDirection = localMiddlePIP.subtract(localMiddleMCP).normalize()
    const reference = new Vector3(-0.830394244494938, -0.5566311582035673, 0.024640463198495298).normalize()

    return {
      name: "右中指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, middleDirection, new Quaternion()),
    }
  }

  private solveRightMiddle2(): BoneState {
    return this.solveFingerJoint("right_middle_1", "右中指２", new Vector3(0.03, 0.0, 0.996).normalize(), 0.9)
  }

  private solveRightMiddle3(): BoneState {
    return this.solveFingerJoint("right_middle_1", "右中指３", new Vector3(0.03, 0.0, 0.996).normalize(), 0.65)
  }

  private solveRightRing1(): BoneState {
    const ringMCP = this.getRightHandLandmark("ring_mcp")
    const ringPIP = this.getRightHandLandmark("ring_pip")
    if (!ringMCP || !ringPIP) return { name: "右薬指１", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const rightArmQuat = this.boneStates["right_arm"].rotation
    const rightElbowQuat = this.boneStates["right_elbow"].rotation
    const rightWristTwistQuat = this.boneStates["right_wrist_twist"].rotation
    const rightWristQuat = this.boneStates["right_wrist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(rightArmQuat)
      .multiply(rightElbowQuat)
      .multiply(rightWristTwistQuat)
      .multiply(rightWristQuat)

    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localRingMCP = Vector3.TransformCoordinates(ringMCP, worldToFullParent)
    const localRingPIP = Vector3.TransformCoordinates(ringPIP, worldToFullParent)

    const ringDirection = localRingPIP.subtract(localRingMCP).normalize()
    const reference = new Vector3(-0.8076382239720394, -0.5884013252930373, 0.03878633229228).normalize()

    return {
      name: "右薬指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, ringDirection, new Quaternion()),
    }
  }

  private solveRightRing2(): BoneState {
    return this.solveFingerJoint("right_ring_1", "右薬指２", new Vector3(0.048, 0.0, 0.997).normalize(), 0.88)
  }

  private solveRightRing3(): BoneState {
    return this.solveFingerJoint("right_ring_1", "右薬指３", new Vector3(0.048, 0.0, 0.997).normalize(), 0.6)
  }

  private solveRightPinky1(): BoneState {
    const pinkyMCP = this.getRightHandLandmark("pinky_mcp")
    const pinkyPIP = this.getRightHandLandmark("pinky_pip")
    if (!pinkyMCP || !pinkyPIP) return { name: "右小指１", rotation: Quaternion.Identity() }

    const upperBodyQuat = this.boneStates["upper_body"].rotation
    const rightArmQuat = this.boneStates["right_arm"].rotation
    const rightElbowQuat = this.boneStates["right_elbow"].rotation
    const rightWristQuat = this.boneStates["right_wrist"].rotation
    const rightWristTwistQuat = this.boneStates["right_wrist_twist"].rotation

    const fullParentQuat = upperBodyQuat
      .multiply(rightArmQuat)
      .multiply(rightElbowQuat)
      .multiply(rightWristTwistQuat)
      .multiply(rightWristQuat)

    const fullParentMatrix = new Matrix()
    Matrix.FromQuaternionToRef(fullParentQuat, fullParentMatrix)
    const worldToFullParent = fullParentMatrix.invert()

    const localPinkyMCP = Vector3.TransformCoordinates(pinkyMCP, worldToFullParent)
    const localPinkyPIP = Vector3.TransformCoordinates(pinkyPIP, worldToFullParent)

    const pinkyDirection = localPinkyPIP.subtract(localPinkyMCP).normalize()
    const reference = new Vector3(-0.8462155810704232, -0.5276077240369134, 0.07449348891167894).normalize()

    return {
      name: "右小指１",
      rotation: Quaternion.FromUnitVectorsToRef(reference, pinkyDirection, new Quaternion()),
    }
  }

  private solveRightPinky2(): BoneState {
    return this.solveFingerJoint("right_pinky_1", "右小指２", new Vector3(0.088, 0.0, 0.997).normalize(), 0.85)
  }

  private solveRightPinky3(): BoneState {
    return this.solveFingerJoint("right_pinky_1", "右小指３", new Vector3(0.088, 0.0, 0.997).normalize(), 0.55)
  }

  private solveFingerJoint(baseJointName: string, jointName: string, bendAxis: Vector3, ratio: number): BoneState {
    // Extract bend degrees from base joint quaternion
    const baseRotation = this.boneStates[baseJointName].rotation
    const bendDegrees = this.extractBendDegrees(baseRotation, bendAxis)

    // Apply ratio to get degrees for this joint
    const adjustedDegrees = bendDegrees * ratio

    // Create quaternion directly from degrees (following MPL approach)
    const radians = (adjustedDegrees * Math.PI) / 180
    const halfAngle = radians / 2
    const sin = Math.sin(halfAngle)
    const cos = Math.cos(halfAngle)

    const rotation = new Quaternion(bendAxis.x * sin, bendAxis.y * sin, bendAxis.z * sin, cos)

    return {
      name: jointName,
      rotation: rotation,
    }
  }

  private extractBendDegrees(quat: Quaternion, bendAxis: Vector3): number {
    // Extract the total rotation angle from quaternion in degrees
    const totalAngle = 2 * Math.acos(Math.abs(quat.w)) * (180 / Math.PI)

    // Determine the sign based on the bend axis component
    const axisComponent = quat.x * bendAxis.x + quat.y * bendAxis.y + quat.z * bendAxis.z
    const sign = axisComponent < 0 ? -1 : 1

    return totalAngle * sign
  }
}
