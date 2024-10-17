export type Body = {
  mainBody: NormalizedLandmark[] | null
  leftHand: NormalizedLandmark[] | null
  rightHand: NormalizedLandmark[] | null
}

export type BoneFrame = {
  name: string
  rotation: Quaternion
  position: Vector3
}
