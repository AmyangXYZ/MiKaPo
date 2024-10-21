export type Body = {
  mainBody: NormalizedLandmark[] | null
  leftHand: NormalizedLandmark[] | null
  rightHand: NormalizedLandmark[] | null
  face: NormalizedLandmark[] | null
}

export type RecordedFrame = {
  boneFrames: BoneFrame[]
  morphFrames: MorphFrame[]
}

export type BoneFrame = {
  name: string
  rotation: Quaternion
  position: Vector3
}

export type MorphFrame = {
  name: string
  weight: number
}
