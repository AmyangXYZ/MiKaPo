import Encoding from "encoding-japanese"
import { BoneState } from "./solver"
import { FaceMorphWeights } from "./face-blendshape-solver"

export interface RecordedFrame {
  boneStates: BoneState[]
  morphWeights: FaceMorphWeights | null
}

function encodeShiftJIS(str: string): Uint8Array {
  const unicodeArray = Encoding.stringToCode(str)
  const sjisArray = Encoding.convert(unicodeArray, {
    to: "SJIS",
    from: "UNICODE",
  })
  return new Uint8Array(sjisArray)
}

// VMD file creation
// frameMultiplier: 1 = 30fps, 2 = 15fps effective (slower), etc.
export function createVMD(frames: RecordedFrame[], frameMultiplier: number = 1): Blob {
  if (frames.length === 0) {
    return new Blob()
  }

  const writeBoneFrame = (
    dataView: DataView,
    offset: number,
    name: string,
    frame: number,
    rotation: { x: number; y: number; z: number; w: number },
  ): number => {
    const nameBytes = encodeShiftJIS(name)
    for (let i = 0; i < 15; i++) {
      dataView.setUint8(offset + i, i < nameBytes.length ? nameBytes[i] : 0)
    }
    offset += 15

    dataView.setUint32(offset, frame, true)
    offset += 4

    // Translation x, y, z — always zero (rotation-only capture)
    dataView.setFloat32(offset, 0, true)
    offset += 4
    dataView.setFloat32(offset, 0, true)
    offset += 4
    dataView.setFloat32(offset, 0, true)
    offset += 4

    dataView.setFloat32(offset, rotation.x, true)
    offset += 4
    dataView.setFloat32(offset, rotation.y, true)
    offset += 4
    dataView.setFloat32(offset, rotation.z, true)
    offset += 4
    dataView.setFloat32(offset, rotation.w, true)
    offset += 4

    // Interpolation parameters (64 bytes) - linear interpolation
    for (let i = 0; i < 64; i++) {
      dataView.setUint8(offset + i, 20)
    }
    offset += 64

    return offset
  }

  const writeMorphFrame = (dataView: DataView, offset: number, name: string, frame: number, weight: number): number => {
    const nameBytes = encodeShiftJIS(name)
    for (let i = 0; i < 15; i++) {
      dataView.setUint8(offset + i, i < nameBytes.length ? nameBytes[i] : 0)
    }
    offset += 15

    dataView.setUint32(offset, frame, true)
    offset += 4

    dataView.setFloat32(offset, weight, true)
    offset += 4

    return offset
  }

  const frameCount = frames.length
  const boneCnt = frames[0].boneStates.length

  // Count morph frames (only if we have morph data)
  const morphNames = frames[0].morphWeights ? Object.keys(frames[0].morphWeights) : []
  const morphCnt = morphNames.length

  const headerSize = 30 + 20
  const boneFrameSize = 15 + 4 + 12 + 16 + 64
  const morphFrameSize = 15 + 4 + 4
  const totalSize =
    headerSize + 4 + boneFrameSize * frameCount * boneCnt + 4 + morphFrameSize * frameCount * morphCnt + 4 + 4 + 4

  const buffer = new ArrayBuffer(totalSize)
  const dataView = new DataView(buffer)
  let offset = 0

  // Write header
  const header = "Vocaloid Motion Data 0002"
  for (let i = 0; i < 30; i++) {
    dataView.setUint8(offset + i, i < header.length ? header.charCodeAt(i) : 0)
  }
  offset += 30

  // Write model name (empty)
  for (let i = 0; i < 20; i++) {
    dataView.setUint8(offset + i, 0)
  }
  offset += 20

  // Write bone frame count
  dataView.setUint32(offset, frameCount * boneCnt, true)
  offset += 4

  // Generate bone keyframes
  // Frame numbers are multiplied to adjust playback speed
  // frameMultiplier=1 means 30fps, frameMultiplier=2 means 15fps effective (slower)
  for (let i = 0; i < frameCount; i++) {
    const frameNumber = i * frameMultiplier
    for (const boneState of frames[i].boneStates) {
      offset = writeBoneFrame(dataView, offset, boneState.name, frameNumber, boneState.rotation)
    }
  }

  // Write morph frame count
  dataView.setUint32(offset, frameCount * morphCnt, true)
  offset += 4

  // Generate morph keyframes
  for (let i = 0; i < frameCount; i++) {
    const frameNumber = i * frameMultiplier
    const morphWeights = frames[i].morphWeights
    if (morphWeights) {
      for (const morphName of morphNames) {
        const weight = morphWeights[morphName] ?? 0
        offset = writeMorphFrame(dataView, offset, morphName, frameNumber, weight)
      }
    }
  }

  // Write counts for other frame types (all 0)
  dataView.setUint32(offset, 0, true) // Camera keyframe count
  offset += 4
  dataView.setUint32(offset, 0, true) // Light keyframe count
  offset += 4
  dataView.setUint32(offset, 0, true) // Self shadow keyframe count
  offset += 4

  return new Blob([buffer], { type: "application/octet-stream" })
}
