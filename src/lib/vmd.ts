import Encoding from "encoding-japanese"
import { Quat, Vec3, type AnimationClip, type BoneInterpolation, type BoneKeyframe, type MorphKeyframe } from "reze-engine"
import { BoneState } from "./solver"
import { FaceMorphWeights } from "./face-blendshape-solver"

/**
 * One captured pose, stamped with the time it belongs to.
 *
 * The stamp is what makes an export usable elsewhere. Wall-clock pacing drifts
 * whenever detection stalls or the tab is backgrounded, so a capture taken from
 * an uploaded video stamps `time` with the VIDEO's own clock and the result lines
 * up with the source frame for frame.
 */
export interface RecordedFrame {
  /** Seconds from the start of the capture, on whichever clock drives it. */
  time: number
  boneStates: BoneState[]
  morphWeights: FaceMorphWeights | null
}

/** VMD runs at 30 frames per second. Not a preference — it is the format. */
export const VMD_FPS = 30

/** VMD's standard linear handles, in its 127-space. */
const LINEAR: BoneInterpolation = {
  rotation: [
    { x: 20, y: 20 },
    { x: 107, y: 107 },
  ],
  translationX: [
    { x: 20, y: 20 },
    { x: 107, y: 107 },
  ],
  translationY: [
    { x: 20, y: 20 },
    { x: 107, y: 107 },
  ],
  translationZ: [
    { x: 20, y: 20 },
    { x: 107, y: 107 },
  ],
}

const ZERO = new Vec3(0, 0, 0)

/**
 * Captured poses → an engine animation clip.
 *
 * Frame numbers come from each pose's timestamp rather than its position in the
 * array, so a capture that dropped frames exports as motion with gaps at the
 * right moments instead of motion that has quietly sped up. Two poses landing on
 * one VMD frame keep the later one.
 *
 * The clip is handed to `model.loadClip()` and written out by the engine's own
 * `exportVmd()` — the same writer Reze Studio exports through, so a file from
 * here opens there, and in MMD, without a second implementation to keep honest.
 */
export function buildClip(frames: RecordedFrame[]): AnimationClip {
  const boneTracks = new Map<string, BoneKeyframe[]>()
  const morphTracks = new Map<string, MorphKeyframe[]>()
  if (frames.length === 0) return { boneTracks, morphTracks, frameCount: 0 }

  const origin = frames[0].time
  // Index by frame so a duplicate stamp overwrites rather than appending a
  // second key at the same time, which VMD readers disagree about.
  const bones = new Map<string, Map<number, BoneKeyframe>>()
  const morphs = new Map<string, Map<number, MorphKeyframe>>()
  let last = 0

  for (const captured of frames) {
    const frame = Math.max(0, Math.round((captured.time - origin) * VMD_FPS))
    if (frame > last) last = frame

    for (const bone of captured.boneStates) {
      let track = bones.get(bone.name)
      if (!track) bones.set(bone.name, (track = new Map()))
      track.set(frame, {
        boneName: bone.name,
        frame,
        rotation: new Quat(bone.rotation.x, bone.rotation.y, bone.rotation.z, bone.rotation.w),
        // MiKaPo solves rotation only; every bone keeps its rest translation.
        translation: ZERO,
        interpolation: LINEAR,
      })
    }

    if (!captured.morphWeights) continue
    for (const [morphName, weight] of Object.entries(captured.morphWeights)) {
      let track = morphs.get(morphName)
      if (!track) morphs.set(morphName, (track = new Map()))
      track.set(frame, { morphName, frame, weight })
    }
  }

  for (const [name, track] of bones) {
    boneTracks.set(name, [...track.values()].sort((a, b) => a.frame - b.frame))
  }
  for (const [name, track] of morphs) {
    morphTracks.set(name, [...track.values()].sort((a, b) => a.frame - b.frame))
  }
  return { boneTracks, morphTracks, frameCount: last + 1 }
}

/** What the export produced, for telling the user what they just got. */
export function clipSummary(clip: AnimationClip): { frames: number; seconds: number; bones: number; morphs: number } {
  return {
    frames: clip.frameCount,
    seconds: clip.frameCount / VMD_FPS,
    bones: clip.boneTracks.size,
    morphs: clip.morphTracks.size,
  }
}

// ─── IK disable block ────────────────────────────────────────────────────────
//
// MiKaPo solves FK rotations for every bone, legs included. MMD enables leg IK
// by default, and an enabled IK chain OVERRIDES whatever the leg bones were
// rotated to — driving them toward the IK bone instead, which this file never
// keyframes, so it sits at rest. The captured leg motion would be discarded and
// the character would stand there with moving arms.
//
// The fix is the one hand-keyed FK motions have always used: a single frame-0
// record switching the leg IK chains off. The engine's writer stops after the
// morph block (a valid motion-only VMD), so the remaining sections are appended
// here — camera, light and self-shadow counts of zero, then the IK block.
const IK_BONES_OFF = ["左足ＩＫ", "右足ＩＫ", "左つま先ＩＫ", "右つま先ＩＫ"]
const IK_NAME_SIZE = 20

function encodeShiftJIS(str: string): Uint8Array {
  const unicode = Encoding.stringToCode(str)
  return new Uint8Array(Encoding.convert(unicode, { to: "SJIS", from: "UNICODE" }))
}

/** VMD tail: `uint32 camera, uint32 light, uint32 shadow, uint32 ikCount, [ik record]`. */
export function withIkDisabled(motion: ArrayBuffer, boneNames: string[] = IK_BONES_OFF): ArrayBuffer {
  // frame(4) + visible(1) + ikCount(4) + per bone (20 name + 1 enabled)
  const record = 4 + 1 + 4 + boneNames.length * (IK_NAME_SIZE + 1)
  const out = new ArrayBuffer(motion.byteLength + 4 * 4 + record)
  new Uint8Array(out).set(new Uint8Array(motion))

  const view = new DataView(out)
  let offset = motion.byteLength
  // Camera, light and self-shadow: none. Written explicitly so readers that walk
  // the file sequentially reach the IK block instead of stopping at EOF.
  for (let i = 0; i < 3; i++, offset += 4) view.setUint32(offset, 0, true)

  view.setUint32(offset, 1, true) // one IK record
  offset += 4
  view.setUint32(offset, 0, true) // at frame 0
  offset += 4
  view.setUint8(offset, 1) // model visible
  offset += 1
  view.setUint32(offset, boneNames.length, true)
  offset += 4
  for (const name of boneNames) {
    const bytes = new Uint8Array(out, offset, IK_NAME_SIZE)
    bytes.fill(0)
    bytes.set(encodeShiftJIS(name).subarray(0, IK_NAME_SIZE))
    offset += IK_NAME_SIZE
    view.setUint8(offset, 0) // off
    offset += 1
  }
  return out
}
