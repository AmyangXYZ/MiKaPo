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
  /** Bone translations — センター and the foot IK targets. Offsets from rest, as
   *  VMD stores them. */
  translations?: Record<string, { x: number; y: number; z: number }> | null
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
        translation: ZERO,
        interpolation: LINEAR,
      })
    }

    // Translation-only bones: センター carries the body, the foot IK bones carry
    // the feet. They take no rotation, so an identity quaternion goes with them.
    for (const [name, t] of Object.entries(captured.translations ?? {})) {
      let track = bones.get(name)
      if (!track) bones.set(name, (track = new Map()))
      track.set(frame, {
        boneName: name,
        frame,
        rotation: new Quat(0, 0, 0, 1),
        translation: new Vec3(t.x, t.y, t.z),
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
