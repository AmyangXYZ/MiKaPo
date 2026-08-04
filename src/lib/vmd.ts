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
        // Only センター and the leg IK bones move; everything else keeps its
        // rest translation, which is what MMD expects of a rotation rig.
        translation: bone.translation
          ? new Vec3(bone.translation.x, bone.translation.y, bone.translation.z)
          : ZERO,
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
  // This capture solves FK rotations for the legs, so the leg IK chains have to
  // stand down or they would drive the feet toward IK bones this motion never
  // keyframes. Declared in the clip; the engine writes it into the file and
  // honours it on playback.
  // Leg IK is switched ON exactly when the capture actually keyframes the IK
  // bones. Without those tracks MMD's solver would drag the legs toward IK
  // bones the motion never touches, overriding the FK it does carry — with
  // them, the file is a native MMD leg rig and editable as one. つま先ＩＫ stays
  // down either way: nothing here solves toe direction.
  const drivesFootIk = bones.has("左足ＩＫ") || bones.has("右足ＩＫ")
  const ikTracks = new Map<string, { frame: number; enabled: boolean }[]>([
    ["左足ＩＫ", [{ frame: 0, enabled: drivesFootIk }]],
    ["右足ＩＫ", [{ frame: 0, enabled: drivesFootIk }]],
    ["左つま先ＩＫ", [{ frame: 0, enabled: false }]],
    ["右つま先ＩＫ", [{ frame: 0, enabled: false }]],
  ])
  return { boneTracks, morphTracks, ikTracks, frameCount: last + 1 }
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
