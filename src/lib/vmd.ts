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


// ─── Keyframe reduction ────────────────────────────────────────────────────
//
// A capture writes a key on every frame of a bone that barely moves, which is
// how a two-minute take becomes a file an animator has to clean before they
// can touch it. Held poses and steady arcs are the same motion with far fewer
// keys, and a track carrying only the frames that decide its shape is one a
// human can actually edit.
//
// Top-down and error-bounded: keep the ends, find the frame that a straight
// run between them gets most wrong, and split there if it is off by more than
// the tolerance. Because this writer emits VMD's linear handles, playback
// between two kept keys IS the straight run being measured against, so the
// error checked here is the error MMD will show.
//
// Tolerances are fixed, and are the same ones Reze Studio simplifies to.

/** Rotation drift nobody sees on a character, in degrees. */
const SIMPLIFY_ROT_DEG = 0.5
/** ~3mm at character scale. */
const SIMPLIFY_TRANS = 0.01
/** A morph weight runs 0..1; a hundredth of that is invisible. */
const SIMPLIFY_MORPH = 0.01

const _slerp = Quat.identity()

/** How far frame `k` sits from the straight run a→b, as a multiple of the
 *  tolerance — so rotation and translation share one threshold. */
function boneDeviation(k: BoneKeyframe, a: BoneKeyframe, b: BoneKeyframe, t: number): number {
  Quat.slerpInto(a.rotation, b.rotation, t, _slerp)
  const rotDeg = (Quat.angleTo(k.rotation, _slerp) * 180) / Math.PI
  const tx = a.translation.x + (b.translation.x - a.translation.x) * t
  const ty = a.translation.y + (b.translation.y - a.translation.y) * t
  const tz = a.translation.z + (b.translation.z - a.translation.z) * t
  const dist = Math.hypot(k.translation.x - tx, k.translation.y - ty, k.translation.z - tz)
  return Math.max(rotDeg / SIMPLIFY_ROT_DEG, dist / SIMPLIFY_TRANS)
}

function reduceBoneTrack(keys: BoneKeyframe[]): BoneKeyframe[] {
  if (keys.length <= 2) return keys
  const keep = new Uint8Array(keys.length)
  keep[0] = 1
  keep[keys.length - 1] = 1
  const stack: [number, number][] = [[0, keys.length - 1]]
  while (stack.length > 0) {
    const [i0, i1] = stack.pop()!
    if (i1 - i0 < 2) continue
    const a = keys[i0]
    const b = keys[i1]
    const span = b.frame - a.frame
    let worst = -1
    let worstErr = 1
    for (let i = i0 + 1; i < i1; i++) {
      const t = span > 0 ? (keys[i].frame - a.frame) / span : 0
      const err = boneDeviation(keys[i], a, b, t)
      if (err > worstErr) {
        worstErr = err
        worst = i
      }
    }
    if (worst < 0) continue
    keep[worst] = 1
    stack.push([i0, worst], [worst, i1])
  }
  return keys.filter((_, i) => keep[i] === 1)
}

function reduceMorphTrack(keys: MorphKeyframe[]): MorphKeyframe[] {
  if (keys.length <= 2) return keys
  const keep = new Uint8Array(keys.length)
  keep[0] = 1
  keep[keys.length - 1] = 1
  const stack: [number, number][] = [[0, keys.length - 1]]
  while (stack.length > 0) {
    const [i0, i1] = stack.pop()!
    if (i1 - i0 < 2) continue
    const a = keys[i0]
    const b = keys[i1]
    const span = b.frame - a.frame
    let worst = -1
    let worstErr = SIMPLIFY_MORPH
    for (let i = i0 + 1; i < i1; i++) {
      const t = span > 0 ? (keys[i].frame - a.frame) / span : 0
      const err = Math.abs(keys[i].weight - (a.weight + (b.weight - a.weight) * t))
      if (err > worstErr) {
        worstErr = err
        worst = i
      }
    }
    if (worst < 0) continue
    keep[worst] = 1
    stack.push([i0, worst], [worst, i1])
  }
  return keys.filter((_, i) => keep[i] === 1)
}

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
    boneTracks.set(name, reduceBoneTrack([...track.values()].sort((a, b) => a.frame - b.frame)))
  }
  for (const [name, track] of morphs) {
    morphTracks.set(name, reduceMorphTrack([...track.values()].sort((a, b) => a.frame - b.frame)))
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
