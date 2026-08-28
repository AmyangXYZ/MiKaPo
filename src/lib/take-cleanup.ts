import { Landmark } from "@mediapipe/tasks-vision"
import { FacingTracker, MIRRORED, mirrorFrame } from "./facing"

/**
 * Landmark cleanup for a finished take.
 *
 * Live capture sees one frame at a time and has to decide immediately whether
 * a landmark that jumped is a fast move or a bad detection. A recorded take
 * has both neighbours, which makes that question answerable: a point that
 * departs and returns within a frame is noise, and a point that departs and
 * stays is motion.
 *
 * Both passes run on the LANDMARKS, before any solving. Noise removed here
 * never reaches the geometry that amplifies it — a centimetre on a 7cm ear
 * span is eight degrees of head rotation, and no amount of smoothing applied
 * to the output recovers what the input got wrong.
 */

/** One frame's worth of the landmark sets the solver reads. */
export interface TakeFrame {
  pose: Landmark[] | null
  leftHand: Landmark[] | null
  rightHand: Landmark[] | null
  /** Whether the face detector found a face on this frame — the one honest
   *  signal in a monocular stream for which way the body is facing. */
  faceSeen?: boolean
}

type Channel = "x" | "y" | "z"
const CHANNELS: Channel[] = ["x", "y", "z"]

/** Quadratic Savitzky-Golay, 7 wide — the same fit the output pass uses. */
const SG7 = [-2, 3, 6, 7, 6, 3, -2]
const SG7_HALF = 3

type LandmarkSet = "pose" | "leftHand" | "rightHand"

function seriesOf(frames: TakeFrame[], key: LandmarkSet): (Landmark[] | null)[] {
  return frames.map((f) => f[key])
}

/**
 * Median of three, per landmark per channel.
 *
 * A single frame where the detector put a wrist somewhere impossible is the
 * classic export artefact: it survives smoothing as a softened spike, and it
 * is what the model does a twitch for. Taking the middle of three neighbours
 * removes it outright while leaving a genuine step — where the new value is
 * the median — untouched.
 */
function medianOfThree(sets: (Landmark[] | null)[]): void {
  const n = sets.length
  if (n < 3) return
  const prev: Landmark[][] = []
  for (let i = 0; i < n; i++) {
    const cur = sets[i]
    if (cur) prev.push(cur.map((p) => ({ ...p })))
    else prev.push([])
  }
  for (let i = 1; i < n - 1; i++) {
    const a = prev[i - 1]
    const b = prev[i]
    const c = prev[i + 1]
    const out = sets[i]
    if (!out || a.length !== out.length || b.length !== out.length || c.length !== out.length) continue
    for (let j = 0; j < out.length; j++) {
      for (const ch of CHANNELS) {
        const x = a[j][ch]
        const y = b[j][ch]
        const z = c[j][ch]
        out[j][ch] = Math.max(Math.min(x, y), Math.min(Math.max(x, y), z))
      }
    }
  }
}

/**
 * Zero-phase smoothing per landmark per channel.
 *
 * A polynomial fit rather than an average: it removes shake while keeping the
 * height of a real transient, and reading both directions at once means no
 * timing shift at all. `zGain` smooths the z channel harder, because z is
 * inferred rather than measured and carries most of the noise the solver
 * turns into rotation.
 */
function savitzkyGolay(sets: (Landmark[] | null)[], zGain: number): void {
  const n = sets.length
  if (n < SG7.length) return
  const src: (Landmark[] | null)[] = sets.map((s) => (s ? s.map((p) => ({ ...p })) : null))
  for (let i = 0; i < n; i++) {
    const out = sets[i]
    if (!out) continue
    for (let j = 0; j < out.length; j++) {
      for (const ch of CHANNELS) {
        let sum = 0
        let wsum = 0
        for (let k = -SG7_HALF; k <= SG7_HALF; k++) {
          const f = src[i + k]
          if (!f || j >= f.length) continue
          const w = SG7[k + SG7_HALF]
          sum += f[j][ch] * w
          wsum += w
        }
        if (wsum === 0) continue
        const fitted = sum / wsum
        // z leans on the fit; x and y are measured and mostly keep themselves.
        const gain = ch === "z" ? zGain : 1
        out[j][ch] = out[j][ch] + (fitted - out[j][ch]) * gain
      }
    }
  }
}


// ─── Left/right continuity ─────────────────────────────────────────────────
//
// A pose model decides which limb is which from what it can see, and a body
// turned away from the camera gives it very little to go on. The usual result
// is that the labels swap for the frames where the subject faces away: the
// shoulder line reverses between one frame and the next, the solver rotates
// the trunk 180° to match, and a full turn plays back as a half turn that
// changes its mind.
//
// A turn is continuous and a mislabel is not, so the two are easy to tell
// apart with the neighbours in hand: at 30fps, the shoulder line moving more
// than 140° in a single frame would be four full revolutions a second.
//
// Each frame is taken as it comes or with its sides exchanged, whichever
// continues the previous frame — a decision that costs nothing when the labels
// were right, because then the unswapped reading is always the closer one.

/** Bearing of the shoulder line in the ground plane. */
function shoulderYaw(pose: Landmark[], swapped: boolean): number {
  const l = pose[swapped ? 12 : 11]
  const r = pose[swapped ? 11 : 12]
  if (!l || !r) return NaN
  return Math.atan2(l.x - r.x, l.z - r.z)
}

function angleGap(a: number, b: number): number {
  let d = Math.abs(a - b) % (2 * Math.PI)
  if (d > Math.PI) d = 2 * Math.PI - d
  return d
}

/**
 * Hold left and right consistent across a take, exchanging the sides on frames
 * where the detector read the body from the wrong side. The hands swap with
 * the pose they belong to.
 *
 * The first frame is taken on trust: nothing precedes it to be continuous
 * with, and a subject almost always starts facing the camera.
 */
export function stabilizeHandedness(frames: TakeFrame[]): number {
  let previous = NaN
  let swapped = false
  let repaired = 0
  for (const f of frames) {
    const pose = f.pose
    if (!pose || pose.length < 33) continue
    const asIs = shoulderYaw(pose, swapped)
    const flipped = shoulderYaw(pose, !swapped)
    if (Number.isNaN(asIs) || Number.isNaN(flipped)) continue
    if (!Number.isNaN(previous)) {
      // Only a reading that is WILDLY discontinuous gets overruled, and only
      // when exchanging the sides actually restores continuity.
      const keepGap = angleGap(asIs, previous)
      const swapGap = angleGap(flipped, previous)
      if (keepGap > 140 * (Math.PI / 180) && swapGap < keepGap * 0.5) {
        swapped = !swapped
        repaired++
      }
    }
    if (swapped) {
      for (const [a, b] of MIRRORED) {
        if (a < pose.length && b < pose.length) {
          const t = pose[a]
          pose[a] = pose[b]
          pose[b] = t
        }
      }
      const hand = f.leftHand
      f.leftHand = f.rightHand
      f.rightHand = hand
    }
    previous = shoulderYaw(pose, false)
  }
  return repaired
}


// ─── Dropouts ──────────────────────────────────────────────────────────────
//
// A limb that the detector loses for a moment — crossing the body, blurred by
// its own speed, briefly behind the torso — comes back where the motion says
// it should. Live capture cannot know that, so it holds and then eases the
// bone toward rest, which is the right call when the limb might be gone for
// good and reads as the arm resetting when it was gone for a third of a
// second.
//
// A take knows. A gap with confident readings on both sides is a hole to
// bridge, and the only honest question is how long a hole is still a hole.

/** Below this the solver treats a landmark as unmeasured. */
const CONFIDENT = 0.35
/** Longer than this and the limb was genuinely away; the crossfade should
 *  have it. Two thirds of a second at VMD's 30fps. */
const MAX_BRIDGE_FRAMES = 20

/**
 * Fill short low-confidence spans by running the motion straight through
 * them, and hand the filled frames the confidence of the readings they were
 * interpolated from.
 */
function bridgeDropouts(sets: (Landmark[] | null)[]): number {
  const n = sets.length
  if (n < 3) return 0
  let bridged = 0
  const width = sets.find((s) => s)?.length ?? 0
  for (let j = 0; j < width; j++) {
    let i = 0
    while (i < n) {
      const cur = sets[i]
      const confident = cur && j < cur.length && (cur[j].visibility ?? 1) >= CONFIDENT
      if (confident) {
        i++
        continue
      }
      // Walk to the end of the gap, and take the confident frames on each side.
      const start = i
      while (i < n) {
        const f = sets[i]
        if (f && j < f.length && (f[j].visibility ?? 1) >= CONFIDENT) break
        i++
      }
      const before = start - 1
      const after = i
      const span = after - before
      if (before < 0 || after >= n || span - 1 > MAX_BRIDGE_FRAMES) continue
      const a = sets[before]![j]
      const b = sets[after]![j]
      for (let k = start; k < after; k++) {
        const f = sets[k]
        if (!f || j >= f.length) continue
        const t = (k - before) / span
        f[j].x = a.x + (b.x - a.x) * t
        f[j].y = a.y + (b.y - a.y) * t
        f[j].z = a.z + (b.z - a.z) * t
        // The bridge is only as trustworthy as its ends.
        f[j].visibility = Math.min(a.visibility ?? 1, b.visibility ?? 1)
        bridged++
      }
    }
  }
  return bridged
}


// ─── Facing ────────────────────────────────────────────────────────────────
//
// A single camera cannot see which way a body faces. Depth is inferred, and
// the inference has two answers that fit the same picture equally well: a
// person turned a quarter to the left, and a person turned a quarter to the
// right. Pose models resolve it from training bias, which means a subject
// turning through profile is usually reported as turning back the way they
// came. A full spin plays as a half spin that reconsiders — smoothly, with no
// discontinuity anywhere for a continuity check to catch.
//
// The face is the tiebreaker. A face detector finds a face when someone is
// facing the camera and finds nothing when they turn their back, which is
// exactly the fact the pose stream is missing. Where the face is gone and the
// body is side-on — the moment the two readings agree, so switching between
// them is seamless — the pose is turned to face away.
//
// Rotating a bilaterally symmetric body 180° about its own vertical is the
// same as mirroring its depth and exchanging its sides, and that is how it is
// applied here.

/**
 * Carry a turn through the half of it the camera cannot see.
 *
 * The same tracker the live path runs, over a take that is already recorded:
 * the decision is per frame either way, and the reasoning does not improve by
 * knowing the ending. It resolves both halves of the problem — a detector that
 * reports a spin as walking up to profile and back down, and a detector that
 * alternates between the two readings while the subject's back is turned.
 */
export function continueTurns(frames: TakeFrame[]): number {
  const tracker = new FacingTracker()
  let turned = 0
  for (const f of frames) {
    if (!f.pose) continue
    if (tracker.update(f.pose, f.faceSeen !== false)) {
      mirrorFrame(f)
      turned++
    }
  }
  return turned
}

/**
 * Clean a whole take in place: turns carried through, sides made consistent,
 * brief dropouts bridged,
 * spikes removed, then smoothed without phase shift.
 *
 * A frame the detector missed entirely stays missing: with nothing of the body
 * in it, there is no gap to bridge, and the solver's crossfades know what to
 * do with an absence.
 */
export function cleanTake(
  frames: TakeFrame[],
  opts?: { zGain?: number },
): { sideRepairs: number; bridged: number; turned: number } {
  const zGain = opts?.zGain ?? 1
  // Sides first, facing second. A detector that merely exchanges left and
  // right leaves a half-turn jump, and turning the body around is the wrong
  // repair for it — fixing the labels first leaves the facing pass looking at
  // an honest turn.
  const sideRepairs = stabilizeHandedness(frames)
  const turned = continueTurns(frames)
  let bridged = 0
  for (const key of ["pose", "leftHand", "rightHand"] as const) {
    const sets = seriesOf(frames, key)
    bridged += bridgeDropouts(sets)
    medianOfThree(sets)
    savitzkyGolay(sets, zGain)
  }
  return { sideRepairs, bridged, turned }
}
