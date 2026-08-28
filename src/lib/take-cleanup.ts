import { Landmark } from "@mediapipe/tasks-vision"

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
}

type Channel = "x" | "y" | "z"
const CHANNELS: Channel[] = ["x", "y", "z"]

/** Quadratic Savitzky-Golay, 7 wide — the same fit the output pass uses. */
const SG7 = [-2, 3, 6, 7, 6, 3, -2]
const SG7_HALF = 3

function seriesOf(frames: TakeFrame[], key: keyof TakeFrame): (Landmark[] | null)[] {
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

/**
 * Clean a whole take in place: despike, then smooth without phase shift.
 *
 * Missing frames are left missing — a hole in the detection is information the
 * solver's own crossfades know how to handle, and filling it here would invent
 * a pose nobody struck.
 */
export function cleanTake(frames: TakeFrame[], opts?: { zGain?: number }): void {
  const zGain = opts?.zGain ?? 1
  for (const key of ["pose", "leftHand", "rightHand"] as const) {
    const sets = seriesOf(frames, key)
    medianOfThree(sets)
    savitzkyGolay(sets, zGain)
  }
}
