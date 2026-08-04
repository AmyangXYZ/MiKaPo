import { Quat } from "reze-engine"

// One-Euro filter (Casiez et al. 2012): adaptive low-pass whose cutoff rises with
// speed — smooths jitter at rest without lagging fast motion.
export class OneEuroFilter {
  private prev: number | null = null
  private prevDeriv = 0
  private prevTs: number | null = null

  constructor(
    private minCutoff: number,
    private beta: number,
    private dCutoff: number,
  ) {}

  /** `ts` is in milliseconds on any monotonic-per-stream clock (media time or wall time). */
  filter(value: number, ts: number): number {
    if (this.prev === null || this.prevTs === null) {
      this.prev = value
      this.prevTs = ts
      return value
    }
    const dt = (ts - this.prevTs) / 1000
    // Discontinuity (seek backward, long stall): reseed instead of smoothing across the cut.
    if (dt <= 0 || dt > 1.0) {
      this.prev = value
      this.prevDeriv = 0
      this.prevTs = ts
      return value
    }

    const rawDeriv = (value - this.prev) / dt
    const aD = OneEuroFilter.smoothing(this.dCutoff, dt)
    const filteredDeriv = aD * rawDeriv + (1 - aD) * this.prevDeriv

    const cutoff = this.minCutoff + this.beta * Math.abs(filteredDeriv)
    const a = OneEuroFilter.smoothing(cutoff, dt)
    const filtered = a * value + (1 - a) * this.prev

    this.prev = filtered
    this.prevDeriv = filteredDeriv
    this.prevTs = ts
    return filtered
  }

  reset(): void {
    this.prev = null
    this.prevDeriv = 0
    this.prevTs = null
  }

  private static smoothing(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff)
    return 1 / (1 + tau / dt)
  }
}

export class QuaternionOneEuroFilter {
  private fx: OneEuroFilter
  private fy: OneEuroFilter
  private fz: OneEuroFilter
  private fw: OneEuroFilter
  private prev = Quat.identity()
  private hasPrev = false

  constructor(minCutoff: number, beta: number, dCutoff: number) {
    this.fx = new OneEuroFilter(minCutoff, beta, dCutoff)
    this.fy = new OneEuroFilter(minCutoff, beta, dCutoff)
    this.fz = new OneEuroFilter(minCutoff, beta, dCutoff)
    this.fw = new OneEuroFilter(minCutoff, beta, dCutoff)
  }

  /** Filters `q` into `out` (allocation-free; `out` may be a persistent per-bone quat). */
  filterInto(q: Quat, ts: number, out: Quat): Quat {
    let x = q.x,
      y = q.y,
      z = q.z,
      w = q.w
    // Hemisphere flip: keep dot(prev, raw) >= 0 so component-wise filtering
    // doesn't take the long way around the 4D sphere.
    if (this.hasPrev && Quat.dot(this.prev, q) < 0) {
      x = -x
      y = -y
      z = -z
      w = -w
    }
    out.setXYZW(this.fx.filter(x, ts), this.fy.filter(y, ts), this.fz.filter(z, ts), this.fw.filter(w, ts))
    out.normalize()
    this.prev.set(out)
    this.hasPrev = true
    return out
  }

  reset(): void {
    this.fx.reset()
    this.fy.reset()
    this.fz.reset()
    this.fw.reset()
    this.hasPrev = false
  }
}

/**
 * Zero-phase smoothing for an already-captured take (offline export only).
 *
 * The live solver filters causally — it must, it cannot see the future — and
 * every causal filter trades jitter against lag. A finished sequence has no
 * such constraint, so this pass reads both directions at once.
 *
 * Savitzky-Golay, not another low-pass: it fits a local quadratic and takes its
 * centre value, which removes shake while KEEPING the amplitude of a fast
 * transient. Running One-Euro backwards over the already-forward-filtered take
 * cancels phase lag but filters twice, and the second pass flattens exactly the
 * poses that matter — a kick, a snap, a hit. A polynomial fit does not.
 *
 * Above `keepFastAbove` the original samples are kept outright (blending back in
 * over a ramp): during genuinely fast motion the residual shake is invisible,
 * and the peak is worth more than the polish.
 *
 * Per bone, hemisphere-aligned, in place.
 */
export function smoothTakeZeroPhase(
  frames: { time: number; boneStates: { name: string; rotation: Quat }[] }[],
  opts?: { keepFastAbove?: number; fullyKeepAbove?: number },
): void {
  if (frames.length < 5) return
  // Quadratic SG, 7-wide (±3 frames ≈ ±0.1 s at 30 fps).
  const K = [-2, 3, 6, 7, 6, 3, -2]
  const HALF = 3
  const lo = opts?.keepFastAbove ?? 2.5 // rad/s — brisk gesture
  const hi = opts?.fullyKeepAbove ?? 6.0 // rad/s — a kick or a snap

  const byBone = new Map<string, { q: Quat; frame: number }[]>()
  for (let i = 0; i < frames.length; i++) {
    for (const bs of frames[i].boneStates) {
      let seq = byBone.get(bs.name)
      if (!seq) byBone.set(bs.name, (seq = []))
      seq.push({ q: bs.rotation, frame: i })
    }
  }

  const aligned = new Quat(0, 0, 0, 1)
  const smoothed = new Quat(0, 0, 0, 1)
  for (const seq of byBone.values()) {
    if (seq.length < 7) continue
    // Hemisphere-align the whole sequence first: component-wise weighting is
    // meaningless across a sign flip.
    for (let i = 1; i < seq.length; i++) {
      if (Quat.dot(seq[i - 1].q, seq[i].q) < 0) {
        seq[i].q.setXYZW(-seq[i].q.x, -seq[i].q.y, -seq[i].q.z, -seq[i].q.w)
      }
    }
    const src = seq.map((e) => e.q.clone())
    for (let i = 0; i < seq.length; i++) {
      let x = 0,
        y = 0,
        z = 0,
        w = 0,
        wsum = 0
      for (let k = -HALF; k <= HALF; k++) {
        const j = i + k
        if (j < 0 || j >= src.length) continue // shrink at the edges
        const c = K[k + HALF]
        x += src[j].x * c
        y += src[j].y * c
        z += src[j].z * c
        w += src[j].w * c
        wsum += c
      }
      if (wsum === 0) continue
      smoothed.setXYZW(x / wsum, y / wsum, z / wsum, w / wsum)
      smoothed.normalize()

      // Local angular speed from the ORIGINAL samples, so the gate reads the
      // performance rather than its own output.
      const a = src[Math.max(0, i - 1)]
      const b = src[Math.min(src.length - 1, i + 1)]
      const dt = Math.max(1e-3, frames[seq[Math.min(seq.length - 1, i + 1)].frame].time - frames[seq[Math.max(0, i - 1)].frame].time)
      const speed = Quat.angleTo(a, b) / dt
      const keep = Math.min(1, Math.max(0, (speed - lo) / (hi - lo)))

      if (keep <= 0) aligned.set(smoothed)
      else if (keep >= 1) continue // fast: original stands
      else {
        Quat.nlerpInto(smoothed, src[i], keep, aligned)
        aligned.normalize()
      }
      seq[i].q.set(aligned)
    }
  }
}
