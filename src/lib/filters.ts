import { Quat } from "reze-engine"
import { quatDot } from "./math-utils"

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
    if (this.hasPrev && quatDot(this.prev, q) < 0) {
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
