import { Quat, Vec3 } from "reze-engine"

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
  private prev = Quat.identity()
  private hasPrev = false
  private prevTs = 0
  private speedFilter: OneEuroFilter
  private readonly minCutoff: number
  private readonly beta: number
  /** Fastest rotation treated as real, radians/second.
   *
   *  Set against what a joint actually does, not what is comfortable: 30 rad/s
   *  is about 1700°/s, which a dance snap or a thrown punch reaches easily —
   *  an elbow in a throw passes 2000°/s. The old 14 (800°/s) sat below the peak
   *  of nearly every deliberate fast action, so those actions were clipped at
   *  the exact frames they were supposed to be fastest, and arrived short. The
   *  acceleration limiter below is what rejects glitches; this only has to
   *  refuse the physically impossible. */
  maxSpeed = 30
  /** Fastest CHANGE in rotation speed, radians/second². This is what separates a
   *  detection glitch from a fast move: a real limb accelerates over several
   *  frames, while an outlier arrives at full speed from nothing. A velocity cap
   *  alone cannot tell them apart, because the first frame of a genuine snap
   *  looks identical to a spike.
   *
   *  At 30 fps this admits ~15 rad/s of new speed per frame, so a real strike
   *  is at full speed by its second frame instead of its fourth. A one-frame
   *  spike still gets throttled, and what survives meets a cutoff that is still
   *  narrow — the speed estimate it would need to widen has not risen yet. */
  maxAccel = 450
  private prevSpeed = 0

  constructor(minCutoff: number, beta: number, dCutoff: number) {
    this.minCutoff = minCutoff
    this.beta = beta
    // The speed signal gets its own gentle low-pass, as One-Euro prescribes.
    this.speedFilter = new OneEuroFilter(dCutoff, 0, dCutoff)
  }

  /**
   * Filters `q` into `out` (allocation-free; `out` may be a persistent quat).
   *
   * One-Euro driven by ANGULAR VELOCITY rather than per-component derivatives.
   * Four independent scalar filters each estimate their own "speed" from a
   * quaternion component, which is not a rate of anything physical: the same
   * rotation splits differently across components depending where it sits on
   * the sphere, so the adaptive term — the whole point of One-Euro, the part
   * that is supposed to open up and let a fast move through — fires
   * inconsistently and a quick pose arrives softened. Measuring the actual
   * rotation between frames gives one honest speed, shared by the blend.
   */
  filterInto(q: Quat, ts: number, out: Quat): Quat {
    let x = q.x,
      y = q.y,
      z = q.z,
      w = q.w
    // Hemisphere flip: keep dot(prev, raw) >= 0 so the blend takes the short way
    // around the 4D sphere.
    if (this.hasPrev && Quat.dot(this.prev, q) < 0) {
      x = -x
      y = -y
      z = -z
      w = -w
    }
    if (!this.hasPrev) {
      out.setXYZW(x, y, z, w)
      this.prev.set(out)
      this.prevTs = ts
      this.hasPrev = true
      return out
    }
    const dt = (ts - this.prevTs) / 1000
    // Discontinuity (seek backward, long stall): reseed rather than smooth across it.
    if (dt <= 0 || dt > 1.0) {
      out.setXYZW(x, y, z, w)
      this.prev.set(out)
      this.prevTs = ts
      this.speedFilter.reset()
      return out
    }

    // Angle between the last output and this sample, in radians per second.
    const dot = Math.min(1, Math.abs(this.prev.x * x + this.prev.y * y + this.prev.z * z + this.prev.w * w))
    const step = 2 * Math.acos(dot)
    let speed = step / dt

    // Admit only what a limb could physically have done since the last frame:
    // no faster than maxSpeed, and no more than maxAccel faster than it was
    // already going. A glitch is throttled to near nothing because it comes
    // from rest; a real snap builds over two or three frames and arrives
    // essentially intact. The clamped speed is what feeds the cutoff, so an
    // outlier can never widen the filter that is meant to suppress it.
    const allowed = Math.min(this.maxSpeed, this.prevSpeed + this.maxAccel * dt)
    if (speed > allowed && step > 1e-6) {
      const t = (allowed * dt) / step
      x = this.prev.x + (x - this.prev.x) * t
      y = this.prev.y + (y - this.prev.y) * t
      z = this.prev.z + (z - this.prev.z) * t
      w = this.prev.w + (w - this.prev.w) * t
      const len = Math.hypot(x, y, z, w)
      if (len > 1e-9) {
        x /= len
        y /= len
        z /= len
        w /= len
      }
      speed = allowed
    }
    this.prevSpeed = speed

    const cutoff = this.minCutoff + this.beta * this.speedFilter.filter(speed, ts)
    const tau = 1 / (2 * Math.PI * cutoff)
    const alpha = 1 / (1 + tau / dt)

    // One blend for the whole rotation, at one honest cutoff.
    out.setXYZW(
      this.prev.x + (x - this.prev.x) * alpha,
      this.prev.y + (y - this.prev.y) * alpha,
      this.prev.z + (z - this.prev.z) * alpha,
      this.prev.w + (w - this.prev.w) * alpha,
    )
    out.normalize()
    this.prev.set(out)
    this.prevTs = ts
    return out
  }

  reset(): void {
    this.hasPrev = false
    this.prevTs = 0
    this.prevSpeed = 0
    this.speedFilter.reset()
  }
}

/** Three One-Euro filters for a position, with the same physical plausibility
 *  limits the rotation filter uses.
 *
 *  These carry the IK targets, which sit at the end of a long lever: a rotation
 *  glitch the filter shaved down to a few degrees still throws a foot a long
 *  way. Limiting the position directly is what stops that becoming a visible
 *  kick. Units are the model's own (an MMD unit is roughly 8 cm), so the
 *  defaults allow a fast step and refuse a teleport. */
export class Vec3OneEuroFilter {
  private fx: OneEuroFilter
  private fy: OneEuroFilter
  private fz: OneEuroFilter
  private px = 0
  private py = 0
  private pz = 0
  private hasPrev = false
  private prevTs = 0
  private prevSpeed = 0
  /** An MMD unit is roughly 8 cm, so 160 is about 13 m/s — the speed a foot
   *  actually reaches in a kick. The old 55 (4.4 m/s) was a brisk walking
   *  step, and capped every kick and stamp at a fraction of its travel. */
  maxSpeed = 160
  maxAccel = 1200

  constructor(minCutoff: number, beta: number, dCutoff: number) {
    this.fx = new OneEuroFilter(minCutoff, beta, dCutoff)
    this.fy = new OneEuroFilter(minCutoff, beta, dCutoff)
    this.fz = new OneEuroFilter(minCutoff, beta, dCutoff)
  }

  filterInto(x: number, y: number, z: number, ts: number, out: Vec3): Vec3 {
    if (this.hasPrev) {
      const dt = (ts - this.prevTs) / 1000
      if (dt > 0 && dt <= 1.0) {
        const step = Math.hypot(x - this.px, y - this.py, z - this.pz)
        const speed = step / dt
        const allowed = Math.min(this.maxSpeed, this.prevSpeed + this.maxAccel * dt)
        if (speed > allowed && step > 1e-6) {
          const t = (allowed * dt) / step
          x = this.px + (x - this.px) * t
          y = this.py + (y - this.py) * t
          z = this.pz + (z - this.pz) * t
          this.prevSpeed = allowed
        } else {
          this.prevSpeed = speed
        }
      } else {
        this.prevSpeed = 0
      }
    }
    out.setXYZ(this.fx.filter(x, ts), this.fy.filter(y, ts), this.fz.filter(z, ts))
    this.px = out.x
    this.py = out.y
    this.pz = out.z
    this.prevTs = ts
    this.hasPrev = true
    return out
  }

  reset(): void {
    this.fx.reset()
    this.fy.reset()
    this.fz.reset()
    this.hasPrev = false
    this.prevSpeed = 0
    this.prevTs = 0
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
