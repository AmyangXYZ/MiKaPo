import { Landmark } from "@mediapipe/tasks-vision"

/**
 * Which way a body faces.
 *
 * A single camera cannot see it. Depth is inferred, and the inference has two
 * answers that fit the same picture equally well: a person turned a quarter to
 * the left, and a person turned a quarter to the right. Pose models resolve it
 * from training bias, which means a subject turning through profile is usually
 * reported as turning back the way they came — smoothly, with nothing
 * discontinuous anywhere to catch. A full spin plays as a half spin that
 * reconsiders.
 *
 * The face is the tiebreaker the pose stream lacks: a face detector finds a
 * face when someone faces the camera and finds nothing when they turn their
 * back. Rotating a bilaterally symmetric body 180° about its own vertical is
 * the same as mirroring its depth and exchanging its sides, which is how the
 * correction is applied.
 */

/** Index pairs that exchange when a pose is read from the wrong side. */
export const MIRRORED: [number, number][] = [
  [1, 4], [2, 5], [3, 6], // eyes
  [7, 8], // ears
  [9, 10], // mouth
  [11, 12], [13, 14], [15, 16], // shoulders, elbows, wrists
  [17, 18], [19, 20], [21, 22], // pinky, index, thumb
  [23, 24], [25, 26], [27, 28], // hips, knees, ankles
  [29, 30], [31, 32], // heels, foot indices
]

/** How side-on the body is: 0 square to the camera, 1 fully in profile. */
export function profileness(pose: Landmark[]): number {
  const l = pose[11]
  const r = pose[12]
  if (!l || !r) return 0
  const dx = Math.abs(l.x - r.x)
  const dz = Math.abs(l.z - r.z)
  const span = Math.hypot(dx, dz)
  return span > 1e-6 ? dz / span : 0
}

/** The landmark sets one frame carries, whatever holds them. */
export interface MirrorableFrame {
  pose: Landmark[] | null
  leftHand: Landmark[] | null
  rightHand: Landmark[] | null
}

/** Turn a frame's pose to face the other way, in place. */
export function mirrorFrame(f: MirrorableFrame): void {
  const pose = f.pose
  if (!pose) return
  let cz = 0
  for (const p of pose) cz += p.z
  cz /= pose.length || 1
  for (const p of pose) p.z = 2 * cz - p.z
  for (const hand of [f.leftHand, f.rightHand]) {
    if (!hand) continue
    let hz = 0
    for (const p of hand) hz += p.z
    hz /= hand.length || 1
    for (const p of hand) p.z = 2 * hz - p.z
  }
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

/** Bearing of the shoulder line in the ground plane. */
function shoulderYaw(pose: Landmark[]): number {
  const l = pose[11]
  const r = pose[12]
  if (!l || !r) return NaN
  return Math.atan2(l.x - r.x, l.z - r.z)
}

/**
 * Whether a reading has the chest pointing away from the camera.
 *
 * forward = shoulderLine × up with y pointing down, which comes out as
 * (sz, 0, −sx); z grows with distance, so a forward with positive z faces
 * away. Mirroring exchanges the sides, so sx changes sign with it.
 */
function facesAway(pose: Landmark[], mirrored: boolean): boolean {
  const l = pose[11]
  const r = pose[12]
  if (!l || !r) return false
  const sx = (l.x - r.x) * (mirrored ? -1 : 1)
  return -sx > 0
}

/**
 * Frame by frame, for capture that cannot look ahead.
 *
 * Two problems, one decision. A subject facing away makes the detector
 * unstable as well as biased: it alternates between the two readings from one
 * frame to the next, which is the character snapping front, back, front. So
 * each frame is taken as whichever of the two continues the last one — that
 * alone holds the body still — and the face decides which branch is right
 * when the evidence for a branch has built up over several frames.
 */
export class FacingTracker {
  private mirrored = false
  private prevYaw = NaN
  private absent = 0
  private present = 0
  /** A face gone this long is a back, not a bad frame — about a third of a
   *  second at the rate capture actually runs. */
  private static readonly AWAY_RUN = 5
  /** A face found again this many times running has the subject back. */
  private static readonly HOME_RUN = 4

  reset(): void {
    this.mirrored = false
    this.prevYaw = NaN
    this.absent = 0
    this.present = 0
  }

  /** Whether this frame should be turned to face away. */
  update(pose: Landmark[] | null, faceSeen: boolean): boolean {
    if (!pose) return this.mirrored
    const raw = shoulderYaw(pose)
    if (Number.isNaN(raw)) return this.mirrored
    // Exchanging the sides and mirroring depth maps a shoulder bearing θ to
    // −θ: the two readings agree in profile, where the switch is free, and sit
    // a half turn apart square to the camera, where the difference is the
    // whole question.
    const yawFor = (m: boolean) => (m ? -raw : raw)

    if (faceSeen) {
      this.present++
      this.absent = 0
    } else {
      this.absent++
      this.present = 0
    }

    // Continuity decides first, and decides most: whichever reading follows on
    // from the last one. This is what holds the body still while the detector
    // alternates between its two answers.
    let chosen = false
    if (!Number.isNaN(this.prevYaw)) {
      const gap = (m: boolean) => {
        let d = yawFor(m) - this.prevYaw
        while (d > Math.PI) d -= 2 * Math.PI
        while (d < -Math.PI) d += 2 * Math.PI
        return Math.abs(d)
      }
      chosen = gap(true) < gap(false)
    }

    // Then the face overrules it, in the one direction the evidence is strong.
    //
    // Where the two readings cross — profile — continuity has nothing to say,
    // and a face is still perfectly detectable there, so nothing at that
    // moment reveals which way the turn went. What reveals it is what happens
    // next: the face goes, or it does not. A face missing for a third of a
    // second means the subject turned their back, whatever was chosen at the
    // crossing, so the choice is remade.
    if (this.absent >= FacingTracker.AWAY_RUN && !facesAway(pose, chosen)) chosen = !chosen
    else if (this.present >= FacingTracker.HOME_RUN && facesAway(pose, chosen)) chosen = !chosen

    this.mirrored = chosen
    this.prevYaw = yawFor(chosen)
    return chosen
  }
}
