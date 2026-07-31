import { Vec3 } from "reze-engine"
import { OneEuroFilter } from "./filters"
import { PoseLandmarksTable } from "./landmarks"
import type { SolverInput } from "./solver"

/**
 * Root translation and foot IK targets.
 *
 * The FK solver turns landmarks into bone rotations. This turns them into the
 * two things rotations cannot express: where the body IS, and where the feet
 * are planted.
 *
 * Why it has to be separate data: MediaPipe's world landmarks are defined with
 * their ORIGIN AT THE MID-HIP. The hip is (0,0,0) in every frame, so the stream
 * carries no translation at all — a character driven from rotations alone can
 * only ever pivot in place. Height comes back by measuring how far the hips sit
 * above the feet and comparing that to the model's own rest proportions.
 *
 * Everything below is in MMD space (Y up, model units). Landmarks arrive in
 * MediaPipe space (Y down, metres) and are flipped and scaled on the way in.
 */

const L = PoseLandmarksTable

/** Translations for one frame, as VMD writes them: offsets from rest position. */
export interface RootIkPose {
  /** センター offset. Y only for now — X/Z need image-space landmarks. */
  center: Vec3
  leftFootIk: Vec3
  rightFootIk: Vec3
}

/** Rest-pose facts the solver measures itself against, from the loaded model. */
interface Rest {
  hipY: number
  ankleY: number
  /** 足→ひざ→足首 chain length: the model's own leg, in model units. */
  legLength: number
  leftAnkle: Vec3
  rightAnkle: Vec3
}

/** How far the hips may sink, as a fraction of leg span. Past this the pose is a
 *  detection failure rather than a crouch. */
const MAX_CROUCH = 0.55

const dist = (a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }): number =>
  Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)

export class RootIkSolver {
  private rest: Rest | null = null

  // Scale is a property of the PERSON, not of the frame — how big they are
  // relative to the model does not change while they dance. Measuring it per
  // frame made it noise, and because it multiplies every position downstream,
  // that noise came out as the hips wandering.
  //
  // Estimated from the longest leg seen so far. Foreshortening — a leg swinging
  // toward or away from the camera — can only ever make the chain measure
  // SHORTER than it is, never longer, so the running maximum converges on the
  // true length and stops moving. The slow decay lets it re-settle if the
  // performer changes, without chasing single-frame spikes.
  private legLengthSeen = 0
  private centerY = new OneEuroFilter(1.2, 1.0, 1.0)
  private ik = {
    lx: new OneEuroFilter(1.5, 2.0, 1.0),
    ly: new OneEuroFilter(1.5, 2.0, 1.0),
    // Depth is MediaPipe's weakest axis; a foot that jitters in Z reads as
    // sliding, so Z is filtered harder than X and Y.
    lz: new OneEuroFilter(0.7, 0.8, 1.0),
    rx: new OneEuroFilter(1.5, 2.0, 1.0),
    ry: new OneEuroFilter(1.5, 2.0, 1.0),
    rz: new OneEuroFilter(0.7, 0.8, 1.0),
  }

  /** True once the model's rest proportions are known. */
  get ready(): boolean {
    return this.rest !== null
  }

  /**
   * Measure the model. `restWorldPos` is the same table the FK solver calibrates
   * from — world positions of each bone in the loaded model's rest pose.
   */
  calibrate(restWorldPos: Record<string, { x: number; y: number; z: number }>): void {
    const need = ["左足", "右足", "左ひざ", "右ひざ", "左足首", "右足首"]
    if (need.some((b) => !restWorldPos[b])) {
      this.rest = null
      return
    }
    const legLength =
      (dist(restWorldPos["左足"], restWorldPos["左ひざ"]) +
        dist(restWorldPos["左ひざ"], restWorldPos["左足首"]) +
        dist(restWorldPos["右足"], restWorldPos["右ひざ"]) +
        dist(restWorldPos["右ひざ"], restWorldPos["右足首"])) /
      2
    const hipY = (restWorldPos["左足"].y + restWorldPos["右足"].y) / 2
    const ankleY = (restWorldPos["左足首"].y + restWorldPos["右足首"].y) / 2
    this.rest = {
      hipY,
      ankleY,
      legLength,
      leftAnkle: new Vec3(restWorldPos["左足首"].x, restWorldPos["左足首"].y, restWorldPos["左足首"].z),
      rightAnkle: new Vec3(restWorldPos["右足首"].x, restWorldPos["右足首"].y, restWorldPos["右足首"].z),
    }
  }

  /** Drop filter history — call when the source changes (new video, new take). */
  reset(): void {
    this.legLengthSeen = 0
    this.centerY = new OneEuroFilter(1.2, 1.0, 1.0)
  }

  solve(input: SolverInput, timestampMs: number): RootIkPose | null {
    const rest = this.rest
    const lms = input.poseWorldLandmarks?.[0]
    if (!rest || !lms) return null

    // MediaPipe Y grows downward; MMD's grows up.
    const at = (i: number): Vec3 | null => {
      const p = lms[i]
      return p ? new Vec3(p.x, -p.y, p.z) : null
    }
    const lHip = at(L.left_hip)
    const rHip = at(L.right_hip)
    const lKnee = at(L.left_knee)
    const rKnee = at(L.right_knee)
    const lAnkle = at(L.left_ankle)
    const rAnkle = at(L.right_ankle)
    if (!lHip || !rHip || !lKnee || !rKnee || !lAnkle || !rAnkle) return null

    // Metres → model units, from the one thing both skeletons share: leg length.
    const detected = (dist(lHip, lKnee) + dist(lKnee, lAnkle) + dist(rHip, rKnee) + dist(rKnee, rAnkle)) / 2
    if (detected < 1e-4) return null
    this.legLengthSeen = Math.max(detected, this.legLengthSeen * 0.9995)
    const scale = rest.legLength / this.legLengthSeen

    // The hips are the landmark origin, so a foot's Y IS its distance below the
    // hips. The lower foot is the one standing on something.
    const stance = Math.min(lAnkle.y, rAnkle.y) * scale
    // Drop the hips by however far the legs have folded up under them. Clamped
    // to a squat: a bad frame — legs foreshortened, a foot out of shot — reads
    // as a very short leg span, and unclamped that buries the character in the
    // floor for as long as the detection stays bad.
    const legSpan = rest.hipY - rest.ankleY
    const raw = Math.max(-legSpan * MAX_CROUCH, Math.min(0, stance + legSpan))
    const centerY = this.centerY.filter(raw, timestampMs)

    // Hip position once the body has been placed at that height. X and Z stay at
    // rest until image-space landmarks give us somewhere real to put them.
    const hipY = rest.hipY + centerY

    const target = (ankle: Vec3, restAnkle: Vec3, fx: OneEuroFilter, fy: OneEuroFilter, fz: OneEuroFilter): Vec3 =>
      new Vec3(
        fx.filter(ankle.x * scale - restAnkle.x, timestampMs),
        fy.filter(hipY + ankle.y * scale - restAnkle.y, timestampMs),
        fz.filter(ankle.z * scale - restAnkle.z, timestampMs),
      )

    return {
      // A VMD translation is an offset from the bone's rest position, and
      // センター rests at the origin — so the offset IS the position.
      center: new Vec3(0, centerY, 0),
      leftFootIk: target(lAnkle, rest.leftAnkle, this.ik.lx, this.ik.ly, this.ik.lz),
      rightFootIk: target(rAnkle, rest.rightAnkle, this.ik.rx, this.ik.ry, this.ik.rz),
    }
  }
}

/** The bones this solver drives, in the names MMD rigs use. */
export const ROOT_IK_BONES = { center: "センター", leftFoot: "左足ＩＫ", rightFoot: "右足ＩＫ" } as const
