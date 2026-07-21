import { Quat, Vec3 } from "reze-engine"

// Solver math helpers on reze-engine types. These fill the gaps in reze's Quat/Vec3
// API (quat-from-two-vectors, basis→quat, direct vector rotation) so the solver
// never needs a 4x4 matrix or a matrix inverse: rotating by the inverse of a unit
// quaternion is just rotating by its conjugate.

/** out = v rotated by unit quaternion q. Safe when out === v. */
export function rotateVec(q: Quat, v: Vec3, out: Vec3): Vec3 {
  // v' = v + 2*qw*(qv × v) + 2*(qv × (qv × v))
  const qx = q.x,
    qy = q.y,
    qz = q.z,
    qw = q.w
  const vx = v.x,
    vy = v.y,
    vz = v.z
  // t = 2 * (qv × v)
  const tx = 2 * (qy * vz - qz * vy)
  const ty = 2 * (qz * vx - qx * vz)
  const tz = 2 * (qx * vy - qy * vx)
  out.x = vx + qw * tx + qy * tz - qz * ty
  out.y = vy + qw * ty + qz * tx - qx * tz
  out.z = vz + qw * tz + qx * ty - qy * tx
  return out
}

/** out = v rotated by the inverse (conjugate) of unit quaternion q. Safe when out === v. */
export function rotateVecInv(q: Quat, v: Vec3, out: Vec3): Vec3 {
  const qx = -q.x,
    qy = -q.y,
    qz = -q.z,
    qw = q.w
  const vx = v.x,
    vy = v.y,
    vz = v.z
  const tx = 2 * (qy * vz - qz * vy)
  const ty = 2 * (qz * vx - qx * vz)
  const tz = 2 * (qx * vy - qy * vx)
  out.x = vx + qw * tx + qy * tz - qz * ty
  out.y = vy + qw * ty + qz * tx - qx * tz
  out.z = vz + qw * tz + qx * ty - qy * tx
  return out
}

/**
 * out = shortest-arc rotation taking unit vector `from` to unit vector `to`.
 * Matches Babylon's FromUnitVectorsToRef exactly, including the near-antiparallel
 * branch (w = 1 + dot < 0.001 → 180° about a perpendicular picked the same way).
 */
export function quatFromUnitVectors(from: Vec3, to: Vec3, out: Quat): Quat {
  const r = from.x * to.x + from.y * to.y + from.z * to.z + 1
  if (r < 0.001) {
    if (Math.abs(from.x) > Math.abs(from.z)) {
      out.setXYZW(-from.y, from.x, 0, 0)
    } else {
      out.setXYZW(0, -from.z, from.y, 0)
    }
  } else {
    // q = (from × to, 1 + from·to)
    out.setXYZW(
      from.y * to.z - from.z * to.y,
      from.z * to.x - from.x * to.z,
      from.x * to.y - from.y * to.x,
      r,
    )
  }
  const invLen = 1 / Math.sqrt(out.x * out.x + out.y * out.y + out.z * out.z + out.w * out.w)
  out.setXYZW(out.x * invLen, out.y * invLen, out.z * invLen, out.w * invLen)
  return out
}

/**
 * out = rotation whose local X/Y/Z axes map to the given orthonormal basis vectors.
 * Equivalent to decomposing the row-major matrix [x; y; z] the Babylon solver used.
 */
export function quatFromBasis(x: Vec3, y: Vec3, z: Vec3, out: Quat): Quat {
  // Shepperd's method on the 3x3 rotation matrix with rows x, y, z
  // (row-vector convention: v_world = v_local * M, matching Babylon decompose).
  const m00 = x.x,
    m01 = x.y,
    m02 = x.z
  const m10 = y.x,
    m11 = y.y,
    m12 = y.z
  const m20 = z.x,
    m21 = z.y,
    m22 = z.z
  const trace = m00 + m11 + m22
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1)
    out.setXYZW((m12 - m21) * s, (m20 - m02) * s, (m01 - m10) * s, 0.25 / s)
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22)
    out.setXYZW(0.25 * s, (m10 + m01) / s, (m20 + m02) / s, (m12 - m21) / s)
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22)
    out.setXYZW((m10 + m01) / s, 0.25 * s, (m21 + m12) / s, (m20 - m02) / s)
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11)
    out.setXYZW((m20 + m02) / s, (m21 + m12) / s, 0.25 * s, (m01 - m10) / s)
  }
  return out
}

/**
 * out = twist component of `q` around unit axis `a` (q = swing ∘ twist).
 * Singular when q is ~180° about an axis perpendicular to `a` — returns identity.
 */
export function quatTwistAroundAxis(q: Quat, a: Vec3, out: Quat): Quat {
  const d = q.x * a.x + q.y * a.y + q.z * a.z
  const px = a.x * d
  const py = a.y * d
  const pz = a.z * d
  const len = Math.sqrt(px * px + py * py + pz * pz + q.w * q.w)
  if (len < 1e-8) {
    out.setIdentity()
    return out
  }
  out.setXYZW(px / len, py / len, pz / len, q.w / len)
  return out
}

export function quatDot(a: Quat, b: Quat): number {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

/** out = a·b lerp-normalized (nlerp); assumes hemisphere-aligned inputs. Safe when out === a or b. */
export function quatNlerp(a: Quat, b: Quat, t: number, out: Quat): Quat {
  const x = a.x + (b.x - a.x) * t
  const y = a.y + (b.y - a.y) * t
  const z = a.z + (b.z - a.z) * t
  const w = a.w + (b.w - a.w) * t
  const invLen = 1 / Math.sqrt(x * x + y * y + z * z + w * w)
  out.setXYZW(x * invLen, y * invLen, z * invLen, w * invLen)
  return out
}
