// Replay a recorded take through the real solver and measure it.
//
//   npm run harness -- flash [--export]
//
// The fixture holds what MediaPipe actually said about a real video, so an
// experiment here is answerable in seconds and answers about the footage
// rather than about synthetic input invented to match the hypothesis.

import { readFileSync, readdirSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { Quat, Vec3 } from "reze-engine"
import { Solver, type BoneState } from "../src/lib/solver"
import { cleanTake, type TakeFrame } from "../src/lib/take-cleanup"
import { smoothTakeZeroPhase } from "../src/lib/filters"

// The bundle runs from .build/, so fixtures are found relative to the harness
// rather than to wherever the built file happens to sit.
const HERE = fileURLToPath(new URL(".", import.meta.url))
const FIXTURES = HERE.includes(".build") ? join(HERE, "..", "fixtures") : join(HERE, "fixtures")

type Pt = { x: number; y: number; z: number; visibility: number }
interface Frame {
  time: number
  pose: Pt[] | null
  pose2d: Pt[] | null
  leftHand: Pt[] | null
  rightHand: Pt[] | null
  faceSeen: boolean
}
interface Fixture {
  video: string
  fps: number
  delegate: string
  frames: Frame[]
}

// A model to calibrate against. Real rest positions would come from a PMX;
// these are a standard MMD build, which is what the solver's defaults assume.
const REST: Record<string, { x: number; y: number; z: number }> = {
  左腕: { x: 0.9, y: 16, z: 0 }, 右腕: { x: -0.9, y: 16, z: 0 },
  左ひじ: { x: 2.5, y: 16, z: 0 }, 右ひじ: { x: -2.5, y: 16, z: 0 },
  左手首: { x: 4.0, y: 16, z: 0 }, 右手首: { x: -4.0, y: 16, z: 0 },
  左足: { x: 0.6, y: 9, z: 0 }, 右足: { x: -0.6, y: 9, z: 0 },
  左ひざ: { x: 0.6, y: 5, z: 0 }, 右ひざ: { x: -0.6, y: 5, z: 0 },
  左足首: { x: 0.6, y: 1, z: 0 }, 右足首: { x: -0.6, y: 1, z: 0 },
  左つま先: { x: 0.6, y: 0.2, z: -1 }, 右つま先: { x: -0.6, y: 0.2, z: -1 },
  下半身: { x: 0, y: 10, z: 0 }, 上半身: { x: 0, y: 11, z: 0 }, 上半身2: { x: 0, y: 13, z: 0 },
  センター: { x: 0, y: 8, z: 0 }, 首: { x: 0, y: 16.5, z: 0 }, 頭: { x: 0, y: 17.5, z: 0 },
}

const WATCHED = ["上半身", "上半身2", "頭", "左腕", "右腕", "左ひじ", "右ひじ", "左足", "右足"]

const deg = (rad: number) => (rad * 180) / Math.PI

/** The bearing the shoulders face, straight from the landmarks. */
function shoulderYaw(pose: Pt[] | null): number {
  if (!pose) return NaN
  const l = pose[11]
  const r = pose[12]
  return Math.atan2(l.x - r.x, l.z - r.z)
}

/** Signed step between two bearings, wrapped. */
function wrap(d: number): number {
  while (d > Math.PI) d -= 2 * Math.PI
  while (d < -Math.PI) d += 2 * Math.PI
  return d
}

const vy = new Vec3(0, 1, 0)

/** The chain each watched bone hangs from. Measuring a local rotation is
 *  measuring the wrong thing: parents and children counter-rotate, so a local
 *  quat reports shake that the world pose does not have, and hides shake it
 *  does. */
const PARENTS: Record<string, string | null> = {
  上半身: null,
  上半身2: "上半身",
  首: "上半身2",
  頭: "首",
  左肩: "上半身2",
  右肩: "上半身2",
  左腕: "左肩",
  右腕: "右肩",
  左ひじ: "左腕",
  右ひじ: "右腕",
  下半身: null,
  左足: "下半身",
  右足: "下半身",
}

const _chain = Quat.identity()
const _acc = Quat.identity()
const twistScratch = Quat.identity()

/** The composed world rotation of a bone, walking its parents. */
function worldOf(states: Map<string, Quat>, name: string): Quat | null {
  if (!states.get(name)) return null
  _acc.setIdentity()
  const stack: string[] = []
  for (let n: string | null = name; n; n = PARENTS[n] ?? null) stack.push(n)
  for (let i = stack.length - 1; i >= 0; i--) {
    const q = states.get(stack[i])
    if (!q) continue
    Quat.multiplyInto(_acc, q, _chain)
    _acc.set(_chain)
  }
  return _acc
}

/** Where a bone points in the world, walking its parents. */
function worldDirection(states: Map<string, Quat>, name: string, out: Vec3): Vec3 {
  _acc.setIdentity()
  const stack: string[] = []
  for (let n: string | null = name; n; n = PARENTS[n] ?? null) stack.push(n)
  for (let i = stack.length - 1; i >= 0; i--) {
    const q = states.get(stack[i])
    if (!q) continue
    Quat.multiplyInto(_acc, q, _chain)
    _acc.set(_chain)
  }
  Quat.rotateVecInto(_acc, vy, out)
  return out
}

function solveTake(frames: Frame[], mode: "live" | "export") {
  const solver = new Solver()
  solver.calibrate(REST)
  const poses: BoneState[][] = []

  if (mode === "live") {
    for (const f of frames) {
      const out = solver.solve(
        {
          poseWorldLandmarks: f.pose ? [f.pose] : [],
          leftHandWorldLandmarks: f.leftHand ? [f.leftHand] : [],
          rightHandWorldLandmarks: f.rightHand ? [f.rightHand] : [],
        } as never,
        f.time * 1000,
      )
      poses.push(out.map((b) => ({ name: b.name, rotation: b.rotation.clone() })))
    }
    return poses
  }

  // The export path, as the app runs it: clean the landmarks, solve once with
  // the filters opened up, then fit the finished take without phase shift.
  const take: TakeFrame[] = frames.map((f) => ({
    pose: (f.pose as never) ?? null,
    leftHand: (f.leftHand as never) ?? null,
    rightHand: (f.rightHand as never) ?? null,
  }))
  cleanTake(take, { zGain: 1 })
  solver.reset()
  solver.setSmoothing(3, 6, 6)
  const out: BoneState[][] = []
  for (let i = 0; i < take.length; i++) {
    const pose = solver.solve(
      {
        poseWorldLandmarks: take[i].pose ? [take[i].pose] : [],
        leftHandWorldLandmarks: take[i].leftHand ? [take[i].leftHand] : [],
        rightHandWorldLandmarks: take[i].rightHand ? [take[i].rightHand] : [],
      } as never,
      frames[i].time * 1000,
    )
    out.push(pose.map((b) => ({ name: b.name, rotation: b.rotation.clone() })))
  }
  smoothTakeZeroPhase(out.map((boneStates, i) => ({ time: frames[i].time, boneStates })))
  return out
}

/** A row of characters showing a signal over time — readable in a terminal,
 *  which is where this gets read. */
function sparkline(values: number[], lo: number, hi: number, width = 100): string {
  const bars = " ▁▂▃▄▅▆▇█"
  let out = ""
  for (let i = 0; i < width; i++) {
    const a = Math.floor((i * values.length) / width)
    const b = Math.max(a + 1, Math.floor(((i + 1) * values.length) / width))
    let peak = -Infinity
    for (let k = a; k < b && k < values.length; k++) peak = Math.max(peak, values[k])
    if (!Number.isFinite(peak)) {
      out += "·"
      continue
    }
    const t = Math.max(0, Math.min(1, (peak - lo) / (hi - lo)))
    out += bars[Math.round(t * (bars.length - 1))]
  }
  return out
}

function band(flags: boolean[], width = 100): string {
  let out = ""
  for (let i = 0; i < width; i++) {
    const a = Math.floor((i * flags.length) / width)
    const b = Math.max(a + 1, Math.floor(((i + 1) * flags.length) / width))
    let any = false
    for (let k = a; k < b && k < flags.length; k++) any ||= flags[k]
    out += any ? "█" : "·"
  }
  return out
}

// ---------------------------------------------------------------------------

const name = process.argv[2] ?? readdirSync(FIXTURES)[0]?.replace(/\.json$/, "")
if (!name) {
  console.error("no fixtures — run: node harness/record.mjs <video>")
  process.exit(1)
}
const fixture: Fixture = JSON.parse(readFileSync(join(FIXTURES, `${name}.json`), "utf8"))
const frames = fixture.frames
console.log(`${fixture.video} · ${frames.length} frames @ ${fixture.fps}fps · detected on ${fixture.delegate}\n`)

// --- what the detector gave us -------------------------------------------
const posed = frames.map((f) => Boolean(f.pose))
const faced = frames.map((f) => f.faceSeen)
const leftHand = frames.map((f) => Boolean(f.leftHand))
const rightHand = frames.map((f) => Boolean(f.rightHand))
let longestFaceGap = 0
let run = 0
for (const f of faced) {
  run = f ? 0 : run + 1
  longestFaceGap = Math.max(longestFaceGap, run)
}
console.log("DETECTION")
console.log(`  pose   ${band(posed)}  ${((posed.filter(Boolean).length / frames.length) * 100).toFixed(0)}%`)
console.log(`  face   ${band(faced)}  ${((faced.filter(Boolean).length / frames.length) * 100).toFixed(0)}%, longest gap ${longestFaceGap} frames`)
console.log(`  hand L ${band(leftHand)}  ${((leftHand.filter(Boolean).length / frames.length) * 100).toFixed(0)}%`)
console.log(`  hand R ${band(rightHand)}  ${((rightHand.filter(Boolean).length / frames.length) * 100).toFixed(0)}%`)

// --- facing, straight from the landmarks ----------------------------------
const yaws = frames.map((f) => shoulderYaw(f.pose))
const steps: number[] = []
let flips = 0
let worstStep = 0
for (let i = 1; i < yaws.length; i++) {
  if (Number.isNaN(yaws[i]) || Number.isNaN(yaws[i - 1])) {
    steps.push(0)
    continue
  }
  const d = Math.abs(deg(wrap(yaws[i] - yaws[i - 1])))
  steps.push(d)
  if (d > 90) flips++
  worstStep = Math.max(worstStep, d)
}
console.log("\nFACING (shoulder bearing, from the landmarks themselves)")
console.log(`  step   ${sparkline(steps, 0, 180)}  worst ${worstStep.toFixed(0)}°/frame`)
console.log(`  half-turn steps: ${flips}   (a real body cannot do this at ${fixture.fps}fps)`)

// --- the solve ------------------------------------------------------------
for (const mode of ["live", "export"] as const) {
  const poses = solveTake(frames, mode)
  const byName = poses.map((p) => new Map(p.map((b) => [b.name, b.rotation])))
  console.log(`\n${mode.toUpperCase()} SOLVE`)
  const rows: string[] = []
  for (const bone of WATCHED) {
    let jitter = 0
    let pops = 0
    let worst = 0
    let n = 0
    const a = new Vec3(0, 0, 0)
    const b = new Vec3(0, 0, 0)
    for (let i = 1; i < byName.length; i++) {
      if (!byName[i].get(bone)) continue
      worldDirection(byName[i - 1], bone, a)
      worldDirection(byName[i], bone, b)
      const d = deg(Math.acos(Math.max(-1, Math.min(1, a.dot(b)))))
      jitter += d
      worst = Math.max(worst, d)
      if (d > 20) pops++
      n++
    }
    // How far the bone ranges, measured against its own average direction
    // rather than against its first frame — a take starts with the pose
    // fading in, and a pass that begins at the other end of the video does
    // not, so anchoring on frame zero compares two different things and
    // reports the difference as lost motion.
    const mean = new Vec3(0, 0, 0)
    const EDGE = Math.min(10, Math.floor(byName.length / 10))
    for (let i = EDGE; i < byName.length - EDGE; i++) {
      worldDirection(byName[i], bone, b)
      mean.setXYZ(mean.x + b.x, mean.y + b.y, mean.z + b.z)
    }
    const len = Math.hypot(mean.x, mean.y, mean.z) || 1
    mean.setXYZ(mean.x / len, mean.y / len, mean.z / len)
    let range = 0
    for (let i = EDGE; i < byName.length - EDGE; i++) {
      worldDirection(byName[i], bone, b)
      range = Math.max(range, deg(Math.acos(Math.max(-1, Math.min(1, mean.dot(b))))))
    }
    // Roll: the turn about a bone's own axis. A landmark preview cannot show
    // it — the stick figure is identical whichever way the limb is twisted —
    // so a roll that flips is invisible everywhere except on the character,
    // which is exactly the report that keeps arriving.
    let rollJitter = 0
    let rollPops = 0
    let rollWorst = 0
    let rollN = 0
    let prevRoll = NaN
    const axis = new Vec3(0, 0, 0)
    for (let i = 0; i < byName.length; i++) {
      const q = worldOf(byName[i], bone)
      if (!q) continue
      Quat.rotateVecInto(q, vy, axis)
      // Twist of the world rotation about the direction the bone points.
      Quat.twistAroundAxisInto(q, axis, twistScratch)
      const k = twistScratch.x * axis.x + twistScratch.y * axis.y + twistScratch.z * axis.z
      let roll = 2 * Math.atan2(k, twistScratch.w)
      if (roll > Math.PI) roll -= 2 * Math.PI
      else if (roll < -Math.PI) roll += 2 * Math.PI
      if (!Number.isNaN(prevRoll)) {
        let d = roll - prevRoll
        while (d > Math.PI) d -= 2 * Math.PI
        while (d < -Math.PI) d += 2 * Math.PI
        const dd = Math.abs(deg(d))
        rollJitter += dd
        rollWorst = Math.max(rollWorst, dd)
        if (dd > 20) rollPops++
        rollN++
      }
      prevRoll = roll
    }

    rows.push(
      `  ${bone.padEnd(6)} ${(jitter / Math.max(1, n)).toFixed(2).padStart(6)}°/frame   worst ${worst
        .toFixed(0)
        .padStart(3)}°   pops ${String(pops).padStart(3)}   range ${range.toFixed(0).padStart(3)}°` +
        `   │ roll ${(rollJitter / Math.max(1, rollN)).toFixed(2).padStart(6)}°/frame worst ${rollWorst
          .toFixed(0)
          .padStart(3)}° pops ${String(rollPops).padStart(3)}`,
    )
  }
  console.log(rows.join("\n"))
}
