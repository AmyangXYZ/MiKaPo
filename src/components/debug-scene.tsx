"use client"

// Raw-landmark inset: what the pose model actually measured, drawn with a
// plain 2D canvas — same approach as Mixamo-MMD's source inset. Its job is
// blame assignment: when the character looks wrong here too, the model did it;
// when this looks right, the solver did. Drag to orbit — depth errors are
// invisible from the front, and depth is exactly where these models fail.

import { memo, useEffect, useRef } from "react"
import type { PoseWorkerResult } from "@/lib/pose-worker"

const W = 300
const H = 188

// MediaPipe 33-point pose topology (the ONNX adapter emits the same shape).
const POSE_CONNECTIONS: [number, number][] = [
  [0, 2], [0, 5], [2, 7], [5, 8], [9, 10],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
]
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]

type LmSet = PoseWorkerResult

/** Lerp every landmark between two results; t clamps at 1 (hold). */
function lerpResults(a: LmSet, b: LmSet, t: number): LmSet {
  const lerpArr = (pa?: { x: number; y: number; z: number; visibility?: number }[], pb?: typeof pa) => {
    if (!pa || !pb || pa.length !== pb.length) return pb ? [pb] : []
    return [
      pb.map((q, i) => ({
        x: pa[i].x + (q.x - pa[i].x) * t,
        y: pa[i].y + (q.y - pa[i].y) * t,
        z: pa[i].z + (q.z - pa[i].z) * t,
        visibility: q.visibility,
      })),
    ]
  }
  return {
    poseWorldLandmarks: lerpArr(a.poseWorldLandmarks[0], b.poseWorldLandmarks[0]) as LmSet["poseWorldLandmarks"],
    leftHandWorldLandmarks: lerpArr(a.leftHandWorldLandmarks[0], b.leftHandWorldLandmarks[0]) as LmSet["leftHandWorldLandmarks"],
    rightHandWorldLandmarks: lerpArr(a.rightHandWorldLandmarks[0], b.rightHandWorldLandmarks[0]) as LmSet["rightHandWorldLandmarks"],
    faceLandmarks: [],
  }
}

/** Applied-pose connections, by MMD bone name (drawn when both ends exist). */
const BONE_PAIRS: [string, string][] = [
  ["下半身", "上半身"], ["上半身", "首"], ["首", "頭"],
  ["上半身", "左肩"], ["左肩", "左腕"], ["左腕", "左ひじ"], ["左ひじ", "左手首"],
  ["上半身", "右肩"], ["右肩", "右腕"], ["右腕", "右ひじ"], ["右ひじ", "右手首"],
  ["下半身", "左足"], ["左足", "左ひざ"], ["左ひざ", "左足首"], ["左足首", "左つま先"],
  ["下半身", "右足"], ["右足", "右ひざ"], ["右ひざ", "右足首"], ["右足首", "右つま先"],
]

function DebugScene({
  landmarks,
  getLivePose,
}: {
  landmarks: PoseWorkerResult | null
  getLivePose?: () => { name: string; x: number; y: number; z: number }[]
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const yawRef = useRef(0)
  const dragRef = useRef<{ x: number; yaw: number } | null>(null)
  // Interpolation state: the inset tweens between the last two results over
  // the measured arrival interval, so it moves at display rate like the main
  // scene instead of stepping at capture rate.
  const prevRef = useRef<PoseWorkerResult | null>(null)
  const currRef = useRef<PoseWorkerResult | null>(null)
  const currAtRef = useRef(0)
  const intervalRef = useRef(200)
  if (landmarks && landmarks !== currRef.current) {
    const now = performance.now()
    if (currRef.current) {
      const dt = now - currAtRef.current
      if (dt < 2000) intervalRef.current = intervalRef.current * 0.7 + dt * 0.3
    }
    prevRef.current = currRef.current
    currRef.current = landmarks
    currAtRef.current = now
  }
  const getLivePoseRef = useRef(getLivePose)
  getLivePoseRef.current = getLivePose

  const drawRef = useRef<() => void>(() => {})
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = W * dpr
    canvas.height = H * dpr
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // World landmarks are hip-centred meters; a standing body spans ~1.9m, so
    // a fixed scale frames it without per-frame zoom jitter.
    const scale = H / 2.4
    const PITCH = 0.26
    const cosP = Math.cos(PITCH)
    const sinP = Math.sin(PITCH)

    drawRef.current = () => {
      const curr = currRef.current
      const prev = prevRef.current
      let lms = curr
      if (curr && prev) {
        const t = Math.min(1, (performance.now() - currAtRef.current) / Math.max(16, intervalRef.current))
        lms = lerpResults(prev, curr, t)
      }
      const yaw = yawRef.current
      const cos = Math.cos(yaw)
      const sin = Math.sin(yaw)
      const half = W / 2

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, H)
      ctx.strokeStyle = "rgba(255,255,255,0.08)"
      ctx.beginPath()
      ctx.moveTo(half, 0)
      ctx.lineTo(half, H)
      ctx.stroke()
      ctx.fillStyle = "rgba(255,255,255,0.35)"
      ctx.font = "9px monospace"
      ctx.fillText("landmarks", 4, 10)
      ctx.fillText("applied", half + 4, 10)

      // Left: raw landmarks (MediaPipe world: y down, z away — flip y).
      const project = (lm: { x: number; y: number; z: number }): [number, number] => {
        const rx = lm.x * cos - lm.z * sin
        const rz = lm.x * sin + lm.z * cos
        return [half / 2 + rx * scale, H / 2 - (-lm.y * cosP - rz * sinP) * scale]
      }
      const drawSet = (
        pts: { x: number; y: number; z: number; visibility?: number }[] | undefined,
        connections: [number, number][],
        stroke: string,
        dim: string,
      ) => {
        if (!pts || pts.length === 0) return
        const projected = pts.map(project)
        for (const [a, b] of connections) {
          const pa = projected[a]
          const pb = projected[b]
          if (!pa || !pb) continue
          const faint = (pts[a].visibility ?? 1) < 0.35 || (pts[b].visibility ?? 1) < 0.35
          ctx.strokeStyle = faint ? dim : stroke
          ctx.lineWidth = faint ? 1 : 1.5
          ctx.beginPath()
          ctx.moveTo(pa[0], pa[1])
          ctx.lineTo(pb[0], pb[1])
          ctx.stroke()
        }
        for (let i = 0; i < projected.length; i++) {
          const faint = (pts[i].visibility ?? 1) < 0.35
          ctx.fillStyle = faint ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.9)"
          ctx.beginPath()
          ctx.arc(projected[i][0], projected[i][1], faint ? 1.2 : 1.7, 0, Math.PI * 2)
          ctx.fill()
        }
      }
      drawSet(lms?.poseWorldLandmarks[0], POSE_CONNECTIONS, "rgba(102,153,255,0.95)", "rgba(102,153,255,0.3)")
      drawSet(lms?.leftHandWorldLandmarks[0], HAND_CONNECTIONS, "rgba(120,220,180,0.9)", "rgba(120,220,180,0.3)")
      drawSet(lms?.rightHandWorldLandmarks[0], HAND_CONNECTIONS, "rgba(255,180,120,0.9)", "rgba(255,180,120,0.3)")

      // Right: the engine's live bone positions (MMD model space, y UP).
      const live = getLivePoseRef.current?.()
      if (live && live.length > 0) {
        const byName: Record<string, { x: number; y: number; z: number }> = {}
        let minY = Infinity
        let maxY = -Infinity
        let cx = 0
        for (const b of live) {
          byName[b.name] = b
          if (b.y < minY) minY = b.y
          if (b.y > maxY) maxY = b.y
        }
        const hips = byName["下半身"]
        if (hips) cx = hips.x
        const cz = hips?.z ?? 0
        const s2 = (H * 0.8) / Math.max(1e-3, maxY - minY)
        const cy = (minY + maxY) / 2
        const proj2 = (p: { x: number; y: number; z: number }): [number, number] => {
          const x = p.x - cx
          const z = p.z - cz
          const rx = x * cos - z * sin
          const rz = x * sin + z * cos
          return [half + half / 2 + rx * s2, H / 2 - ((p.y - cy) * cosP - rz * sinP) * s2]
        }
        ctx.strokeStyle = "rgba(255,210,100,0.95)"
        ctx.lineWidth = 1.5
        for (const [a, b] of BONE_PAIRS) {
          const pa = byName[a]
          const pb = byName[b]
          if (!pa || !pb) continue
          const qa = proj2(pa)
          const qb = proj2(pb)
          ctx.beginPath()
          ctx.moveTo(qa[0], qa[1])
          ctx.lineTo(qb[0], qb[1])
          ctx.stroke()
        }
        ctx.fillStyle = "rgba(255,255,255,0.9)"
        for (const b of live) {
          const q = proj2(b)
          ctx.beginPath()
          ctx.arc(q[0], q[1], 1.7, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }
    // Continuous redraw: interpolation means every frame differs. The canvas
    // is small; this costs far less than the WebGPU viewport beside it.
    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      drawRef.current()
    }
    loop()
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: W, height: H }}
      className="block h-full w-full cursor-ew-resize touch-none"
      onPointerDown={(e) => {
        dragRef.current = { x: e.clientX, yaw: yawRef.current }
        e.currentTarget.setPointerCapture(e.pointerId)
      }}
      onPointerMove={(e) => {
        const d = dragRef.current
        if (d) {
          yawRef.current = d.yaw + (e.clientX - d.x) * 0.01
          drawRef.current()
        }
      }}
      onPointerUp={() => {
        dragRef.current = null
      }}
    />
  )
}

export default memo(DebugScene)
