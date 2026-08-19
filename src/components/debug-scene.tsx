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

function DebugScene({ landmarks }: { landmarks: PoseWorkerResult | null }) {
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
    const scale = H / 2.1
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
      // MediaPipe world convention is y down, z away — flip y for drawing.
      const project = (lm: { x: number; y: number; z: number }): [number, number] => {
        const rx = lm.x * cos - lm.z * sin
        const rz = lm.x * sin + lm.z * cos
        return [W / 2 + rx * scale, H / 2 - (-lm.y * cosP - rz * sinP) * scale]
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, H)

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
