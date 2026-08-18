/// <reference lib="webworker" />
// RTMW3D detection worker — the ONNX/WebGPU counterpart of pose-worker.ts,
// speaking the same message protocol so the capture panel can swap engines
// with one constant.
//
// One model does body + feet + hands in 3D (COCO-WholeBody 133 keypoints,
// SimCC x/y/z). There is no person-detector model: the first frame runs on a
// full-frame crop, and every frame after crops to the previous frame's own
// keypoints — the single-person case needs nothing more. The model is
// stateless frame to frame (no internal tracker), so "reset" only clears the
// crop box and scale estimate.
import * as ort from "onnxruntime-web/webgpu"
import type { PoseWorkerRequest, PoseWorkerResponse } from "./pose-worker"
import {
  MODEL_W,
  MODEL_H,
  SIMCC_RATIO,
  bboxCenterScale,
  warpTransform,
  imageDataToTensor,
  decodeSimCC3D,
  bboxFromKeypoints,
  estimateMetersPerPixel,
  toWorkerResult,
  type Decoded,
} from "./onnx/rtmw3d"

// Preferred first: local fp16 (half the download, and WebGPU runs f16 shaders
// on Apple Silicon). Falls back to the fp32 HuggingFace original when the
// local file isn't deployed.
const MODEL_SOURCES = [
  "/models/rtmw3d-x-fp16.onnx",
  "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx",
]
const MODEL_CACHE = "mikapo-onnx-models"

// Serve the ORT wasm runtime from the CDN, mirroring how the MediaPipe worker
// loads its fileset — keep the version in lock-step with package.json.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/"
ort.env.webgpu.powerPreference = "high-performance"

let session: ort.InferenceSession | null = null
let inputName = ""
// Output names resolved to axes after warmup: y is the unique MODEL_H×2-bin
// tensor; x and z share a bin count (width and depth dims are both 288) and
// keep their graph order, x first — for this export: output, 1554, 1556.
let outX = ""
let outY = ""
let outZ = ""
let runningMode: "VIDEO" | "IMAGE" = "VIDEO"

/** Crop box carried between frames; null = full frame next. */
let trackedBbox: number[] | null = null
/** EMA'd meters-per-pixel — camera distance changes slowly. */
let metersPerPixel: number | null = null

const canvas = new OffscreenCanvas(MODEL_W, MODEL_H)
const ctx = canvas.getContext("2d", { willReadFrequently: true })!
const inputData = new Float32Array(3 * MODEL_W * MODEL_H)

const post = (msg: PoseWorkerResponse) => (self as unknown as Worker).postMessage(msg)

async function fetchModel(url: string): Promise<ArrayBuffer> {
  // Same-origin files come off local disk — Cache Storage would only duplicate
  // them; the cache earns its keep on the cross-origin fallback.
  const cache = url.startsWith("/") ? null : await caches.open(MODEL_CACHE).catch(() => null)
  const hit = await cache?.match(url)
  if (hit) return hit.arrayBuffer()

  const resp = await fetch(url)
  if (!resp.ok || !resp.body) throw new Error(`model fetch failed: HTTP ${resp.status}`)
  const total = Number(resp.headers.get("Content-Length") ?? 0)
  const reader = resp.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  let lastPost = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    const now = performance.now()
    if (now - lastPost > 200) {
      lastPost = now
      post({ type: "progress", loaded, total })
    }
  }
  post({ type: "progress", loaded, total })
  const buf = new Uint8Array(loaded)
  let off = 0
  for (const c of chunks) {
    buf.set(c, off)
    off += c.length
  }
  // Cache best-effort — a 370MB put can exceed quota, and that only costs the
  // next visit a re-download.
  await cache?.put(url, new Response(buf)).catch(() => {})
  return buf.buffer
}

/** Which provider actually took the session — surfaced because a silent wasm
 * fallback is a 100× slowdown that otherwise looks like "the model is slow". */
let epUsed = ""

async function init(): Promise<void> {
  let model: ArrayBuffer | null = null
  let src = ""
  for (const url of MODEL_SOURCES) {
    try {
      model = await fetchModel(url)
      src = url
      break
    } catch (e) {
      console.warn(`[rtmw3d] model source failed: ${url}`, e)
    }
  }
  if (!model) throw new Error("no model source reachable")

  // Attempt ladder: WebGPU with graph capture (static shapes make it legal,
  // and it strips per-run CPU overhead), plain WebGPU, then wasm as the
  // last resort. Warmup runs INSIDE each attempt — WebGPU failures can
  // surface at first run rather than at create.
  const attempts: [string, ort.InferenceSession.SessionOptions][] = [
    ["webgpu+capture", { executionProviders: ["webgpu"], graphOptimizationLevel: "all", enableGraphCapture: true }],
    ["webgpu", { executionProviders: ["webgpu"], graphOptimizationLevel: "all" }],
    ["wasm", { executionProviders: ["wasm"] }],
  ]
  let warm: Awaited<ReturnType<ort.InferenceSession["run"]>> | null = null
  for (const [name, opts] of attempts) {
    try {
      const s = await ort.InferenceSession.create(model, opts)
      const t0 = performance.now()
      warm = await s.run({
        [s.inputNames[0]]: new ort.Tensor("float32", inputData, [1, 3, MODEL_H, MODEL_W]),
      })
      console.log(`[rtmw3d] ${name} warmup ${(performance.now() - t0).toFixed(0)}ms (${src.split("/").pop()})`)
      session = s
      epUsed = name
      break
    } catch (e) {
      console.warn(`[rtmw3d] ${name} failed:`, e)
    }
  }
  if (!session || !warm) throw new Error("no execution provider worked")
  inputName = session.inputNames[0]

  const xz: string[] = []
  for (const name of session.outputNames) {
    const bins = warm[name].dims[warm[name].dims.length - 1]
    if (bins === MODEL_H * SIMCC_RATIO) outY = name
    else xz.push(name)
  }
  if (!outY || xz.length !== 2) throw new Error(`unexpected model outputs: ${session.outputNames.join(", ")}`)
  ;[outX, outZ] = xz

  post({ type: "ready", ep: epUsed })
}

let inferEma = 0
let inferCount = 0

async function runOnce(bitmap: ImageBitmap, bbox: readonly number[]): Promise<Decoded> {
  const cs = bboxCenterScale(bbox)
  const warp = warpTransform(cs)
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.fillStyle = "#000"
  ctx.fillRect(0, 0, MODEL_W, MODEL_H)
  ctx.setTransform(warp.k, 0, 0, warp.k, warp.tx, warp.ty)
  ctx.drawImage(bitmap, 0, 0)
  const rgba = ctx.getImageData(0, 0, MODEL_W, MODEL_H).data
  imageDataToTensor(rgba, inputData)
  const t0 = performance.now()
  const outputs = await session!.run({
    [inputName]: new ort.Tensor("float32", inputData, [1, 3, MODEL_H, MODEL_W]),
  })
  const ms = performance.now() - t0
  inferEma = inferEma === 0 ? ms : inferEma * 0.9 + ms * 0.1
  if (++inferCount % 30 === 0)
    console.log(`[rtmw3d] ${epUsed}: ${inferEma.toFixed(0)}ms/inference (~${(1000 / inferEma).toFixed(1)} Hz)`)
  return decodeSimCC3D(
    outputs[outX].data as Float32Array,
    outputs[outY].data as Float32Array,
    outputs[outZ].data as Float32Array,
    warp,
  )
}

async function detect(bitmap: ImageBitmap, mediaTs: number, still: boolean): Promise<void> {
  const w = bitmap.width
  const h = bitmap.height
  let bbox = still ? null : trackedBbox
  let decoded = await runOnce(bitmap, bbox ?? [0, 0, w, h])

  // A still gets a second pass: crop to the person found in the full frame,
  // which is what a detector would have provided.
  if (still) {
    const refined = bboxFromKeypoints(decoded.kpts, decoded.scores, w, h)
    if (refined) decoded = await runOnce(bitmap, refined)
  }

  trackedBbox = bboxFromKeypoints(decoded.kpts, decoded.scores, w, h)

  const estimate = estimateMetersPerPixel(decoded.kpts, decoded.scores)
  if (estimate) {
    metersPerPixel = still || metersPerPixel === null ? estimate : metersPerPixel * 0.8 + estimate * 0.2
  } else if (metersPerPixel === null) {
    // Bootstrap from the crop: its height is a padded person, roughly 1.7m.
    bbox = trackedBbox ?? [0, 0, w, h]
    metersPerPixel = 1.7 / Math.max(1, (bbox[3] - bbox[1]))
  }

  post({ type: "result", mediaTs, result: toWorkerResult(decoded.kpts, decoded.scores, metersPerPixel) })
}

self.onmessage = async (e: MessageEvent<PoseWorkerRequest>) => {
  const msg = e.data
  try {
    switch (msg.type) {
      case "init":
        await init()
        break
      case "mode":
        runningMode = msg.running
        trackedBbox = null
        break
      case "reset":
        trackedBbox = null
        metersPerPixel = null
        break
      case "video":
        if (session && runningMode === "VIDEO") await detect(msg.bitmap, msg.mediaTs, false)
        msg.bitmap.close()
        break
      case "image":
        if (session && runningMode === "IMAGE") await detect(msg.bitmap, msg.mediaTs, true)
        msg.bitmap.close()
        break
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) })
  }
}
