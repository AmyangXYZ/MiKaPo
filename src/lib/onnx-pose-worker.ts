/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
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
  bodyScore,
  estimateMetersPerPixel,
  toWorkerResult,
  type Decoded,
} from "./onnx/rtmw3d"

// RTMW3D-L first: 0.678 whole-body AP vs the X model's 0.680 at roughly half
// the compute (decode verified equivalent offline). fp16 goes to the GPU
// providers only; the wasm fallback gets fp32 — CPUs have no fp16 arithmetic,
// so ORT emulates it per-op and fp16-on-wasm comes out SLOWER than fp32.
const HF_X_FP32 =
  "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
const MODELS_GPU = ["/models/rtmw3d-l-fp16.onnx", "/models/rtmw3d-x-fp16.onnx", HF_X_FP32]
const MODELS_CPU = ["/models/rtmw3d-l.onnx", HF_X_FP32]
// DepthToSpace rewritten as a batch-1 5D chain — WebNN caps tensor rank at 5
// and refuses the standard rank-6 decomposition.
const MODELS_WEBNN = ["/models/rtmw3d-l-webnn.onnx"]
const MODEL_CACHE = "mikapo-onnx-models"

// Serve the ORT wasm runtime from the CDN, mirroring how the MediaPipe worker
// loads its fileset — keep the version in lock-step with package.json.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/"
ort.env.webgpu.powerPreference = "high-performance"
// Takes effect only when the page is cross-origin isolated (see next.config
// headers); harmless otherwise. Without threads a wasm fallback is 1-thread.
ort.env.wasm.numThreads = Math.min(8, (navigator.hardwareConcurrency || 4) - 1)

let session: ort.InferenceSession | null = null
/** GPU providers get two preprocessing slots so two frames pipeline: one
 * frame's pre/post-processing overlaps the other's GPU time. Each slot owns
 * its canvas and input buffer — sharing them across in-flight frames is a
 * race. The SESSION stays singular and its run() calls are serialized by
 * `gpuLock`: ORT-web's WebGPU sessions share one device and buffer manager
 * per worker, and concurrent OrtRun races buffer downloads ("Buffer was
 * unmapped before mapping was resolved"). */
interface Slot {
  session: ort.InferenceSession
  canvas: OffscreenCanvas
  ctx: OffscreenCanvasRenderingContext2D
  input: Float32Array
  busy: boolean
}
let slots: Slot[] = []
let gpuLock: Promise<void> = Promise.resolve()
/** Graph-capture mode: I/O lives in externally-owned GPU buffers. */
let gpuIO = false
let gpuDevice: GPUDevice | null = null
let gpuInput: GPUBuffer | null = null

function runFeeds(
  s: ort.InferenceSession,
  data: Float32Array,
  useGpu: boolean,
): ReturnType<ort.InferenceSession["run"]> {
  const name = s.inputNames[0]
  if (useGpu && gpuDevice && gpuInput) {
    gpuDevice.queue.writeBuffer(gpuInput, 0, data.buffer, data.byteOffset, data.byteLength)
    return s.run({
      [name]: ort.Tensor.fromGpuBuffer(gpuInput as never, { dataType: "float32", dims: [1, 3, MODEL_H, MODEL_W] }),
    })
  }
  return s.run({ [name]: new ort.Tensor("float32", data, [1, 3, MODEL_H, MODEL_W]) })
}
// Output names resolved to axes after warmup: y is the unique MODEL_H×2-bin
// tensor; x and z share a bin count (width and depth dims are both 288) and
// keep their graph order, x first — for this export: output, 1554, 1556.
let outX = ""
let outY = ""
let outZ = ""
let runningMode: "VIDEO" | "IMAGE" = "VIDEO"

/** Crop box carried between frames; null = full frame next. */
let trackedBbox: number[] | null = null
/** Media time of the frame trackedBbox was measured on — its staleness clock. */
let trackedTs = 0
/** EMA'd meters-per-pixel — camera distance changes slowly. */
let metersPerPixel: number | null = null

const inputData = new Float32Array(3 * MODEL_W * MODEL_H)

const makeSlot = (s: ort.InferenceSession): Slot => {
  const canvas = new OffscreenCanvas(MODEL_W, MODEL_H)
  return {
    session: s,
    canvas,
    ctx: canvas.getContext("2d", { willReadFrequently: true })!,
    input: new Float32Array(3 * MODEL_W * MODEL_H),
    busy: false,
  }
}

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
/** Why the preferred providers refused, surfaced through the ready message so
 * diagnosing a fallback doesn't require devtools archaeology. */
let epNote = ""

async function init(): Promise<void> {
  const cached: Record<string, ArrayBuffer> = {}
  const firstAvailable = async (urls: string[]): Promise<ArrayBuffer> => {
    for (const url of urls) {
      if (cached[url]) return cached[url]
      try {
        return (cached[url] = await fetchModel(url))
      } catch (e) {
        console.warn(`[rtmw3d] model source failed: ${url}`, e)
      }
    }
    throw new Error("no model source reachable")
  }
  const gpuModel = () => firstAvailable(MODELS_GPU)
  const cpuModel = () => firstAvailable(MODELS_CPU)
  const webnnModel = () => firstAvailable(MODELS_WEBNN)

  // Attempt ladder: WebNN first (Chrome can route it through CoreML on Macs,
  // which beats generic WebGPU shaders when it works), then WebGPU with graph
  // capture (static shapes make it legal, and replaying a captured graph
  // strips the per-dispatch overhead that dominates this model's cost), plain
  // WebGPU, then wasm as the last resort — each with the model that suits it.
  // Warmup runs INSIDE each attempt: provider failures can surface at first
  // run, not at create. Graph capture requires externally-owned GPU buffers
  // for I/O (`gpuIO`), so its runs feed a persistent device buffer.
  const attempts: [string, () => Promise<ArrayBuffer>, ort.InferenceSession.SessionOptions, boolean?][] = [
    [
      "webnn",
      webnnModel,
      { executionProviders: [{ name: "webnn", deviceType: "gpu" } as never], graphOptimizationLevel: "all" },
    ],
    [
      "webgpu+capture",
      gpuModel,
      {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
        enableGraphCapture: true,
        preferredOutputLocation: "gpu-buffer",
      } as ort.InferenceSession.SessionOptions,
      true,
    ],
    ["webgpu", gpuModel, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" }],
    ["wasm", cpuModel, { executionProviders: ["wasm"] }],
  ]
  let warm: Awaited<ReturnType<ort.InferenceSession["run"]>> | null = null
  for (const [name, pickModel, opts, wantGpuIO] of attempts) {
    try {
      const s = await ort.InferenceSession.create(await pickModel(), opts)
      if (wantGpuIO) {
        const device = (ort.env.webgpu as unknown as { device?: GPUDevice }).device
        if (!device) throw new Error("no webgpu device exposed for gpu I/O")
        gpuInput = device.createBuffer({
          size: inputData.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        })
        gpuDevice = device
      }
      const t0 = performance.now()
      warm = await runFeeds(s, inputData, wantGpuIO === true)
      console.log(`[rtmw3d] ${name} warmup ${(performance.now() - t0).toFixed(0)}ms`)
      session = s
      epUsed = name
      gpuIO = wantGpuIO === true
      break
    } catch (e) {
      console.warn(`[rtmw3d] ${name} failed:`, e)
      epNote += `${name}: ${e instanceof Error ? e.message : String(e)}; `
      gpuInput?.destroy()
      gpuInput = null
      gpuDevice = null
    }
  }
  if (!session || !warm) throw new Error("no execution provider worked")
  slots = [makeSlot(session)]
  // Second SLOT (same session): lets the next frame preprocess while this one
  // is on the GPU. Meaningless on single-thread wasm, where run() occupies
  // this very thread.
  if (epUsed !== "wasm") slots.push(makeSlot(session))

  const xz: string[] = []
  for (const name of session.outputNames) {
    const bins = warm[name].dims[warm[name].dims.length - 1]
    if (bins === MODEL_H * SIMCC_RATIO) outY = name
    else xz.push(name)
  }
  if (!outY || xz.length !== 2) throw new Error(`unexpected model outputs: ${session.outputNames.join(", ")}`)
  ;[outX, outZ] = xz

  post({ type: "ready", ep: epUsed, note: epNote || undefined, parallel: slots.length })
}

let inferEma = 0
let inferCount = 0

async function runOnce(slot: Slot, bitmap: ImageBitmap, bbox: readonly number[]): Promise<Decoded> {
  const cs = bboxCenterScale(bbox)
  const warp = warpTransform(cs)
  const { ctx } = slot
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.fillStyle = "#000"
  ctx.fillRect(0, 0, MODEL_W, MODEL_H)
  ctx.setTransform(warp.k, 0, 0, warp.k, warp.tx, warp.ty)
  ctx.drawImage(bitmap, 0, 0)
  const rgba = ctx.getImageData(0, 0, MODEL_W, MODEL_H).data
  imageDataToTensor(rgba, slot.input)
  // Serialize the GPU part only — preprocessing above runs concurrently
  // across slots; run() calls (and, under capture, output downloads — the
  // captured replay reuses the same output buffers) must not.
  const prev = gpuLock
  let release!: () => void
  gpuLock = new Promise((r) => (release = r))
  await prev
  const t0 = performance.now()
  let x: Float32Array
  let y: Float32Array
  let z: Float32Array
  try {
    const outputs = await runFeeds(slot.session, slot.input, gpuIO)
    ;[x, y, z] = (await Promise.all([
      outputs[outX].getData(),
      outputs[outY].getData(),
      outputs[outZ].getData(),
    ])) as [Float32Array, Float32Array, Float32Array]
  } finally {
    release()
  }
  const ms = performance.now() - t0
  inferEma = inferEma === 0 ? ms : inferEma * 0.9 + ms * 0.1
  if (++inferCount % 30 === 0)
    console.log(`[rtmw3d] ${epUsed}: ${inferEma.toFixed(0)}ms GPU/inference (~${(1000 / inferEma).toFixed(1)} Hz ceiling)`)
  return decodeSimCC3D(x, y, z, warp)
}

// Two frames can be in flight; results must still leave IN ORDER — the
// solver's filters and the tween assume monotonic media time.
let seqNext = 0
let seqEmit = 0
interface HeldResult {
  mediaTs: number
  result: ReturnType<typeof toWorkerResult>
  echoSeq?: number
  tracker?: { bbox: number[] | null; mpp: number | null }
}
const held = new Map<number, HeldResult>()

function emitOrdered(
  seq: number,
  mediaTs: number,
  result: ReturnType<typeof toWorkerResult>,
  echoSeq?: number,
  tracker?: { bbox: number[] | null; mpp: number | null },
): void {
  held.set(seq, { mediaTs, result, echoSeq, tracker })
  while (held.has(seqEmit)) {
    const h = held.get(seqEmit)!
    held.delete(seqEmit)
    post({ type: "result", mediaTs: h.mediaTs, result: h.result, seq: h.echoSeq, tracker: h.tracker })
    seqEmit++
  }
}

async function detect(
  slot: Slot,
  bitmap: ImageBitmap,
  mediaTs: number,
  still: boolean,
  seq: number,
  echoSeq?: number,
  shared?: { bbox: number[] | null; mpp: number | null },
): Promise<void> {
  // When the main thread carries tracker state (parallel engines), it is the
  // authority: two engines each keeping their own crop box and scale means
  // alternating frames are cropped and scaled slightly differently, and the
  // solver turns that into a limb angle that oscillates at the alternation
  // rate — a vibration no amount of filtering can distinguish from motion.
  if (shared) {
    trackedBbox = shared.bbox
    if (shared.mpp) metersPerPixel = shared.mpp
  }
  const w = bitmap.width
  const h = bitmap.height
  let bbox = still ? null : trackedBbox
  let decoded = await runOnce(slot, bitmap, bbox ?? [0, 0, w, h])

  // A still gets a second pass: crop to the person found in the full frame,
  // which is what a detector would have provided.
  if (still) {
    const refined = bboxFromKeypoints(decoded.kpts, decoded.scores, w, h)
    if (refined) decoded = await runOnce(slot, bitmap, refined)
  } else if (bbox && bodyScore(decoded.scores) < 0.45) {
    // The tracked crop went bad (subject moved out of the stale box, and a
    // bad crop feeds a worse box). Re-detect on the full frame NOW rather
    // than emitting a broken pose and hoping next frame recovers.
    decoded = await runOnce(slot, bitmap, [0, 0, w, h])
    bbox = null
  }
  bitmap.close()

  // The next crop must cover wherever the subject can move before the next
  // detection — the box is used one-to-two inferences later, so its margin
  // grows with the time since ITS OWN frame (a dancer covers about a
  // body-width per second). A pipelined completion can land out of order;
  // only ever track forward in media time.
  if (mediaTs >= trackedTs) {
    const dtSec = trackedTs > 0 ? Math.min(1, Math.max(0, (mediaTs - trackedTs) / 1000)) : 0.1
    trackedBbox = bboxFromKeypoints(decoded.kpts, decoded.scores, w, h, 0.1 + 1.2 * dtSec)
    trackedTs = mediaTs
  }

  const estimate = estimateMetersPerPixel(decoded.kpts, decoded.scores)
  if (estimate) {
    metersPerPixel = still || metersPerPixel === null ? estimate : metersPerPixel * 0.8 + estimate * 0.2
  } else if (metersPerPixel === null) {
    // Bootstrap from the crop: its height is a padded person, roughly 1.7m.
    bbox = trackedBbox ?? [0, 0, w, h]
    metersPerPixel = 1.7 / Math.max(1, bbox[3] - bbox[1])
  }

  emitOrdered(seq, mediaTs, toWorkerResult(decoded.kpts, decoded.scores, metersPerPixel, decoded.depth), echoSeq, {
    bbox: trackedBbox,
    mpp: metersPerPixel,
  })
}

const EMPTY_RESULT = () => ({
  poseWorldLandmarks: [],
  leftHandWorldLandmarks: [],
  rightHandWorldLandmarks: [],
  faceLandmarks: [],
})

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
        trackedTs = 0
        metersPerPixel = null
        break
      case "video": {
        const slot = slots.find((s) => !s.busy)
        if (!slot || runningMode !== "VIDEO") {
          msg.bitmap.close()
          // A dropped frame must still answer, or the main thread's sequence
          // weaver waits on this seq forever.
          if (msg.seq !== undefined) post({ type: "result", mediaTs: msg.mediaTs, result: EMPTY_RESULT(), seq: msg.seq })
          break
        }
        // Fire-and-forget: a second frame may start while this one is on the
        // GPU. detect() closes the bitmap and emits in sequence order.
        const seq = seqNext++
        slot.busy = true
        detect(slot, msg.bitmap, msg.mediaTs, false, seq, msg.seq, msg.tracker)
          .catch((err) => {
            // The empty result keeps the sequence emitter flowing AND settles
            // the main thread's in-flight accounting — do not ALSO post an
            // error for this frame (that would decrement pending twice).
            console.warn("[rtmw3d] frame failed:", err)
            emitOrdered(seq, msg.mediaTs, EMPTY_RESULT(), msg.seq)
          })
          .finally(() => {
            slot.busy = false
          })
        break
      }
      case "image":
        if (slots[0] && runningMode === "IMAGE") {
          slots[0].busy = true
          try {
            await detect(slots[0], msg.bitmap, msg.mediaTs, true, seqNext++)
          } finally {
            slots[0].busy = false
          }
        } else {
          msg.bitmap.close()
        }
        break
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) })
  }
}
