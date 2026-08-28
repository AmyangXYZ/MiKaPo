/// <reference lib="webworker" />
// MediaPipe detection worker. Detection runs off the main thread so the WebGPU
// render loop never blocks on it — frames arrive as transferred ImageBitmaps,
// results go back as plain landmark arrays.
//
// The body comes from PoseLandmarker rather than from HolisticLandmarker.
// Holistic bundles a pose model that cannot be chosen, and on recorded footage
// it swings the body a half turn between adjacent frames: four such flips in
// eight seconds, worst 165°. The standalone model, given the same frames at
// the same thresholds, produces none at all — and costs less than half as
// much per frame. Hands and face are the two models still to come back.
import { FilesetResolver, PoseLandmarker, PoseLandmarkerResult, type NormalizedLandmark, type Landmark } from "@mediapipe/tasks-vision"

export type PoseWorkerRequest =
  | { type: "init" }
  | { type: "mode"; running: "VIDEO" | "IMAGE" }
  | { type: "video"; bitmap: ImageBitmap; ts: number; mediaTs: number }
  | { type: "image"; bitmap: ImageBitmap; mediaTs: number }
  | { type: "reset" }

/** What the app consumes, structured-clone friendly. The hand and face arrays
 * stay empty while those models are out: the solver reads an absent set the
 * same way it reads a lost one, and crossfades those bones to rest. */
export interface PoseWorkerResult {
  poseWorldLandmarks: Landmark[][]
  leftHandWorldLandmarks: Landmark[][]
  rightHandWorldLandmarks: Landmark[][]
  faceLandmarks: NormalizedLandmark[][]
  /** Image-space pose landmarks: the solver's projective depth rebuild reads
   * the 2D spine length, which the hip-centred world landmarks cannot carry. */
  poseLandmarks: NormalizedLandmark[][]
  /** Frame width/height — 2D landmark x is width-normalized. */
  imageAspect: number
  /** Wall time the detector spent on this frame. The capture rate is set by
   *  this number, so it is worth showing rather than guessing at. */
  inferenceMs: number
}

export type PoseWorkerResponse =
  | { type: "ready" }
  /** The graph has been rebuilt for a new source and will detect the very next
   *  frame it is handed. Playing before this lands means frames going by with
   *  no pose behind them. */
  | { type: "prepared" }
  | { type: "result"; result: PoseWorkerResult; mediaTs: number }
  | { type: "error"; message: string }

let landmarker: PoseLandmarker | null = null
let runningMode: "VIDEO" | "IMAGE" = "VIDEO"

const post = (msg: PoseWorkerResponse) => (self as unknown as Worker).postMessage(msg)

const NONE: never[] = []

const emit = (result: PoseLandmarkerResult, mediaTs: number, imageAspect: number, inferenceMs: number) => {
  post({
    type: "result",
    mediaTs,
    result: {
      inferenceMs,
      poseWorldLandmarks: result.worldLandmarks,
      poseLandmarks: result.landmarks,
      leftHandWorldLandmarks: NONE,
      rightHandWorldLandmarks: NONE,
      faceLandmarks: NONE,
      imageAspect,
    },
  })
}

async function init(): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm",
  )

  const createOptions = {
    baseOptions: {
      // Served from public/ for now. It belongs in the assets bucket like the
      // rest, once it is uploaded there.
      modelAssetPath: "/pose_landmarker_full.task",
      delegate: "GPU" as const,
    },
    minPosePresenceConfidence: 0.7,
    minPoseDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
    numPoses: 1,
    runningMode: "VIDEO" as const,
  }

  try {
    landmarker = await PoseLandmarker.createFromOptions(vision, createOptions)
  } catch (gpuError) {
    console.warn("GPU delegate failed in worker, falling back to CPU:", gpuError)
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      ...createOptions,
      baseOptions: { ...createOptions.baseOptions, delegate: "CPU" },
    })
  }

  // Warm up: force shader compilation / tensor allocation before the first real frame.
  try {
    const canvas = new OffscreenCanvas(256, 256)
    const ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.fillStyle = "#808080"
      ctx.fillRect(0, 0, 256, 256)
    }
    await new Promise<void>((resolve) => {
      landmarker!.detectForVideo(canvas, performance.now(), () => resolve())
    })
  } catch (warmupError) {
    console.warn("MediaPipe warmup failed (non-fatal):", warmupError)
  }

  post({ type: "ready" })
}

self.onmessage = async (e: MessageEvent<PoseWorkerRequest>) => {
  const msg = e.data
  try {
    switch (msg.type) {
      case "init":
        await init()
        break
      case "mode":
        if (landmarker && msg.running !== runningMode) {
          await landmarker.setOptions({ runningMode: msg.running })
          runningMode = msg.running
        }
        post({ type: "prepared" })
        break
      case "reset":
        // Between stills, the landmarker must forget the previous one. Its
        // graph carries tracker state even in IMAGE mode, so two uploads in a
        // row show the second easing out of the first — visible in the raw
        // landmarks, before any solving. setOptions rebuilds the graph, which
        // is the documented way to clear it without rebuilding the model.
        if (landmarker) await landmarker.setOptions({ runningMode })
        post({ type: "prepared" })
        break
      case "video": {
        if (landmarker && runningMode === "VIDEO") {
          // Read before close — the result callback outlives the bitmap.
          const aspect = msg.bitmap.width / Math.max(1, msg.bitmap.height)
          const t0 = performance.now()
          landmarker.detectForVideo(msg.bitmap, msg.ts, (result) =>
            emit(result, msg.mediaTs, aspect, performance.now() - t0),
          )
        }
        msg.bitmap.close()
        break
      }
      case "image": {
        if (landmarker && runningMode === "IMAGE") {
          const aspect = msg.bitmap.width / Math.max(1, msg.bitmap.height)
          const t0 = performance.now()
          landmarker.detect(msg.bitmap, (result) => emit(result, msg.mediaTs, aspect, performance.now() - t0))
        }
        msg.bitmap.close()
        break
      }
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) })
  }
}
