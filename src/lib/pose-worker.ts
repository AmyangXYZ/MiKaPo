/// <reference lib="webworker" />
// MediaPipe holistic detection worker. Detection takes ~20-30ms per frame; on
// the main thread that blocked the WebGPU render loop and capped the app at
// ~20 FPS. Here it shares nothing with rendering — frames arrive as transferred
// ImageBitmaps, results go back as plain landmark arrays.
import { ASSETS } from "@/lib/assets"
import { FilesetResolver, HolisticLandmarker, HolisticLandmarkerResult } from "@mediapipe/tasks-vision"

export type PoseWorkerRequest =
  | { type: "init" }
  | { type: "mode"; running: "VIDEO" | "IMAGE" }
  | { type: "video"; bitmap: ImageBitmap; ts: number; mediaTs: number }
  | { type: "image"; bitmap: ImageBitmap; mediaTs: number }
  | { type: "reset" }

/** Subset of HolisticLandmarkerResult the app consumes (structured-clone friendly).
 * Face ships as mesh landmarks: the blendshape subgraph doesn't run on the
 * holistic GPU delegate ("No support of const"), so the face solver measures
 * geometry on the mesh instead. */
export interface PoseWorkerResult {
  poseWorldLandmarks: HolisticLandmarkerResult["poseWorldLandmarks"]
  leftHandWorldLandmarks: HolisticLandmarkerResult["leftHandWorldLandmarks"]
  rightHandWorldLandmarks: HolisticLandmarkerResult["rightHandWorldLandmarks"]
  faceLandmarks: HolisticLandmarkerResult["faceLandmarks"]
  /** Image-space pose landmarks: the solver's projective depth rebuild reads
   * the 2D spine length, which the hip-centred world landmarks cannot carry. */
  poseLandmarks: HolisticLandmarkerResult["poseLandmarks"]
  /** Frame width/height — 2D landmark x is width-normalized. */
  imageAspect: number
}

export type PoseWorkerResponse =
  | { type: "ready" }
  | { type: "result"; result: PoseWorkerResult; mediaTs: number }
  | { type: "error"; message: string }

let landmarker: HolisticLandmarker | null = null
let runningMode: "VIDEO" | "IMAGE" = "VIDEO"

const post = (msg: PoseWorkerResponse) => (self as unknown as Worker).postMessage(msg)

const emit = (result: HolisticLandmarkerResult, mediaTs: number, imageAspect: number) => {
  post({
    type: "result",
    mediaTs,
    result: {
      poseWorldLandmarks: result.poseWorldLandmarks,
      leftHandWorldLandmarks: result.leftHandWorldLandmarks,
      rightHandWorldLandmarks: result.rightHandWorldLandmarks,
      faceLandmarks: result.faceLandmarks,
      poseLandmarks: result.poseLandmarks,
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
      // Self-hosted snapshot of Google's float16 holistic_landmarker — their
      // /latest/ URL can change bytes under us; this one cannot.
      modelAssetPath: `${ASSETS}/holistic_landmarker.task`,
      delegate: "GPU" as const,
    },
    minPosePresenceConfidence: 0.7,
    minPoseDetectionConfidence: 0.7,
    minFaceDetectionConfidence: 0.4,
    minHandLandmarksConfidence: 0.95,
    runningMode: "VIDEO" as const,
  }

  try {
    landmarker = await HolisticLandmarker.createFromOptions(vision, createOptions)
  } catch (gpuError) {
    console.warn("GPU delegate failed in worker, falling back to CPU:", gpuError)
    landmarker = await HolisticLandmarker.createFromOptions(vision, {
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
        break
      case "reset":
        // Between stills, the landmarker must forget the previous one. Its
        // graph carries tracker state even in IMAGE mode, so two uploads in a
        // row show the second easing out of the first — visible in the raw
        // landmarks, before any solving. setOptions rebuilds the graph, which
        // is the documented way to clear it without rebuilding the model.
        if (landmarker) await landmarker.setOptions({ runningMode })
        break
      case "video": {
        if (landmarker && runningMode === "VIDEO") {
          // Read before close — the result callback outlives the bitmap.
          const aspect = msg.bitmap.width / Math.max(1, msg.bitmap.height)
          landmarker.detectForVideo(msg.bitmap, msg.ts, (result) => emit(result, msg.mediaTs, aspect))
        }
        msg.bitmap.close()
        break
      }
      case "image": {
        if (landmarker && runningMode === "IMAGE") {
          const aspect = msg.bitmap.width / Math.max(1, msg.bitmap.height)
          landmarker.detect(msg.bitmap, (result) => emit(result, msg.mediaTs, aspect))
        }
        msg.bitmap.close()
        break
      }
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) })
  }
}
