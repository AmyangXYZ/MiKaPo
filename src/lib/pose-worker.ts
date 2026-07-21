/// <reference lib="webworker" />
// MediaPipe holistic detection worker. Detection takes ~20-30ms per frame; on
// the main thread that blocked the WebGPU render loop and capped the app at
// ~20 FPS. Here it shares nothing with rendering — frames arrive as transferred
// ImageBitmaps, results go back as plain landmark arrays.
import { FilesetResolver, HolisticLandmarker, HolisticLandmarkerResult } from "@mediapipe/tasks-vision"

export type PoseWorkerRequest =
  | { type: "init" }
  | { type: "mode"; running: "VIDEO" | "IMAGE" }
  | { type: "video"; bitmap: ImageBitmap; ts: number; mediaTs: number }
  | { type: "image"; bitmap: ImageBitmap; mediaTs: number }

/** Subset of HolisticLandmarkerResult the app consumes (structured-clone friendly).
 * Face ships as mesh landmarks: the blendshape subgraph doesn't run on the
 * holistic GPU delegate ("No support of const"), so the face solver measures
 * geometry on the mesh instead. */
export interface PoseWorkerResult {
  poseWorldLandmarks: HolisticLandmarkerResult["poseWorldLandmarks"]
  leftHandWorldLandmarks: HolisticLandmarkerResult["leftHandWorldLandmarks"]
  rightHandWorldLandmarks: HolisticLandmarkerResult["rightHandWorldLandmarks"]
  faceLandmarks: HolisticLandmarkerResult["faceLandmarks"]
}

export type PoseWorkerResponse =
  | { type: "ready" }
  | { type: "result"; result: PoseWorkerResult; mediaTs: number }
  | { type: "error"; message: string }

let landmarker: HolisticLandmarker | null = null
let runningMode: "VIDEO" | "IMAGE" = "VIDEO"

const post = (msg: PoseWorkerResponse) => (self as unknown as Worker).postMessage(msg)

const emit = (result: HolisticLandmarkerResult, mediaTs: number) => {
  post({
    type: "result",
    mediaTs,
    result: {
      poseWorldLandmarks: result.poseWorldLandmarks,
      leftHandWorldLandmarks: result.leftHandWorldLandmarks,
      rightHandWorldLandmarks: result.rightHandWorldLandmarks,
      faceLandmarks: result.faceLandmarks,
    },
  })
}

async function init(): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm",
  )

  const createOptions = {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
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
      case "video":
        if (landmarker && runningMode === "VIDEO") {
          landmarker.detectForVideo(msg.bitmap, msg.ts, (result) => emit(result, msg.mediaTs))
        }
        msg.bitmap.close()
        break
      case "image":
        if (landmarker && runningMode === "IMAGE") {
          landmarker.detect(msg.bitmap, (result) => emit(result, msg.mediaTs))
        }
        msg.bitmap.close()
        break
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) })
  }
}
