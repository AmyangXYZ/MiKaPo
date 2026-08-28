// Runs inside a headless browser: the only place MediaPipe will run, because
// its vision tasks build a WebGL canvas through `document` before they will
// look at a frame.
//
// It steps a video at a fixed rate, detects every frame, and posts the raw
// landmarks back. Nothing here solves anything — solving is what the replay
// does, over and over, on the recording this leaves behind.

import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision"

const params = new URLSearchParams(location.search)
const VIDEO = params.get("video") ?? "flash.mp4"
const FPS = Number(params.get("fps") ?? 30)
const WIDTH = Number(params.get("width") ?? 960)
const START = Number(params.get("start") ?? 0)
const SECONDS = Number(params.get("seconds") ?? 0)

const say = (msg) => {
  console.log(msg)
  void fetch("/log", { method: "POST", body: String(msg) })
}

const fail = (err) => {
  void fetch("/fail", { method: "POST", body: String(err?.stack ?? err) })
}
window.addEventListener("error", (e) => fail(e.error ?? e.message))
window.addEventListener("unhandledrejection", (e) => fail(e.reason))

/** Landmarks come back as class instances; only the numbers travel. */
const plain = (set) =>
  set ? set.map((p) => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility ?? 1 })) : null

async function main() {
  const fileset = await FilesetResolver.forVisionTasks("/wasm")
  say("fileset ready")

  const options = {
    baseOptions: { modelAssetPath: "/media/holistic_landmarker.task", delegate: "GPU" },
    minPosePresenceConfidence: 0.7,
    minPoseDetectionConfidence: 0.7,
    minFaceDetectionConfidence: 0.4,
    minHandLandmarksConfidence: 0.95,
    runningMode: "VIDEO",
  }
  let landmarker
  let delegate = "GPU"
  try {
    landmarker = await HolisticLandmarker.createFromOptions(fileset, options)
  } catch {
    delegate = "CPU"
    landmarker = await HolisticLandmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "CPU" },
    })
  }
  say(`landmarker ready on ${delegate}`)

  // Frames come from the server as images ffmpeg decoded, one per file.
  //
  // A <video> was the obvious source and the wrong one: seeking it in a
  // headless browser resolves `seeked` before the new picture exists, so the
  // detector is handed the same frame over and over and every measurement
  // taken from it describes a still. Decoded files cannot do that.
  const manifest = await (await fetch("/frames.json")).json()
  say(`${manifest.count} frames of ${VIDEO} at ${manifest.width}×${manifest.height}`)

  const frames = []
  // detectForVideo insists on a clock that only goes forward.
  let tick = 0
  for (let n = 0; n < manifest.count; n++) {
    const blob = await (await fetch(`/frame/${n}`)).blob()
    const bitmap = await createImageBitmap(blob)
    tick += 1000 / FPS
    const result = landmarker.detectForVideo(bitmap, tick)
    bitmap.close()
    frames.push({
      time: manifest.start + n / FPS,
      pose: plain(result.poseWorldLandmarks?.[0]),
      pose2d: plain(result.poseLandmarks?.[0]),
      leftHand: plain(result.leftHandWorldLandmarks?.[0]),
      rightHand: plain(result.rightHandWorldLandmarks?.[0]),
      faceSeen: Boolean(result.faceLandmarks?.[0]?.length),
    })
    if (frames.length % 60 === 0) say(`${frames.length}/${manifest.count} frames`)
  }

  say(`done: ${frames.length} frames`)
  await fetch("/result", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      video: VIDEO,
      fps: FPS,
      width: WIDTH,
      delegate,
      sourceSize: [manifest.width, manifest.height],
      frames,
    }),
  })
}

main().catch(fail)
