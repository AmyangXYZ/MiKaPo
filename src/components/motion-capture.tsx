import { useEffect, useRef, useState, useCallback } from "react"
import Image from "next/image"
import DebugScene from "./debug-scene"
import { BoneState, Solver, type BodyCollider } from "@/lib/solver"
import { FaceBlendshapeSolver, FaceSolverResult, FaceMorphWeights } from "@/lib/face-blendshape-solver"
import { buildClip, clipSummary, RecordedFrame } from "@/lib/vmd"
import { smoothTakeZeroPhase } from "@/lib/filters"
import type { PoseWorkerRequest, PoseWorkerResponse, PoseWorkerResult } from "@/lib/pose-worker"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Camera, Image as ImageIcon, Video, Webcam, Pause, Play, Download, Square } from "lucide-react"

/** Pose engine. RTMW3D: one ONNX model, whole-body 3D (body+hands), WebGPU.
 * Flip to false to fall back to MediaPipe holistic for A/B comparison. */
const USE_RTMW3D = true

/**
 * One tuning row: name, setting, slider — and for anything with a live signal, a
 * meter under it showing what the solver is currently reading.
 *
 * The meter is the point. A threshold is meaningless on its own; watched against
 * the level it gates, "drag until blinking registers" becomes something you can
 * see rather than guess.
 */

type InputMode = "image" | "video" | "camera" | null

/** Debug skeleton preview refresh (React re-render); the model itself is driven
 * directly from the detection callback and doesn't wait for React. */
const DEBUG_PREVIEW_INTERVAL_MS = 66

export const MotionCapture = ({
  applyPose,
  applyFace,
  getLivePose,
  modelLoaded,
  onMediaPipeReadyChange,
  resetModel,
  restPose,
  colliders,
  modelMorphs,
  exportVmd,
}: {
  applyPose: (boneStates: BoneState[], tweenMs: number) => void
  applyFace: (faceResult: FaceSolverResult, tweenMs: number) => void
  /** Live engine bone positions, for the debug inset's "applied" panel. */
  getLivePose?: () => { name: string; x: number; y: number; z: number }[]
  modelLoaded: boolean
  onMediaPipeReadyChange?: (ready: boolean) => void
  resetModel?: () => void
  /** Hands a finished clip to the model, which writes the VMD. The engine owns
   *  the format — the same writer Reze Studio exports through. */
  exportVmd?: (clip: ReturnType<typeof buildClip>) => void
  // MMD rest-pose world bone positions, keyed by Japanese bone name.
  restPose?: Record<string, { x: number; y: number; z: number }> | null
  colliders?: BodyCollider[] | null
  // Morph names present on the loaded model — resolves blendshape mappings.
  modelMorphs?: string[] | null
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const workerRef = useRef<Worker | null>(null)
  // All live workers — mode/reset must reach every engine, not just the primary.
  const enginesRef = useRef<{ worker: Worker }[]>([])
  const broadcast = (msg: PoseWorkerRequest) => {
    for (const e of enginesRef.current) e.worker.postMessage(msg)
  }
  const [mediaPipeReady, setMediaPipeReady] = useState(false)
  // ONNX model download progress (0..1); null once loaded or when unknown.
  const [modelLoadPct, setModelLoadPct] = useState<number | null>(null)
  // Which execution provider the ONNX engine landed on (webgpu vs wasm), and
  // why the better ones refused if it fell back — shown in the badge tooltip.
  const [engineEp, setEngineEp] = useState<string | null>(null)
  const [engineNote, setEngineNote] = useState<string | null>(null)
  const [landmarks, setLandmarks] = useState<PoseWorkerResult | null>(null)
  const [inputMode, setInputMode] = useState<InputMode>("video")
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [currentImage, setCurrentImage] = useState<string>("/4.png")
  const [videoSrc, setVideoSrc] = useState<string>("/Stellar (스텔라) - Vibrato (떨려요)- DANCE COVER.mp4")
  const [lastMedia, setLastMedia] = useState<"IMAGE" | "VIDEO">("VIDEO")
  const solverRef = useRef<Solver>(new Solver())
  const faceBlendshapeSolverRef = useRef<FaceBlendshapeSolver>(new FaceBlendshapeSolver())
  const onMediaPipeReadyChangeRef = useRef(onMediaPipeReadyChange)
  useEffect(() => {
    onMediaPipeReadyChangeRef.current = onMediaPipeReadyChange
  }, [onMediaPipeReadyChange])

  // Tuning. Applied straight to the live solvers — no re-detection needed, so a
  // drag shows its effect on the next frame.
  //
  // Every face control is a SENSITIVITY: 0 needs a deliberate expression, 1
  // triggers readily. The underlying thresholds do not agree on direction — a
  // higher eye-closed ratio triggers a blink sooner, while a higher mouth or
  // smile threshold triggers later — so exposing them raw meant three sliders
  // where "more" meant different things. The mapping lives here; the UI only
  // ever says "more sensitive to the right".
  // Face capture runs with the body. The solver's thresholds are its own
  // business — surfacing them as unlabelled dials asked people to tune what the
  // defaults already handle.
  const faceEnabledRef = useRef(true)

  // Offline conversion state. Nothing is "recorded" as it plays — the video is
  // stepped frame by frame, so the result does not depend on how fast this
  // machine happens to be.
  const [converting, setConverting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [exported, setExported] = useState<string | null>(null)
  const convertingRef = useRef(false)
  const cancelRef = useRef(false)
  // Set while a conversion frame is in flight; the worker's reply resolves it
  // instead of going down the live path.
  const awaitingRef = useRef<((r: PoseWorkerResult | null) => void) | null>(null)

  // Current pose/face state
  const currentBoneStatesRef = useRef<BoneState[]>([])
  const currentMorphWeightsRef = useRef<FaceMorphWeights | null>(null)

  // Custom video controls — replaces native browser chrome to match the panel style.
  const [videoPlaying, setVideoPlaying] = useState(false)
  const [videoTime, setVideoTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  /** Accept a duration only once it is a real number. When the browser says
   *  Infinity (common for WebM and anything streamed), nudge it into resolving:
   *  seeking far past the end forces the demuxer to find the true end, and the
   *  durationchange that follows carries the answer. Guarded by a flag so the
   *  nudge happens once per file. */
  const durationNudged = useRef(false)
  const resolveDuration = useCallback((video: HTMLVideoElement) => {
    const d = video.duration
    if (Number.isFinite(d) && d > 0) {
      setVideoDuration(d)
      durationNudged.current = false
      return
    }
    if (durationNudged.current) return
    durationNudged.current = true
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked)
      video.currentTime = 0
    }
    video.addEventListener("seeked", onSeeked)
    video.currentTime = 1e9
  }, [])
  const formatTime = (s: number): string => {
    if (!Number.isFinite(s) || s < 0) return "0:00"
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, "0")}`
  }
  const toggleVideoPlay = () => {
    // The conversion owns the playhead while it steps through the take.
    if (!videoRef.current || convertingRef.current) return
    if (videoRef.current.paused) videoRef.current.play()
    else videoRef.current.pause()
  }

  // Space toggles playback, the way every player does. A focused control keeps
  // its own Space (that is the control's job), and typing is never intercepted.
  const toggleVideoPlayRef = useRef(toggleVideoPlay)
  useEffect(() => {
    toggleVideoPlayRef.current = toggleVideoPlay
  })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space") return
      const t = e.target as HTMLElement | null
      if (t && (["INPUT", "TEXTAREA", "BUTTON", "SELECT"].includes(t.tagName) || t.isContentEditable)) return
      e.preventDefault()
      toggleVideoPlayRef.current()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  // Re-calibrate solver reference directions when a (new) model's rest pose arrives.
  useEffect(() => {
    if (restPose) solverRef.current.calibrate(restPose)
    if (restPose && colliders) solverRef.current.calibrateColliders(colliders, restPose)
  }, [restPose, colliders])

  // Bisection switches: ?ground=0 ?clear=0 ?rhythm=0 ?witness=0 disable one
  // post-solve stage each, to isolate which one breaks a pose live.
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    const s = solverRef.current
    if (p.get("ground") === "0") s.groundingGain = 0
    if (p.get("clear") === "0") s.clearanceSlack = 1e9
    if (p.get("rhythm") === "0") s.shoulderRatio = 0
    if (p.get("witness") === "0") s.witnessEnabled = false
  }, [])

  // Resolve blendshape→morph mappings against the loaded model's morph list.
  useEffect(() => {
    if (modelMorphs && modelMorphs.length > 0) faceBlendshapeSolverRef.current.configure(modelMorphs)
  }, [modelMorphs])

  // Hot path: detection result → solve → apply, all through refs — no React
  // state or effects between a video frame and the model moving.
  const modelLoadedRef = useRef(modelLoaded)
  useEffect(() => {
    modelLoadedRef.current = modelLoaded
  }, [modelLoaded])
  const applyPoseRef = useRef(applyPose)
  const applyFaceRef = useRef(applyFace)
  useEffect(() => {
    applyPoseRef.current = applyPose
    applyFaceRef.current = applyFace
  }, [applyPose, applyFace])

  // The live loop reads this without re-subscribing: a still is solved
  // unfiltered (no time axis to smooth along).
  const inputModeRef = useRef<InputMode>(null)
  useEffect(() => {
    inputModeRef.current = inputMode
  }, [inputMode])

  const lastDebugUpdateRef = useRef(0)
  // Detection results arrive at ~25-35 Hz while the renderer runs at 60 —
  // tween each pose over the measured inter-result interval (EMA, slight
  // overlap) so the model is always mid-motion between results instead of
  // reaching its target early and stepping.
  const lastResultAtRef = useRef(0)
  const resultIntervalEmaRef = useRef(33)
  const handleResult = useCallback((result: PoseWorkerResult, timestampMs: number) => {
    // Throttled React update — feeds only the debug skeleton preview.
    const now = performance.now()
    if (now - lastDebugUpdateRef.current >= DEBUG_PREVIEW_INTERVAL_MS) {
      lastDebugUpdateRef.current = now
      setLandmarks(result)
    }

    if (lastResultAtRef.current > 0) {
      const dt = now - lastResultAtRef.current
      if (dt < 500) resultIntervalEmaRef.current = resultIntervalEmaRef.current * 0.8 + dt * 0.2
    }
    lastResultAtRef.current = now
    // 0.9×: finish the tween just before the average next result — chasing with
    // a longer duration compounds into extra latency and shaved motion peaks.
    // The cap only binds below ~2.5 Hz: at slow capture rates a snap-and-wait
    // reads worse than the latency, so let the tween span the real interval.
    const tweenMs = Math.min(400, resultIntervalEmaRef.current * 0.9)

    if (!modelLoadedRef.current) return

    const pose = solverRef.current.solve(result, timestampMs, inputModeRef.current === "image")
    currentBoneStatesRef.current = pose
    // A still has nothing to tween FROM — easing into it just replays the
    // previous upload on the way to this one.
    applyPoseRef.current(pose, inputModeRef.current === "image" ? 0 : tweenMs)

    if (faceEnabledRef.current && result.faceLandmarks?.[0]) {
      const faceResult = faceBlendshapeSolverRef.current.solve(result.faceLandmarks[0], timestampMs)
      currentMorphWeightsRef.current = faceResult.morphWeights
      applyFaceRef.current(faceResult, tweenMs)
    }
  }, [])
  const handleResultRef = useRef(handleResult)
  handleResultRef.current = handleResult
  const exportVmdRef = useRef(exportVmd)
  exportVmdRef.current = exportVmd

  // Initialize the MediaPipe detection worker and the frame-feed loop.
  // Detection runs off the main thread so the WebGPU render loop never blocks
  // on it; this loop only snapshots frames (createImageBitmap) and ships them.
  useEffect(() => {
    let rafId = 0
    let lastSentAt = 0

    // Engine pool. One worker normally; when the ONNX engine lands on a GPU
    // provider, a SECOND worker spawns with its own WebGPU device — ORT-web
    // serializes runs per device, so a second device is the only way two
    // frames actually overlap on the GPU. Results are re-woven by `seq`.
    interface Engine {
      worker: Worker
      inFlight: number
      parallel: number
      ready: boolean
    }
    const engines: Engine[] = []
    enginesRef.current = engines
    let sendSeq = 0
    let emitSeq = 0
    const heldResults = new Map<number, { result: PoseWorkerResult; mediaTs: number }>()

    const spawn = (primary: boolean): void => {
      const worker = USE_RTMW3D
        ? new Worker(new URL("../lib/onnx-pose-worker.ts", import.meta.url))
        : new Worker(new URL("../lib/pose-worker.ts", import.meta.url))
      const eng: Engine = { worker, inFlight: 0, parallel: 1, ready: false }
      engines.push(eng)
      if (primary) workerRef.current = worker

      worker.onmessage = (e: MessageEvent<PoseWorkerResponse>) => {
        const msg = e.data
        if (msg.type === "ready") {
          eng.ready = true
          eng.parallel = msg.parallel ?? 1
          if (primary) {
            setMediaPipeReady(true)
            setModelLoadPct(null)
            if (msg.ep) setEngineEp(msg.ep)
            if (msg.note) setEngineNote(msg.note)
            onMediaPipeReadyChangeRef.current?.(true)
            if (USE_RTMW3D && msg.ep && msg.ep !== "wasm" && engines.length < 2) spawn(false)
          } else {
            setEngineEp((prev) => (prev && !prev.endsWith("×2") ? `${prev} ×2` : prev))
          }
        } else if (msg.type === "progress") {
          if (primary) setModelLoadPct(msg.total > 0 ? msg.loaded / msg.total : null)
        } else if (msg.type === "result") {
          eng.inFlight = Math.max(0, eng.inFlight - 1)
          // A conversion owns the primary worker while it runs; its awaiting
          // frame takes the reply instead of the live path.
          const awaiting = awaitingRef.current
          if (awaiting) {
            awaitingRef.current = null
            awaiting(msg.result)
          } else if (msg.seq !== undefined) {
            // Weave results from both engines back into send order — the
            // solver's filters need monotonic media time.
            heldResults.set(msg.seq, { result: msg.result, mediaTs: msg.mediaTs })
            while (heldResults.has(emitSeq)) {
              const h = heldResults.get(emitSeq)!
              heldResults.delete(emitSeq)
              if (h.result.poseWorldLandmarks[0]) handleResultRef.current(h.result, h.mediaTs)
              emitSeq++
            }
          } else if (msg.result.poseWorldLandmarks[0]) {
            handleResultRef.current(msg.result, msg.mediaTs)
          }
        } else if (msg.type === "error") {
          eng.inFlight = Math.max(0, eng.inFlight - 1)
          const awaiting = awaitingRef.current
          if (awaiting) {
            awaitingRef.current = null
            awaiting(null)
          }
          console.error("Pose worker error:", msg.message)
        }
      }
      worker.onerror = (e) => console.error("Failed to initialize pose worker:", e.message)
      worker.postMessage({ type: "init" } satisfies PoseWorkerRequest)
    }
    spawn(true)

    let lastVideoTime = -1
    let lastImgSrc = ""

    const detect = () => {
      rafId = requestAnimationFrame(detect)
      // A conversion steps the video itself; the opportunistic loop would fight
      // it for the workers and re-solve the same frames.
      if (convertingRef.current) return
      const ready = engines.filter((en) => en.ready)
      if (ready.length === 0) return
      const now = performance.now()
      if (ready.every((en) => en.inFlight >= en.parallel)) {
        // Recover if a worker dropped frames (e.g. mode switch mid-flight).
        if (now - lastSentAt > 2000) ready.forEach((en) => (en.inFlight = 0))
        else return
      }
      const video = videoRef.current
      if (video && video.videoWidth > 0 && video.currentTime !== lastVideoTime) {
        // Pacing: the in-flight guard above plus the new-frame gate (source
        // fps) — no artificial rate floor, so result cadence stays as steady
        // as the workers can deliver.
        lastVideoTime = video.currentTime
        // Media time drives the solver's smoothing filters so pause/seek
        // reset them correctly; `ts` gets wall time (MediaPipe's clock).
        const mediaTs = video.currentTime * 1000
        // Least-loaded engine takes the frame.
        const target = ready.reduce((a, b) => (a.inFlight / a.parallel <= b.inFlight / b.parallel ? a : b))
        if (target.inFlight >= target.parallel) return
        target.inFlight++
        lastSentAt = now
        // Downscale at capture: the crop model input is only 288×384, so a
        // 1080p transfer + draw per frame is pure overhead.
        const resize = video.videoWidth > 800 ? { resizeWidth: 720 } : undefined
        createImageBitmap(video, resize as ImageBitmapOptions)
          .then((bitmap) => {
            // seq assigned at SEND so the weave order matches what reaches
            // the workers, even if two bitmap creations resolve out of order.
            const seq = sendSeq++
            target.worker.postMessage(
              { type: "video", bitmap, ts: performance.now(), mediaTs, seq } satisfies PoseWorkerRequest,
              [bitmap],
            )
          })
          .catch(() => {
            target.inFlight = Math.max(0, target.inFlight - 1)
          })
      } else if (
        imageRef.current &&
        imageRef.current.src.length > 0 &&
        imageRef.current.src !== lastImgSrc &&
        imageRef.current.complete &&
        imageRef.current.naturalWidth > 0
      ) {
        const img = imageRef.current
        lastImgSrc = img.src
        const primary = engines[0]
        if (!primary?.ready) return
        primary.inFlight++
        lastSentAt = now
        createImageBitmap(img)
          .then((bitmap) =>
            primary.worker.postMessage({ type: "image", bitmap, mediaTs: performance.now() } satisfies PoseWorkerRequest, [
              bitmap,
            ]),
          )
          .catch(() => {
            primary.inFlight = Math.max(0, primary.inFlight - 1)
          })
      }
    }
    detect()

    return () => {
      cancelAnimationFrame(rafId)
      for (const en of engines) en.worker.terminate()
      engines.length = 0
      workerRef.current = null
    }
  }, [])

  // Handle image upload
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.includes("image")) {
      const url = URL.createObjectURL(file)
      resetModel?.()
      solverRef.current.reset()
      faceBlendshapeSolverRef.current.reset()
      // Worker messages are FIFO — the mode switch lands before the next frame.
      broadcast({ type: "mode", running: "IMAGE" })
      // ...and the landmarker forgets the previous still, so this one is solved
      // on its own merits rather than tracked from the last.
      broadcast({ type: "reset" })
      setCurrentImage(url)
      setVideoSrc("")
      setInputMode("image")
      setLastMedia("IMAGE")
    }
  }

  // Handle video upload
  const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.includes("video")) {
      const url = URL.createObjectURL(file)
      resetModel?.()
      solverRef.current.reset()
      faceBlendshapeSolverRef.current.reset()
      if (lastMedia === "IMAGE") {
        broadcast({ type: "mode", running: "VIDEO" })
        setCurrentImage("")
      }
      setVideoSrc(url)
      setInputMode("video")
      if (videoRef.current) {
        videoRef.current.currentTime = 0
      }
      setLastMedia("VIDEO")
    }
  }

  // Stop current input
  const stopCurrentInput = () => {
    if (isStreamActive && videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.src = ""
      videoRef.current.load()
    }
    setIsStreamActive(false)
    setInputMode(null)
  }

  // Camera functions
  const toggleCamera = async () => {
    if (isStreamActive) {
      stopCurrentInput()
    } else {
      try {
        stopCurrentInput()
        resetModel?.()
        solverRef.current.reset()
      faceBlendshapeSolverRef.current.reset()
        setInputMode("camera")
        setIsStreamActive(true)

        if (lastMedia === "IMAGE") {
          broadcast({ type: "mode", running: "VIDEO" })
        }

        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
        setLastMedia("VIDEO")
      } catch (error) {
        console.error("Error accessing camera:", error)
        setIsStreamActive(false)
        setInputMode(null)
      }
    }
  }

  /**
   * Video → VMD, offline.
   *
   * The video is STEPPED, not played: seek to each 1/30s mark, detect, solve,
   * keep the pose. Nothing is paced by wall time, so a slow machine produces the
   * same file as a fast one — it only takes longer — and no frame is ever
   * skipped because detection fell behind. Media time drives the solver's
   * filters, which is what they were designed for.
   */
  /**
   * A still → a one-frame VMD.
   *
   * There is no timeline to walk: the live loop has already solved the image and
   * left the pose in the refs, so exporting is a matter of writing down what is
   * already on screen. A single-frame motion is how MMD carries a pose.
   */
  const exportPoseVmd = useCallback(() => {
    const pose = currentBoneStatesRef.current
    if (pose.length === 0) return
    const clip = buildClip([
      {
        time: 0,
        boneStates: pose.map((bs) => ({ name: bs.name, rotation: bs.rotation.clone() })),
        morphWeights: faceEnabledRef.current ? currentMorphWeightsRef.current : null,
      },
    ])
    exportVmdRef.current?.(clip)
    setExported("pose saved")
  }, [])

  const convertVideoToVmd = useCallback(async () => {
    const video = videoRef.current
    const worker = workerRef.current
    // A non-finite duration (unresolved WebM/stream) would make the frame loop
    // below run forever — refuse until it is a real number.
    if (!video || !worker || !Number.isFinite(video.duration) || video.duration <= 0 || convertingRef.current) return

    video.pause()
    cancelRef.current = false
    convertingRef.current = true
    setConverting(true)
    setProgress(0)
    setExported(null)

    const step = 1 / 30
    const frames: RecordedFrame[] = []
    // detectForVideo demands a monotonically rising clock of its own, separate
    // from the media time the solver reads.
    let tick = performance.now()

    const seek = (t: number) =>
      new Promise<void>((resolve) => {
        const done = () => {
          video.removeEventListener("seeked", done)
          resolve()
        }
        video.addEventListener("seeked", done)
        video.currentTime = t
      })

    const detectAt = (bitmap: ImageBitmap, mediaTs: number) =>
      new Promise<PoseWorkerResult | null>((resolve) => {
        awaitingRef.current = resolve
        tick += 33
        const msg: PoseWorkerRequest = { type: "video", bitmap, ts: tick, mediaTs }
        worker.postMessage(msg, [bitmap])
      })

    // Seek and detect are the two expensive steps and neither needs the other's
    // result, so the next frame is decoding while the current one is in the
    // worker. Serially they added up; overlapped, the slower of the two sets the
    // pace. (The worker handles one frame at a time, so one lookahead is all
    // there is to win.)
    const grab = async (t: number): Promise<ImageBitmap | null> => {
      if (t >= video.duration) return null
      await seek(t)
      return createImageBitmap(video)
    }

    let lastPaint = 0
    try {
      let bitmap = await grab(0)
      for (let t = 0; t < video.duration && !cancelRef.current; t += step) {
        if (!bitmap) break
        const ahead = grab(t + step)
        const result = await detectAt(bitmap, t * 1000)
        bitmap = await ahead
        if (cancelRef.current) break
        if (!result?.poseWorldLandmarks[0]) continue

        const pose = solverRef.current.solve(result, t * 1000)
        currentBoneStatesRef.current = pose
        // Applied untweened: the character steps through the take as it converts,
        // which is the progress bar people actually watch.
        applyPoseRef.current(pose, 0)

        let morphWeights: FaceMorphWeights | null = null
        if (faceEnabledRef.current && result.faceLandmarks?.[0]) {
          const face = faceBlendshapeSolverRef.current.solve(result.faceLandmarks[0], t * 1000)
          morphWeights = face.morphWeights
          currentMorphWeightsRef.current = morphWeights
          applyFaceRef.current(face, 0)
        }

        frames.push({
          time: t,
          boneStates: pose.map((bs) => ({ name: bs.name, rotation: bs.rotation.clone() })),
          morphWeights,
        })
        // Throttled: a React render per frame is a render the conversion pays
        // for thousands of times, for a number that changes by a fraction.
        const now = performance.now()
        if (now - lastPaint > 100) {
          lastPaint = now
          setProgress(t / video.duration)
        }
      }
    } finally {
      awaitingRef.current = null
      convertingRef.current = false
      setConverting(false)
      setProgress(0)
    }

    if (frames.length === 0) return
    // The live solve was causal — it filtered each frame knowing only the past.
    // The finished take can be read in both directions: a zero-phase polynomial
    // fit removes the residual shake without shifting timing, and hands fast
    // transients (a kick, a snap) back their original amplitude. Export only;
    // the on-screen preview is inherently real-time.
    smoothTakeZeroPhase(frames)
    const clip = buildClip(frames)
    exportVmdRef.current?.(clip)
    const { frames: n, seconds } = clipSummary(clip)
    setExported(`${n}f · ${seconds.toFixed(1)}s`)
  }, [])

  const loadingLabel =
    modelLoadPct != null ? `Loading model… ${Math.round(modelLoadPct * 100)}%` : "Loading…"

  const statusPill =
    !mediaPipeReady && modelLoadPct != null ? (
      <span className="font-mono text-[10px] tabular-nums text-blue-300/90">{Math.round(modelLoadPct * 100)}%</span>
    ) : inputMode === "camera" ? (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-red-500/40 bg-red-500/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-red-300">
        <span className="size-1.5 animate-pulse rounded-full bg-red-500" />
        Live
      </span>
    ) : inputMode === "video" ? (
      <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white/60">
        Video
      </span>
    ) : inputMode === "image" ? (
      <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white/60">
        Image
      </span>
    ) : null

  return (
    <div className="absolute left-2 top-2 z-10 w-[148px] max-w-[calc(100vw-1rem)] md:left-3 md:top-12 md:w-[300px]">
      <div className="overflow-hidden rounded-xl border border-white/10 bg-zinc-950/35 shadow-2xl shadow-black/40 backdrop-blur-md md:bg-zinc-950/60">
        {/* Toolbar — mode buttons + status + record (status/record desktop-only). */}
        <div className="flex items-center gap-0.5 border-b border-white/5 px-1.5 py-1.5 md:gap-1 md:px-3 md:py-2">
          <TooltipProvider delayDuration={150}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={toggleCamera}
                  variant="ghost"
                  size="icon"
                  className={`size-7 ${
                    isStreamActive
                      ? "bg-white/10 text-white hover:bg-white/15"
                      : "text-white/70 hover:bg-white/10 hover:text-white"
                  }`}
                  disabled={!mediaPipeReady}
                >
                  {isStreamActive ? <Pause className="size-3.5" /> : <Webcam className="size-3.5" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {!mediaPipeReady ? loadingLabel : isStreamActive ? "Stop webcam" : "Start webcam"}
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={() => imageInputRef.current?.click()}
                  variant="ghost"
                  size="icon"
                  className="size-7 text-white/70 hover:bg-white/10 hover:text-white"
                  disabled={!mediaPipeReady}
                >
                  <ImageIcon className="size-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{!mediaPipeReady ? loadingLabel : "Upload image"}</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={() => videoInputRef.current?.click()}
                  variant="ghost"
                  size="icon"
                  className="size-7 text-white/70 hover:bg-white/10 hover:text-white"
                  disabled={!mediaPipeReady}
                >
                  <Video className="size-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{!mediaPipeReady ? loadingLabel : "Upload video"}</TooltipContent>
            </Tooltip>

            <div className="ml-auto hidden items-center gap-1.5 md:flex">
              {engineEp && (
                <span
                  className={`rounded-full border px-1.5 py-0.5 font-mono text-[9px] lowercase tracking-wide ${
                    engineEp.startsWith("webgpu")
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300/80"
                      : "border-amber-500/40 bg-amber-500/15 text-amber-300/90"
                  }`}
                  title={`ONNX execution provider: ${engineEp}${engineNote ? ` — ${engineNote}` : ""}`}
                >
                  {engineEp}
                </span>
              )}
              {converting ? (
                <span className="font-mono text-[10px] tabular-nums text-blue-300/90">
                  {Math.round(progress * 100)}%
                </span>
              ) : exported ? (
                <span className="font-mono text-[10px] tabular-nums text-emerald-300/90">{exported} saved</span>
              ) : (
                statusPill
              )}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={() =>
                      converting
                        ? (cancelRef.current = true)
                        : inputMode === "image"
                          ? exportPoseVmd()
                          : void convertVideoToVmd()
                    }
                    variant="ghost"
                    size="icon"
                    className={`size-7 ${
                      converting
                        ? "bg-blue-500/10 text-blue-300 hover:bg-blue-500/15"
                        : "text-white/70 hover:bg-white/10 hover:text-white"
                    }`}
                    // A webcam has no end, so there is nothing to convert. A
                    // video is walked frame by frame; a still is written as it
                    // stands.
                    disabled={(inputMode !== "video" && inputMode !== "image") || !mediaPipeReady}
                  >
                    {converting ? <Square className="size-3.5 fill-current" /> : <Download className="size-3.5" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {converting
                    ? "Stop converting"
                    : inputMode === "image"
                      ? "Save this pose as a VMD file"
                      : "Convert this video to a VMD file"}
                </TooltipContent>
              </Tooltip>
            </div>
          </TooltipProvider>
        </div>

        <input ref={imageInputRef} type="file" accept="image/*" onChange={handleImageUpload} hidden />
        <input ref={videoInputRef} type="file" accept="video/*" onChange={handleVideoUpload} hidden />

        {/* Media — mobile uses opacity-70 so the model bleeds through; tap/hover restores
            full clarity. Desktop stays opaque. */}
        <div className="group/media relative aspect-video bg-black/30 opacity-70 transition-opacity duration-200 hover:opacity-100 active:opacity-100 md:bg-black/50 md:opacity-100">
          {inputMode === "image" && (
            <div className="flex h-full w-full items-center justify-center">
              <Image
                src={currentImage}
                alt="Motion capture input"
                ref={imageRef}
                width={320}
                height={320}
                priority
                className="max-h-full max-w-full object-contain"
              />
            </div>
          )}

          {(inputMode === "video" || inputMode === "camera") && (
            <>
              <video
                ref={videoRef}
                className={`h-full w-full object-contain ${inputMode === "camera" ? "scale-x-[-1]" : ""}`}
                playsInline
                autoPlay={inputMode === "camera"}
                disablePictureInPicture
                controlsList="nofullscreen noremoteplayback nodownload"
                src={isStreamActive ? undefined : videoSrc}
                onPlay={() => setVideoPlaying(true)}
                onPause={() => setVideoPlaying(false)}
                onTimeUpdate={(e) => setVideoTime(e.currentTarget.currentTime)}
                onLoadedMetadata={(e) => resolveDuration(e.currentTarget)}
                // Browsers report Infinity at loadedmetadata for WebM/streamed
                // and variable-frame-rate files, then refine it later — so take
                // the update too, not just the first announcement.
                onDurationChange={(e) => resolveDuration(e.currentTarget)}
              />

              {inputMode === "video" && videoSrc && (
                <div className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-black/70 via-black/30 to-transparent px-2 py-1.5">
                  <button
                    type="button"
                    onClick={toggleVideoPlay}
                    disabled={converting}
                    className="flex size-6 shrink-0 items-center justify-center rounded text-white/90 transition-colors hover:bg-white/10 hover:text-white disabled:opacity-40"
                    aria-label={videoPlaying ? "Pause" : "Play"}
                  >
                    {videoPlaying ? <Pause className="size-3.5" /> : <Play className="size-3.5 translate-x-[1px]" />}
                  </button>
                  <input
                    type="range"
                    min={0}
                    max={videoDuration || 1}
                    step={0.01}
                    value={videoTime}
                    disabled={converting}
                    onChange={(e) => {
                      if (videoRef.current && !convertingRef.current) videoRef.current.currentTime = Number(e.target.value)
                    }}
                    className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-white/20 disabled:cursor-not-allowed accent-white outline-none [&::-moz-range-thumb]:size-2.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-white [&::-webkit-slider-thumb]:size-2.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
                  />
                  <span className="hidden font-mono text-[10px] tabular-nums text-white/70 sm:block">
                    {formatTime(videoTime)} / {formatTime(videoDuration)}
                  </span>
                </div>
              )}
            </>
          )}

          {!inputMode && (
            <div className="flex h-full w-full items-center justify-center">
              <Camera className="size-8 text-white/30" />
            </div>
          )}
        </div>

        {/* Landmarks (left) vs applied engine pose (right) — desktop only.
            Drag to orbit both. */}
        <div className="hidden aspect-[16/10] border-t border-white/5 bg-black/50 md:block">
          <DebugScene landmarks={landmarks} getLivePose={getLivePose} />
        </div>
      </div>
    </div>
  )
}
