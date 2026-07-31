import { useEffect, useRef, useState, useCallback, type ComponentType } from "react"
import Image from "next/image"
import { BoneState, Solver } from "@/lib/solver"
import { FaceBlendshapeSolver, FaceSolverResult, FaceMorphWeights } from "@/lib/face-blendshape-solver"
import { buildClip, clipSummary, RecordedFrame } from "@/lib/vmd"
import type { PoseWorkerRequest, PoseWorkerResponse, PoseWorkerResult } from "@/lib/pose-worker"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Camera, ChevronDown, Image as ImageIcon, Video, Webcam, Pause, Play, Download, Square } from "lucide-react"

type DebugSceneProps = { landmarks: PoseWorkerResult | null }

/** One tuning row: name, live value, and the slider that moves it. */
function Knob({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  hint: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <label className="block" title={hint}>
      <span className="flex items-baseline justify-between">
        <span className="text-[11px] text-white/70">{label}</span>
        <span className="font-mono text-[10px] tabular-nums text-white/40">{value.toFixed(3)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 h-1 w-full cursor-pointer appearance-none rounded-full bg-white/15 accent-white outline-none [&::-moz-range-thumb]:size-2.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-white [&::-webkit-slider-thumb]:size-2.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
      />
    </label>
  )
}

type InputMode = "image" | "video" | "camera" | null

/** Debug skeleton preview refresh (React re-render); the model itself is driven
 * directly from the detection callback and doesn't wait for React. */
const DEBUG_PREVIEW_INTERVAL_MS = 66

export const MotionCapture = ({
  applyPose,
  applyFace,
  modelLoaded,
  onMediaPipeReadyChange,
  resetModel,
  restPose,
  modelMorphs,
  exportVmd,
}: {
  applyPose: (boneStates: BoneState[], tweenMs: number) => void
  applyFace: (faceResult: FaceSolverResult, tweenMs: number) => void
  modelLoaded: boolean
  onMediaPipeReadyChange?: (ready: boolean) => void
  resetModel?: () => void
  /** Hands a finished clip to the model, which writes the VMD. The engine owns
   *  the format — the same writer Reze Studio exports through. */
  exportVmd?: (clip: ReturnType<typeof buildClip>) => void
  // MMD rest-pose world bone positions, keyed by Japanese bone name.
  restPose?: Record<string, { x: number; y: number; z: number }> | null
  // Morph names present on the loaded model — resolves blendshape mappings.
  modelMorphs?: string[] | null
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const workerRef = useRef<Worker | null>(null)
  const [mediaPipeReady, setMediaPipeReady] = useState(false)
  const [landmarks, setLandmarks] = useState<PoseWorkerResult | null>(null)
  const [inputMode, setInputMode] = useState<InputMode>("video")
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [currentImage, setCurrentImage] = useState<string>("/4.png")
  const [videoSrc, setVideoSrc] = useState<string>("/flash.mp4")
  const [lastMedia, setLastMedia] = useState<"IMAGE" | "VIDEO">("VIDEO")
  const solverRef = useRef<Solver>(new Solver())
  const faceBlendshapeSolverRef = useRef<FaceBlendshapeSolver>(new FaceBlendshapeSolver())
  const onMediaPipeReadyChangeRef = useRef(onMediaPipeReadyChange)
  useEffect(() => {
    onMediaPipeReadyChangeRef.current = onMediaPipeReadyChange
  }, [onMediaPipeReadyChange])

  // Tuning. Applied straight to the live solvers — no re-detection needed, so a
  // drag shows its effect on the next frame.
  const [tuning, setTuning] = useState({ smoothing: 1.5, beta: 1.5, eyeClosed: 0.1, mouthOpen: 0.18, smile: 0.008 })
  const [tuningOpen, setTuningOpen] = useState(false)
  useEffect(() => {
    solverRef.current.setSmoothing(tuning.smoothing, tuning.beta)
    faceBlendshapeSolverRef.current.setThresholds({
      eyeClosed: tuning.eyeClosed,
      mouthOpen: tuning.mouthOpen,
      smile: tuning.smile,
    })
  }, [tuning])

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

  // Babylon-based skeleton preview, loaded client-side only so @babylonjs/*
  // code-splits out of the initial bundle (the main viewport is reze-engine/WebGPU).
  const [DebugScene, setDebugScene] = useState<ComponentType<DebugSceneProps> | null>(null)
  useEffect(() => {
    let mounted = true
    import("./debug-scene").then((mod) => {
      if (mounted) setDebugScene(() => mod.default)
    })
    return () => {
      mounted = false
    }
  }, [])

  // Re-calibrate solver reference directions when a (new) model's rest pose arrives.
  useEffect(() => {
    if (restPose) solverRef.current.calibrate(restPose)
  }, [restPose])

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
    const tweenMs = Math.min(100, resultIntervalEmaRef.current * 0.9)

    if (!modelLoadedRef.current) return

    const pose = solverRef.current.solve(result, timestampMs)
    currentBoneStatesRef.current = pose
    applyPoseRef.current(pose, tweenMs)

    if (result.faceLandmarks?.[0]) {
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
    let ready = false
    // In-flight guard: never queue a second frame while the worker is busy —
    // detection latency then paces capture instead of building a frame backlog.
    let pending = false
    let pendingSince = 0

    const worker = new Worker(new URL("../lib/pose-worker.ts", import.meta.url))
    workerRef.current = worker
    const send = (msg: PoseWorkerRequest, transfer?: Transferable[]) =>
      worker.postMessage(msg, transfer ?? [])

    worker.onmessage = (e: MessageEvent<PoseWorkerResponse>) => {
      const msg = e.data
      if (msg.type === "ready") {
        ready = true
        setMediaPipeReady(true)
        onMediaPipeReadyChangeRef.current?.(true)
      } else if (msg.type === "result") {
        pending = false
        // A conversion owns the worker while it runs; its awaiting frame takes
        // the reply instead of the live path.
        const awaiting = awaitingRef.current
        if (awaiting) {
          awaitingRef.current = null
          awaiting(msg.result)
        } else if (msg.result.poseWorldLandmarks[0]) {
          handleResultRef.current(msg.result, msg.mediaTs)
        }
      } else if (msg.type === "error") {
        pending = false
        const awaiting = awaitingRef.current
        if (awaiting) {
          awaitingRef.current = null
          awaiting(null)
        }
        console.error("Pose worker error:", msg.message)
      }
    }
    worker.onerror = (e) => console.error("Failed to initialize pose worker:", e.message)
    send({ type: "init" })

    let lastVideoTime = -1
    let lastImgSrc = ""

    const detect = () => {
      rafId = requestAnimationFrame(detect)
      // A conversion steps the video itself; the opportunistic loop would fight
      // it for the worker and re-solve the same frames.
      if (!ready || convertingRef.current) return
      const now = performance.now()
      if (pending) {
        // Recover if the worker dropped a frame (e.g. mode switch mid-flight).
        if (now - pendingSince > 2000) pending = false
        else return
      }
      const video = videoRef.current
      if (video && video.videoWidth > 0 && video.currentTime !== lastVideoTime) {
        // Pacing: the in-flight guard above (one frame in the worker at a time)
        // plus the new-frame gate (source fps) — no artificial rate floor, so
        // result cadence stays as steady as the worker can deliver.
        lastVideoTime = video.currentTime
        // Media time drives the solver's smoothing filters so pause/seek
        // reset them correctly; detectForVideo gets wall time because it
        // requires a monotonically increasing clock.
        const mediaTs = video.currentTime * 1000
        pending = true
        pendingSince = now
        createImageBitmap(video)
          .then((bitmap) => send({ type: "video", bitmap, ts: performance.now(), mediaTs }, [bitmap]))
          .catch(() => {
            pending = false
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
        pending = true
        pendingSince = now
        createImageBitmap(img)
          .then((bitmap) => send({ type: "image", bitmap, mediaTs: performance.now() }, [bitmap]))
          .catch(() => {
            pending = false
          })
      }
    }
    detect()

    return () => {
      cancelAnimationFrame(rafId)
      worker.terminate()
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
      workerRef.current?.postMessage({ type: "mode", running: "IMAGE" } satisfies PoseWorkerRequest)
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
        workerRef.current?.postMessage({ type: "mode", running: "VIDEO" } satisfies PoseWorkerRequest)
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
          workerRef.current?.postMessage({ type: "mode", running: "VIDEO" } satisfies PoseWorkerRequest)
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
  const convertVideoToVmd = useCallback(async () => {
    const video = videoRef.current
    const worker = workerRef.current
    if (!video || !worker || !video.duration || convertingRef.current) return

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
        if (result.faceLandmarks?.[0]) {
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
    const clip = buildClip(frames)
    exportVmdRef.current?.(clip)
    const { frames: n, seconds } = clipSummary(clip)
    setExported(`${n}f · ${seconds.toFixed(1)}s`)
  }, [])

  const statusPill =
    inputMode === "camera" ? (
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
                {!mediaPipeReady ? "Loading…" : isStreamActive ? "Stop webcam" : "Start webcam"}
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
              <TooltipContent>{!mediaPipeReady ? "Loading…" : "Upload image"}</TooltipContent>
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
              <TooltipContent>{!mediaPipeReady ? "Loading…" : "Upload video"}</TooltipContent>
            </Tooltip>

            <div className="ml-auto hidden items-center gap-1.5 md:flex">
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
                    onClick={() => (converting ? (cancelRef.current = true) : void convertVideoToVmd())}
                    variant="ghost"
                    size="icon"
                    className={`size-7 ${
                      converting
                        ? "bg-blue-500/10 text-blue-300 hover:bg-blue-500/15"
                        : "text-white/70 hover:bg-white/10 hover:text-white"
                    }`}
                    // Only a video has a timeline to walk. A webcam has no end
                    // and a still has no motion.
                    disabled={inputMode !== "video" || !mediaPipeReady}
                  >
                    {converting ? <Square className="size-3.5 fill-current" /> : <Download className="size-3.5" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {converting ? "Stop converting" : "Convert this video to a VMD file"}
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
                onLoadedMetadata={(e) => setVideoDuration(e.currentTarget.duration || 0)}
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

        {/* Tuning — desktop only, folded away until asked for. Every value here
            feeds the live solvers, so a drag is visible on the next frame. */}
        <div className="hidden border-t border-white/5 md:block">
          <button
            type="button"
            onClick={() => setTuningOpen((v) => !v)}
            className="flex w-full items-center justify-between px-3 py-1.5 text-[10px] font-medium uppercase tracking-wide text-white/50 transition-colors hover:text-white/80"
          >
            Tuning
            <ChevronDown className={`size-3 transition-transform ${tuningOpen ? "rotate-180" : ""}`} />
          </button>
          {tuningOpen && (
            <div className="space-y-2 px-3 pb-2.5">
              <Knob
                label="Smoothing"
                hint="Lower is calmer at rest, and laggier"
                value={tuning.smoothing}
                min={0.2}
                max={5}
                step={0.1}
                onChange={(v) => setTuning((t) => ({ ...t, smoothing: v }))}
              />
              <Knob
                label="Responsiveness"
                hint="How fast smoothing opens up on quick motion"
                value={tuning.beta}
                min={0}
                max={6}
                step={0.1}
                onChange={(v) => setTuning((t) => ({ ...t, beta: v }))}
              />
              <Knob
                label="Blink"
                hint="Eye must close this far to register"
                value={tuning.eyeClosed}
                min={0.02}
                max={0.25}
                step={0.005}
                onChange={(v) => setTuning((t) => ({ ...t, eyeClosed: v }))}
              />
              <Knob
                label="Mouth"
                hint="Opening needed before the mouth moves"
                value={tuning.mouthOpen}
                min={0.05}
                max={0.4}
                step={0.005}
                onChange={(v) => setTuning((t) => ({ ...t, mouthOpen: v }))}
              />
              <Knob
                label="Smile"
                hint="Corner spread before a smile registers"
                value={tuning.smile}
                min={0.001}
                max={0.03}
                step={0.001}
                onChange={(v) => setTuning((t) => ({ ...t, smile: v }))}
              />
            </div>
          )}
        </div>

        {/* Skeleton preview — desktop only */}
        <div className="hidden aspect-[16/10] border-t border-white/5 bg-black/50 md:block">
          {DebugScene && <DebugScene landmarks={landmarks} />}
        </div>
      </div>
    </div>
  )
}
