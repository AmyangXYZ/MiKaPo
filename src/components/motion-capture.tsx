import { useEffect, useRef, useState, useCallback, type ComponentType } from "react"
import Image from "next/image"
import { BoneState, Solver } from "@/lib/solver"
import { FaceBlendshapeSolver, FaceSolverResult, FaceMorphWeights } from "@/lib/face-blendshape-solver"
import { createVMD, RecordedFrame } from "@/lib/vmd"
import type { PoseWorkerRequest, PoseWorkerResponse, PoseWorkerResult } from "@/lib/pose-worker"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Camera, Image as ImageIcon, Video, Webcam, Pause, Play, Circle } from "lucide-react"

type DebugSceneProps = { landmarks: PoseWorkerResult | null }

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
}: {
  applyPose: (boneStates: BoneState[], tweenMs: number) => void
  applyFace: (faceResult: FaceSolverResult, tweenMs: number) => void
  modelLoaded: boolean
  onMediaPipeReadyChange?: (ready: boolean) => void
  resetModel?: () => void
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

  // VMD Recording state
  const [isRecordingVMD, setIsRecordingVMD] = useState(false)
  const isRecordingRef = useRef(false)
  const recordedFramesRef = useRef<RecordedFrame[]>([])
  const [recordedFrameCount, setRecordedFrameCount] = useState(0)

  // Current pose/face state for recording
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
    if (!videoRef.current) return
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

  // VMD Recording loop
  useEffect(() => {
    if (isRecordingVMD) {
      recordedFramesRef.current = []
      setRecordedFrameCount(0)
      isRecordingRef.current = true
      let lastRecordTime = performance.now()
      const targetInterval = 1000 / 30 // 30 FPS

      const recordFrame = () => {
        if (!isRecordingRef.current) return

        const currentTime = performance.now()
        const elapsedTime = currentTime - lastRecordTime

        if (elapsedTime >= targetInterval) {
          // Record current pose and morph state
          if (currentBoneStatesRef.current.length > 0) {
            const frame: RecordedFrame = {
              boneStates: currentBoneStatesRef.current.map((bs) => ({
                name: bs.name,
                rotation: bs.rotation.clone(),
              })),
              morphWeights: currentMorphWeightsRef.current ? { ...currentMorphWeightsRef.current } : null,
            }
            recordedFramesRef.current.push(frame)
            setRecordedFrameCount(recordedFramesRef.current.length)
          }
          lastRecordTime = currentTime - (elapsedTime % targetInterval)
        }

        if (isRecordingRef.current) {
          requestAnimationFrame(recordFrame)
        }
      }

      requestAnimationFrame(recordFrame)
    } else {
      isRecordingRef.current = false
    }
  }, [isRecordingVMD])

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
        if (msg.result.poseWorldLandmarks[0]) {
          handleResultRef.current(msg.result, msg.mediaTs)
        }
      } else if (msg.type === "error") {
        pending = false
        console.error("Pose worker error:", msg.message)
      }
    }
    worker.onerror = (e) => console.error("Failed to initialize pose worker:", e.message)
    send({ type: "init" })

    let lastVideoTime = -1
    let lastImgSrc = ""

    const detect = () => {
      rafId = requestAnimationFrame(detect)
      if (!ready) return
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

  // Toggle VMD recording
  const toggleRecording = useCallback(() => {
    if (isRecordingVMD) {
      // Stop recording and auto-export
      setIsRecordingVMD(false)
      isRecordingRef.current = false

      // Auto export after a small delay to ensure last frames are captured
      setTimeout(() => {
        const frames = recordedFramesRef.current
        if (frames.length > 0) {
          // frameMultiplier=2 because VMD standard is 30fps but most players run at 60fps
          const vmdBlob = createVMD(frames, 2)
          const url = URL.createObjectURL(vmdBlob)
          const link = document.createElement("a")
          link.href = url
          link.download = "mikapo_animation.vmd"
          link.click()
          URL.revokeObjectURL(url)

          // Clear recorded frames
          recordedFramesRef.current = []
          setRecordedFrameCount(0)
        }
      }, 100)
    } else {
      // Start recording
      setIsRecordingVMD(true)
    }
  }, [isRecordingVMD])

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
              {isRecordingVMD ? (
                <span className="font-mono text-[10px] tabular-nums text-red-300/90">
                  {recordedFrameCount}f · {(recordedFrameCount / 30).toFixed(1)}s
                </span>
              ) : (
                statusPill
              )}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={toggleRecording}
                    variant="ghost"
                    size="icon"
                    className={`size-7 ${
                      isRecordingVMD
                        ? "bg-red-500/10 text-red-400 hover:bg-red-500/15 hover:text-red-300"
                        : "text-white/70 hover:bg-white/10 hover:text-white"
                    }`}
                    disabled={!isRecordingVMD && inputMode === "image"}
                  >
                    <Circle className={`size-3.5 ${isRecordingVMD ? "fill-current" : ""}`} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{isRecordingVMD ? "Stop & export VMD" : "Record VMD"}</TooltipContent>
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
                    className="flex size-6 shrink-0 items-center justify-center rounded text-white/90 transition-colors hover:bg-white/10 hover:text-white"
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
                    onChange={(e) => {
                      if (videoRef.current) videoRef.current.currentTime = Number(e.target.value)
                    }}
                    className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-white/20 accent-white outline-none [&::-moz-range-thumb]:size-2.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-white [&::-webkit-slider-thumb]:size-2.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
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

        {/* Skeleton preview — desktop only */}
        <div className="hidden aspect-[16/10] border-t border-white/5 bg-black/50 md:block">
          {DebugScene && <DebugScene landmarks={landmarks} />}
        </div>
      </div>
    </div>
  )
}
