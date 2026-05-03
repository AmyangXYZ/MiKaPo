import { FilesetResolver, HolisticLandmarker, HolisticLandmarkerResult } from "@mediapipe/tasks-vision"
import { useEffect, useRef, useState, useCallback } from "react"
import Image from "next/image"
import { Quaternion, Vector3 } from "@babylonjs/core"
import Encoding from "encoding-japanese"
import { BoneState, Solver } from "@/lib/solver"
import { FaceBlendshapeSolver, FaceSolverResult, FaceMorphWeights } from "@/lib/face-blendshape-solver"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Camera, Image as ImageIcon, Video, Webcam, Pause, Play, Circle } from "lucide-react"
import DebugScene from "./debug-scene"

type InputMode = "image" | "video" | "camera" | null

interface RecordedFrame {
  boneStates: BoneState[]
  morphWeights: FaceMorphWeights | null
}

export const MotionCapture = ({
  applyPose,
  applyFace,
  modelLoaded,
  onMediaPipeReadyChange,
  resetModel,
  restPose,
}: {
  applyPose: (boneStates: BoneState[]) => void
  applyFace: (faceResult: FaceSolverResult) => void
  modelLoaded: boolean
  onMediaPipeReadyChange?: (ready: boolean) => void
  resetModel?: () => void
  // MMD rest-pose world bone positions, keyed by Japanese bone name.
  restPose?: Record<string, Vector3> | null
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const holisticLandmarkerRef = useRef<HolisticLandmarker | null>(null)
  const [mediaPipeReady, setMediaPipeReady] = useState(false)
  const [landmarks, setLandmarks] = useState<HolisticLandmarkerResult | null>(null)
  const [inputMode, setInputMode] = useState<InputMode>("video")
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [currentImage, setCurrentImage] = useState<string>("/4.png")
  const [videoSrc, setVideoSrc] = useState<string>("/flash.mp4")
  const [lastMedia, setLastMedia] = useState<"IMAGE" | "VIDEO">("VIDEO")
  const solverRef = useRef<Solver | null>(null)
  const faceBlendshapeSolverRef = useRef<FaceBlendshapeSolver | null>(null)
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

  // Initialize solvers and apply poses
  useEffect(() => {
    if (!solverRef.current) {
      solverRef.current = new Solver()
    }
    if (!faceBlendshapeSolverRef.current) {
      faceBlendshapeSolverRef.current = new FaceBlendshapeSolver({ smoothingFactor: 0.4 })
    }
    if (restPose && solverRef.current) {
      solverRef.current.calibrate(restPose)
    }
    if (landmarks && modelLoaded) {
      // Apply body pose
      if (solverRef.current) {
        const pose = solverRef.current.solve(landmarks)
        if (pose) {
          currentBoneStatesRef.current = pose
          applyPose(pose)
        }
      }
      // Apply face (eye rotations + morphs)
      if (faceBlendshapeSolverRef.current && landmarks.faceLandmarks?.[0]) {
        const faceResult = faceBlendshapeSolverRef.current.solve(landmarks.faceLandmarks[0])
        currentMorphWeightsRef.current = faceResult.morphWeights
        applyFace(faceResult)
      }
    }
  }, [landmarks, applyPose, applyFace, modelLoaded, restPose])

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

  // Initialize MediaPipe landmarker
  useEffect(() => {
    let isMounted = true

    const initLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm",
        )

        if (!isMounted || holisticLandmarkerRef.current) return

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
          holisticLandmarkerRef.current = await HolisticLandmarker.createFromOptions(vision, createOptions)
        } catch (gpuError) {
          console.warn("GPU delegate failed, falling back to CPU:", gpuError)
          holisticLandmarkerRef.current = await HolisticLandmarker.createFromOptions(vision, {
            ...createOptions,
            baseOptions: { ...createOptions.baseOptions, delegate: "CPU" },
          })
        }

        if (!isMounted) return

        // Warm up: force GPU shader compilation / tensor allocation now, not on the
        // user's first real frame. Result is discarded — `mediaPipeReady` is still
        // false so the detect loop below isn't running yet.
        try {
          const warmupCanvas = document.createElement("canvas")
          warmupCanvas.width = 256
          warmupCanvas.height = 256
          const ctx = warmupCanvas.getContext("2d")
          if (ctx) {
            ctx.fillStyle = "#808080"
            ctx.fillRect(0, 0, 256, 256)
          }
          await new Promise<void>((resolve) => {
            holisticLandmarkerRef.current!.detectForVideo(warmupCanvas, performance.now(), () => {
              resolve()
            })
          })
        } catch (warmupError) {
          console.warn("MediaPipe warmup failed (non-fatal):", warmupError)
        }

        if (!isMounted) return
        setMediaPipeReady(true)
        onMediaPipeReadyChangeRef.current?.(true)

        let lastTime = performance.now()
        let lastImgSrc = ""
        let frameCounter = 0
        const FRAME_SKIP = 2

        const detect = () => {
          frameCounter++
          const shouldProcess = frameCounter % FRAME_SKIP === 0

          if (videoRef.current && lastTime !== videoRef.current.currentTime && videoRef.current.videoWidth > 0) {
            lastTime = videoRef.current.currentTime
            if (shouldProcess) {
              holisticLandmarkerRef.current!.detectForVideo(videoRef.current, performance.now(), (result) => {
                if (result.poseWorldLandmarks[0]) {
                  setLandmarks(result)
                }
              })
            }
          } else if (
            imageRef.current &&
            imageRef.current.src.length > 0 &&
            imageRef.current.src !== lastImgSrc &&
            imageRef.current.complete &&
            imageRef.current.naturalWidth > 0
          ) {
            lastImgSrc = imageRef.current.src
            holisticLandmarkerRef.current!.detect(imageRef.current, (result) => {
              if (result.poseWorldLandmarks.length > 0) {
                setLandmarks(result)
              }
            })
          }
          requestAnimationFrame(detect)
        }
        detect()
      } catch (error) {
        console.error("Failed to initialize MediaPipe:", error)
      }
    }

    initLandmarker()

    return () => {
      isMounted = false
      holisticLandmarkerRef.current?.close()
      holisticLandmarkerRef.current = null
    }
  }, [])

  // Handle image upload
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.includes("image")) {
      const url = URL.createObjectURL(file)
      resetModel?.()
      solverRef.current?.reset()
      holisticLandmarkerRef.current?.setOptions({ runningMode: "IMAGE" }).then(() => {
        setCurrentImage(url)
        setVideoSrc("")
        setInputMode("image")
      })
      setLastMedia("IMAGE")
    }
  }

  // Handle video upload
  const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.includes("video")) {
      const url = URL.createObjectURL(file)
      resetModel?.()
      solverRef.current?.reset()
      if (lastMedia === "IMAGE") {
        holisticLandmarkerRef.current?.setOptions({ runningMode: "VIDEO" }).then(() => {
          setVideoSrc(url)
          setCurrentImage("")
          setInputMode("video")
          if (videoRef.current) {
            videoRef.current.currentTime = 0
          }
        })
      } else {
        setVideoSrc(url)
        setInputMode("video")
        if (videoRef.current) {
          videoRef.current.currentTime = 0
        }
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
        solverRef.current?.reset()
        setInputMode("camera")
        setIsStreamActive(true)

        if (lastMedia === "IMAGE") {
          await holisticLandmarkerRef.current?.setOptions({ runningMode: "VIDEO" })
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
          <DebugScene landmarks={landmarks} />
        </div>
      </div>
    </div>
  )
}

// VMD file creation
// frameMultiplier: 1 = 30fps, 2 = 15fps effective (slower), etc.
function createVMD(frames: RecordedFrame[], frameMultiplier: number = 1): Blob {
  if (frames.length === 0) {
    return new Blob()
  }

  function encodeShiftJIS(str: string): Uint8Array {
    const unicodeArray = Encoding.stringToCode(str)
    const sjisArray = Encoding.convert(unicodeArray, {
      to: "SJIS",
      from: "UNICODE",
    })
    return new Uint8Array(sjisArray)
  }

  const writeBoneFrame = (
    dataView: DataView,
    offset: number,
    name: string,
    frame: number,
    position: Vector3,
    rotation: Quaternion,
  ): number => {
    const nameBytes = encodeShiftJIS(name)
    for (let i = 0; i < 15; i++) {
      dataView.setUint8(offset + i, i < nameBytes.length ? nameBytes[i] : 0)
    }
    offset += 15

    dataView.setUint32(offset, frame, true)
    offset += 4

    dataView.setFloat32(offset, position.x, true)
    offset += 4
    dataView.setFloat32(offset, position.y, true)
    offset += 4
    dataView.setFloat32(offset, position.z, true)
    offset += 4

    dataView.setFloat32(offset, rotation.x, true)
    offset += 4
    dataView.setFloat32(offset, rotation.y, true)
    offset += 4
    dataView.setFloat32(offset, rotation.z, true)
    offset += 4
    dataView.setFloat32(offset, rotation.w, true)
    offset += 4

    // Interpolation parameters (64 bytes) - linear interpolation
    for (let i = 0; i < 64; i++) {
      dataView.setUint8(offset + i, 20)
    }
    offset += 64

    return offset
  }

  const writeMorphFrame = (dataView: DataView, offset: number, name: string, frame: number, weight: number): number => {
    const nameBytes = encodeShiftJIS(name)
    for (let i = 0; i < 15; i++) {
      dataView.setUint8(offset + i, i < nameBytes.length ? nameBytes[i] : 0)
    }
    offset += 15

    dataView.setUint32(offset, frame, true)
    offset += 4

    dataView.setFloat32(offset, weight, true)
    offset += 4

    return offset
  }

  const frameCount = frames.length
  const boneCnt = frames[0].boneStates.length

  // Count morph frames (only if we have morph data)
  const morphNames = frames[0].morphWeights ? Object.keys(frames[0].morphWeights) : []
  const morphCnt = morphNames.length

  const headerSize = 30 + 20
  const boneFrameSize = 15 + 4 + 12 + 16 + 64
  const morphFrameSize = 15 + 4 + 4
  const totalSize =
    headerSize + 4 + boneFrameSize * frameCount * boneCnt + 4 + morphFrameSize * frameCount * morphCnt + 4 + 4 + 4

  const buffer = new ArrayBuffer(totalSize)
  const dataView = new DataView(buffer)
  let offset = 0

  // Write header
  const header = "Vocaloid Motion Data 0002"
  for (let i = 0; i < 30; i++) {
    dataView.setUint8(offset + i, i < header.length ? header.charCodeAt(i) : 0)
  }
  offset += 30

  // Write model name (empty)
  for (let i = 0; i < 20; i++) {
    dataView.setUint8(offset + i, 0)
  }
  offset += 20

  // Write bone frame count
  dataView.setUint32(offset, frameCount * boneCnt, true)
  offset += 4

  // Generate bone keyframes
  // Frame numbers are multiplied to adjust playback speed
  // frameMultiplier=1 means 30fps, frameMultiplier=2 means 15fps effective (slower)
  for (let i = 0; i < frameCount; i++) {
    const frameNumber = i * frameMultiplier
    for (const boneState of frames[i].boneStates) {
      offset = writeBoneFrame(
        dataView,
        offset,
        boneState.name,
        frameNumber,
        new Vector3(0, 0, 0), // No translation
        boneState.rotation,
      )
    }
  }

  // Write morph frame count
  dataView.setUint32(offset, frameCount * morphCnt, true)
  offset += 4

  // Generate morph keyframes
  for (let i = 0; i < frameCount; i++) {
    const frameNumber = i * frameMultiplier
    const morphWeights = frames[i].morphWeights
    if (morphWeights) {
      for (const morphName of morphNames) {
        const weight = morphWeights[morphName as keyof FaceMorphWeights] ?? 0
        offset = writeMorphFrame(dataView, offset, morphName, frameNumber, weight)
      }
    }
  }

  // Write counts for other frame types (all 0)
  dataView.setUint32(offset, 0, true) // Camera keyframe count
  offset += 4
  dataView.setUint32(offset, 0, true) // Light keyframe count
  offset += 4
  dataView.setUint32(offset, 0, true) // Self shadow keyframe count
  offset += 4

  return new Blob([buffer], { type: "application/octet-stream" })
}
