import { FilesetResolver, HolisticLandmarker, HolisticLandmarkerResult } from "@mediapipe/tasks-vision"
import { useEffect, useRef, useState, useCallback } from "react"
import Image from "next/image"
import { Quaternion, Vector3 } from "@babylonjs/core"
import Encoding from "encoding-japanese"
import { BoneState, Solver } from "@/lib/solver"
import { FaceBlendshapeSolver, FaceSolverResult, FaceMorphWeights } from "@/lib/face-blendshape-solver"
import { Button } from "@/components/ui/button"
import { Camera, Image as ImageIcon, Video, Webcam, Pause, Circle } from "lucide-react"
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
}: {
  applyPose: (boneStates: BoneState[]) => void
  applyFace: (faceResult: FaceSolverResult) => void
  modelLoaded: boolean
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const holisticLandmarkerRef = useRef<HolisticLandmarker | null>(null)
  const [landmarks, setLandmarks] = useState<HolisticLandmarkerResult | null>(null)
  const [inputMode, setInputMode] = useState<InputMode>("video")
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [currentImage, setCurrentImage] = useState<string>("/4.png")
  const [videoSrc, setVideoSrc] = useState<string>("/flash.mp4")
  const [lastMedia, setLastMedia] = useState<"IMAGE" | "VIDEO">("VIDEO")
  const solverRef = useRef<Solver | null>(null)
  const faceBlendshapeSolverRef = useRef<FaceBlendshapeSolver | null>(null)

  // VMD Recording state
  const [isRecordingVMD, setIsRecordingVMD] = useState(false)
  const isRecordingRef = useRef(false)
  const recordedFramesRef = useRef<RecordedFrame[]>([])
  const [recordedFrameCount, setRecordedFrameCount] = useState(0)

  // Current pose/face state for recording
  const currentBoneStatesRef = useRef<BoneState[]>([])
  const currentMorphWeightsRef = useRef<FaceMorphWeights | null>(null)

  // Initialize solvers and apply poses
  useEffect(() => {
    if (!solverRef.current) {
      solverRef.current = new Solver()
    }
    if (!faceBlendshapeSolverRef.current) {
      faceBlendshapeSolverRef.current = new FaceBlendshapeSolver({ smoothingFactor: 0.4 })
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
  }, [landmarks, applyPose, applyFace, modelLoaded])

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
              boneStates: currentBoneStatesRef.current.map(bs => ({
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
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
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


  return (
    <div className="absolute top-0 left-0 z-10 p-4 max-w-[180px] md:max-w-sm w-full">
      <div className="bg-white/30 backdrop-blur-xs shadow-sm rounded-lg p-1 md:p-4 flex flex-col items-center justify-center">
        {/* Controls */}
        <div className="w-full flex justify-center md:justify-between items-center mb-1 md:mb-2">
          <div className="text-white text-lg font-medium hidden md:block">Motion Capture</div>

          <div className="flex gap-1 md:gap-2 items-center justify-center flex-wrap">
            <Button onClick={toggleCamera} variant={isStreamActive ? "destructive" : "secondary"} className="size-6 md:size-8">
              {isStreamActive ? <Pause /> : <Webcam />}
            </Button>

            <Button onClick={() => imageInputRef.current?.click()} variant="secondary" className="size-6 md:size-8">
              <ImageIcon className="h-4 w-4" />
            </Button>

            <Button onClick={() => videoInputRef.current?.click()} variant="secondary" className="size-6 md:size-8">
              <Video className="h-4 w-4" />
            </Button>

            {/* Record VMD button - always allow stopping if recording */}
            <Button
              onClick={toggleRecording}
              variant={isRecordingVMD ? "destructive" : "secondary"}
              className="size-6 md:size-8 hidden md:flex"
              disabled={!isRecordingVMD && inputMode === "image"}
              title={isRecordingVMD ? "Stop recording & export VMD" : "Start VMD recording"}
            >
              <Circle className={`h-4 w-4 ${isRecordingVMD ? "fill-current" : ""}`} />
            </Button>
          </div>

          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            style={{ display: "none" }}
          />

          <input
            ref={videoInputRef}
            type="file"
            accept="video/*"
            onChange={handleVideoUpload}
            style={{ display: "none" }}
          />
        </div>

        {/* Recording indicator */}
        {isRecordingVMD && (
          <div className="w-full mb-2 px-2">
            <div className="text-white text-xs text-center flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              Recording: {recordedFrameCount} frames ({(recordedFrameCount / 30).toFixed(1)}s)
            </div>
          </div>
        )}

        {/* Media Container */}
        <div className="w-full h-28 md:h-80 bg-black/10 rounded-lg border border-white/20 overflow-hidden">
          {inputMode === "image" && (
            <div className="w-full h-full flex items-center justify-center">
              <Image
                src={currentImage}
                alt="Motion capture input"
                ref={imageRef}
                width={320}
                height={320}
                priority
                className="max-w-full max-h-full object-contain"
              />
            </div>
          )}

          {(inputMode === "video" || inputMode === "camera") && (
            <div className="relative w-full h-full">
              <video
                ref={videoRef}
                className={`w-full h-full object-contain ${inputMode === "camera" ? "scale-x-[-1]" : ""}`}
                muted
                playsInline
                autoPlay={inputMode === "camera"}
                controls={inputMode === "video"}
                disablePictureInPicture
                controlsList="nofullscreen noremoteplayback"
                src={isStreamActive ? undefined : videoSrc}
              />
              {inputMode === "camera" && (
                <div className="absolute top-2 right-2">
                  <div className="bg-red-500 text-white text-xs px-2 py-1 rounded-full animate-pulse">LIVE</div>
                </div>
              )}
            </div>
          )}

          {!inputMode && (
            <div className="w-full h-full flex items-center justify-center">
              <Camera className="h-12 w-12 text-white/50" />
            </div>
          )}
        </div>
        <div className="md:block hidden w-[320px] h-[200px] rounded-lg overflow-hidden z-10 mt-2">
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
    rotation: Quaternion
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

  const writeMorphFrame = (
    dataView: DataView,
    offset: number,
    name: string,
    frame: number,
    weight: number
  ): number => {
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
        boneState.rotation
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
