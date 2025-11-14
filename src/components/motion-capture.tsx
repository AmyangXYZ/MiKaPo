import { FilesetResolver, HolisticLandmarker, HolisticLandmarkerResult } from "@mediapipe/tasks-vision"
import { useEffect, useRef, useState } from "react"
import Image from "next/image"
import { BoneState, Solver } from "@/lib/solver"
import { Button } from "@/components/ui/button"
import { Camera, Image as ImageIcon, Video, Webcam, Pause } from "lucide-react"

type InputMode = "image" | "video" | "camera" | null

export const MotionCapture = ({
  applyPose,
  modelLoaded,
}: {
  applyPose: (boneStates: BoneState[]) => void
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

  // Initialize solver
  useEffect(() => {
    if (!solverRef.current) {
      solverRef.current = new Solver()
    }
    if (landmarks && solverRef.current && modelLoaded) {
      const pose = solverRef.current?.solve(landmarks)
      if (pose) {
        applyPose(pose)
      }
    }
  }, [landmarks, applyPose, modelLoaded])

  // Initialize MediaPipe landmarker
  useEffect(() => {
    FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm").then(
      async (vision) => {
        holisticLandmarkerRef.current = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
            delegate: "GPU",
          },
          minPosePresenceConfidence: 0.7,
          minPoseDetectionConfidence: 0.7,
          minFaceDetectionConfidence: 0.95,
          minHandLandmarksConfidence: 0.95,
          runningMode: "VIDEO",
        })

        let lastTime = performance.now()
        let lastImgSrc = ""

        const detect = () => {
          if (videoRef.current && lastTime !== videoRef.current.currentTime && videoRef.current.videoWidth > 0) {
            lastTime = videoRef.current.currentTime
            holisticLandmarkerRef.current!.detectForVideo(videoRef.current, performance.now(), (result) => {
              if (result.poseWorldLandmarks[0]) {
                setLandmarks(result)
              }
            })
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
      }
    )

    return () => {
      holisticLandmarkerRef.current?.close()
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

  return (
    <div className="absolute top-0 left-0 z-10 p-4 max-w-[180px] md:max-w-sm w-full">
      <div className="bg-white/30 backdrop-blur-xs shadow-sm rounded-lg p-1 md:p-4">
        {/* Controls */}
        <div className="flex justify-center md:justify-between items-center mb-2">
          <div className="text-white text-lg font-medium hidden md:block">Motion Capture</div>

          <div className="flex gap-2 items-center justify-center">
            <Button onClick={toggleCamera} variant={isStreamActive ? "destructive" : "secondary"} className="size-6 md:size-8">
              {isStreamActive ? <Pause /> : <Webcam />}
            </Button>

            <Button onClick={() => imageInputRef.current?.click()} variant="secondary" className="size-6 md:size-8">
              <ImageIcon className="h-4 w-4" />
            </Button>

            <Button onClick={() => videoInputRef.current?.click()} variant="secondary" className="size-6 md:size-8">
              <Video className="h-4 w-4" />
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
      </div>
    </div>
  )
}
