import { FilesetResolver, HolisticLandmarker, HolisticLandmarkerResult } from "@mediapipe/tasks-vision"
import { useEffect, useRef, useState } from "react"
import { BoneState, Solver } from "@/lib/solver"
import DebugScene from "./debug-scene"

export const MotionCapture = ({
  applyPose,
  modelLoaded,
}: {
  applyPose: (boneStates: BoneState[]) => void
  modelLoaded: boolean
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const holisticLandmarkerRef = useRef<HolisticLandmarker | null>(null)
  const [landmarks, setLandmarks] = useState<HolisticLandmarkerResult | null>(null)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [isVideoLoaded, setIsVideoLoaded] = useState(false)
  const [inputMode, setInputMode] = useState<"camera" | "video" | null>(null)
  const solverRef = useRef<Solver | null>(null)

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

  const stopCurrentInput = () => {
    if (isCameraActive && videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    if (isVideoLoaded && videoRef.current) {
      videoRef.current.src = ""
      videoRef.current.load()
    }
    setIsCameraActive(false)
    setIsVideoLoaded(false)
    setInputMode(null)
  }

  const toggleCamera = async () => {
    if (isCameraActive) {
      stopCurrentInput()
    } else {
      try {
        stopCurrentInput()
        setIsCameraActive(true)
        setInputMode("camera")
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
      } catch (error) {
        console.error("Error accessing camera:", error)
        setIsCameraActive(false)
        setInputMode(null)
      }
    }
  }

  const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && videoRef.current) {
      stopCurrentInput()
      setIsVideoLoaded(true)
      setInputMode("video")
      const videoUrl = URL.createObjectURL(file)
      videoRef.current.src = videoUrl
      videoRef.current.play()
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  useEffect(() => {
    FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm").then(
      async (vision) => {
        holisticLandmarkerRef.current = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
        })

        let lastTime = performance.now()
        const detect = () => {
          if (videoRef.current && lastTime !== videoRef.current.currentTime && videoRef.current.videoWidth > 0) {
            lastTime = videoRef.current.currentTime
            holisticLandmarkerRef.current!.detectForVideo(videoRef.current, performance.now(), (result) => {
              if (result.poseWorldLandmarks[0]) {
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

  return (
    <div className="absolute top-0 left-0 z-10 p-4 max-w-md w-full flex flex-col items-center gap-4">
      <div className="flex flex-col items-center gap-2">
        <video
          ref={videoRef}
          width={400}
          height={300}
          src="/flash.mp4"
          style={{ width: "400px", height: "300px", transform: inputMode === "camera" ? "scaleX(-1)" : "none" }}
          className="border border-gray-300 rounded-lg"
          muted
          playsInline
          controls={inputMode === "video"}
        />

        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleVideoUpload}
          style={{ display: "none" }}
        />

        <div className="flex gap-2">
          <button
            onClick={toggleCamera}
            className={`px-4 py-2 text-white rounded ${
              isCameraActive ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
            }`}
          >
            {isCameraActive ? "Stop Webcam" : "Start Webcam"}
          </button>

          <button
            onClick={triggerFileInput}
            className={`px-4 py-2 text-white rounded ${
              isVideoLoaded ? "bg-red-500 hover:bg-red-600" : "bg-blue-500 hover:bg-blue-600"
            }`}
          >
            {isVideoLoaded ? "Stop Video" : "Load Video"}
          </button>
        </div>

        {isCameraActive && <div className="text-sm text-green-600">Webcam Active - Motion Capture Running</div>}
        {isVideoLoaded && <div className="text-sm text-blue-600">Video Loaded - Motion Capture Running</div>}
      </div>
      <DebugScene landmarks={landmarks} />
    </div>
  )
}
