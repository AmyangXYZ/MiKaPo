import { useEffect, useRef, useState } from "react"

import { FilesetResolver, PoseLandmarker, NormalizedLandmark, DrawingUtils } from "@mediapipe/tasks-vision"

function Video({ setPose }: { setPose: (pose: NormalizedLandmark[]) => void }): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [videoSrc, setVideoSrc] = useState<string>("./zhiyin.mp4")

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setVideoSrc(url)
      if (videoRef.current) {
        videoRef.current.currentTime = 0
      }
    }
  }

  useEffect(() => {
    const initPoseDetector = async (): Promise<void> => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      )
      const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        minPosePresenceConfidence: 0.5,
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false,
      })
      let lastTime = performance.now()
      const drawPose = (landmarks: NormalizedLandmark[]) => {
        const canvasCtx = canvasRef.current?.getContext("2d")
        if (canvasCtx) {
          canvasCtx.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height)
        }
        const drawingUtils = new DrawingUtils(canvasCtx as CanvasRenderingContext2D)
        drawingUtils.drawLandmarks(landmarks, {
          radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
        })
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS)
      }
      const detect = () => {
        if (videoRef.current && lastTime != videoRef.current.currentTime && videoRef.current.videoWidth > 0) {
          lastTime = videoRef.current.currentTime
          poseLandmarker.detectForVideo(videoRef.current, performance.now(), (result) => {
            setPose(result.worldLandmarks[0])
            if (canvasRef.current) {
              drawPose(result.landmarks[0])
            }
          })
        }
        requestAnimationFrame(detect)
      }
      detect()
    }
    initPoseDetector()
  }, [setPose])

  useEffect(() => {
    const resizeCanvas = () => {
      if (videoRef.current && canvasRef.current) {
        const videoWidth = videoRef.current.videoWidth
        const videoHeight = videoRef.current.videoHeight
        const containerWidth = videoRef.current.clientWidth
        const containerHeight = videoRef.current.clientHeight

        const scale = Math.min(containerWidth / videoWidth, containerHeight / videoHeight)
        const scaledWidth = videoWidth * scale
        const scaledHeight = videoHeight * scale

        canvasRef.current.width = scaledWidth
        canvasRef.current.height = scaledHeight
        canvasRef.current.style.left = `${(containerWidth - scaledWidth) / 2}px`
        canvasRef.current.style.top = `${(containerHeight - scaledHeight) / 2}px`
      }
    }

    const videoElement = videoRef.current
    if (videoElement) {
      videoElement.addEventListener("loadedmetadata", resizeCanvas)
      window.addEventListener("resize", resizeCanvas)
    }

    return () => {
      if (videoElement) {
        videoElement.removeEventListener("loadedmetadata", resizeCanvas)
      }
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [])
  return (
    <div className="videoPlayer" style={{ position: "relative" }}>
      <input
        type="file"
        accept="video/*"
        onChange={handleFileUpload}
        style={{ position: "absolute", zIndex: "1002", left: "2rem", top: "1rem" }}
      />
      <video
        ref={videoRef}
        controls
        playsInline
        disablePictureInPicture
        controlsList="nofullscreen noremoteplayback"
        src={videoSrc}
        style={{ width: "100%", height: "100%" }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          zIndex: "1001",
          pointerEvents: "none",
        }}
      />
    </div>
  )
}

export default Video
