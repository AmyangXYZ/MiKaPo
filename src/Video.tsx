import { useEffect, useRef, useState } from "react"

import {
  FilesetResolver,
  PoseLandmarker,
  NormalizedLandmark,
  DrawingUtils,
  FaceLandmarker,
} from "@mediapipe/tasks-vision"

function Video({
  setPose,
  setFace,
}: {
  setPose: (pose: NormalizedLandmark[]) => void
  setFace: (face: NormalizedLandmark[]) => void
}): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [videoSrc, setVideoSrc] = useState<string>("./blue.mp4")
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false)
  const isDebug = useRef<boolean>(true)

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

  const toggleCamera = async () => {
    if (isCameraActive) {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
        tracks.forEach((track) => track.stop())
      }
      setIsCameraActive(false)
      // Set the video source after disabling the camera
      setVideoSrc("./blue.mp4")
      if (videoRef.current) {
        videoRef.current.srcObject = null
        videoRef.current.src = "./blue.mp4"
        videoRef.current.load()
      }
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
        setIsCameraActive(true)
      } catch (error) {
        console.error("Error accessing camera:", error)
      }
    }
  }

  const toggleDebug = () => {
    isDebug.current = !isDebug.current
    if (!isDebug.current && canvasRef.current) {
      canvasRef.current.getContext("2d")?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
  }

  useEffect(() => {
    FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm").then(
      async (vision) => {
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
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },

          runningMode: "VIDEO",
          outputFaceBlendshapes: true,
          numFaces: 1,
          minFacePresenceConfidence: 0.2,
          minFaceDetectionConfidence: 0.2,
        })

        const canvasCtx = canvasRef.current?.getContext("2d")
        if (canvasCtx) {
          canvasCtx.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height)
        }
        const drawingUtils = new DrawingUtils(canvasCtx as CanvasRenderingContext2D)

        const drawPose = (landmarks: NormalizedLandmark[]) => {
          drawingUtils.drawLandmarks(landmarks, {
            radius: 2,
          })
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: "white",
            lineWidth: 3,
          })
        }

        const drawFace = (landmarks: NormalizedLandmark[]) => {
          drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
            color: "white",
            lineWidth: 1,
          })
        }

        let lastTime = performance.now()
        const detect = () => {
          if (videoRef.current && lastTime != videoRef.current.currentTime && videoRef.current.videoWidth > 0) {
            if (canvasCtx) {
              canvasCtx.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height)
            }

            lastTime = videoRef.current.currentTime
            poseLandmarker.detectForVideo(videoRef.current, performance.now(), (result) => {
              setPose(result.worldLandmarks[0])
              if (canvasRef.current && isDebug.current) {
                drawPose(result.landmarks[0])
              }
            })
            const faceResult = faceLandmarker.detectForVideo(videoRef.current, performance.now(), {})
            setFace(faceResult.faceLandmarks[0])
            if (canvasRef.current && faceResult.faceLandmarks.length > 0 && isDebug.current) {
              drawFace(faceResult.faceLandmarks[0])
            }
          }
          requestAnimationFrame(detect)
        }
        detect()
      }
    )
  }, [setPose, setFace, isDebug])

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
    <div className="videoContainer">
      <div className="toolbar">
        <input
          type="file"
          accept="video/*"
          className="toolbar-item"
          onChange={handleFileUpload}
          disabled={isCameraActive}
        />
        <button className="toolbar-item" onClick={toggleCamera}>
          {isCameraActive ? "Disable Camera" : "Enable Camera"}
        </button>
        <button className="toolbar-item" onClick={toggleDebug}>
          {isDebug.current ? "Hide Landmarks" : "Show Landmarks"}
        </button>
      </div>
      <div className="videoPlayer" style={{ position: "relative" }}>
        <video
          ref={videoRef}
          controls={!isCameraActive}
          playsInline
          disablePictureInPicture
          controlsList="nofullscreen noremoteplayback"
          src={isCameraActive ? undefined : videoSrc}
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
    </div>
  )
}

export default Video
