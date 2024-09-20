import { useEffect, useRef, useState } from "react"

import {
  FilesetResolver,
  PoseLandmarker,
  NormalizedLandmark,
  DrawingUtils,
  FaceLandmarker,
} from "@mediapipe/tasks-vision"
import { Button, FormControlLabel, Switch, Typography } from "@mui/material"
import { Videocam, CloudUpload } from "@mui/icons-material"
import { styled } from "@mui/material/styles"
import { blue, green } from "@mui/material/colors"

const defaultVideoSrc = "./zhiyin.mp4"

const VisuallyHiddenInput = styled("input")({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
})

function Video({
  setPose,
  setFace,
}: {
  setPose: (pose: NormalizedLandmark[]) => void
  setFace: (face: NormalizedLandmark[]) => void
}): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [videoSrc, setVideoSrc] = useState<string>(defaultVideoSrc)
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false)
  const isDebug = useRef<boolean>(true)
  const isPoseDetectionEnabled = useRef<boolean>(true)
  const isFaceDetectionEnabled = useRef<boolean>(true)

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
      setVideoSrc(defaultVideoSrc)
      if (videoRef.current) {
        videoRef.current.srcObject = null
        videoRef.current.src = defaultVideoSrc
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
            if (isPoseDetectionEnabled.current) {
              poseLandmarker.detectForVideo(videoRef.current, performance.now(), (result) => {
                setPose(result.worldLandmarks[0])
                if (canvasRef.current && isDebug.current) {
                  drawPose(result.landmarks[0])
                }
              })
            } else {
              setPose([])
            }

            if (isFaceDetectionEnabled.current) {
              const faceResult = faceLandmarker.detectForVideo(videoRef.current, performance.now(), {})
              setFace(faceResult.faceLandmarks[0])
              if (canvasRef.current && faceResult.faceLandmarks.length > 0 && isDebug.current) {
                drawFace(faceResult.faceLandmarks[0])
              }
            } else {
              setFace([])
            }
          }
          requestAnimationFrame(detect)
        }
        detect()
      }
    )
  }, [setPose, setFace])

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
        <Button
          className="toolbar-item"
          component="label"
          role={undefined}
          variant="contained"
          tabIndex={-1}
          startIcon={<CloudUpload />}
          sx={{ backgroundColor: blue[500], "&:hover": { backgroundColor: blue[700] } }}
        >
          <Typography>Upload</Typography>
          <VisuallyHiddenInput type="file" onChange={handleFileUpload} accept="video/*" disabled={isCameraActive} />
        </Button>

        <Button
          className="toolbar-item"
          variant="contained"
          endIcon={<Videocam />}
          onClick={toggleCamera}
          sx={{ backgroundColor: green[500], "&:hover": { backgroundColor: green[700] } }}
        >
          <Typography>Camera</Typography>
        </Button>

        <FormControlLabel
          className="toolbar-item"
          style={{ marginLeft: "1.5rem" }}
          control={
            <Switch
              checked={isPoseDetectionEnabled.current}
              onChange={(e) => (isPoseDetectionEnabled.current = e.target.checked)}
              color="secondary"
            />
          }
          label="Pose"
        />
        <FormControlLabel
          className="toolbar-item"
          control={
            <Switch
              checked={isFaceDetectionEnabled.current}
              onChange={(e) => (isFaceDetectionEnabled.current = e.target.checked)}
              color="secondary"
            />
          }
          label="Face"
        />
        <FormControlLabel
          className="toolbar-item"
          control={<Switch checked={isDebug.current} onChange={toggleDebug} color="warning" />}
          label="Landmark"
        />
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
