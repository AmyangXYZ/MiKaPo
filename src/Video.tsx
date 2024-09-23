import { useEffect, useRef, useState } from "react"

import { FilesetResolver, NormalizedLandmark, HolisticLandmarker } from "@mediapipe/tasks-vision"
import { FormControlLabel, IconButton, Switch, Tooltip } from "@mui/material"
import { Videocam, CloudUpload } from "@mui/icons-material"
import { styled } from "@mui/material/styles"

const defaultVideoSrc = "./video/flash.mp4"

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
  setLerpFactor,
}: {
  setPose: (pose: NormalizedLandmark[]) => void
  setFace: (face: NormalizedLandmark[]) => void
  setLerpFactor: (lerpFactor: number) => void
}): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [videoSrc, setVideoSrc] = useState<string>(defaultVideoSrc)
  const [imgSrc, setImgSrc] = useState<string>("")
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false)
  const isPoseDetectionEnabled = useRef<boolean>(true)
  const isFaceDetectionEnabled = useRef<boolean>(true)
  const holisticLandmarkerRef = useRef<HolisticLandmarker | null>(null)
  const [lastMedia, setLastMedia] = useState<string>("VIDEO")

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      if (file.type.includes("video")) {
        if (lastMedia === "IMAGE") {
          setLerpFactor(0.5)
          holisticLandmarkerRef.current?.setOptions({ runningMode: "VIDEO" }).then(() => {
            setVideoSrc(url)
            setImgSrc("")
            if (videoRef.current) {
              videoRef.current.currentTime = 0
            }
          })
        } else {
          setVideoSrc(url)
          if (videoRef.current) {
            videoRef.current.currentTime = 0
          }
        }
        setLastMedia("VIDEO")
      } else if (file.type.includes("image")) {
        if (lastMedia === "VIDEO") {
          setLerpFactor(1)
          holisticLandmarkerRef.current?.setOptions({ runningMode: "IMAGE" }).then(() => {
            setVideoSrc("")
            setImgSrc(url)
          })
        } else {
          setImgSrc(url)
        }
        setLastMedia("IMAGE")
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
        videoRef.current.load()
      }
    } else {
      try {
        setIsCameraActive(true)
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (lastMedia === "IMAGE") {
          await holisticLandmarkerRef.current?.setOptions({ runningMode: "VIDEO" })
          setLerpFactor(0.5)
          setImgSrc("")
        }
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
        setLastMedia("VIDEO")
      } catch (error) {
        console.error("Error accessing camera:", error)
      }
    }
  }

  useEffect(() => {
    FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm").then(
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
        let lastImgSrc = ""
        const detect = () => {
          if (videoRef.current && lastTime != videoRef.current.currentTime && videoRef.current.videoWidth > 0) {
            lastTime = videoRef.current.currentTime
            holisticLandmarkerRef.current!.detectForVideo(videoRef.current, performance.now(), (result) => {
              if (isPoseDetectionEnabled.current && result.poseWorldLandmarks[0]) {
                setPose(result.poseWorldLandmarks[0])
              } else {
                setPose([])
              }
              if (isFaceDetectionEnabled.current && result.faceLandmarks && result.faceLandmarks.length > 0) {
                setFace(result.faceLandmarks[0])
              } else {
                setFace([])
              }
            })
          } else if (
            imgRef.current &&
            imgRef.current.src.length > 0 &&
            imgRef.current.src != lastImgSrc &&
            imgRef.current.complete
          ) {
            lastImgSrc = imgRef.current.src
            holisticLandmarkerRef.current!.detect(imgRef.current, (result) => {
              if (isPoseDetectionEnabled.current && result.poseLandmarks[0]) {
                setPose(result.poseLandmarks[0])
              } else {
                setPose([])
              }
              if (isFaceDetectionEnabled.current && result.faceLandmarks && result.faceLandmarks.length > 0) {
                setFace(result.faceLandmarks[0])
              }
            })
          }
          requestAnimationFrame(detect)
        }
        detect()
      }
    )
  }, [setPose, setFace, imgRef, videoRef])

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
      if (imgRef.current && imgRef.current.complete && canvasRef.current) {
        const containerWidth = imgRef.current.clientWidth
        const containerHeight = imgRef.current.clientHeight
        console.log("containerWidth", containerWidth)
        canvasRef.current.width = containerWidth
        canvasRef.current.height = containerHeight
        canvasRef.current.style.width = `${containerWidth}px`
        canvasRef.current.style.height = `${containerHeight}px`
        canvasRef.current.style.left = "0"
        canvasRef.current.style.top = "0"
      }
    }
    resizeCanvas()
  }, [videoSrc, imgSrc])

  return (
    <>
      <div className="toolbar">
        <Tooltip title="Upload a video or image">
          <IconButton className="toolbar-item" color="primary" component="label" disabled={isCameraActive}>
            <CloudUpload />
            <VisuallyHiddenInput
              type="file"
              onChange={handleFileUpload}
              accept="video/*, image/*"
              disabled={isCameraActive}
            />
          </IconButton>
        </Tooltip>
        <Tooltip title="Use your camera">
          <IconButton className="toolbar-item" onClick={toggleCamera} color={isCameraActive ? "error" : "success"}>
            <Videocam />
          </IconButton>
        </Tooltip>

        <FormControlLabel
          className="toolbar-item"
          control={
            <Switch
              checked={isPoseDetectionEnabled.current}
              onChange={(e) => (isPoseDetectionEnabled.current = e.target.checked)}
              color="secondary"
              size="small"
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
              size="small"
            />
          }
          label="Face"
        />
      </div>
      <div className="video-player">
        {videoSrc || isCameraActive ? (
          <video
            ref={videoRef}
            controls={!isCameraActive}
            playsInline
            disablePictureInPicture
            controlsList="nofullscreen noremoteplayback"
            src={isCameraActive ? undefined : videoSrc}
          />
        ) : (
          <img ref={imgRef} src={imgSrc} style={{ width: "100%", height: "auto" }} />
        )}
      </div>
    </>
  )
}

export default Video
