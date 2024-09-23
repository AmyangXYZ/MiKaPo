import { useEffect, useRef, useState } from "react"

import {
  FilesetResolver,
  NormalizedLandmark,
  HolisticLandmarker,
  HolisticLandmarkerResult,
} from "@mediapipe/tasks-vision"
import { FormControlLabel, IconButton, Switch, Tooltip } from "@mui/material"
import { Videocam, CloudUpload, Replay } from "@mui/icons-material"
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
  const [videoSrc, setVideoSrc] = useState<string>(defaultVideoSrc)
  const [imgSrc, setImgSrc] = useState<string>("")
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false)
  const isPoseDetectionEnabled = useRef<boolean>(true)
  const isFaceDetectionEnabled = useRef<boolean>(true)
  const holisticLandmarkerRef = useRef<HolisticLandmarker | null>(null)
  const [lastMedia, setLastMedia] = useState<string>("VIDEO")

  const landmarkHistoryRef = useRef<HolisticLandmarkerResult[]>([])

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
    landmarkHistoryRef.current = []
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
              if (!isCameraActive) {
                landmarkHistoryRef.current.push(result)
              }

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
  }, [setPose, setFace, imgRef, videoRef, isCameraActive])

  const replayCallback = () => {
    let currentIndex = 0
    const frameInterval = 1000 / 60 // 60 FPS

    const playNextFrame = () => {
      if (currentIndex < landmarkHistoryRef.current.length) {
        const result = landmarkHistoryRef.current[currentIndex]

        if (isPoseDetectionEnabled.current && result.poseWorldLandmarks && result.poseWorldLandmarks[0]) {
          setPose(result.poseWorldLandmarks[0])
        } else {
          setPose([])
        }

        if (isFaceDetectionEnabled.current && result.faceLandmarks && result.faceLandmarks.length > 0) {
          setFace(result.faceLandmarks[0])
        } else {
          setFace([])
        }

        currentIndex++
        setTimeout(() => requestAnimationFrame(playNextFrame), frameInterval)
      }
    }

    requestAnimationFrame(playNextFrame)
  }

  useEffect(() => {
    const clearLandmarkHistory = () => {
      landmarkHistoryRef.current = []
    }

    const videoElement = videoRef.current
    if (videoElement) {
      videoElement.addEventListener("play", clearLandmarkHistory)
      return () => {
        videoElement.removeEventListener("play", clearLandmarkHistory)
      }
    }
  }, [])

  return (
    <>
      <div className="toolbar">
        <Tooltip title="Upload a video or image">
          <IconButton className="toolbar-item" color="primary" component="label" disabled={isCameraActive} size="small">
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
        <Tooltip title="Replay last captured motion">
          <IconButton className="toolbar-item" onClick={replayCallback} color="secondary" size="small">
            <Replay />
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
