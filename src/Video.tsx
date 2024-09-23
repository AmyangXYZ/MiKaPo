import { useEffect, useRef, useState } from "react"

import {
  FilesetResolver,
  NormalizedLandmark,
  HolisticLandmarker,
  HolisticLandmarkerResult,
} from "@mediapipe/tasks-vision"
import { IconButton, Tooltip } from "@mui/material"
import { Videocam, CloudUpload, Replay, RadioButtonChecked, StopCircle } from "@mui/icons-material"
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
  const [isRecording, setIsRecording] = useState<boolean>(true)
  const isRecordingRef = useRef<boolean>(true)
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

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false)
    } else {
      setIsRecording(true)
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
              if (isRecordingRef.current) {
                landmarkHistoryRef.current.push(result)
              }

              if (result.poseWorldLandmarks[0]) {
                setPose(result.poseWorldLandmarks[0])
              } else {
                setPose([])
              }
              if (result.faceLandmarks && result.faceLandmarks.length > 0) {
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
              if (result.poseLandmarks[0]) {
                setPose(result.poseLandmarks[0])
              } else {
                setPose([])
              }
              if (result.faceLandmarks && result.faceLandmarks.length > 0) {
                setFace(result.faceLandmarks[0])
              }
            })
          }
          requestAnimationFrame(detect)
        }
        detect()
      }
    )
  }, [setPose, setFace, imgRef, videoRef, isRecordingRef])

  const replayCallback = () => {
    let currentIndex = 0
    const frameInterval = 1000 / 60 // 60 FPS

    const playNextFrame = () => {
      if (currentIndex < landmarkHistoryRef.current.length) {
        const result = landmarkHistoryRef.current[currentIndex]

        if (result.poseWorldLandmarks && result.poseWorldLandmarks[0]) {
          setPose(result.poseWorldLandmarks[0])
        } else {
          setPose([])
        }

        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
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
    if (isRecording) {
      landmarkHistoryRef.current = []
      isRecordingRef.current = true
    } else {
      isRecordingRef.current = false
      // save recorded
    }
  }, [isRecording, isRecordingRef])

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
        <Tooltip title={isRecording ? "Stop recording" : "Record motion capture"}>
          <IconButton className="toolbar-item" onClick={toggleRecording} color="secondary" size="small">
            {isRecording ? <StopCircle /> : <RadioButtonChecked />}
          </IconButton>
        </Tooltip>
        <Tooltip title="Replay last capture">
          <IconButton
            className="toolbar-item"
            onClick={replayCallback}
            color="secondary"
            size="small"
            disabled={isCameraActive}
          >
            <Replay />
          </IconButton>
        </Tooltip>
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
