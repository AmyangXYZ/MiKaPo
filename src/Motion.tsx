import { useEffect, useRef, useState } from "react"

import {
  FilesetResolver,
  NormalizedLandmark,
  HolisticLandmarker,
  HolisticLandmarkerResult,
} from "@mediapipe/tasks-vision"
import { Badge, BadgeProps, IconButton, Tooltip } from "@mui/material"
import { Videocam, CloudUpload, Replay, RadioButtonChecked, StopCircle, LocalFireDepartment } from "@mui/icons-material"
import { styled } from "@mui/material/styles"
import DebugScene from "./DebugScene"

const defaultVideoSrc = "./video/flash.mp4"

const StyledBadge = styled(Badge)<BadgeProps>(({ theme }) => ({
  "& .MuiBadge-badge": {
    right: -3,
    top: 4,
    fontSize: ".66rem",
    minWidth: "12px",
    height: "12px",
    border: `1px solid ${theme.palette.background.paper}`,
    padding: "2px",
  },
}))

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
  pose,
  leftHand,
  rightHand,
  setPose,
  setFace,
  setLeftHand,
  setRightHand,
  setLerpFactor,
  style,
}: {
  pose: NormalizedLandmark[] | null
  leftHand: NormalizedLandmark[] | null
  rightHand: NormalizedLandmark[] | null
  setPose: (pose: NormalizedLandmark[]) => void
  setFace: (face: NormalizedLandmark[]) => void
  setLeftHand: (leftHand: NormalizedLandmark[]) => void
  setRightHand: (rightHand: NormalizedLandmark[]) => void
  setLerpFactor: (lerpFactor: number) => void
  style: React.CSSProperties
}): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [videoSrc, setVideoSrc] = useState<string>(defaultVideoSrc)
  const [imgSrc, setImgSrc] = useState<string>("")
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false)
  const [isRecording, setIsRecording] = useState<boolean>(true)
  const isRecordingRef = useRef<boolean>(true)
  const [isReplaying, setIsReplaying] = useState<boolean>(false)
  const holisticLandmarkerRef = useRef<HolisticLandmarker | null>(null)
  const [lastMedia, setLastMedia] = useState<string>("VIDEO")
  const isOfflineProcessingRef = useRef<boolean>(false)
  const offlineProcessingProgressRef = useRef<number>(0)
  const landmarkHistoryRef = useRef<HolisticLandmarkerResult[]>([])

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (landmarkHistoryRef.current) {
      landmarkHistoryRef.current = []
    }
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
        setLerpFactor(1)
        holisticLandmarkerRef.current?.setOptions({ runningMode: "IMAGE" }).then(() => {
          setVideoSrc("")
          setImgSrc(url)
        })
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
      isRecordingRef.current = false
    } else {
      landmarkHistoryRef.current = []
      setIsRecording(true)
      isRecordingRef.current = true
    }
  }

  const toggleProcessCurrentVideoOffline = async () => {
    if (!videoRef.current) return
    if (isOfflineProcessingRef.current) {
      isOfflineProcessingRef.current = false
      videoRef.current.controls = true
      return
    }
    const videoElement = videoRef.current
    videoElement.currentTime = 0
    videoElement.controls = false

    landmarkHistoryRef.current = []
    isOfflineProcessingRef.current = true
    const processFrame = async () => {
      if (!isOfflineProcessingRef.current) {
        return
      }
      if (videoElement.currentTime < videoElement.duration) {
        offlineProcessingProgressRef.current = videoElement.currentTime / videoElement.duration
        holisticLandmarkerRef.current!.detectForVideo(videoElement, performance.now(), (result) => {
          landmarkHistoryRef.current.push(result)
        })
        videoElement.currentTime += 1 / 60 // Process at 60 FPS
        await new Promise((resolve) => {
          videoElement.onseeked = () => {
            resolve(null)
          }
        })
        await processFrame()
      }
    }

    await processFrame()
    // replayCallback(60)
    isOfflineProcessingRef.current = false
    videoElement.controls = true
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
          if (
            videoRef.current &&
            lastTime != videoRef.current.currentTime &&
            videoRef.current.videoWidth > 0 &&
            !isOfflineProcessingRef.current
          ) {
            lastTime = videoRef.current.currentTime
            holisticLandmarkerRef.current!.detectForVideo(videoRef.current, performance.now(), (result) => {
              if (isRecordingRef.current && result.poseWorldLandmarks[0]) {
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
              if (result.leftHandWorldLandmarks && result.leftHandWorldLandmarks.length > 0) {
                setLeftHand(result.leftHandWorldLandmarks[0])
              } else {
                setLeftHand([])
              }
              if (result.rightHandWorldLandmarks && result.rightHandWorldLandmarks.length > 0) {
                setRightHand(result.rightHandWorldLandmarks[0])
              } else {
                setRightHand([])
              }
            })
          } else if (
            imgRef.current &&
            imgRef.current.src.length > 0 &&
            imgRef.current.src != lastImgSrc &&
            imgRef.current.complete &&
            imgRef.current.naturalWidth > 0
          ) {
            lastImgSrc = imgRef.current.src

            holisticLandmarkerRef.current!.detect(imgRef.current!, (result) => {
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
              if (result.leftHandWorldLandmarks && result.leftHandWorldLandmarks.length > 0) {
                setLeftHand(result.leftHandWorldLandmarks[0])
              } else {
                setLeftHand([])
              }
              if (result.rightHandWorldLandmarks && result.rightHandWorldLandmarks.length > 0) {
                setRightHand(result.rightHandWorldLandmarks[0])
              } else {
                setRightHand([])
              }
            })
          }
          requestAnimationFrame(detect)
        }
        detect()
      }
    )
  }, [setPose, setFace, setLeftHand, setRightHand, imgRef, videoRef, isRecordingRef])

  const replayCallback = (fps: number) => {
    setIsReplaying(true)
    if (videoRef.current) {
      videoRef.current.controls = false
    }
    let currentIndex = 0
    const frameInterval = 1000 / fps

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

        if (result.leftHandWorldLandmarks && result.leftHandWorldLandmarks.length > 0) {
          setLeftHand(result.leftHandWorldLandmarks[0])
        } else {
          setLeftHand([])
        }

        if (result.rightHandWorldLandmarks && result.rightHandWorldLandmarks.length > 0) {
          setRightHand(result.rightHandWorldLandmarks[0])
        } else {
          setRightHand([])
        }

        currentIndex++
        setTimeout(() => requestAnimationFrame(playNextFrame), frameInterval)
      } else {
        setIsReplaying(false)
        if (videoRef.current) {
          videoRef.current.controls = true
        }
      }
    }

    requestAnimationFrame(playNextFrame)
  }

  return (
    <div className="motion" style={style}>
      <div className="toolbar">
        <Tooltip title="Upload a video or image">
          <IconButton
            className="toolbar-item"
            color="primary"
            component="label"
            disabled={isCameraActive || isOfflineProcessingRef.current}
          >
            <CloudUpload />
            <VisuallyHiddenInput type="file" onChange={handleFileUpload} accept="video/*, image/*" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Use your camera">
          <IconButton
            className="toolbar-item"
            onClick={toggleCamera}
            color={isCameraActive ? "error" : "success"}
            disabled={isOfflineProcessingRef.current}
          >
            <Videocam />
          </IconButton>
        </Tooltip>
        <Tooltip title={isOfflineProcessingRef.current ? "Stop processing" : "Process video offline at 60 FPS"}>
          <IconButton
            className="toolbar-item"
            onClick={toggleProcessCurrentVideoOffline}
            color="error"
            disabled={isCameraActive}
          >
            {!isOfflineProcessingRef.current ? (
              <LocalFireDepartment />
            ) : (
              <p style={{ fontSize: ".7rem" }}>{Math.round(offlineProcessingProgressRef.current * 100) + "%"}</p>
            )}
          </IconButton>
        </Tooltip>

        <Tooltip title={isRecording ? "Stop recording" : "Record motion capture"}>
          <IconButton
            className="toolbar-item"
            onClick={toggleRecording}
            color="secondary"
            size="small"
            disabled={isOfflineProcessingRef.current}
          >
            {isRecording ? (
              <>
                <StyledBadge badgeContent={landmarkHistoryRef.current.length} color="secondary" max={999}>
                  <StopCircle />
                </StyledBadge>
              </>
            ) : (
              <>
                <StyledBadge badgeContent={landmarkHistoryRef.current.length} color="secondary" max={999}>
                  <RadioButtonChecked />
                </StyledBadge>
              </>
            )}
          </IconButton>
        </Tooltip>
        <Tooltip title="Replay last capture">
          <IconButton
            className="toolbar-item"
            onClick={() => replayCallback(60)}
            color="secondary"
            size="small"
            disabled={isCameraActive || isReplaying || isOfflineProcessingRef.current}
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

      {style.display == "block" && <DebugScene pose={pose} leftHand={leftHand} rightHand={rightHand} />}
    </div>
  )
}

export default Video
