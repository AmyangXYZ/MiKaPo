import { useEffect, useRef, useState } from "react"

import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision"
import { IconButton, Tooltip } from "@mui/material"
import { Videocam, CloudUpload, Stop } from "@mui/icons-material"
import { styled } from "@mui/material/styles"
import DebugScene from "./DebugScene"
import { Body } from "./index"
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
  body,
  setBody,
  setLerpFactor,
  style,
}: {
  body: Body
  setBody: (body: Body) => void
  setLerpFactor: (lerpFactor: number) => void
  style: React.CSSProperties
}): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [videoSrc, setVideoSrc] = useState<string>(defaultVideoSrc)
  const [imgSrc, setImgSrc] = useState<string>("")
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false)
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
              setBody({
                mainBody: result.poseWorldLandmarks[0],
                leftHand: result.leftHandWorldLandmarks[0],
                rightHand: result.rightHandWorldLandmarks[0],
                face: result.faceLandmarks[0],
              })
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
              setBody({
                mainBody: result.poseWorldLandmarks[0],
                leftHand: result.leftHandWorldLandmarks[0],
                rightHand: result.rightHandWorldLandmarks[0],
                face: result.faceLandmarks[0],
              })
            })
          }
          requestAnimationFrame(detect)
        }
        detect()
      }
    )
  }, [setBody, imgRef, videoRef])

  return (
    <div className="motion" style={style}>
      <div className="toolbar">
        <Tooltip title="Upload a video or image">
          <IconButton className="toolbar-item" color="primary" component="label" disabled={isCameraActive}>
            <CloudUpload />
            <VisuallyHiddenInput type="file" onChange={handleFileUpload} accept="video/*, image/*" />
          </IconButton>
        </Tooltip>
        <Tooltip title={isCameraActive ? "Stop webcam" : "Use your webcam"}>
          <IconButton className="toolbar-item" onClick={toggleCamera}>
            {isCameraActive ? <Stop sx={{ color: "red" }} /> : <Videocam sx={{ color: "green" }} />}
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

      {style.display == "block" && <DebugScene body={body} />}
    </div>
  )
}

export default Video
