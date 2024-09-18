import { useEffect, useRef } from "react"

import { FilesetResolver, PoseLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision"

function Video({ setPose }: { setPose: (pose: NormalizedLandmark[]) => void }): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
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
      })
      let lastTime = performance.now()
      const detect = () => {
        if (videoRef.current && lastTime != videoRef.current.currentTime) {
          lastTime = videoRef.current.currentTime
          poseLandmarker.detectForVideo(videoRef.current, performance.now(), (result) => {
            setPose(result.landmarks[0])
          })
        }
        requestAnimationFrame(detect)
      }
      detect()
    }
    initPoseDetector()
  }, [setPose])

  return (
    <video ref={videoRef} className="videoPlayer" controls muted>
      <source src="./blue.mp4" type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  )
}

export default Video
