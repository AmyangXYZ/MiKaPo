import { useState } from "react"
import Video from "./Video"
import MMDScene from "./MMDScene"
import { NormalizedLandmark } from "@mediapipe/tasks-vision"

function App(): JSX.Element {
  const [pose, setPose] = useState<NormalizedLandmark[] | null>(null)
  const [fps, setFps] = useState<number>(0)
  return (
    <>
      <a href="https://github.com/AmyangXYZ/MiKaPo" target="_blank">
        MiKaPo
      </a>
      <p>FPS: {fps}</p>
      <Video setPose={setPose}></Video>
      <MMDScene pose={pose} setFps={setFps}></MMDScene>
    </>
  )
}

export default App
