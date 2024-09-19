import { useState } from "react"
import Video from "./Video"
import MMDScene from "./MMDScene"
import { NormalizedLandmark } from "@mediapipe/tasks-vision"

function App(): JSX.Element {
  const [pose, setPose] = useState<NormalizedLandmark[] | null>(null)
  const [face, setFace] = useState<NormalizedLandmark[] | null>(null)
  const [fps, setFps] = useState<number>(0)
  return (
    <>
      {pose === null && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <h3>Initializing AI and MMD...</h3>
        </div>
      )}
      <header className="header">
        <a href="https://github.com/AmyangXYZ/MiKaPo" target="_blank">
          <h2>MiKaPo</h2>
        </a>
        <p>FPS: {fps}</p>
        <a href="https://github.com/AmyangXYZ/MiKaPo-electron" target="_blank">
          <h4>Download</h4>
        </a>
      </header>
      <div className="container">
        <Video setPose={setPose} setFace={setFace}></Video>
        <MMDScene pose={pose} face={face} setFps={setFps}></MMDScene>
      </div>
    </>
  )
}

export default App
