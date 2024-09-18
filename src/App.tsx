import { useState } from "react"
import Video from "./Video"
import MMDScene from "./MMDScene"
import { NormalizedLandmark } from "@mediapipe/tasks-vision"
import Header from "./Header"

function App(): JSX.Element {
  const [pose, setPose] = useState<NormalizedLandmark[] | null>(null)
  const [fps, setFps] = useState<number>(0)
  return (
    <>
      {pose === null && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <h3>Initializing AI and MMD...</h3>
        </div>
      )}
      <Header fps={fps}></Header>
      <div className="container">
        <Video setPose={setPose}></Video>
        <MMDScene pose={pose} setFps={setFps}></MMDScene>
      </div>
    </>
  )
}

export default App
