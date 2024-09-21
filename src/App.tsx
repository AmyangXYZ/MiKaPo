import { useState } from "react"
import Video from "./Video"
import MMDScene from "./MMDScene"
import { NormalizedLandmark } from "@mediapipe/tasks-vision"
import { IconButton } from "@mui/material"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faGithub } from "@fortawesome/free-brands-svg-icons"
import { Download } from "@mui/icons-material"

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
        <h3>MiKaPo</h3>

        <p>FPS: {fps}</p>
        <div className="header-item">
          <a href="https://github.com/AmyangXYZ/MiKaPo" target="_blank">
            <IconButton>
              <FontAwesomeIcon icon={faGithub} color="white" size="sm" />
            </IconButton>
          </a>
          <a href="https://github.com/AmyangXYZ/MiKaPo-Electron" target="_blank">
            <IconButton size="small" color="inherit">
              <Download sx={{ color: "white", fontSize: "1.5rem", marginTop: ".2rem" }} />
            </IconButton>
          </a>
        </div>
      </header>
      <Video setPose={setPose} setFace={setFace}></Video>
      <MMDScene pose={pose} face={face} setFps={setFps}></MMDScene>
    </>
  )
}

export default App
