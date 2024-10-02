import { useState } from "react"
import Video from "./Video"
import MMDScene from "./MMDScene"
import { NormalizedLandmark } from "@mediapipe/tasks-vision"
import { Avatar, IconButton } from "@mui/material"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faGithub } from "@fortawesome/free-brands-svg-icons"
import { Download } from "@mui/icons-material"

function App(): JSX.Element {
  const [pose, setPose] = useState<NormalizedLandmark[] | null>(null)
  const [face, setFace] = useState<NormalizedLandmark[] | null>(null)
  const [leftHand, setLeftHand] = useState<NormalizedLandmark[] | null>(null)
  const [rightHand, setRightHand] = useState<NormalizedLandmark[] | null>(null)
  const [lerpFactor, setLerpFactor] = useState<number>(0.5)
  const [fps, setFps] = useState<number>(0)
  const [isInitializedAI, setIsInitializedAI] = useState<boolean>(false)
  const [isInitializedMMD, setIsInitializedMMD] = useState<boolean>(false)
  return (
    <>
      {(!isInitializedAI || !isInitializedMMD) && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <h3>Initializing AI and MMD...</h3>
        </div>
      )}
      <header className="header">
        <div className="header-item" style={{ justifyContent: "flex-start" }}>
          <Avatar
            alt="MiKaPo"
            src="/logo.png"
            sx={{
              width: 36,
              height: 36,
              marginRight: ".5rem",
              transition: "transform 2s ease-in-out",
              "&:hover": {
                transform: "rotate(360deg)",
              },
            }}
          />
          <h2>MiKaPo </h2>
        </div>

        <div className="header-item" style={{ marginTop: "-.7rem" }}>
          <p>FPS: {fps}</p>
        </div>
        <div className="header-item" style={{ justifyContent: "flex-end" }}>
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
          <a href="https://www.buymeacoffee.com/amyang" target="_blank">
            <img src="/coffee.png" alt="Buy Me A Coffee" width={140} height={34} />
          </a>
        </div>
      </header>
      <Video
        setIsInitializedAI={setIsInitializedAI}
        setPose={setPose}
        setFace={setFace}
        setLeftHand={setLeftHand}
        setRightHand={setRightHand}
        setLerpFactor={setLerpFactor}
      ></Video>
      <MMDScene
        setIsInitializedMMD={setIsInitializedMMD}
        pose={pose}
        face={face}
        leftHand={leftHand}
        rightHand={rightHand}
        lerpFactor={lerpFactor}
        setFps={setFps}
      ></MMDScene>
    </>
  )
}

export default App
