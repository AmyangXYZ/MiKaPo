import { useState } from "react"
import Motion from "./Motion"
import MMD from "./MMD"
import Outfit from "./Outfit"
import Model from "./Model"
import Animation from "./Animation"
import Header from "./Header"
import Footer from "./Footer"
import Skeleton from "./Skeleton"
import Background from "./Background"
import { NormalizedLandmark } from "@mediapipe/tasks-vision"
import { Drawer, IconButton } from "@mui/material"
import { KeyboardBackspace } from "@mui/icons-material"

function App(): JSX.Element {
  const [pose, setPose] = useState<NormalizedLandmark[] | null>(null)
  const [face, setFace] = useState<NormalizedLandmark[] | null>(null)
  const [leftHand, setLeftHand] = useState<NormalizedLandmark[] | null>(null)
  const [rightHand, setRightHand] = useState<NormalizedLandmark[] | null>(null)
  const [lerpFactor, setLerpFactor] = useState<number>(0.5)
  const [fps, setFps] = useState<number>(0)
  const [openDrawer, setOpenDrawer] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<string>("motion")

  const [selectedModel, setSelectedModel] = useState<string>("深空之眼-托特")
  const [selectedBackground, setSelectedBackground] = useState<string>("Static")

  return (
    <>
      <Header fps={fps}></Header>

      <MMD
        selectedModel={selectedModel}
        selectedBackground={selectedBackground}
        pose={pose}
        face={face}
        leftHand={leftHand}
        rightHand={rightHand}
        lerpFactor={lerpFactor}
        setFps={setFps}
      ></MMD>
      <Drawer
        variant="persistent"
        open={openDrawer}
        onClose={() => setOpenDrawer(false)}
        sx={{
          [`& .MuiDrawer-paper`]: {
            width: "calc(400px + 2rem)",
            backgroundColor: "rgba(0, 0, 0, 0.8)",
          },
        }}
      >
        <IconButton onClick={() => setOpenDrawer(false)} sx={{ position: "absolute", top: 0, right: ".5rem" }}>
          <KeyboardBackspace sx={{ color: "white" }} />
        </IconButton>

        <Motion
          setPose={setPose}
          setFace={setFace}
          setLeftHand={setLeftHand}
          setRightHand={setRightHand}
          setLerpFactor={setLerpFactor}
          style={{ display: activeTab === "motion" ? "block" : "none" }}
        ></Motion>
        <Outfit style={{ display: activeTab === "outfit" ? "block" : "none" }}></Outfit>
        <Skeleton style={{ display: activeTab === "skeleton" ? "block" : "none" }}></Skeleton>
        <Animation style={{ display: activeTab === "animation" ? "block" : "none" }}></Animation>
        <Model
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          style={{ display: activeTab === "model" ? "block" : "none" }}
        ></Model>
        <Background
          selectedBackground={selectedBackground}
          setSelectedBackground={setSelectedBackground}
          style={{ display: activeTab === "background" ? "block" : "none" }}
        ></Background>
      </Drawer>
      <Footer setOpenDrawer={setOpenDrawer} setActiveTab={setActiveTab}></Footer>
    </>
  )
}

export default App
