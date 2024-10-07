import { useState } from "react"
import Motion from "./Motion"
import MMDScene from "./MMDScene"
import Materials from "./Materials"
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

  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [selectedAnimation, setSelectedAnimation] = useState<string>("")
  const [currentAnimationTime, setCurrentAnimationTime] = useState<number>(0)
  const [animationSeekTime, setAnimationSeekTime] = useState<number>(0)
  const [animationDuration, setAnimationDuration] = useState<number>(0)

  const [boneRotation, setBoneRotation] = useState<{ name: string; axis: string; value: number } | null>(null)
  const [materials, setMaterials] = useState<string[]>([])
  const [materialVisible, setMaterialVisible] = useState<{ name: string; visible: boolean } | null>(null)
  return (
    <>
      <Header fps={fps}></Header>

      <MMDScene
        selectedModel={selectedModel}
        selectedBackground={selectedBackground}
        selectedAnimation={selectedAnimation}
        setSelectedAnimation={setSelectedAnimation}
        pose={pose}
        face={face}
        leftHand={leftHand}
        rightHand={rightHand}
        lerpFactor={lerpFactor}
        setFps={setFps}
        boneRotation={boneRotation}
        setMaterials={setMaterials}
        materialVisible={materialVisible}
        setCurrentAnimationTime={setCurrentAnimationTime}
        setAnimationDuration={setAnimationDuration}
        animationSeekTime={animationSeekTime}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
      ></MMDScene>
      <Drawer
        variant="persistent"
        open={openDrawer}
        onClose={() => setOpenDrawer(false)}
        sx={{
          [`& .MuiDrawer-paper`]: {
            backgroundColor: "rgba(0, 0, 0, 0.6)",
            minWidth: "210px",
          },
        }}
      >
        <IconButton onClick={() => setOpenDrawer(false)} sx={{ position: "absolute", top: 0, right: ".5rem" }}>
          <KeyboardBackspace sx={{ color: "white" }} />
        </IconButton>

        <Motion
          pose={pose}
          face={face}
          leftHand={leftHand}
          rightHand={rightHand}
          setPose={setPose}
          setFace={setFace}
          setLeftHand={setLeftHand}
          setRightHand={setRightHand}
          setLerpFactor={setLerpFactor}
          style={{ display: activeTab === "motion" ? "block" : "none" }}
        ></Motion>
        {activeTab === "material" && (
          <Materials materials={materials} setMaterialVisible={setMaterialVisible}></Materials>
        )}
        {activeTab === "skeleton" && <Skeleton setBoneRotation={setBoneRotation}></Skeleton>}
        {activeTab === "animation" && (
          <Animation
            isPlaying={isPlaying}
            setIsPlaying={setIsPlaying}
            setSelectedAnimation={setSelectedAnimation}
            currentAnimationTime={currentAnimationTime}
            setAnimationSeekTime={setAnimationSeekTime}
            animationDuration={animationDuration}
          ></Animation>
        )}
        {activeTab === "model" && <Model setSelectedModel={setSelectedModel}></Model>}
        {activeTab === "background" && (
          <Background
            selectedBackground={selectedBackground}
            setSelectedBackground={setSelectedBackground}
          ></Background>
        )}
      </Drawer>
      <Footer setOpenDrawer={setOpenDrawer} setActiveTab={setActiveTab}></Footer>
    </>
  )
}

export default App
