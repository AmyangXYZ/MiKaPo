import { useEffect, useRef, useState } from "react"
import {
  ArcRotateCamera,
  BackgroundMaterial,
  Color3,
  Color4,
  CreateScreenshotAsync,
  DirectionalLight,
  Engine,
  HemisphericLight,
  Material,
  Mesh,
  MeshBuilder,
  PhotoDome,
  Quaternion,
  registerSceneLoaderPlugin,
  Scene,
  SceneLoader,
  ShadowGenerator,
  Space,
  Texture,
  Vector3,
  Viewport,
} from "@babylonjs/core"
import {
  getMmdWasmInstance,
  MmdWasmAnimation,
  MmdWasmInstance,
  MmdWasmInstanceTypeMPD,
  MmdWasmModel,
  MmdWasmPhysics,
  MmdWasmRuntime,
  PmxLoader,
  SdefInjector,
  VmdLoader,
} from "babylon-mmd"
import backgroundGroundUrl from "./assets/backgroundGround.png"
import type { IMmdRuntimeLinkedBone } from "babylon-mmd/esm/Runtime/IMmdRuntimeLinkedBone"

import "@babylonjs/core/Engines/shaderStore"
import { Badge, BadgeProps, IconButton, styled, Tooltip } from "@mui/material"
import { BorderAll, Camera, CenterFocusWeak, RadioButtonChecked, StopCircle } from "@mui/icons-material"
import Encoding from "encoding-japanese"
import { BoneFrame, MorphFrame, RecordedFrame, Body } from "."

import init, { PoseSolver, PoseSolverResult, Rotation } from "pose_solver"

registerSceneLoaderPlugin(new PmxLoader())

ArcRotateCamera.prototype.spinTo = function (
  this: ArcRotateCamera,
  targetPosition: Vector3,
  targetTarget: Vector3 = new Vector3(0, 12, 0),
  duration: number = 1000
): void {
  const startPosition = this.position.clone()
  const startTarget = this.target.clone()
  const startTime = performance.now()

  const smoothStep = (x: number): number => {
    return x * x * (3 - 2 * x)
  }

  const animate = (currentTime: number) => {
    const elapsedTime = currentTime - startTime
    const progress = Math.min(elapsedTime / duration, 1)

    const easedProgress = smoothStep(progress)

    const newPosition = Vector3.Lerp(startPosition, targetPosition, easedProgress)
    const newTarget = Vector3.Lerp(startTarget, targetTarget, easedProgress)

    this.position = newPosition
    this.setTarget(newTarget)

    if (progress < 1) {
      requestAnimationFrame(animate)
    }
  }

  requestAnimationFrame(animate)
}

// Declare the spinTo method on the ArcRotateCamera interface
declare module "@babylonjs/core/Cameras/arcRotateCamera" {
  interface ArcRotateCamera {
    spinTo(targetPosition: Vector3, targetTarget?: Vector3, duration?: number): void
  }
}

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

const usedKeyBones: string[] = [
  "センター",
  "首",
  "頭",
  "上半身",
  "下半身",
  "左足",
  "右足",
  "左ひざ",
  "右ひざ",
  "左足首",
  "右足首",
  "左腕",
  "右腕",
  "左ひじ",
  "右ひじ",
  "左手首",
  "右手首",
  "左足ＩＫ",
  "右足ＩＫ",
  "左目",
  "右目",
  "右親指１",
  "右親指２",
  "右人指１",
  "右人指２",
  "右人指３",
  "右中指１",
  "右中指２",
  "右中指３",
  "右薬指１",
  "右薬指２",
  "右薬指３",
  "右小指１",
  "右小指２",
  "右小指３",
  "左親指１",
  "左親指２",
  "左人指１",
  "左人指２",
  "左人指３",
  "左中指１",
  "左中指２",
  "左中指３",
  "左薬指１",
  "左薬指２",
  "左薬指３",
  "左小指１",
  "左小指２",
  "左小指３",
]

function MMDScene({
  body,
  lerpFactor,
  setFps,
  selectedModel,
  selectedBackground,
  selectedAnimation,
  setSelectedAnimation,
  boneRotation,
  setMaterials,
  materialVisible,
  isPlaying,
  setIsPlaying,
  setCurrentAnimationTime,
  animationSeekTime,
  setAnimationDuration,
}: {
  body: Body
  lerpFactor: number
  setFps: (fps: number) => void
  selectedModel: string
  selectedBackground: string
  selectedAnimation: string
  setSelectedAnimation: (animation: string) => void
  boneRotation: { name: string; axis: string; value: number } | null
  setMaterials: (materials: string[]) => void
  materialVisible: { name: string; visible: boolean } | null
  isPlaying: boolean
  setIsPlaying: (isPlaying: boolean) => void
  setCurrentAnimationTime: (time: number) => void
  setAnimationDuration: (duration: number) => void
  animationSeekTime: number
}): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<Scene | null>(null)
  const engineRef = useRef<Engine | null>(null)
  const cameraRef = useRef<ArcRotateCamera | null>(null)
  const [sceneRendered, setSceneRendered] = useState<boolean>(false)
  const shadowGeneratorRef = useRef<ShadowGenerator | null>(null)
  const domeRef = useRef<PhotoDome | null>(null)
  const groundRef = useRef<Mesh | null>(null)

  const mmdWasmInstanceRef = useRef<MmdWasmInstance | null>(null)
  const mmdModelRef = useRef<MmdWasmModel | null>(null)
  const mmdRuntimeRef = useRef<MmdWasmRuntime | null>(null)
  const keyBones = useRef<{ [key: string]: IMmdRuntimeLinkedBone }>({})

  const poseSolverRef = useRef<PoseSolver | null>(null)

  const [enableSplitView, setEnableSplitView] = useState<boolean>(false)

  const [isRecordingVMD, setIsRecordingVMD] = useState<boolean>(false)
  const isRecordingRef = useRef<boolean>(false)
  const recordedFramesRef = useRef<RecordedFrame[]>([])

  const getBone = (name: string): IMmdRuntimeLinkedBone | null => {
    return keyBones.current[name]
  }

  useEffect(() => {
    init().then(() => {
      poseSolverRef.current = new PoseSolver()
    })
  }, [])
  useEffect(() => {
    if (mmdModelRef.current && boneRotation) {
      if (mmdRuntimeRef.current && mmdRuntimeRef.current.isAnimationPlaying) {
        mmdRuntimeRef.current.pauseAnimation()
        mmdModelRef.current.removeAnimation(0)
      }
      const bone = getBone(boneRotation.name)
      if (bone) {
        bone.setRotationQuaternion(
          Quaternion.FromEulerAngles(
            boneRotation.axis === "x" ? boneRotation.value : bone.rotationQuaternion!.toEulerAngles().x,
            boneRotation.axis === "y" ? boneRotation.value : bone.rotationQuaternion!.toEulerAngles().y,
            boneRotation.axis === "z" ? boneRotation.value : bone.rotationQuaternion!.toEulerAngles().z
          ),
          Space.LOCAL
        )
      }
    }
  }, [boneRotation])

  useEffect(() => {
    if (materialVisible) {
      const material = mmdModelRef.current!.mesh.metadata.materials.find(
        (m: Material) => m.name === materialVisible.name
      )
      const mesh = mmdModelRef.current!.mesh.metadata.meshes.find((m: Mesh) => m.name === materialVisible.name)
      if (material && mesh) {
        material.alpha = materialVisible.visible ? 1 : 0
        mesh.visibility = materialVisible.visible ? 1 : 0
      }
    }
  }, [materialVisible])

  useEffect(() => {
    if (mmdRuntimeRef.current) {
      if (isPlaying) {
        mmdRuntimeRef.current.playAnimation()
      } else {
        mmdRuntimeRef.current.pauseAnimation()
      }
    }
  }, [isPlaying])

  useEffect(() => {
    if (mmdRuntimeRef.current) {
      mmdRuntimeRef.current.seekAnimation(animationSeekTime * 30, true)
    }
  }, [animationSeekTime])

  useEffect(() => {
    if (sceneRef.current) {
      if (enableSplitView) {
        const topCamera = new ArcRotateCamera(
          "TopCamera",
          -Math.PI / 2,
          -Math.PI,
          30,
          new Vector3(0, 10, 0),
          sceneRef.current!
        )

        const sideCamera = new ArcRotateCamera(
          "SideCamera",
          Math.PI,
          Math.PI / 2,
          30,
          new Vector3(0, 10, 0),
          sceneRef.current!
        )
        const frontCamera = new ArcRotateCamera(
          "FrontCamera",
          -Math.PI / 2,
          Math.PI / 2,
          30,
          new Vector3(0, 10, 0),
          sceneRef.current!
        )

        topCamera.inputs.clear()
        topCamera.inputs.addMouseWheel()
        topCamera.attachControl(canvasRef.current!, false)

        sideCamera.inputs.clear()
        sideCamera.inputs.addMouseWheel()
        sideCamera.attachControl(canvasRef.current!, false)

        frontCamera.inputs.clear()
        frontCamera.inputs.addMouseWheel()
        frontCamera.attachControl(canvasRef.current!, false)

        // Set camera viewports
        cameraRef.current!.viewport = new Viewport(0, 0.5, 0.5, 0.5)
        topCamera.viewport = new Viewport(0.5, 0.5, 0.5, 0.5)
        sideCamera.viewport = new Viewport(0, 0, 0.5, 0.5)
        frontCamera.viewport = new Viewport(0.5, 0, 0.5, 0.5)

        sceneRef.current.activeCameras = [cameraRef.current!, topCamera, sideCamera, frontCamera]
      } else {
        cameraRef.current!.viewport = new Viewport(0, 0, 1, 1)
        sceneRef.current.activeCameras = [cameraRef.current!]
      }
    }
  }, [enableSplitView])

  useEffect(() => {
    const createScene = async (canvas: HTMLCanvasElement): Promise<Scene> => {
      const engine = new Engine(
        canvas,
        true,
        {
          preserveDrawingBuffer: false,
          stencil: false,
          antialias: false,
          alpha: true,
          premultipliedAlpha: false,
          powerPreference: "high-performance",
          doNotHandleTouchAction: false,
          doNotHandleContextLost: true,
          audioEngine: false,
        },
        true
      )
      engineRef.current = engine
      SdefInjector.OverrideEngineCreateEffect(engine)
      const scene = new Scene(engine)
      scene.clearColor = new Color4(0, 0, 0, 0)
      mmdWasmInstanceRef.current = await getMmdWasmInstance(new MmdWasmInstanceTypeMPD(), 2)
      mmdRuntimeRef.current = new MmdWasmRuntime(mmdWasmInstanceRef.current, scene, new MmdWasmPhysics(scene))
      mmdRuntimeRef.current.register(scene)
      mmdRuntimeRef.current!.onAnimationTickObservable.add(() => {
        setCurrentAnimationTime(mmdRuntimeRef.current!.currentTime)
      })
      mmdRuntimeRef.current!.onPauseAnimationObservable.add(() => {
        if (mmdRuntimeRef.current!.currentTime == mmdRuntimeRef.current!.animationDuration) {
          mmdRuntimeRef.current!.seekAnimation(0, true)
          mmdRuntimeRef.current!.playAnimation()
        }
      })

      const camera = new ArcRotateCamera("ArcRotateCamera", 0, 0, 45, new Vector3(0, 12, 0), scene)
      camera.setPosition(new Vector3(0, 19, -25))
      camera.attachControl(canvas, false)
      camera.inertia = 0.8
      camera.speed = 10
      cameraRef.current = camera
      scene.activeCameras = [camera]

      const hemisphericLight = new HemisphericLight("HemisphericLight", new Vector3(0, 1, 0), scene)
      hemisphericLight.intensity = 0.3
      hemisphericLight.specular = new Color3(0, 0, 0)
      hemisphericLight.groundColor = new Color3(1, 1, 1)

      const directionalLight = new DirectionalLight("DirectionalLight", new Vector3(8, -15, 10), scene)
      directionalLight.intensity = 0.7

      shadowGeneratorRef.current = new ShadowGenerator(4096, directionalLight, true)
      shadowGeneratorRef.current.usePercentageCloserFiltering = true
      shadowGeneratorRef.current.forceBackFacesOnly = true
      shadowGeneratorRef.current.filteringQuality = ShadowGenerator.QUALITY_HIGH
      shadowGeneratorRef.current.frustumEdgeFalloff = 0.1
      shadowGeneratorRef.current.transparencyShadow = true

      const backgroundMaterial = new BackgroundMaterial("backgroundMaterial", scene)
      backgroundMaterial.diffuseTexture = new Texture(backgroundGroundUrl, scene)
      backgroundMaterial.diffuseTexture.hasAlpha = true
      backgroundMaterial.opacityFresnel = false
      backgroundMaterial.shadowLevel = 0.4
      backgroundMaterial.useRGBColor = false
      backgroundMaterial.primaryColor = Color3.Magenta()

      groundRef.current = MeshBuilder.CreateGround(
        "Ground11",
        {
          width: 48,
          height: 48,
          subdivisions: 2,
          updatable: false,
        },
        scene
      )
      groundRef.current!.material = backgroundMaterial
      groundRef.current!.receiveShadows = true

      engine.runRenderLoop(() => {
        setFps(Math.round(engine.getFps()))
        engine.resize()
        scene!.render()
      })
      return scene
    }

    if (canvasRef.current) {
      createScene(canvasRef.current).then((scene) => {
        sceneRef.current = scene
        setSceneRendered(true)
      })
    }
  }, [setFps, setSceneRendered, setCurrentAnimationTime])

  useEffect(() => {
    if (domeRef.current) {
      domeRef.current.dispose()
    }

    if (sceneRef.current) {
      if (selectedBackground !== "Static") {
        if (groundRef.current) {
          groundRef.current.material!.alpha = 0
        }
        domeRef.current = new PhotoDome(
          "testdome",
          `/background/${selectedBackground}.jpeg`,
          {
            resolution: 32,
            size: 500,
          },
          sceneRef.current
        )
        domeRef.current!.rotation.y = Math.PI / 2
      } else {
        if (groundRef.current) {
          groundRef.current.material!.alpha = 1
        }
      }
    }
  }, [sceneRendered, sceneRef, selectedBackground])

  useEffect(() => {
    const loadMMD = async (): Promise<void> => {
      if (!sceneRendered || !selectedModel || !mmdWasmInstanceRef.current || !mmdRuntimeRef.current) return
      if (mmdModelRef.current) {
        mmdRuntimeRef.current.destroyMmdModel(mmdModelRef.current)
        mmdModelRef.current.mesh.dispose()
      }
      setSelectedAnimation("")
      SceneLoader.ImportMeshAsync(undefined, `/model/${selectedModel}/`, `${selectedModel}.pmx`, sceneRef.current).then(
        (result) => {
          const mesh = result.meshes[0]
          for (const m of mesh.metadata.meshes) {
            m.receiveShadows = true
          }
          setMaterials(mesh.metadata.materials.map((m: Material) => m.name))
          shadowGeneratorRef.current!.addShadowCaster(mesh)
          mmdModelRef.current = mmdRuntimeRef.current!.createMmdModel(mesh as Mesh, {
            buildPhysics: {
              worldId: 0,
            },
          })

          for (const bone of mmdModelRef.current!.skeleton.bones) {
            if (usedKeyBones.includes(bone.name)) {
              keyBones.current[bone.name] = bone
            }
          }
        }
      )
    }
    loadMMD()
  }, [sceneRendered, sceneRef, mmdWasmInstanceRef, mmdRuntimeRef, selectedModel, setSelectedAnimation, setMaterials])

  useEffect(() => {
    if (
      !sceneRef.current ||
      !mmdWasmInstanceRef.current ||
      !mmdRuntimeRef.current ||
      !mmdModelRef.current ||
      selectedAnimation === ""
    )
      return

    const loadAnimation = async (): Promise<void> => {
      const vmd = await new VmdLoader(sceneRef.current!).loadAsync("vmd", selectedAnimation)
      const animation = new MmdWasmAnimation(vmd, mmdWasmInstanceRef.current!, sceneRef.current!)
      mmdModelRef.current?.addAnimation(animation)
      mmdModelRef.current?.setAnimation("vmd")
      mmdRuntimeRef.current!.seekAnimation(0, true)
      mmdRuntimeRef.current!.playAnimation()
      setIsPlaying(true)
      setAnimationDuration(mmdRuntimeRef.current!.animationDuration)
    }
    loadAnimation()
  }, [sceneRef, mmdWasmInstanceRef, mmdRuntimeRef, mmdModelRef, selectedAnimation, setAnimationDuration, setIsPlaying])

  useEffect(() => {
    if (mmdRuntimeRef.current && mmdRuntimeRef.current.isAnimationPlaying) {
      mmdRuntimeRef.current.pauseAnimation()
      mmdModelRef.current!.removeAnimation(0)
    }
    const setBoneRotation = (bone: IMmdRuntimeLinkedBone | null, rotation: Rotation): void => {
      if (!bone) return
      bone.setRotationQuaternion(
        Quaternion.Slerp(
          bone.rotationQuaternion || new Quaternion(),
          new Quaternion(rotation.x, rotation.y, rotation.z, rotation.w),
          lerpFactor
        ),
        Space.LOCAL
      )
    }
    const updateMMDPose = (mmdModel: MmdWasmModel | null, body: Body): void => {
      if (!mmdModel || !body || !body.mainBody || !poseSolverRef.current) {
        return
      }

      const result: PoseSolverResult = poseSolverRef.current.solve(
        body.mainBody,
        body.leftHand || [],
        body.rightHand || [],
        body.face || []
      )

      setBoneRotation(getBone("上半身"), result.upper_body)
      setBoneRotation(getBone("下半身"), result.lower_body)
      setBoneRotation(getBone("首"), result.neck)
      setBoneRotation(getBone("左腕"), result.left_upper_arm)
      setBoneRotation(getBone("左ひじ"), result.left_lower_arm)
      setBoneRotation(getBone("右腕"), result.right_upper_arm)
      setBoneRotation(getBone("右ひじ"), result.right_lower_arm)
      setBoneRotation(getBone("左足"), result.left_hip)
      setBoneRotation(getBone("右足"), result.right_hip)
      setBoneRotation(getBone("左足首"), result.left_foot)
      setBoneRotation(getBone("右足首"), result.right_foot)
      setBoneRotation(getBone("左手首"), result.left_wrist)
      setBoneRotation(getBone("右手首"), result.right_wrist)

      // setBoneRotation(getBone("左親指１"), result.left_thumb_cmc)
      // setBoneRotation(getBone("左親指２"), result.left_thumb_mcp)
      // setBoneRotation(getBone("左人指１"), result.left_index_finger_mcp)
      // setBoneRotation(getBone("左人指２"), result.left_index_finger_pip)
      // setBoneRotation(getBone("左人指３"), result.left_index_finger_dip)
      // setBoneRotation(getBone("左中指１"), result.left_middle_finger_mcp)
      // setBoneRotation(getBone("左中指２"), result.left_middle_finger_pip)
      // setBoneRotation(getBone("左中指３"), result.left_middle_finger_dip)
      // setBoneRotation(getBone("左薬指１"), result.left_ring_finger_mcp)
      // setBoneRotation(getBone("左薬指２"), result.left_ring_finger_pip)
      // setBoneRotation(getBone("左薬指３"), result.left_ring_finger_dip)
      // setBoneRotation(getBone("左小指１"), result.left_pinky_finger_mcp)
      // setBoneRotation(getBone("左小指２"), result.left_pinky_finger_pip)
      // setBoneRotation(getBone("左小指３"), result.left_pinky_finger_dip)

      // setBoneRotation(getBone("右親指１"), result.right_thumb_cmc)
      // setBoneRotation(getBone("右親指２"), result.right_thumb_mcp)
      // setBoneRotation(getBone("右人指１"), result.right_index_finger_mcp)
      // setBoneRotation(getBone("右人指２"), result.right_index_finger_pip)
      // setBoneRotation(getBone("右人指３"), result.right_index_finger_dip)
      // setBoneRotation(getBone("右中指１"), result.right_middle_finger_mcp)
      // setBoneRotation(getBone("右中指２"), result.right_middle_finger_pip)
      // setBoneRotation(getBone("右中指３"), result.right_middle_finger_dip)
      // setBoneRotation(getBone("右薬指１"), result.right_ring_finger_mcp)
      // setBoneRotation(getBone("右薬指２"), result.right_ring_finger_pip)
      // setBoneRotation(getBone("右薬指３"), result.right_ring_finger_dip)
      // setBoneRotation(getBone("右小指１"), result.right_pinky_finger_mcp)
      // setBoneRotation(getBone("右小指２"), result.right_pinky_finger_pip)
      // setBoneRotation(getBone("右小指３"), result.right_pinky_finger_dip)

      setBoneRotation(getBone("左目"), result.left_eye_rotation)
      setBoneRotation(getBone("右目"), result.right_eye_rotation)

      const morphTargets = {
        まばたき: Math.max(Math.pow(1 - result.right_eye_openness, 1.5), Math.pow(1 - result.left_eye_openness, 1.5)),
        あ: Math.pow(result.mouth_openness, 1.5),
        お: Math.max(0, result.mouth_openness - 0.3) * 1.5,
      }

      // Apply morph targets with smoothing
      const smoothingFactor = 0.7
      for (const [morphName, targetValue] of Object.entries(morphTargets)) {
        const currentValue = mmdModel.morph.getMorphWeight(morphName)
        const newValue = currentValue + (targetValue - currentValue) * smoothingFactor
        mmdModel.morph.setMorphWeight(morphName, Math.max(0, Math.min(1, newValue)))
      }

      getBone("左足ＩＫ")!.position = new Vector3(
        body.mainBody![27].x * 10,
        -body.mainBody![27].y * 10 + 7,
        body.mainBody![27].z * 10
      )
      getBone("右足ＩＫ")!.position = new Vector3(
        body.mainBody![28].x * 10,
        -body.mainBody![28].y * 10 + 7,
        body.mainBody![28].z * 10
      )
    }
    if (sceneRef.current && mmdModelRef.current) {
      updateMMDPose(mmdModelRef.current, body)
    }
  }, [body, lerpFactor])

  const handleCaptureScreenshot = () => {
    const button = document.querySelector(".screenshot-button")
    if (button) {
      button.classList.add("animate")

      setTimeout(() => {
        button.classList.remove("animate")
      }, 600)
    }

    CreateScreenshotAsync(engineRef.current!, cameraRef.current!, { precision: 1 }).then((b64) => {
      const link = document.createElement("a")
      link.href = b64
      link.download = "mikapo_screenshot.png"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    })
  }

  useEffect(() => {
    if (isRecordingVMD) {
      recordedFramesRef.current = []
      let lastRecordTime = performance.now()
      const targetInterval = 1000 / 30

      const recordFrame = () => {
        if (!isRecordingRef.current) return
        const currentTime = performance.now()
        const elapsedTime = currentTime - lastRecordTime

        const currentFrames: RecordedFrame = {
          boneFrames: [],
          morphFrames: [],
        }
        if (elapsedTime >= targetInterval) {
          for (const boneName in keyBones.current) {
            const boneFrame: BoneFrame = {
              name: boneName,
              position: new Vector3(0, 0, 0),
              rotation: keyBones.current[boneName].rotationQuaternion.clone(),
            }
            currentFrames.boneFrames.push(boneFrame)
          }

          if (mmdModelRef.current) {
            const morphs = mmdModelRef.current.morph.morphs
            for (const morph of morphs) {
              const morphFrame: MorphFrame = {
                name: morph.name,
                weight: mmdModelRef.current.morph.getMorphWeight(morph.name),
              }
              currentFrames.morphFrames.push(morphFrame)
            }
          }

          recordedFramesRef.current.push(currentFrames)
          lastRecordTime = currentTime - (elapsedTime % targetInterval)
        }

        if (isRecordingVMD) {
          requestAnimationFrame(recordFrame)
        }
      }

      requestAnimationFrame(recordFrame)
    }
  }, [isRecordingVMD])

  function createVMD(frames: RecordedFrame[]): Blob {
    if (frames.length === 0) {
      return new Blob()
    }

    function encodeShiftJIS(str: string): Uint8Array {
      const unicodeArray = Encoding.stringToCode(str)
      const sjisArray = Encoding.convert(unicodeArray, {
        to: "SJIS",
        from: "UNICODE",
      })
      return new Uint8Array(sjisArray)
    }

    const writeBoneFrame = (
      dataView: DataView,
      offset: number,
      name: string,
      frame: number,
      position: Vector3,
      rotation: Quaternion
    ): number => {
      const nameBytes = encodeShiftJIS(name)
      for (let i = 0; i < 15; i++) {
        dataView.setUint8(offset + i, i < nameBytes.length ? nameBytes[i] : 0)
      }
      offset += 15

      dataView.setUint32(offset, frame, true)
      offset += 4

      dataView.setFloat32(offset, position.x, true)
      offset += 4
      dataView.setFloat32(offset, position.y, true)
      offset += 4
      dataView.setFloat32(offset, position.z, true)
      offset += 4

      dataView.setFloat32(offset, rotation.x, true)
      offset += 4
      dataView.setFloat32(offset, rotation.y, true)
      offset += 4
      dataView.setFloat32(offset, rotation.z, true)
      offset += 4
      dataView.setFloat32(offset, rotation.w, true)
      offset += 4

      // Interpolation parameters (64 bytes)
      for (let i = 0; i < 64; i++) {
        dataView.setUint8(offset + i, 20)
      }
      offset += 64

      return offset
    }

    const writeMorphFrame = (
      dataView: DataView,
      offset: number,
      name: string,
      frame: number,
      weight: number
    ): number => {
      const nameBytes = encodeShiftJIS(name)
      for (let i = 0; i < 15; i++) {
        dataView.setUint8(offset + i, i < nameBytes.length ? nameBytes[i] : 0)
      }
      offset += 15

      dataView.setUint32(offset, frame, true)
      offset += 4

      dataView.setFloat32(offset, weight, true)
      offset += 4

      return offset
    }

    const frameCount = frames.length
    const boneCnt = frames[0].boneFrames.length
    const morphCnt = frames[0].morphFrames.length
    const headerSize = 30 + 20
    const boneFrameSize = 15 + 4 + 12 + 16 + 64
    const morphFrameSize = 15 + 4 + 4
    const totalSize =
      headerSize + 4 + boneFrameSize * frameCount * boneCnt + 4 + morphFrameSize * frameCount * morphCnt + 4 + 4 + 4

    const buffer = new ArrayBuffer(totalSize)
    const dataView = new DataView(buffer)
    let offset = 0

    // Write header
    const header = "Vocaloid Motion Data 0002"
    for (let i = 0; i < 30; i++) {
      dataView.setUint8(offset + i, i < header.length ? header.charCodeAt(i) : 0)
    }
    offset += 30

    // Write model name (empty in this case)
    offset += 20

    // Write bone frame count
    dataView.setUint32(offset, frameCount * boneCnt, true)
    offset += 4

    // Generate bone keyframes
    for (let i = 0; i < frameCount; i++) {
      for (const boneFrame of frames[i].boneFrames) {
        offset = writeBoneFrame(dataView, offset, boneFrame.name, i, boneFrame.position, boneFrame.rotation)
      }
    }

    // Write morph frame count
    dataView.setUint32(offset, frameCount * morphCnt, true)
    offset += 4

    // Generate morph keyframes
    for (let i = 0; i < frameCount; i++) {
      for (const morphFrame of frames[i].morphFrames) {
        offset = writeMorphFrame(dataView, offset, morphFrame.name, i, morphFrame.weight)
      }
    }

    // Write counts for other frame types (all 0 in this example)
    dataView.setUint32(offset, 0, true) // Camera keyframe count
    offset += 4
    dataView.setUint32(offset, 0, true) // Light keyframe count
    offset += 4
    dataView.setUint32(offset, 0, true) // Self shadow keyframe count
    offset += 4

    return new Blob([buffer], { type: "application/octet-stream" })
  }

  const handleCreateVMD = () => {
    const vmdBlob = createVMD(recordedFramesRef.current)
    const url = URL.createObjectURL(vmdBlob)
    const link = document.createElement("a")
    link.href = url
    link.download = "mikapo_animation.vmd"
    link.click()
    URL.revokeObjectURL(url)
    setIsRecordingVMD(false)
    isRecordingRef.current = false
  }

  return (
    <>
      <canvas ref={canvasRef} className="scene"></canvas>
      <Tooltip title="Reset camera">
        <IconButton
          style={{ position: "absolute", top: "8rem", right: ".5rem", color: "#f209f5" }}
          onClick={() => {
            cameraRef.current!.spinTo(new Vector3(0, 19, -25), new Vector3(0, 12, 0), 1000)
          }}
        >
          <CenterFocusWeak sx={{ width: "26px", height: "26px" }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Split view">
        <IconButton
          style={{ position: "absolute", top: "10rem", right: ".5rem", color: "#f209f5" }}
          onClick={() => setEnableSplitView(!enableSplitView)}
        >
          <BorderAll sx={{ width: "26px", height: "26px", color: enableSplitView ? "cyan" : "#f209f5" }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Capture screenshot">
        <IconButton
          style={{ position: "absolute", top: "12rem", right: ".5rem", color: "#f209f5" }}
          onClick={handleCaptureScreenshot}
          className="screenshot-button"
        >
          <Camera sx={{ width: "26px", height: "26px" }} />
        </IconButton>
      </Tooltip>

      {isRecordingVMD ? (
        <Tooltip title="Stop recording">
          <IconButton
            style={{ position: "absolute", top: "14rem", right: ".5rem", color: "#f209f5" }}
            onClick={handleCreateVMD}
          >
            <StyledBadge badgeContent={recordedFramesRef.current.length} color="secondary" max={999}>
              <StopCircle sx={{ width: "26px", height: "26px", color: "red" }} />
            </StyledBadge>
          </IconButton>
        </Tooltip>
      ) : (
        <Tooltip title="Record animation for VMD export">
          <IconButton
            style={{ position: "absolute", top: "14rem", right: ".5rem", color: "#f209f5" }}
            onClick={() => {
              isRecordingRef.current = true
              setIsRecordingVMD(true)
            }}
          >
            <RadioButtonChecked sx={{ width: "26px", height: "26px" }} />
          </IconButton>
        </Tooltip>
      )}
    </>
  )
}

export default MMDScene
