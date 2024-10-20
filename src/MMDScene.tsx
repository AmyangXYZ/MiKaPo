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
import { NormalizedLandmark } from "@mediapipe/tasks-vision"
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
import { BoneFrame, MorphFrame, RecordedFrame } from "."

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

const faceKeypoints: { [key: string]: number } = {
  left_eye_upper: 159,
  left_eye_lower: 145,
  left_eye_left: 33,
  left_eye_right: 133,
  left_eye_iris: 468,
  right_eye_upper: 386,
  right_eye_lower: 374,
  right_eye_left: 362,
  right_eye_right: 263,
  right_eye_iris: 473,
  upper_lip_top: 13,
  lower_lip_bottom: 14,
  mouth_left: 61,
  mouth_right: 291,
  upper_lip_center: 0,
  lower_lip_center: 17,
  left_corner: 291,
  right_corner: 61,
  left_ear: 234,
  right_ear: 454,
}

const getFaceKeypoint = (face: NormalizedLandmark[] | null, name: string): Vector3 | null => {
  if (!face || face.length === 0) return null
  const point = face[faceKeypoints[name]]
  const scaleX = 10 // Adjust these values to fit your MMD model's scale
  const scaleY = 10
  const scaleZ = 5
  return point ? new Vector3(point.x * scaleX, point.y * scaleY, point.z * scaleZ) : null
}

function MMDScene({
  body,
  face,
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
  body: {
    mainBody: NormalizedLandmark[] | null
    leftHand: NormalizedLandmark[] | null
    rightHand: NormalizedLandmark[] | null
  }
  face: NormalizedLandmark[] | null
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
    const updateMMDPose = (
      mmdModel: MmdWasmModel | null,
      body: {
        mainBody: NormalizedLandmark[] | null
        leftHand: NormalizedLandmark[] | null
        rightHand: NormalizedLandmark[] | null
      }
    ): void => {
      if (!mmdModel || !body || !body.mainBody || !poseSolverRef.current) {
        return
      }

      const result: PoseSolverResult = poseSolverRef.current.solve(
        body.mainBody,
        body.leftHand || [],
        body.rightHand || []
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
      // setBoneRotation(getBone("左人指１"), result.left_index_finger_mcp)
      // setBoneRotation(getBone("左人指２"), result.left_index_finger_pip)

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

  useEffect(() => {
    const updateMMDFace = (mmdModel: MmdWasmModel | null, face: NormalizedLandmark[] | null): void => {
      if (!face || !mmdModel || face.length === 0) {
        return
      }

      // Eye landmarks
      const leftEyeUpper = getFaceKeypoint(face, "left_eye_upper")
      const leftEyeLower = getFaceKeypoint(face, "left_eye_lower")
      const leftEyeLeft = getFaceKeypoint(face, "left_eye_left")
      const leftEyeRight = getFaceKeypoint(face, "left_eye_right")
      const leftEyeIris = getFaceKeypoint(face, "left_eye_iris")
      const rightEyeUpper = getFaceKeypoint(face, "right_eye_upper")
      const rightEyeLower = getFaceKeypoint(face, "right_eye_lower")
      const rightEyeLeft = getFaceKeypoint(face, "right_eye_left")
      const rightEyeRight = getFaceKeypoint(face, "right_eye_right")
      const rightEyeIris = getFaceKeypoint(face, "right_eye_iris")

      // Calculate eye openness using relative distance
      const calculateEyeOpenness = (
        upper: Vector3 | null,
        lower: Vector3 | null,
        left: Vector3 | null,
        right: Vector3 | null
      ): number => {
        if (!upper || !lower || !left || !right) return 1
        const eyeHeight = Vector3.Distance(upper, lower)
        const eyeWidth = Vector3.Distance(left, right)
        const aspectRatio = eyeHeight / eyeWidth

        const openRatio = 0.28
        const closedRatio = 0.15

        if (aspectRatio <= closedRatio) return 0 // Fully closed
        if (aspectRatio >= openRatio) return 1 // Fully open

        // Linear mapping between closed and open ratios
        return (aspectRatio - closedRatio) / (openRatio - closedRatio)
      }

      const calculateEyeGaze = (
        eyeLeft: Vector3 | null,
        eyeRight: Vector3 | null,
        iris: Vector3 | null
      ): { x: number; y: number } => {
        if (!eyeLeft || !eyeRight || !iris) return { x: 0, y: 0 }

        const eyeCenter = Vector3.Center(eyeLeft, eyeRight)
        const eyeWidth = Vector3.Distance(eyeLeft, eyeRight)
        const eyeHeight = eyeWidth * 0.5 // Approximate eye height

        const x = (iris.x - eyeCenter.x) / (eyeWidth * 0.5)
        const y = (iris.y - eyeCenter.y) / (eyeHeight * 0.5)

        // Constrain the values to a realistic range
        return {
          x: Math.max(-1, Math.min(1, x)),
          y: Math.max(-0.5, Math.min(0.5, y)), // Vertical range is typically smaller
        }
      }

      const leftEyeOpenness = calculateEyeOpenness(leftEyeUpper, leftEyeLower, leftEyeLeft, leftEyeRight)
      const rightEyeOpenness = calculateEyeOpenness(rightEyeUpper, rightEyeLower, rightEyeLeft, rightEyeRight)

      const leftEyeGaze = calculateEyeGaze(leftEyeLeft, leftEyeRight, leftEyeIris)
      const rightEyeGaze = calculateEyeGaze(rightEyeLeft, rightEyeRight, rightEyeIris)

      // Average gaze direction for both eyes
      const averageGaze = {
        x: (leftEyeGaze.x + rightEyeGaze.x) / 2,
        y: (leftEyeGaze.y + rightEyeGaze.y) / 2,
      }

      const leftEyeBone = getBone("左目")
      const rightEyeBone = getBone("右目")

      if (leftEyeBone && rightEyeBone) {
        const maxHorizontalRotation = Math.PI / 6 // 30 degrees max horizontal rotation
        const maxVerticalRotation = Math.PI / 12 // 15 degrees max vertical rotation

        const xRotation = averageGaze.y * maxVerticalRotation
        const yRotation = -averageGaze.x * maxHorizontalRotation

        const targetQuaternion = Quaternion.RotationYawPitchRoll(yRotation, xRotation, 0)

        leftEyeBone.setRotationQuaternion(
          Quaternion.Slerp(leftEyeBone.rotationQuaternion || new Quaternion(), targetQuaternion, lerpFactor),
          Space.LOCAL
        )
        rightEyeBone.setRotationQuaternion(
          Quaternion.Slerp(rightEyeBone.rotationQuaternion || new Quaternion(), targetQuaternion, lerpFactor),
          Space.LOCAL
        )
      }

      // Mouth landmarks
      const upperLipTop = getFaceKeypoint(face, "upper_lip_top")
      const lowerLipBottom = getFaceKeypoint(face, "lower_lip_bottom")
      const mouthLeft = getFaceKeypoint(face, "mouth_left")
      const mouthRight = getFaceKeypoint(face, "mouth_right")
      const upperLipCenter = getFaceKeypoint(face, "upper_lip_center")
      const lowerLipCenter = getFaceKeypoint(face, "lower_lip_center")
      const leftCorner = getFaceKeypoint(face, "left_corner")
      const rightCorner = getFaceKeypoint(face, "right_corner")

      // Calculate mouth shapes using relative distances
      const calculateMouthShape = (): { openness: number; width: number; smile: number } => {
        if (
          !upperLipTop ||
          !lowerLipBottom ||
          !mouthLeft ||
          !mouthRight ||
          !upperLipCenter ||
          !lowerLipCenter ||
          !leftCorner ||
          !rightCorner
        ) {
          return { openness: 0, width: 0, smile: 0 }
        }

        // Calculate mouth openness
        const mouthHeight = Vector3.Distance(upperLipTop, lowerLipBottom)
        const mouthWidth = Vector3.Distance(mouthLeft, mouthRight)
        const openness = Math.min(Math.max((mouthHeight / mouthWidth - 0.1) / 0.5, 0), 0.7)

        // Calculate mouth width relative to face width
        const faceWidth = Vector3.Distance(getFaceKeypoint(face, "left_ear")!, getFaceKeypoint(face, "right_ear")!) // Distance between ears
        const relativeWidth = mouthWidth / faceWidth
        const neutralRelativeWidth = 0.45 // Adjust based on your model's neutral mouth width
        const width = Math.min(Math.max((relativeWidth - neutralRelativeWidth) / 0.1, -1), 1)

        // Calculate smile
        const mouthCenter = Vector3.Center(upperLipCenter, lowerLipCenter)
        const leftLift = Vector3.Distance(leftCorner, mouthCenter)
        const rightLift = Vector3.Distance(rightCorner, mouthCenter)
        const averageLift = (leftLift + rightLift) / 2
        const neutralLift = mouthWidth * 0.3 // Adjust based on your model's neutral mouth shape
        const smile = Math.min(Math.max((averageLift - neutralLift) / (mouthWidth * 0.2), -1), 1)

        return { openness, width, smile }
      }

      const { openness: mouthOpenness, width: mouthWidth, smile: mouthSmile } = calculateMouthShape()

      // Map facial landmarks to morph targets
      const morphTargets = {
        まばたき: Math.pow(1 - leftEyeOpenness, 1.5),
        まばたき右: Math.pow(1 - rightEyeOpenness, 1.5),
        あ: Math.pow(mouthOpenness, 1.5),
        い: Math.max(0, -mouthWidth) * 0.7,
        う: Math.max(0, mouthWidth) * 0.7,
        お: Math.max(0, mouthOpenness - 0.3) * 1.5,
        わ: Math.max(0, mouthSmile) * (1 - Math.min(mouthOpenness, 1) * 0.7), // Closed smile
        にやり: Math.max(0, mouthSmile) * Math.min(mouthOpenness, 1) * 0.8, // Open smile
        "∧": Math.max(0, -mouthSmile) * 0.5, // Slight frown
      }

      // Apply morph targets with smoothing
      const smoothingFactor = 0.7
      for (const [morphName, targetValue] of Object.entries(morphTargets)) {
        const currentValue = mmdModel.morph.getMorphWeight(morphName)
        const newValue = currentValue + (targetValue - currentValue) * smoothingFactor
        mmdModel.morph.setMorphWeight(morphName, Math.max(0, Math.min(1, newValue)))
      }
    }

    if (sceneRef.current && mmdModelRef.current) {
      updateMMDFace(mmdModelRef.current, face)
    }
  }, [face, lerpFactor])

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
