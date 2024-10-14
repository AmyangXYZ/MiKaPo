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
  Matrix,
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
import { IconButton, Tooltip } from "@mui/material"
import { BorderAll, Camera, CenterFocusWeak } from "@mui/icons-material"

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

const poseKeypoints: { [key: string]: number } = {
  nose: 0,
  left_eye_inner: 1,
  left_eye: 2,
  left_eye_outer: 3,
  right_eye_inner: 4,
  right_eye: 5,
  right_eye_outer: 6,
  left_ear: 7,
  right_ear: 8,
  mouth_left: 9,
  mouth_right: 10,
  left_shoulder: 11,
  right_shoulder: 12,
  left_elbow: 13,
  right_elbow: 14,
  left_wrist: 15,
  right_wrist: 16,
  left_pinky: 17,
  right_pinky: 18,
  left_index: 19,
  right_index: 20,
  left_thumb: 21,
  right_thumb: 22,
  left_hip: 23,
  right_hip: 24,
  left_knee: 25,
  right_knee: 26,
  left_ankle: 27,
  right_ankle: 28,
  left_heel: 29,
  right_heel: 30,
  left_foot_index: 31,
  right_foot_index: 32,
}

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

const handKeypoints: { [key: string]: number } = {
  wrist: 0,
  thumb_cmc: 1,
  thumb_mcp: 2,
  thumb_ip: 3,
  thumb_tip: 4,
  index_mcp: 5,
  index_pip: 6,
  index_dip: 7,
  index_tip: 8,
  middle_mcp: 9,
  middle_pip: 10,
  middle_dip: 11,
  middle_tip: 12,
  ring_mcp: 13,
  ring_pip: 14,
  ring_dip: 15,
  ring_tip: 16,
  pinky_mcp: 17,
  pinky_pip: 18,
  pinky_dip: 19,
  pinky_tip: 20,
}

const getPoseKeypoint = (pose: NormalizedLandmark[] | null, name: string): Vector3 | null => {
  if (!pose || pose.length === 0) return null
  const point = pose[poseKeypoints[name]]
  return point && point.visibility > 0.1 ? new Vector3(point.x, point.y, point.z) : null
}

const getFaceKeypoint = (face: NormalizedLandmark[] | null, name: string): Vector3 | null => {
  if (!face || face.length === 0) return null
  const point = face[faceKeypoints[name]]
  const scaleX = 10 // Adjust these values to fit your MMD model's scale
  const scaleY = 10
  const scaleZ = 5
  return point ? new Vector3(point.x * scaleX, point.y * scaleY, point.z * scaleZ) : null
}

const getHandKeypoint = (hand: NormalizedLandmark[] | null, name: string): Vector3 | null => {
  if (!hand || hand.length === 0) return null
  const point = hand[handKeypoints[name]]
  return point ? new Vector3(point.x, point.y, point.z) : null
}

function MMDScene({
  pose,
  face,
  leftHand,
  rightHand,
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
  pose: NormalizedLandmark[] | null
  face: NormalizedLandmark[] | null
  leftHand: NormalizedLandmark[] | null
  rightHand: NormalizedLandmark[] | null
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

  const [enableSplitView, setEnableSplitView] = useState<boolean>(false)

  const getBone = (name: string): IMmdRuntimeLinkedBone | undefined => {
    return keyBones.current[name]
  }

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
      const vmd = await new VmdLoader(sceneRef.current!).loadAsync("vmd", `/animation/${selectedAnimation}.vmd`)
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
    const scale = 10
    const yOffset = 7

    const updateMMDPose = (mmdModel: MmdWasmModel | null, pose: NormalizedLandmark[] | null): void => {
      if (!pose || !mmdModel || pose.length === 0) {
        return
      }

      const rotateHead = (): void => {
        const nose = getPoseKeypoint(pose, "nose")
        const leftShoulder = getPoseKeypoint(pose, "left_shoulder")
        const rightShoulder = getPoseKeypoint(pose, "right_shoulder")
        const neckBone = getBone("首")
        const upperBodyBone = getBone("上半身")

        if (nose && leftShoulder && rightShoulder && neckBone && upperBodyBone) {
          const neckPos = leftShoulder.add(rightShoulder).scale(0.5)
          const headDir = nose.subtract(neckPos).normalize()

          const upperBodyRotation = upperBodyBone.rotationQuaternion || new Quaternion()
          const upperBodyRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(upperBodyRotation, upperBodyRotationMatrix)

          const localHeadDir = Vector3.TransformNormal(headDir, upperBodyRotationMatrix.invert())

          const forwardDir = new Vector3(localHeadDir.x, 0, localHeadDir.z).normalize()

          const tiltAngle = Math.atan2(-localHeadDir.y, forwardDir.length())

          const tiltOffset = -Math.PI / 5
          const adjustedTiltAngle = tiltAngle + tiltOffset

          const horizontalQuat = Quaternion.FromLookDirectionLH(forwardDir, Vector3.Up())

          const tiltQuat = Quaternion.RotationAxis(Vector3.Right(), adjustedTiltAngle)

          const combinedQuat = horizontalQuat.multiply(tiltQuat)

          const maxRotationAngle = Math.PI / 3
          const clampedQuat = Quaternion.FromEulerAngles(
            Math.max(-maxRotationAngle, Math.min(maxRotationAngle, combinedQuat.toEulerAngles().x)),
            Math.max(-maxRotationAngle, Math.min(maxRotationAngle, combinedQuat.toEulerAngles().y)),
            Math.max(-maxRotationAngle, Math.min(maxRotationAngle, combinedQuat.toEulerAngles().z))
          )

          neckBone.setRotationQuaternion(
            Quaternion.Slerp(neckBone.rotationQuaternion || new Quaternion(), clampedQuat, lerpFactor),
            Space.LOCAL
          )
        }
      }

      const rotateUpperBody = (): void => {
        const leftShoulder = getPoseKeypoint(pose, "left_shoulder")
        const rightShoulder = getPoseKeypoint(pose, "right_shoulder")
        const upperBodyBone = getBone("上半身")

        if (leftShoulder && rightShoulder && upperBodyBone) {
          const spineDir = leftShoulder.subtract(rightShoulder).normalize()
          const spineForward = Vector3.Cross(spineDir, Vector3.Up()).normalize()
          const spineRotation = Quaternion.FromLookDirectionRH(spineForward, Vector3.Up())

          upperBodyBone.setRotationQuaternion(
            Quaternion.Slerp(upperBodyBone.rotationQuaternion || new Quaternion(), spineRotation, lerpFactor),
            Space.LOCAL
          )
        }

        const leftHip = getPoseKeypoint(pose, "left_hip")
        const rightHip = getPoseKeypoint(pose, "right_hip")

        if (leftShoulder && rightShoulder && leftHip && rightHip && upperBodyBone) {
          const shoulderCenter = leftShoulder.add(rightShoulder).scale(0.5)
          const hipCenter = leftHip.add(rightHip).scale(0.5)
          const bendDir = hipCenter.subtract(shoulderCenter).normalize()

          const bendAngle = Math.acos(Vector3.Dot(bendDir, Vector3.Up()))
          const bendAxis = Vector3.Cross(Vector3.Up(), bendDir).normalize()
          const bendRotation = Quaternion.RotationAxis(bendAxis, -bendAngle)

          upperBodyBone.rotationQuaternion = Quaternion.Slerp(
            upperBodyBone.rotationQuaternion || new Quaternion(),
            bendRotation.multiply(upperBodyBone.rotationQuaternion || new Quaternion()),
            lerpFactor
          )
        }
      }

      const rotateLowerBody = (): void => {
        const leftHip = getPoseKeypoint(pose, "left_hip")
        const rightHip = getPoseKeypoint(pose, "right_hip")
        const lowerBodyBone = getBone("下半身")
        if (leftHip && rightHip && lowerBodyBone) {
          const hipDir = leftHip.subtract(rightHip).normalize()
          const lowerBodyUp = Vector3.Up()
          const lowerBodyForward = Vector3.Cross(hipDir, lowerBodyUp).normalize()
          const lowerBodyRotation = Quaternion.FromLookDirectionRH(lowerBodyForward, lowerBodyUp)
          lowerBodyBone.setRotationQuaternion(
            Quaternion.Slerp(lowerBodyBone.rotationQuaternion || new Quaternion(), lowerBodyRotation, lerpFactor),
            Space.LOCAL
          )
        }
      }

      const rotateHip = (side: "left" | "right"): void => {
        const hip = getPoseKeypoint(pose, `${side}_hip`)
        const knee = getPoseKeypoint(pose, `${side}_knee`)
        const hipBone = getBone(`${side === "left" ? "左" : "右"}足`)
        const lowerBodyBone = getBone("下半身")

        if (hip && knee && hipBone && lowerBodyBone) {
          const legDir = knee.subtract(hip).normalize()
          legDir.y *= -1

          const lowerBodyRotation = lowerBodyBone.rotationQuaternion || new Quaternion()
          const lowerBodyRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(lowerBodyRotation, lowerBodyRotationMatrix)

          const localLegDir = Vector3.TransformNormal(legDir, lowerBodyRotationMatrix.invert())

          const defaultDir = new Vector3(0, -1, 0).normalize()

          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, localLegDir, new Quaternion())

          hipBone.setRotationQuaternion(
            Quaternion.Slerp(hipBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
            Space.LOCAL
          )
        }
      }

      const moveFoot = (side: "right" | "left"): void => {
        const ankle = getPoseKeypoint(pose, `${side}_ankle`)
        const hip = getPoseKeypoint(pose, `${side}_hip`)
        const bone = getBone(`${side === "right" ? "右" : "左"}足ＩＫ`)

        if (ankle && hip && bone) {
          const targetPosition = new Vector3(ankle.x * scale, -ankle.y * scale + yOffset, ankle.z * scale)

          bone.position = Vector3.Lerp(bone.position, targetPosition, lerpFactor)
        }
      }

      const rotateFoot = (side: "right" | "left"): void => {
        const hip = getPoseKeypoint(pose, `${side}_hip`)
        const ankle = getPoseKeypoint(pose, `${side}_ankle`)
        const footBone = getBone(`${side === "right" ? "右" : "左"}足首`)

        if (hip && ankle && footBone) {
          const footDir = ankle.subtract(hip).normalize()

          const defaultDir = new Vector3(0, 0, 1)

          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, footDir, new Quaternion())

          footBone.setRotationQuaternion(
            Quaternion.Slerp(footBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
            Space.WORLD
          )
        }
      }

      const rotateUpperArm = (side: "left" | "right"): void => {
        const shoulder = getPoseKeypoint(pose, `${side}_shoulder`)
        const elbow = getPoseKeypoint(pose, `${side}_elbow`)
        const upperArmBone = getBone(`${side === "left" ? "左" : "右"}腕`)
        const upperBodyBone = getBone("上半身")

        if (shoulder && elbow && upperArmBone && upperBodyBone) {
          const armDir = elbow.subtract(shoulder).normalize()
          armDir.y *= -1

          const upperBodyRotation = upperBodyBone.rotationQuaternion || new Quaternion()
          const upperBodyRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(upperBodyRotation, upperBodyRotationMatrix)

          const localArmDir = Vector3.TransformNormal(armDir, upperBodyRotationMatrix.invert())

          const defaultDir = new Vector3(side === "left" ? 1 : -1, -1, 0).normalize()

          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, localArmDir, new Quaternion())

          upperArmBone.setRotationQuaternion(
            Quaternion.Slerp(upperArmBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
            Space.LOCAL
          )
        }
      }

      const rotateLowerArm = (side: "left" | "right"): void => {
        const elbow = getPoseKeypoint(pose, `${side}_elbow`)
        const wrist = getPoseKeypoint(pose, `${side}_wrist`)
        const lowerArmBone = getBone(`${side === "left" ? "左" : "右"}ひじ`)
        const upperArmBone = getBone(`${side === "left" ? "左" : "右"}腕`)

        if (elbow && wrist && lowerArmBone && upperArmBone) {
          const lowerArmDir = wrist.subtract(elbow).normalize()
          lowerArmDir.y *= -1

          const upperArmRotation = upperArmBone.rotationQuaternion || new Quaternion()
          const upperArmRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(upperArmRotation, upperArmRotationMatrix)

          const localLowerArmDir = Vector3.TransformNormal(lowerArmDir, upperArmRotationMatrix.invert())

          const defaultDir = new Vector3(side === "left" ? 1 : -1, -1, 0).normalize()

          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, localLowerArmDir, new Quaternion())

          lowerArmBone.setRotationQuaternion(
            Quaternion.Slerp(lowerArmBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
            Space.LOCAL
          )
        }
      }

      rotateUpperBody()
      rotateHead()
      rotateLowerBody()
      moveFoot("right")
      moveFoot("left")
      rotateFoot("right")
      rotateFoot("left")
      rotateHip("right")
      rotateHip("left")
      rotateUpperArm("right")
      rotateUpperArm("left")
      rotateLowerArm("right")
      rotateLowerArm("left")
    }
    if (sceneRef.current && mmdModelRef.current) {
      updateMMDPose(mmdModelRef.current, pose)
    }
  }, [pose, lerpFactor])

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

  useEffect(() => {
    const rotateWrist = (hand: NormalizedLandmark[] | null, side: "left" | "right"): void => {
      if (!hand || hand.length === 0 || !pose || pose.length === 0) return

      const wrist = getPoseKeypoint(pose, `${side}_wrist`)
      const middleFinger = getHandKeypoint(hand, "pinky_dip")
      const handBone = getBone(`${side === "left" ? "左" : "右"}手首`)
      const lowerArmBone = getBone(`${side === "left" ? "左" : "右"}ひじ`)

      if (wrist && middleFinger && handBone && lowerArmBone) {
        const wristDir = middleFinger.subtract(wrist).normalize()
        wristDir.y *= -1

        const lowerArmRotation = lowerArmBone.rotationQuaternion || new Quaternion()
        const lowerArmRotationMatrix = new Matrix()
        Matrix.FromQuaternionToRef(lowerArmRotation, lowerArmRotationMatrix)

        const localWristDir = Vector3.TransformNormal(wristDir, lowerArmRotationMatrix.invert())

        const defaultDir = new Vector3(side === "left" ? 1 : -1, -1, 0).normalize()

        const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, localWristDir, new Quaternion())

        handBone.setRotationQuaternion(
          Quaternion.Slerp(handBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
          Space.LOCAL
        )
      }
    }

    const rotateFingers = (hand: NormalizedLandmark[] | null, side: "left" | "right"): void => {
      if (!hand || hand.length === 0) return

      const fingerNames = ["親指", "人指", "中指", "薬指", "小指"]
      const fingerJoints = ["", "１", "２", "３"]
      const maxAngle = Math.PI / 2
      const maxEndSegmentAngle = (Math.PI * 2) / 3
      const fingerBaseIndices = [1, 5, 9, 13, 17]

      fingerNames.forEach((fingerName, fingerIndex) => {
        fingerJoints.forEach((joint, jointIndex) => {
          const boneName = `${side === "left" ? "左" : "右"}${fingerName}${joint}`
          const bone = getBone(boneName)

          if (bone) {
            const baseIndex = fingerBaseIndices[fingerIndex]
            const currentIndex = baseIndex + jointIndex
            const nextIndex = baseIndex + jointIndex + 1

            let rotationAngle = 0

            if (nextIndex < hand.length) {
              const currentPoint = new Vector3(hand[currentIndex].x, hand[currentIndex].y, hand[currentIndex].z)
              const nextPoint = new Vector3(hand[nextIndex].x, hand[nextIndex].y, hand[nextIndex].z)

              const segmentVector = nextPoint.subtract(currentPoint)

              let defaultVector: Vector3
              if (fingerName === "親指") {
                defaultVector = new Vector3(side === "left" ? -1 : 1, 1, 0)
              } else {
                defaultVector = new Vector3(0, -1, 0)
              }
              rotationAngle = Vector3.GetAngleBetweenVectors(segmentVector, defaultVector, new Vector3(1, 0, 0))

              const isEndSegment = jointIndex === 3
              const currentMaxAngle = isEndSegment ? maxEndSegmentAngle : maxAngle

              rotationAngle = Math.min(Math.max(rotationAngle, 0), currentMaxAngle)

              if (isEndSegment && rotationAngle > maxAngle) {
                rotationAngle = 0
              }
            }

            let defaultDir: Vector3

            if (boneName.includes("親指")) {
              defaultDir = new Vector3(-1, side === "left" ? -1 : 1, 0).normalize()
            } else {
              defaultDir = new Vector3(0, 0, side === "left" ? -1 : 1).normalize()
            }

            const rotation = defaultDir.scale(rotationAngle)

            bone.setRotationQuaternion(
              Quaternion.Slerp(
                bone.rotationQuaternion || new Quaternion(),
                Quaternion.FromEulerAngles(rotation.x, rotation.y, rotation.z),
                lerpFactor
              ),
              Space.LOCAL
            )
          }
        })
      })
    }

    if (sceneRef.current && mmdModelRef.current) {
      rotateWrist(leftHand, "left")
      rotateWrist(rightHand, "right")
      rotateFingers(leftHand, "left")
      rotateFingers(rightHand, "right")
    }
  }, [leftHand, pose, rightHand, lerpFactor])

  const handleCaptureScreenshot = () => {
    CreateScreenshotAsync(engineRef.current!, cameraRef.current!, { precision: 1 }).then((b64) => {
      const link = document.createElement("a")
      link.href = b64
      link.download = "mikapo_screenshot.png"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    })
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
          <BorderAll sx={{ width: "26px", height: "26px" }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Capture screenshot">
        <IconButton
          style={{ position: "absolute", top: "12rem", right: ".5rem", color: "#f209f5" }}
          onClick={handleCaptureScreenshot}
        >
          <Camera sx={{ width: "26px", height: "26px" }} />
        </IconButton>
      </Tooltip>
    </>
  )
}

export default MMDScene
