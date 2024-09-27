import { useEffect, useRef, useState } from "react"
import {
  ArcRotateCamera,
  BackgroundMaterial,
  Color3,
  Color4,
  DirectionalLight,
  Engine,
  HemisphericLight,
  Matrix,
  Mesh,
  MeshBuilder,
  PhotoDome,
  Quaternion,
  Scene,
  SceneLoader,
  ShadowGenerator,
  Space,
  Texture,
  Vector3,
} from "@babylonjs/core"
import { NormalizedLandmark } from "@mediapipe/tasks-vision"
import { MmdAmmoJSPlugin, MmdAmmoPhysics, MmdModel, MmdRuntime } from "babylon-mmd"
import backgroundGroundUrl from "./assets/backgroundGround.png"
import type { IMmdRuntimeLinkedBone } from "babylon-mmd/esm/Runtime/IMmdRuntimeLinkedBone"
import ammoPhysics from "./ammo/ammo.wasm"
import { Box, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from "@mui/material"

const defaultScene = "Static"
const availableScenes = ["Static", "Office", "Beach", "Bedroom"]

const defaultModel = "深空之眼-托特"
const availableModels = ["深空之眼-托特", "深空之眼-托特2", "深空之眼-大梵天", "鸣潮-吟霖", "原神-荧"]

function MMDScene({
  pose,
  face,
  leftHand,
  rightHand,
  lerpFactor,
  setFps,
}: {
  pose: NormalizedLandmark[] | null
  face: NormalizedLandmark[] | null
  leftHand: NormalizedLandmark[] | null
  rightHand: NormalizedLandmark[] | null
  lerpFactor: number
  setFps: (fps: number) => void
}): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<Scene | null>(null)
  const [sceneRendered, setSceneRendered] = useState<boolean>(false)
  const [selectedScene, setSelectedScene] = useState<string>(defaultScene)
  const [selectedModel, setSelectedModel] = useState<string>(defaultModel)
  const mmdModelRef = useRef<MmdModel | null>(null)
  const mmdRuntimeRef = useRef<MmdRuntime | null>(null)
  const shadowGeneratorRef = useRef<ShadowGenerator | null>(null)
  const domeRef = useRef<PhotoDome | null>(null)

  const handleSceneChange = (event: SelectChangeEvent): void => {
    setSelectedScene(event.target.value)
  }

  const handleModelChange = (event: SelectChangeEvent): void => {
    setSelectedModel(event.target.value)
  }

  useEffect(() => {
    const createScene = async (canvas: HTMLCanvasElement): Promise<Scene> => {
      const engine = new Engine(canvas, true, {}, true)
      const scene = new Scene(engine)
      scene.clearColor = new Color4(0, 0, 0, 0)
      const physicsInstance = await ammoPhysics()
      const physicsPlugin = new MmdAmmoJSPlugin(true, physicsInstance)
      scene.enablePhysics(new Vector3(0, -98, 0), physicsPlugin)
      mmdRuntimeRef.current = new MmdRuntime(scene, new MmdAmmoPhysics(scene))
      mmdRuntimeRef.current.register(scene)

      const camera = new ArcRotateCamera("ArcRotateCamera", 0, 0, 45, new Vector3(0, 12, 0), scene)
      camera.setPosition(new Vector3(0, 15, -27))
      camera.attachControl(canvas, false)
      camera.inertia = 0.8
      camera.speed = 10

      const hemisphericLight = new HemisphericLight("HemisphericLight", new Vector3(0, 1, 0), scene)
      hemisphericLight.intensity = 0.4
      hemisphericLight.specular = new Color3(0, 0, 0)
      hemisphericLight.groundColor = new Color3(1, 1, 1)

      const directionalLight = new DirectionalLight("DirectionalLight", new Vector3(8, -15, 10), scene)
      directionalLight.intensity = 0.8

      shadowGeneratorRef.current = new ShadowGenerator(1024, directionalLight, true)
      shadowGeneratorRef.current.usePercentageCloserFiltering = true
      shadowGeneratorRef.current.forceBackFacesOnly = true
      shadowGeneratorRef.current.filteringQuality = ShadowGenerator.QUALITY_MEDIUM
      shadowGeneratorRef.current.frustumEdgeFalloff = 0.1
      shadowGeneratorRef.current.transparencyShadow = true

      const backgroundMaterial = new BackgroundMaterial("backgroundMaterial", scene)
      backgroundMaterial.diffuseTexture = new Texture(backgroundGroundUrl, scene)
      backgroundMaterial.diffuseTexture.hasAlpha = true
      backgroundMaterial.opacityFresnel = false
      backgroundMaterial.shadowLevel = 0.4
      backgroundMaterial.useRGBColor = false
      backgroundMaterial.primaryColor = Color3.Magenta()
      const ground = MeshBuilder.CreateGround("Ground", {
        width: 28,
        height: 28,
        subdivisions: 2,
        updatable: false,
      })
      ground.material = backgroundMaterial
      ground.receiveShadows = true

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
  }, [setFps, setSceneRendered])

  useEffect(() => {
    if (domeRef.current) {
      domeRef.current.dispose()
    }
    if (sceneRef.current && selectedScene !== "Static") {
      domeRef.current = new PhotoDome(
        "testdome",
        `/textures/${selectedScene}.jpeg`,
        {
          resolution: 32,
          size: 500,
          useDirectMapping: false,
        },
        sceneRef.current
      )
      domeRef.current.imageMode = PhotoDome.MODE_MONOSCOPIC
      domeRef.current.position.y = 0
    }
  }, [sceneRendered, sceneRef, selectedScene])

  useEffect(() => {
    const loadMMD = async (): Promise<void> => {
      if (!sceneRendered || !selectedModel || !mmdRuntimeRef.current) return
      if (mmdModelRef.current) {
        mmdRuntimeRef.current.destroyMmdModel(mmdModelRef.current)
        mmdModelRef.current.mesh.dispose()
      }
      SceneLoader.ImportMeshAsync(
        undefined,
        `./model/${selectedModel}/`,
        `${selectedModel}.pmx`,
        sceneRef.current
      ).then((result) => {
        const mesh = result.meshes[0]
        for (const m of mesh.metadata.meshes) {
          m.receiveShadows = true
        }
        shadowGeneratorRef.current!.addShadowCaster(mesh)
        mmdModelRef.current = mmdRuntimeRef.current!.createMmdModel(mesh as Mesh)
      })
    }
    loadMMD()
  }, [sceneRendered, sceneRef, mmdRuntimeRef, selectedModel])

  useEffect(() => {
    const scale = 10
    const yOffset = 7
    const visibilityThreshold = 0.1
    const keypointIndexByName: { [key: string]: number } = {
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
    const updateMMDPose = (mmdModel: MmdModel | null, pose: NormalizedLandmark[] | null): void => {
      if (!pose || !mmdModel || pose.length === 0) {
        return
      }

      const getKeypoint = (name: string): Vector3 | null => {
        const point = pose[keypointIndexByName[name]]
        return point.visibility > visibilityThreshold ? new Vector3(point.x, point.y, point.z) : null
      }

      const getBone = (name: string): IMmdRuntimeLinkedBone | undefined => {
        return mmdModel!.skeleton.bones.find((bone) => bone.name === name)
      }

      const rotateHead = (): void => {
        const nose = getKeypoint("nose")
        const leftShoulder = getKeypoint("left_shoulder")
        const rightShoulder = getKeypoint("right_shoulder")
        const neckBone = getBone("首")
        if (nose && leftShoulder && rightShoulder && neckBone) {
          const neckPos = leftShoulder.add(rightShoulder).scale(0.5)
          const headDir = nose.subtract(neckPos).normalize()

          const forwardDir = new Vector3(headDir.x, 0, headDir.z).normalize()

          const tiltAngle = Math.atan2(-headDir.y, forwardDir.length())

          // Add a constant offset to the tilt angle to correct the head orientation
          const tiltOffset = -Math.PI / 5 // Adjust this value as needed
          const adjustedTiltAngle = tiltAngle + tiltOffset

          const horizontalQuat = Quaternion.FromLookDirectionLH(forwardDir, Vector3.Up())

          const tiltQuat = Quaternion.RotationAxis(Vector3.Right(), adjustedTiltAngle)

          const combinedQuat = horizontalQuat.multiply(tiltQuat)

          neckBone.setRotationQuaternion(
            Quaternion.Slerp(neckBone.rotationQuaternion || new Quaternion(), combinedQuat, lerpFactor),
            Space.LOCAL
          )
        }
      }

      const rotateUpperBody = (): void => {
        const leftShoulder = getKeypoint("left_shoulder")
        const rightShoulder = getKeypoint("right_shoulder")
        const upperBodyBone = getBone("上半身")

        if (leftShoulder && rightShoulder && upperBodyBone) {
          // Rotation calculation
          const spineDir = leftShoulder.subtract(rightShoulder).normalize()
          const spineUp = Vector3.Up()
          const spineForward = Vector3.Cross(spineDir, spineUp).normalize()
          const spineRotation = Quaternion.FromLookDirectionRH(spineForward, spineUp)

          upperBodyBone.setRotationQuaternion(
            Quaternion.Slerp(upperBodyBone.rotationQuaternion || new Quaternion(), spineRotation, lerpFactor),
            Space.LOCAL
          )
        }

        const leftHip = getKeypoint("left_hip")
        const rightHip = getKeypoint("right_hip")

        if (leftShoulder && rightShoulder && leftHip && rightHip && upperBodyBone) {
          // Bending calculation
          const shoulderCenter = leftShoulder.add(rightShoulder).scale(0.5)
          const hipCenter = leftHip.add(rightHip).scale(0.5)
          const bendDir = hipCenter.subtract(shoulderCenter).normalize()

          const spineUp = Vector3.Up()
          const bendAngle = Math.acos(Vector3.Dot(bendDir, spineUp))
          const bendAxis = Vector3.Cross(spineUp, bendDir).normalize()
          const bendRotation = Quaternion.RotationAxis(bendAxis, -bendAngle)

          // Apply bend rotation on top of existing rotation
          upperBodyBone.rotationQuaternion = Quaternion.Slerp(
            upperBodyBone.rotationQuaternion || new Quaternion(),
            bendRotation.multiply(upperBodyBone.rotationQuaternion || new Quaternion()),
            lerpFactor
          )
        }
      }

      const rotateLowerBody = (): void => {
        const leftHip = getKeypoint("left_hip")
        const rightHip = getKeypoint("right_hip")
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
        const hip = getKeypoint(`${side}_hip`)
        const knee = getKeypoint(`${side}_knee`)
        const hipBone = getBone(`${side === "left" ? "左" : "右"}足`)
        const lowerBodyBone = getBone("下半身")

        if (hip && knee && hipBone && lowerBodyBone) {
          const desiredLegDir = knee.subtract(hip).normalize()

          const lowerBodyRotation = lowerBodyBone.rotationQuaternion || new Quaternion()
          const lowerBodyRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(lowerBodyRotation, lowerBodyRotationMatrix)

          const localDesiredLegDir = Vector3.TransformNormal(desiredLegDir, lowerBodyRotationMatrix.invert())

          const rotationAxis = Vector3.Cross(Vector3.Down(), localDesiredLegDir).normalize()
          let rotationAngle = Math.acos(Vector3.Dot(Vector3.Down(), localDesiredLegDir))

          const maxRotationAngle = Math.PI / 3
          rotationAngle = Math.min(rotationAngle, maxRotationAngle)

          const hipRotation = Quaternion.RotationAxis(rotationAxis, rotationAngle)

          hipBone.setRotationQuaternion(
            Quaternion.Slerp(hipBone.rotationQuaternion || new Quaternion(), hipRotation, lerpFactor),
            Space.LOCAL
          )
        }
      }

      const moveFoot = (side: "right" | "left"): void => {
        const ankle = getKeypoint(`${side}_ankle`)
        const bone = getBone(`${side === "right" ? "右" : "左"}足ＩＫ`)
        if (ankle && bone) {
          const targetPosition = new Vector3(ankle.x! * scale, -ankle.y! * scale + yOffset, ankle.z! * scale)
          bone.position = Vector3.Lerp(bone.position, targetPosition, lerpFactor)
        }
      }

      const rotateFoot = (side: "right" | "left"): void => {
        const hip = getKeypoint(`${side}_hip`)
        const ankle = getKeypoint(`${side}_ankle`)
        const footBone = getBone(`${side === "right" ? "右" : "左"}足首`)

        if (hip && ankle && footBone) {
          const footDir = ankle.subtract(hip).normalize()
          footDir.y = 0 // Ensure the foot stays level with the ground

          const defaultDir = new Vector3(0, 0, 1) // Assuming default foot direction is forward

          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, footDir, new Quaternion())

          footBone.setRotationQuaternion(
            Quaternion.Slerp(footBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
            Space.WORLD
          )
        }
      }

      const rotateUpperArm = (side: "left" | "right"): void => {
        const shoulder = getKeypoint(`${side}_shoulder`)
        const elbow = getKeypoint(`${side}_elbow`)
        const upperArmBone = getBone(`${side === "left" ? "左" : "右"}腕`)
        const upperBodyBone = getBone("上半身")

        if (shoulder && elbow && upperArmBone && upperBodyBone) {
          // Calculate arm direction (from shoulder to elbow)
          const armDir = elbow.subtract(shoulder).normalize()

          // Ensure Y-axis is always aligned (pointing downwards)
          armDir.y = -Math.abs(armDir.y)

          // Correct X-axis direction based on the side
          armDir.x = side === "left" ? Math.abs(armDir.x) : -Math.abs(armDir.x)

          const upperBodyRotation = upperBodyBone.rotationQuaternion || new Quaternion()
          const upperBodyRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(upperBodyRotation, upperBodyRotationMatrix)

          // Transform arm direction to local space
          const localArmDir = Vector3.TransformNormal(armDir, upperBodyRotationMatrix.invert())

          const defaultDir = new Vector3(side === "left" ? 1 : -1, -1, 0).normalize()

          // Calculate the rotation from default pose to current pose
          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, localArmDir, new Quaternion())

          // Apply rotation with lerp for smooth transition
          upperArmBone.setRotationQuaternion(
            Quaternion.Slerp(upperArmBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
            Space.LOCAL
          )
        }
      }

      const rotateLowerArm = (side: "left" | "right"): void => {
        const elbow = getKeypoint(`${side}_elbow`)
        const wrist = getKeypoint(`${side}_wrist`)
        const lowerArmBone = getBone(`${side === "left" ? "左" : "右"}ひじ`)
        const upperArmBone = getBone(`${side === "left" ? "左" : "右"}腕`)

        if (elbow && wrist && lowerArmBone && upperArmBone) {
          // Calculate lower arm direction (from elbow to wrist)
          const lowerArmDir = wrist.subtract(elbow).normalize()

          // Ensure Z-axis is always pointing forward
          lowerArmDir.z = Math.abs(lowerArmDir.z)
          lowerArmDir.x = -lowerArmDir.x

          const upperArmRotation = upperArmBone.rotationQuaternion || new Quaternion()
          const upperArmRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(upperArmRotation, upperArmRotationMatrix)

          // Transform lower arm direction to local space relative to upper arm
          const localLowerArmDir = Vector3.TransformNormal(lowerArmDir, upperArmRotationMatrix.invert())

          const defaultDir = new Vector3(side === "left" ? -1 : 1, 1, 0)

          // Calculate the rotation from default pose to current pose
          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, localLowerArmDir, new Quaternion())

          // Apply rotation with lerp for smooth transition
          lowerArmBone.setRotationQuaternion(
            Quaternion.Slerp(lowerArmBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
            Space.LOCAL
          )
        }
      }

      const rotateHand = (side: "left" | "right"): void => {
        const wrist = getKeypoint(`${side}_wrist`)
        const indexFinger = getKeypoint(`${side}_index`)
        const handBone = getBone(`${side === "left" ? "左" : "右"}手首`)
        const lowerArmBone = getBone(`${side === "left" ? "左" : "右"}ひじ`)

        if (wrist && indexFinger && handBone && lowerArmBone) {
          // Calculate hand direction
          const handDir = indexFinger.subtract(wrist).normalize()

          // Get lower arm rotation
          const lowerArmRotation = lowerArmBone.rotationQuaternion || new Quaternion()
          const lowerArmRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(lowerArmRotation, lowerArmRotationMatrix)

          // Transform hand direction to local space relative to lower arm
          const localHandDir = Vector3.TransformNormal(handDir, lowerArmRotationMatrix.invert())

          // Define default direction (pointing along the arm)
          const defaultDir = new Vector3(0, -1, 0)

          // Calculate rotation from default to current hand direction
          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, localHandDir, new Quaternion())

          // Decompose the rotation into Euler angles
          const rotation = rotationQuaternion.toEulerAngles()

          // Clamp each rotation axis
          const maxAngle = Math.PI / 3 // 60 degrees
          rotation.x = Math.max(-maxAngle, Math.min(maxAngle, rotation.x))
          rotation.y = Math.max(-maxAngle, Math.min(maxAngle, rotation.y))
          rotation.z = Math.max(-maxAngle, Math.min(maxAngle, rotation.z))

          // Create a new quaternion from the clamped Euler angles
          const clampedQuaternion = Quaternion.FromEulerAngles(rotation.x, rotation.y, rotation.z)

          // Apply rotation with lerp for smooth transition
          handBone.setRotationQuaternion(
            Quaternion.Slerp(handBone.rotationQuaternion || new Quaternion(), clampedQuaternion, lerpFactor),
            Space.LOCAL
          )
        }
      }

      rotateHead()
      rotateUpperBody()
      rotateLowerBody()
      rotateHip("right")
      rotateHip("left")
      moveFoot("right")
      moveFoot("left")
      rotateFoot("right")
      rotateFoot("left")
      rotateUpperArm("right")
      rotateUpperArm("left")
      rotateLowerArm("right")
      rotateLowerArm("left")
      rotateHand("right")
      rotateHand("left")
    }
    if (sceneRef.current && mmdModelRef.current) {
      updateMMDPose(mmdModelRef.current, pose)
    }
  }, [pose, lerpFactor])

  useEffect(() => {
    const updateMMDFace = (mmdModel: MmdModel | null, face: NormalizedLandmark[] | null): void => {
      if (!face || !mmdModel || face.length === 0) {
        return
      }

      // Scaling factors
      const scaleX = 10 // Adjust these values to fit your MMD model's scale
      const scaleY = 10
      const scaleZ = 5

      const getKeypoint = (index: number): Vector3 | null => {
        const point = face[index]
        return point ? new Vector3(point.x * scaleX, point.y * scaleY, point.z * scaleZ) : null
      }

      // Eye landmarks
      const leftEyeUpper = getKeypoint(159)
      const leftEyeLower = getKeypoint(145)
      const leftEyeLeft = getKeypoint(33)
      const leftEyeRight = getKeypoint(133)
      const leftEyeIris = getKeypoint(468)
      const rightEyeUpper = getKeypoint(386)
      const rightEyeLower = getKeypoint(374)
      const rightEyeLeft = getKeypoint(362)
      const rightEyeRight = getKeypoint(263)
      const rightEyeIris = getKeypoint(473)

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

      // Directly control eye bones instead of using morph targets
      const controlEyeBones = (scene: Scene, averageGaze: { x: number; y: number }) => {
        const leftEyeBone = scene.getBoneByName("左目")
        const rightEyeBone = scene.getBoneByName("右目")

        if (leftEyeBone && rightEyeBone) {
          const maxHorizontalRotation = Math.PI / 6 // 30 degrees max horizontal rotation
          const maxVerticalRotation = Math.PI / 12 // 15 degrees max vertical rotation

          const xRotation = averageGaze.y * maxVerticalRotation
          const yRotation = -averageGaze.x * maxHorizontalRotation

          // Apply rotation with smoothing
          const smoothingFactor = 0.7
          leftEyeBone.rotation = Vector3.Lerp(
            leftEyeBone.rotation,
            new Vector3(xRotation, yRotation, 0),
            smoothingFactor
          )
          rightEyeBone.rotation = Vector3.Lerp(
            rightEyeBone.rotation,
            new Vector3(xRotation, yRotation, 0),
            smoothingFactor
          )
        }
      }

      controlEyeBones(sceneRef.current!, averageGaze)

      // Mouth landmarks
      const upperLipTop = getKeypoint(13)
      const lowerLipBottom = getKeypoint(14)
      const mouthLeft = getKeypoint(61)
      const mouthRight = getKeypoint(291)
      const upperLipCenter = getKeypoint(0)
      const lowerLipCenter = getKeypoint(17)
      const leftCorner = getKeypoint(291)
      const rightCorner = getKeypoint(61)

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
        const faceWidth = Vector3.Distance(getKeypoint(234)!, getKeypoint(454)!) // Distance between ears
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
  }, [face])

  useEffect(() => {}, [leftHand, rightHand])
  return (
    <>
      <canvas ref={canvasRef} className="scene"></canvas>
      <Box className="scene-selector">
        <FormControl sx={{ borderColor: "white" }}>
          <InputLabel sx={{ color: "white", fontSize: ".9rem" }}>Scene</InputLabel>
          <Select
            label="Scene"
            value={selectedScene}
            onChange={handleSceneChange}
            sx={{ color: "white", fontSize: ".9rem" }}
            autoWidth
          >
            {availableScenes.map((scene) => (
              <MenuItem sx={{ fontSize: ".8rem" }} key={scene} value={scene}>
                {scene}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
      <Box className="model-selector">
        <FormControl sx={{ borderColor: "white" }}>
          <InputLabel sx={{ color: "white", fontSize: ".9rem" }}>Model</InputLabel>
          <Select
            label="Model"
            value={selectedModel}
            onChange={handleModelChange}
            sx={{ color: "white", fontSize: ".9rem" }}
            autoWidth
          >
            {availableModels.map((model) => (
              <MenuItem sx={{ fontSize: ".8rem" }} key={model} value={model}>
                {model}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
    </>
  )
}

export default MMDScene
