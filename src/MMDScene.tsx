import { useEffect, useRef } from "react"
import {
  ArcRotateCamera,
  BackgroundMaterial,
  Color3,
  DirectionalLight,
  Engine,
  HemisphericLight,
  Matrix,
  Mesh,
  MeshBuilder,
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

function MMDScene({ pose, setFps }: { pose: NormalizedLandmark[] | null; setFps: (fps: number) => void }): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<Scene | null>(null)
  const mmdModelRef = useRef<MmdModel | null>(null)
  const mmdRuntimeRef = useRef<MmdRuntime | null>(null)
  const shadowGeneratorRef = useRef<ShadowGenerator | null>(null)

  useEffect(() => {
    const createScene = (canvas: HTMLCanvasElement): Scene => {
      const engine = new Engine(canvas, true, {}, true)
      const scene = new Scene(engine)
      const camera = new ArcRotateCamera("ArcRotateCamera", 0, 0, 45, new Vector3(0, 12, 0), scene)
      camera.setPosition(new Vector3(0, 20, -25))
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

    const loadMMD = async (scene: Scene | null): Promise<void> => {
      if (!scene) return
      const physicsInstance = await ammoPhysics()
      const physicsPlugin = new MmdAmmoJSPlugin(true, physicsInstance)
      scene.enablePhysics(new Vector3(0, -98, 0), physicsPlugin)

      mmdRuntimeRef.current = new MmdRuntime(scene, new MmdAmmoPhysics(scene))
      mmdRuntimeRef.current.register(scene)

      SceneLoader.ImportMeshAsync(undefined, `./model/Thoth/`, `Thoth.pmx`, scene).then((result) => {
        const mesh = result.meshes[0]
        for (const m of mesh.metadata.meshes) {
          m.receiveShadows = true
        }
        shadowGeneratorRef.current!.addShadowCaster(mesh)
        mmdModelRef.current = mmdRuntimeRef.current!.createMmdModel(mesh as Mesh)
      })
    }

    if (canvasRef.current) {
      sceneRef.current = createScene(canvasRef.current)
      loadMMD(sceneRef.current)
    }
  }, [setFps])

  useEffect(() => {
    const updateMMDPose = (mmdModel: MmdModel | null, pose: NormalizedLandmark[] | null): void => {
      if (!pose || !mmdModel) {
        return
      }

      const lerpFactor = 0.5
      const scale = 10
      const yOffset = 8
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
        const upperBodyBone = getBone("上半身")
        if (nose && leftShoulder && rightShoulder && neckBone && upperBodyBone) {
          const neckPos = leftShoulder.add(rightShoulder).scale(0.5)
          const headDir = nose.subtract(neckPos).normalize()

          const upperBodyRotation = Quaternion.Slerp(
            upperBodyBone.rotationQuaternion || new Quaternion(),
            Quaternion.FromLookDirectionLH(headDir, Vector3.Up()),
            lerpFactor
          )
          const upperBodyRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(upperBodyRotation, upperBodyRotationMatrix)
          const localHeadDir = Vector3.TransformNormal(headDir, upperBodyRotationMatrix.invert())
          const localHeadQuat = Quaternion.FromLookDirectionLH(localHeadDir, Vector3.Up())

          neckBone.setRotationQuaternion(
            Quaternion.Slerp(neckBone.rotationQuaternion || new Quaternion(), localHeadQuat, lerpFactor),
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
          let handDir = indexFinger.subtract(wrist).normalize()

          // Ensure Z-axis is always pointing forward
          handDir.z = Math.abs(handDir.z)

          // Correct X-axis direction based on the side
          handDir.x = side === "left" ? -Math.abs(handDir.x) : Math.abs(handDir.x)

          handDir = handDir.normalize()

          // Get lower arm rotation
          const lowerArmRotation = lowerArmBone.rotationQuaternion || new Quaternion()
          const lowerArmRotationMatrix = new Matrix()
          Matrix.FromQuaternionToRef(lowerArmRotation, lowerArmRotationMatrix)

          // Transform hand direction to local space
          const localHandDir = Vector3.TransformNormal(handDir, lowerArmRotationMatrix.invert())

          // Define default direction
          const defaultDir = new Vector3(side === "left" ? -1 : 1, 0, 0)

          // Calculate rotation
          const rotationQuaternion = Quaternion.FromUnitVectorsToRef(defaultDir, localHandDir, new Quaternion())

          // Apply rotation with lerp for smooth transition
          handBone.setRotationQuaternion(
            Quaternion.Slerp(handBone.rotationQuaternion || new Quaternion(), rotationQuaternion, lerpFactor),
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
  }, [pose])
  return <canvas ref={canvasRef} className="scene"></canvas>
}

export default MMDScene
