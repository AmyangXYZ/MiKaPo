"use client"

import { useRef, useEffect, useCallback, useState } from "react"
import Image from "next/image"
import Link from "next/link"
import { Button } from "@/components/ui/button"

import {
  ArcRotateCamera,
  Color3,
  Color4,
  CreateDisc,
  CreateScreenshotAsync,
  DirectionalLight,
  Engine,
  HemisphericLight,
  LoadAssetContainerAsync,
  Material,
  Mesh,
  Quaternion,
  RegisterSceneLoaderPlugin,
  Scene,
  ShadowGenerator,
  Space,
  StandardMaterial,
  Vector3,
} from "@babylonjs/core"
import {
  MmdWasmModel,
  SdefInjector,
  MmdWasmInstanceTypeMPR,
  GetMmdWasmInstance,
  MmdWasmRuntime,
  MmdWasmPhysics,
  type IMmdWasmInstance,
  MmdStandardMaterialBuilder,
  MmdStandardMaterial,
  BpmxLoader,
} from "babylon-mmd"
import { IMmdRuntimeLinkedBone } from "babylon-mmd/esm/Runtime/IMmdRuntimeLinkedBone"

import { MotionCapture } from "./motion-capture"
import { BoneState, KeyBones } from "@/lib/solver"
import ModelsPanel from "./models-panel"
import { Aperture, User } from "lucide-react"
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip"

export default function MainScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine>(null)
  const sceneRef = useRef<Scene>(null)
  const cameraRef = useRef<ArcRotateCamera>(null)
  const shadowGeneratorRef = useRef<ShadowGenerator>(null)
  const mmdWasmInstanceRef = useRef<IMmdWasmInstance>(null)
  const mmdRuntimeRef = useRef<MmdWasmRuntime>(null)
  const mmdMaterialBuilderRef = useRef<MmdStandardMaterialBuilder>(null)
  const modelRef = useRef<MmdWasmModel>(null)
  const bonesRef = useRef<{ [key: string]: IMmdRuntimeLinkedBone }>({})
  const [modelLoaded, setModelLoaded] = useState(false)
  const [lerp, setLerp] = useState(0.7)

  const [openModelsPanel, setOpenModelsPanel] = useState(false)
  const modelNameRef = useRef(localStorage.getItem("selectedModel") || "深空之眼-梵天")



  const rotateBone = useCallback((name: string, quaternion: Quaternion) => {
    const bone = bonesRef.current[name]
    if (!bone) return
    bone.setRotationQuaternion(Quaternion.Slerp(bone.rotationQuaternion, quaternion, lerp), Space.LOCAL)
  }, [lerp])

  const loadModel = useCallback(async (): Promise<void> => {
    if (!sceneRef.current || !mmdWasmInstanceRef.current || !mmdRuntimeRef.current || !shadowGeneratorRef.current) return
    if (modelRef.current) {
      mmdRuntimeRef.current.destroyMmdModel(modelRef.current)
      modelRef.current.mesh.dispose()
    }

    LoadAssetContainerAsync(`/models/${modelNameRef.current}.bpmx`, sceneRef.current!, {
      pluginOptions: {
        mmdmodel: {
          materialBuilder: mmdMaterialBuilderRef.current || undefined,
        },
      },
    }).then(async (result) => {
      const mesh = result.meshes[0]
      for (const m of mesh.metadata.meshes) {
        m.receiveShadows = true
      }
      shadowGeneratorRef.current!.addShadowCaster(mesh)
      modelRef.current = mmdRuntimeRef.current!.createMmdModel(mesh as Mesh, {
        buildPhysics: {
          disableOffsetForConstraintFrame: true,
        },
      })

      for (const bone of modelRef.current!.skeleton.bones) {
        if (KeyBones.includes(bone.name)) {
          bonesRef.current[bone.name] = bone
        }
      }

      // setTimeout(() => {
      //   modelRef.current!.runtimeBones.forEach((bone) => {
      //     if (KeyBones.includes(bone.name)) {
      //       const worldMatrix = bone.worldMatrix
      //       const position = new Vector3(worldMatrix[12], worldMatrix[13], worldMatrix[14])

      //       const childBones = bone.childBones
      //       if (childBones.length > 0) {
      //         const childBone = childBones[0]
      //         const childWorldMatrix = childBone.worldMatrix
      //         const childPosition = new Vector3(childWorldMatrix[12], childWorldMatrix[13], childWorldMatrix[14])
      //         const direction = childPosition.subtract(position).normalize()
      //         console.log(bone.name, childBone.name, `(${direction.x}, ${direction.y}, ${direction.z})`)
      //       }
      //     }
      //   })
      // }, 1000)

      result.addAllToScene()
      setModelLoaded(true)
    })
  }, [])

  const selectModel = useCallback(
    (model: string) => {
      modelNameRef.current = model
      localStorage.setItem("selectedModel", model)
      loadModel()
    },
    [loadModel]
  )


  useEffect(() => {
    const resize = () => {
      if (sceneRef.current) {
        engineRef.current?.resize()
      }
    }

    const createScene = async () => {
      if (!canvasRef.current) return

      RegisterSceneLoaderPlugin(new BpmxLoader())

      const engine = new Engine(canvasRef.current, true, {}, true)
      SdefInjector.OverrideEngineCreateEffect(engine)

      const scene = new Scene(engine)

      scene.clearColor = new Color4(0.99, 0.44, 0.66, 1.0)
      scene.ambientColor = new Color3(0.5, 0.5, 0.5)

      engineRef.current = engine
      sceneRef.current = scene

      const camera = new ArcRotateCamera("ArcRotateCamera", 0, 0, 45, new Vector3(0, 12, 0), scene)
      camera.setPosition(new Vector3(0, 19, -25))
      camera.attachControl(canvasRef.current, false)
      camera.inertia = 0.8
      camera.speed = 10
      cameraRef.current = camera

      scene.activeCameras = [camera]

      const hemisphericLight = new HemisphericLight("hemisphericLight", new Vector3(0, 1, 0), scene);
      hemisphericLight.intensity = 0.5;
      hemisphericLight.specular = new Color3(0, 0, 0);
      hemisphericLight.groundColor = new Color3(1, 1, 1);

      const directionalLight = new DirectionalLight("directionalLight", new Vector3(0.5, -1, 1), scene);
      directionalLight.intensity = 0.5;
      directionalLight.autoCalcShadowZBounds = false;
      directionalLight.autoUpdateExtends = false;
      directionalLight.shadowMaxZ = 20 * 3;
      directionalLight.shadowMinZ = -30;
      directionalLight.orthoTop = 18 * 3;
      directionalLight.orthoBottom = -1 * 3;
      directionalLight.orthoLeft = -10 * 3;
      directionalLight.orthoRight = 10 * 3;
      directionalLight.shadowOrthoScale = 0;

      const shadowGenerator = new ShadowGenerator(2048, directionalLight, true, camera);
      shadowGenerator.transparencyShadow = true;
      shadowGenerator.usePercentageCloserFiltering = true;
      shadowGenerator.forceBackFacesOnly = false;
      shadowGenerator.bias = 0.01;
      shadowGenerator.filteringQuality = ShadowGenerator.QUALITY_MEDIUM;
      shadowGenerator.frustumEdgeFalloff = 0.1;
      shadowGeneratorRef.current = shadowGenerator

      mmdWasmInstanceRef.current = await GetMmdWasmInstance(new MmdWasmInstanceTypeMPR())
      const mmdRuntime = new MmdWasmRuntime(mmdWasmInstanceRef.current, scene, new MmdWasmPhysics(scene))
      mmdRuntime.register(scene)
      mmdRuntimeRef.current = mmdRuntime

      const ground = CreateDisc("stageGround", { radius: 12, tessellation: 64 }, scene)
      const groundMaterial = new StandardMaterial("groundMaterial", scene)
      groundMaterial.diffuseColor = new Color3(0.95, 0.98, 1.0)
      groundMaterial.emissiveColor = new Color3(0.1, 0.15, 0.25)
      groundMaterial.specularColor = new Color3(0.2, 0.3, 0.5)
      ground.material = groundMaterial
      ground.rotation.x = Math.PI / 2
      ground.receiveShadows = true
      const materialBuilder = new MmdStandardMaterialBuilder()

      materialBuilder.afterBuildSingleMaterial = (material: MmdStandardMaterial): void => {
        material.forceDepthWrite = true
        material.useAlphaFromDiffuseTexture = true
        material.specularColor = new Color3(0, 0, 0)
        if (material.diffuseTexture !== null) material.diffuseTexture.hasAlpha = true

        if (material.transparencyMode === Material.MATERIAL_ALPHABLEND) {
          material.transparencyMode = Material.MATERIAL_ALPHATESTANDBLEND
          material.alphaCutOff = 0.01
        }
      }
      mmdMaterialBuilderRef.current = materialBuilder

      loadModel()

      window.addEventListener("resize", resize)

      engine.runRenderLoop(() => {
        scene.render()
      })
    }
    createScene()

    return () => {
      engineRef.current?.dispose()
      window.removeEventListener("resize", resize)
    }
  }, [loadModel])

  const applyPose = useCallback(
    (boneStates: BoneState[]) => {
      if (!modelRef.current) return
      for (const boneState of boneStates) {
        rotateBone(boneState.name, boneState.rotation)
      }
    },
    [rotateBone]
  )


  const takeScreenshot = useCallback(() => {
    if (!canvasRef.current || !engineRef.current || !cameraRef.current) return
    CreateScreenshotAsync(engineRef.current!, cameraRef.current!, { precision: 1 }).then((b64) => {
      const link = document.createElement("a")
      link.href = b64
      link.download = "mikapo_screenshot.png"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    })
  }, [canvasRef])

  return (
    <div className="w-full h-full">
      <Button
        size="icon"
        asChild
        className="absolute top-3 right-3 bg-white text-black size-7 rounded-full hover:bg-gray-200"
      >
        <Link href="https://github.com/AmyangXYZ/MiKaPo" target="_blank">
          <Image src="/github-mark.svg" alt="GitHub" width={18} height={18} />
        </Link>
      </Button>
      <div className="absolute flex justify-end top-[50%] -translate-y-1/2 right-0 mx-auto flex px-4 z-20">
        <div className="flex flex-col items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="icon"
                className="bg-white text-black size-7 rounded-full hover:bg-pink-100 cursor-pointer"
                onClick={() => setOpenModelsPanel(true)}
              >
                <User />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">
              <p>Switch Models</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="icon"
                className="bg-white text-black size-7 rounded-full hover:bg-pink-100 cursor-pointer"
                onClick={() => takeScreenshot()}
              >
                <Aperture />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">
              <p>Take Screenshot</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      <ModelsPanel
        open={openModelsPanel}
        setOpen={setOpenModelsPanel}
        selectedModel={modelNameRef.current}
        selectModel={selectModel}
      />
      <MotionCapture applyPose={applyPose} modelLoaded={modelLoaded} setLerp={setLerp} />
      <canvas ref={canvasRef} className="w-full h-full z-1 outline-none" />
    </div>
  )
}
