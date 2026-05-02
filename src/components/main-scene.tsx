"use client"

import { useRef, useEffect, useCallback, useState, type ChangeEvent, type InputHTMLAttributes } from "react"
import Image from "next/image"
import Link from "next/link"
import { FolderOpen, X } from "lucide-react"
import { Button } from "@/components/ui/button"

import { Engine, EngineStats, Model, Quat, Vec3, parsePmxFolderInput, pmxFileAtRelativePath } from "reze-engine"

import { MotionCapture } from "./motion-capture"
import Loading from "./loading"
import { BoneState } from "@/lib/solver"
import { FaceSolverResult } from "@/lib/face-blendshape-solver"

/** Stable engine key for the bundled default PMX — folder uploads swap via removeModel + new id. */
const DEFAULT_MODEL_KEY = "mikapo"

function fileStem(filename: string) {
  const i = filename.lastIndexOf(".")
  return i >= 0 ? filename.slice(0, i) : filename
}

/** webkitdirectory attrs — cast kept outside JSX so `<` is not parsed as a tag */
const pmxFolderInputAttrs = {
  webkitdirectory: "",
  mozdirectory: "",
} as InputHTMLAttributes<HTMLInputElement>

export default function MainScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const modelRef = useRef<Model | null>(null)
  const engineRef = useRef<Engine | null>(null)
  /** Engine registry name for removeModel when replacing the avatar */
  const loadedModelNameRef = useRef(DEFAULT_MODEL_KEY)
  const pmxFolderInputRef = useRef<HTMLInputElement>(null)
  /** Bumped on folder upload so a still-in-flight default `loadModel` can discard its result. */
  const loadGenerationRef = useRef(0)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [mediaPipeReady, setMediaPipeReady] = useState(false)
  /** After `engine.init()` — folder picker is safe (loadModel still async for default PMX). */
  const [engineInited, setEngineInited] = useState(false)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [stats, setStats] = useState<EngineStats | null>(null)
  const [pmxPickFiles, setPmxPickFiles] = useState<File[] | null>(null)
  const [pmxPickPaths, setPmxPickPaths] = useState<string[]>([])
  const [pmxPickSelected, setPmxPickSelected] = useState("")

  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      try {
        const engine = new Engine(canvasRef.current, { bloom: { color: new Vec3(0.5, 0.1, 0.9), intensity: 0.03 } })
        engineRef.current = engine
        await engine.init()
        setEngineInited(true)
        engine.addGround({ diffuseColor: new Vec3(0.9, 0.1, 0.9) })
        engine.setIKEnabled(false)
        engine.runRenderLoop(() => {
          setStats(engine.getStats())
        })

        const genBeforeDefault = loadGenerationRef.current
        try {
          const model = await engine.loadModel(DEFAULT_MODEL_KEY, "/models/塞尔凯特/塞尔凯特.pmx")
          if (genBeforeDefault !== loadGenerationRef.current) {
            try {
              engine.removeModel(DEFAULT_MODEL_KEY)
            } catch {
              /* raced folder upload already replaced registry */
            }
            return
          }

          modelRef.current = model
          loadedModelNameRef.current = DEFAULT_MODEL_KEY
          console.log(model.getMaterials())
          engine.setMaterialPresets(loadedModelNameRef.current, {
            eye: ["眼睛", "眼白", "目白", "右瞳", "左瞳", "眉毛", "eyebrow", "eyelash"],
            face: ["脸", "face01"],
            body: ["皮肤", "skin"],
            hair: ["头发", "hair_f"],
            cloth_smooth: [
              "衣服",
              "裙子",
              "裙带",
              "裙布",
              "外套",
              "外套饰",
              "裤子",
              "裤子0",
              "腿环",
              "发饰",
              "鞋子",
              "鞋子饰",
              "shirt",
              "shoes",
              "shorts",
              "trigger",
              "dress",
              "hair_accessory",
              "cloth01_shoes",
            ],
            stockings: ["袜子", "stockings"],
            metal: ["metal01", "earring"],
          })
          setModelLoaded(true)
          setEngineError(null)
        } catch (loadErr) {
          setEngineError(loadErr instanceof Error ? loadErr.message : "Unknown error")
        }

        // await engine.loadAnimation("/mikapo_animation.vmd")
        // engine.playAnimation()
      } catch (error) {
        setEngineError(error instanceof Error ? error.message : "Unknown error")
      }
    }
  }, [])

  useEffect(() => {
    void (async () => {
      initEngine()
    })()

    // Cleanup on unmount
    return () => {
      if (engineRef.current) {
        engineRef.current.dispose()
      }
    }
  }, [initEngine])

  const loadPmxFromFolder = useCallback(async (files: File[], pmxFile: File) => {
    const engine = engineRef.current
    if (!engine) {
      window.alert("Viewport is not ready yet. Wait for initialization, then try again.")
      return
    }
    loadGenerationRef.current += 1
    const stem = fileStem(pmxFile.name)
    const instanceKey = `u_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`
    try {
      try {
        engine.removeModel(loadedModelNameRef.current)
      } catch {
        /* removeModel no-op if name stale */
      }
      const model = await engine.loadModel(instanceKey, { files, pmxFile })
      await new Promise((resolve) => requestAnimationFrame(resolve))
      model.setName(stem)
      modelRef.current = model
      loadedModelNameRef.current = instanceKey
      engine.setMaterialPresets(loadedModelNameRef.current, {
        eye: ["眼睛", "眼白", "目白", "右瞳", "左瞳", "眉毛", "eyebrow", "eyelash"],
        face: ["脸", "face01"],
        body: ["皮肤", "skin"],
        hair: ["头发", "hair_f"],
        cloth_smooth: [
          "衣服",
          "裙子",
          "裙带",
          "裙布",
          "外套",
          "外套饰",
          "裤子",
          "裤子0",
          "腿环",
          "发饰",
          "鞋子",
          "鞋子饰",
          "shirt",
          "shoes",
          "shorts",
          "trigger",
          "dress",
          "hair_accessory",
          "cloth01_shoes",
        ],
        stockings: ["袜子", "stockings"],
        metal: ["metal01", "earring"],
      })
      setModelLoaded(true)
      setEngineError(null)
    } catch (e) {
      console.error("[pmx-upload] loadModel failed:", e)
      window.alert(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const onPickPmxFolder = useCallback(
    async (e: ChangeEvent<HTMLInputElement>) => {
      try {
        const picked = parsePmxFolderInput(e.target.files)
        e.target.value = ""

        if (picked.status === "empty") return
        if (picked.status === "not_directory") {
          window.alert("Please select a folder, not individual files.")
          return
        }
        if (picked.status === "no_pmx") {
          window.alert("No .pmx file in the selected folder.")
          return
        }

        setPmxPickFiles(null)
        setPmxPickPaths([])
        setPmxPickSelected("")

        if (picked.status === "single") {
          await loadPmxFromFolder(picked.files, picked.pmxFile)
        } else {
          setPmxPickFiles(picked.files)
          setPmxPickPaths(picked.pmxRelativePaths)
          setPmxPickSelected(picked.pmxRelativePaths[0] ?? "")
        }
      } catch (err) {
        console.error("[pmx-folder]", err)
        window.alert(err instanceof Error ? err.message : String(err))
      }
    },
    [loadPmxFromFolder],
  )

  const onConfirmPmxPick = useCallback(async () => {
    const files = pmxPickFiles
    const path = pmxPickSelected
    if (!files || !path) return
    const pmxFile = pmxFileAtRelativePath(files, path)
    if (!pmxFile) {
      window.alert("Could not find the selected PMX file.")
      return
    }
    await loadPmxFromFolder(files, pmxFile)
    setPmxPickFiles(null)
    setPmxPickPaths([])
    setPmxPickSelected("")
  }, [loadPmxFromFolder, pmxPickFiles, pmxPickSelected])

  const dismissPmxPickDialog = useCallback(() => {
    setPmxPickFiles(null)
    setPmxPickPaths([])
    setPmxPickSelected("")
  }, [])

  const pmxPickDialogOpen = Boolean(pmxPickFiles && pmxPickPaths.length > 1)

  useEffect(() => {
    if (!pmxPickDialogOpen) return
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") dismissPmxPickDialog()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [dismissPmxPickDialog, pmxPickDialogOpen])

  const applyPose = useCallback(
    (boneStates: BoneState[]) => {
      if (!engineRef.current) return
      const pose: Record<string, Quat> = {}
      for (const bone of boneStates) {
        pose[bone.name] = new Quat(bone.rotation.x, bone.rotation.y, bone.rotation.z, bone.rotation.w)
      }
      if (Object.keys(pose).length > 0) {
        modelRef.current?.rotateBones(pose, 30)
      }
    },
    [engineRef],
  )

  const resetModel = useCallback(() => {
    modelRef.current?.resetAllBones()
    modelRef.current?.resetAllMorphs()
  }, [])

  const applyFace = useCallback(
    (faceResult: FaceSolverResult) => {
      if (!engineRef.current) return

      // Apply eye bone rotations (左目, 右目)
      if (faceResult.boneStates.length > 0) {
        const pose: Record<string, Quat> = {}
        for (const bone of faceResult.boneStates) {
          pose[bone.name] = new Quat(bone.rotation.x, bone.rotation.y, bone.rotation.z, bone.rotation.w)
        }
        modelRef.current?.rotateBones(pose, 30)
      }

      // Apply morph weights to MMD model
      const morphWeights = faceResult.morphWeights

      // Eye morphs
      modelRef.current?.setMorphWeight("まばたき", morphWeights.まばたき, 30)
      modelRef.current?.setMorphWeight("ウィンク", morphWeights.ウィンク, 30)
      modelRef.current?.setMorphWeight("ウィンク右", morphWeights.ウィンク右, 30)

      // Mouth morphs
      modelRef.current?.setMorphWeight("あ", morphWeights.あ, 30)
      modelRef.current?.setMorphWeight("ワ", morphWeights.ワ, 30)
    },
    [engineRef],
  )

  return (
    <div className="w-full h-full">
      <input
        ref={pmxFolderInputRef}
        type="file"
        className="fixed right-0 top-0 -z-10 h-px w-px opacity-0"
        multiple
        {...pmxFolderInputAttrs}
        onChange={onPickPmxFolder}
      />

      <div className="absolute right-0 top-0 z-10 flex w-full flex-row items-center justify-end gap-4 p-4">
        <div className="flex min-w-0 max-w-full shrink-0 flex-row flex-wrap items-center justify-end gap-x-3 gap-y-2">
          {stats && (
            <div className="z-10 flex h-8 mt-0.5 shrink-0 items-center text-sm font-medium leading-none text-white tabular-nums">
              FPS: {stats.fps}
            </div>
          )}

          <div className="flex flex-row items-center gap-2 hidden sm:flex">
            <Button variant="link" size="sm" asChild className="sm:hidden md:flex z-10 text-white underline">
              <Link href="https://reze.one" target="_blank">
                <span className="text-sm">Rendering Engine</span>
              </Link>
            </Button>

            <Button variant="link" size="sm" asChild className="sm:hidden md:flex z-10 text-white underline">
              <Link href="https://reze.studio" target="_blank">
                <span className="text-sm">Animation Editor</span>
              </Link>
            </Button>
          </div>

          <div className="hidden shrink-0 flex-row items-center gap-2 sm:flex">
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={!engineInited}
              className="h-8 shrink-0 gap-1.5 bg-white/90 text-xs text-foreground hover:bg-white disabled:opacity-50"
              onClick={() => pmxFolderInputRef.current?.click()}
            >
              <FolderOpen className="size-4" />
              Use Your Model
            </Button>

            <Button
              size="icon"
              asChild
              className="bg-white text-black size-7 shrink-0 rounded-full z-10 hover:bg-gray-200"
            >
              <Link href="https://github.com/AmyangXYZ/MiKaPo" target="_blank">
                <Image src="/github-mark.svg" alt="GitHub" width={18} height={18} />
              </Link>
            </Button>
          </div>
        </div>
      </div>

      {pmxPickDialogOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <button
            type="button"
            aria-label="Dismiss"
            className="absolute inset-0 bg-black/55 backdrop-blur-[2px]"
            onClick={dismissPmxPickDialog}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="pmx-picker-title"
            className="relative z-[1] w-full max-w-md rounded-xl border border-white/15 bg-zinc-950/95 p-4 text-white shadow-2xl shadow-black/50 backdrop-blur-md"
          >
            <div className="mb-3 flex items-start justify-between gap-3">
              <h2 id="pmx-picker-title" className="text-sm font-semibold leading-tight">
                Multiple .pmx files in folder
              </h2>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-8 shrink-0 text-white hover:bg-white/10"
                aria-label="Close"
                onClick={dismissPmxPickDialog}
              >
                <X className="size-4" />
              </Button>
            </div>
            <p className="mb-2 text-xs text-white/70">Pick which model to load.</p>
            <select
              className="mb-4 w-full rounded-md border border-white/25 bg-black/50 px-2 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-white/30"
              value={pmxPickSelected}
              onChange={(ev) => setPmxPickSelected(ev.target.value)}
            >
              {pmxPickPaths.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <div className="flex flex-row justify-end gap-2">
              <Button type="button" variant="secondary" size="sm" onClick={dismissPmxPickDialog}>
                Cancel
              </Button>
              <Button
                type="button"
                size="sm"
                className="bg-white text-black hover:bg-white/90"
                onClick={() => void onConfirmPmxPick()}
              >
                Load selected
              </Button>
            </div>
          </div>
        </div>
      ) : null}

      <MotionCapture
        applyPose={applyPose}
        applyFace={applyFace}
        modelLoaded={modelLoaded}
        onMediaPipeReadyChange={setMediaPipeReady}
        resetModel={resetModel}
      />

      <Loading modelLoaded={modelLoaded} mediaPipeReady={mediaPipeReady} />

      {engineError && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 font-medium text-white z-10">
          {engineError}
        </div>
      )}
      <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full z-1 outline-none" />
    </div>
  )
}
