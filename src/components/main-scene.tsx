"use client"

import { useRef, useEffect, useCallback, useState, type ChangeEvent, type InputHTMLAttributes } from "react"
import Link from "next/link"
import { FolderOpen, Github, RotateCcw, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { PILL, ICON_BUTTON } from "@/lib/chrome"

import {
  Engine,
  type AnimationClip,
  EngineStats,
  MaterialPresetMap,
  Model,
  Quat,
  Vec3,
  parsePmxFolderInput,
  pmxFileAtRelativePath,
} from "reze-engine"

import { MotionCapture } from "./motion-capture"
import Loading from "./loading"

/** The captured clip is registered under its own name and never played, so
 *  exporting cannot disturb the pose the user is driving live. */
const EXPORT_CLIP_NAME = "mikapo-capture"
import { BoneState, SOLVER_REST_BONES, type BodyCollider } from "@/lib/solver"
import { clearUploads, hasStoredUploads, loadModelUpload, saveModelUpload } from "@/lib/asset-store"
import { FaceSolverResult } from "@/lib/face-blendshape-solver"
import { ASSETS } from "@/lib/assets"

/** Stable engine key for the bundled default PMX — folder uploads swap via removeModel + new id. */
const DEFAULT_MODEL_KEY = "mikapo"

// Whether this build ships the demo model (absent = on). Set
// NEXT_PUBLIC_USE_DEFAULT_ASSETS=false to boot empty; parsed leniently, same
// convention as reze-design. Read at build time (NEXT_PUBLIC_ inlines it).
const NO = ["false", "0", "off", "no"]
const USE_DEFAULT_ASSETS = !NO.includes((process.env.NEXT_PUBLIC_USE_DEFAULT_ASSETS ?? "").trim().toLowerCase())

/** Style-group hints for the bundled 塞尔凯特 PMX (exact material names). Fed to
 *  `engine.autoStyleGroups` as overrides: these win, then the engine's built-in
 *  JP/CN/EN name hints fill in anything else — so an arbitrary standard MMD
 *  upload still auto-styles even though we only enumerate the default model here. */
const DEFAULT_STYLE_OVERRIDES: MaterialPresetMap = {
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
}

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
  const [restPose, setRestPose] = useState<Record<string, Vec3> | null>(null)
  const [colliders, setColliders] = useState<BodyCollider[] | null>(null)
  const [modelMorphs, setModelMorphs] = useState<string[] | null>(null)
  const [mediaPipeReady, setMediaPipeReady] = useState(false)
  /** After `engine.init()` — folder picker is safe (loadModel still async for default PMX). */
  const [engineInited, setEngineInited] = useState(false)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [stats, setStats] = useState<EngineStats | null>(null)
  const [pmxPickFiles, setPmxPickFiles] = useState<File[] | null>(null)
  const [pmxPickPaths, setPmxPickPaths] = useState<string[]>([])
  const [pmxPickSelected, setPmxPickSelected] = useState("")
  /** Whether anything the user uploaded is being remembered — the only case
   *  where "back to the demo" is an action rather than a no-op. */
  const [hasUploads, setHasUploads] = useState(false)
  useEffect(() => {
    void hasStoredUploads().then(setHasUploads)
  }, [])

  /** Forget both uploads and boot clean. A reload is the honest way to do it:
   *  the engine, the solver and the capture pipeline all read their assets
   *  once at start, and unwinding that by hand would be three chances to
   *  leave something stale. */
  const resetToDemo = useCallback(async () => {
    await clearUploads()
    window.location.reload()
  }, [])

  // Build a rest-pose dict from the model's bone world positions. Solver uses
  // these to derive per-bone reference directions instead of the static defaults.
  /** Static orbit centre at the character's rest センター plus a small lift.
   *  センター carries the performance now — height from grounding, distance
   *  from the depth rebuild — and a camera that follows it subtracts that
   *  motion right back out of the frame. */
  const frameModel = useCallback(
    (model: Model) => {
      try {
        const c = model.getBoneWorldPosition("センター")
        if (c) engineRef.current?.setCameraTarget(new Vec3(c.x, c.y + 3, c.z))
      } catch {
        // bone missing — keep the engine's default framing
      }
    },
    [engineRef],
  )

  const buildRestPose = useCallback((model: Model) => {
    const dict: Record<string, Vec3> = {}
    for (const name of SOLVER_REST_BONES) {
      try {
        const p = model.getBoneWorldPosition(name)
        if (p) dict[name] = new Vec3(p.x, p.y, p.z)
      } catch {
        // bone missing — solver falls back to DEFAULT_REFS
      }
    }
    setRestPose(dict)

    // The model's own rigid bodies double as its body volume: the author already
    // shaped capsules to fit this character. The solver uses them to keep arms
    // out of the chest (MMD physics never tests these pairs — they are all
    // bone-following statics, so the broadphase drops them).
    const bones = model.getSkeleton().bones
    setColliders(
      model.getRigidbodies().map((rb) => ({
        bone: bones[rb.boneIndex]?.name ?? "",
        shape: rb.shape as number,
        size: { x: rb.size.x, y: rb.size.y, z: rb.size.z },
        position: { x: rb.shapePosition.x, y: rb.shapePosition.y, z: rb.shapePosition.z },
      })),
    )

    // Morph list for blendshape mapping resolution. reze-engine keeps this
    // private today — worth upstreaming a public getMorphNames() (resetAllMorphs
    // already iterates the same data).
    try {
      const morphs = (model as unknown as { morphing?: { morphs?: { name: string }[] } }).morphing?.morphs
      setModelMorphs(morphs ? morphs.map((m) => m.name) : null)
    } catch {
      setModelMorphs(null)
    }
  }, [])

  /**
   * Load a PMX from picked (or restored) folder files. Returns whether the
   * model is up. A live pick persists to IndexedDB so it survives a refresh;
   * the restore path passes `persist: false` (the bytes are already there)
   * and `quiet: true` (an evicted or stale store must not alert — the app
   * just boots the default).
   */
  const loadPmxFromFolder = useCallback(
    async (files: File[], pmxFile: File, opts?: { persist?: boolean; quiet?: boolean }): Promise<boolean> => {
      const engine = engineRef.current
      if (!engine) {
        if (!opts?.quiet) window.alert("Viewport is not ready yet. Wait for initialization, then try again.")
        return false
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
        frameModel(model)
        await new Promise((resolve) => requestAnimationFrame(resolve))
        model.setName(stem)
        modelRef.current = model
        loadedModelNameRef.current = instanceKey
        await engine.autoStyleGroups(loadedModelNameRef.current, DEFAULT_STYLE_OVERRIDES)
        buildRestPose(model)
        frameModel(model)
        setModelLoaded(true)
        setEngineError(null)
        if (opts?.persist !== false) {
          void saveModelUpload(files, pmxFile, stem)
          setHasUploads(true)
        }
        return true
      } catch (e) {
        console.error("[pmx-upload] loadModel failed:", e)
        if (!opts?.quiet) window.alert(e instanceof Error ? e.message : String(e))
        return false
      }
    },
    [buildRestPose, frameModel],
  )

  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      try {
        const engine = new Engine(canvasRef.current, {
          bloom: { color: new Vec3(0.5, 0.1, 0.9), intensity: 0.03 },
          // Further out than the engine default: a capture is watched whole —
          // raised arms and a deep crouch both have to stay in frame. The
          // target is the chest height of a standard MMD model: framing only
          // once the model's own センター can be read means the first frames
          // are drawn looking somewhere else, and the correction reads as the
          // scene lurching.
          camera: { distance: 30, target: new Vec3(0, 11, 0) },
        })
        engineRef.current = engine
        await engine.init()
        setEngineInited(true)
        // MiKaPo poses the skeleton itself — FK rotations written every frame,
        // no clip playing — so the engine must not also run IK and fight them.
        // The exported motion still carries its own per-chain state for whoever
        // plays it back.
        engine.setIKEnabled(false)
        engine.runRenderLoop(() => {
          setStats(engine.getStats())
        })
        // Before any model: a floor that appears afterwards is one more thing
        // moving in the first second.
        engine.addGround({ diffuseColor: new Vec3(0.9, 0.1, 0.9) })

        // A previously uploaded model survives the refresh — it wins over
        // both the bundled default and the empty boot. Eviction or a failed
        // load falls through to the defaults below.
        const stored = await loadModelUpload()
        if (stored && (await loadPmxFromFolder(stored.files, stored.pmxFile, { persist: false, quiet: true }))) {
          return
        }

        // No bundled model: the scene boots empty (ground only) and the user
        // brings their own PMX via the folder picker.
        if (!USE_DEFAULT_ASSETS) return

        const genBeforeDefault = loadGenerationRef.current
        try {
          const model = await engine.loadModel(DEFAULT_MODEL_KEY, `${ASSETS}/models/塞尔凯特/塞尔凯特.pmx`)
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

          // Frame first, then reveal: the camera reads the model's own センター
          // and the loading pill only clears once it is looking there.
          frameModel(model)
          await engine.autoStyleGroups(loadedModelNameRef.current, DEFAULT_STYLE_OVERRIDES)
          await new Promise((r) => requestAnimationFrame(r))
          buildRestPose(model)
          frameModel(model)
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
  }, [buildRestPose, frameModel, loadPmxFromFolder])

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
    (boneStates: BoneState[], tweenMs: number = 30) => {
      if (!engineRef.current) return
      const pose: Record<string, Quat> = {}
      const moves: Record<string, Vec3> = {}
      for (const bone of boneStates) {
        pose[bone.name] = new Quat(bone.rotation.x, bone.rotation.y, bone.rotation.z, bone.rotation.w)
        // センター and the leg IK bones carry translation — the body's height
        // over the ground and where each foot lands.
        if (bone.translation) {
          moves[bone.name] = new Vec3(bone.translation.x, bone.translation.y, bone.translation.z)
        }
      }
      if (Object.keys(pose).length > 0) {
        modelRef.current?.rotateBones(pose, tweenMs)
      }
      if (Object.keys(moves).length > 0) {
        modelRef.current?.moveBones(moves, tweenMs)
      }
    },
    [engineRef],
  )

  /**
   * Captured motion → a .vmd on disk.
   *
   * The clip goes through the model, so the ENGINE writes the file — the same
   * writer Reze Studio exports through. That is what makes a capture from here
   * open there, and in MMD, without a second VMD implementation to keep correct.
   * The clip is registered under its own name and never played, so the live pose
   * the user is still driving is untouched.
   */
  const exportVmd = useCallback((clip: AnimationClip) => {
    const model = modelRef.current
    if (!model || clip.frameCount === 0) return
    model.loadClip(EXPORT_CLIP_NAME, clip)
    const buffer = model.exportVmd(EXPORT_CLIP_NAME)
    const url = URL.createObjectURL(new Blob([buffer], { type: "application/octet-stream" }))
    const link = document.createElement("a")
    link.href = url
    link.download = `mikapo-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.vmd`
    link.click()
    URL.revokeObjectURL(url)
  }, [])

  const resetModel = useCallback(() => {
    modelRef.current?.resetAllBones()
    modelRef.current?.resetAllMorphs()
  }, [])

  const applyFace = useCallback(
    (faceResult: FaceSolverResult, tweenMs: number = 30) => {
      if (!engineRef.current) return

      // Apply eye bone rotations (左目, 右目)
      if (faceResult.boneStates.length > 0) {
        const pose: Record<string, Quat> = {}
        for (const bone of faceResult.boneStates) {
          pose[bone.name] = new Quat(bone.rotation.x, bone.rotation.y, bone.rotation.z, bone.rotation.w)
        }
        modelRef.current?.rotateBones(pose, tweenMs)
      }

      // Morph weights are already resolved to this model's actual morph names
      // by FaceBlendshapeSolver.configure().
      for (const [name, weight] of Object.entries(faceResult.morphWeights)) {
        modelRef.current?.setMorphWeight(name, weight, tweenMs)
      }
    },
    [engineRef],
  )

  return (
    <main className="relative h-dvh w-full overflow-hidden bg-[#4a044e]">
      <input
        ref={pmxFolderInputRef}
        type="file"
        className="fixed right-0 top-0 -z-10 h-px w-px opacity-0"
        multiple
        {...pmxFolderInputAttrs}
        onChange={onPickPmxFolder}
      />

      {/* Chrome floats over a full-bleed viewport; nothing ever shrinks the
          thing being made. One spacing (0.75rem) repeated is the whole grid. */}
      <div className="pointer-events-none absolute inset-x-3 top-3 z-20 flex items-start gap-2">
        {/* The family wordmark: bare over the viewport, wide-tracked and lit —
            the same title Reze Engine and Reze Rig wear. */}
        <div className="pointer-events-auto flex h-10 shrink-0 items-center pl-1">
          <h1
            // The trailing letter-space that tracking leaves after the last
            // glyph is taken back so the wordmark ends where it looks like it
            // ends. Vertical centring is the flex box's job — h-10 either side
            // of the row, one centre line.
            className="text-base font-light uppercase leading-none tracking-[0.2em] text-white md:text-lg md:tracking-[0.28em] md:-mr-[0.28em]"
            style={{ textShadow: "0 0 20px rgba(255, 255, 255, 0.3), 0 2px 10px rgba(0, 0, 0, 0.5)" }}
          >
            Reze MiPo
          </h1>
        </div>

        <div className="ml-auto flex items-start gap-2">
          {/* Sibling apps — desktop only; on a phone the model needs the room. */}
          <div className={cn(PILL, "pointer-events-auto hidden h-10 items-center gap-1 px-1.5 md:flex")}>
            <span className="px-1.5 font-mono text-[11px] tabular-nums text-muted-foreground">
              {stats ? `${stats.fps} FPS` : "— FPS"}
            </span>
            <div className="h-4 w-px shrink-0 bg-line-strong" />
            {[
              { href: "https://reze.one", label: "Engine" },
              { href: "https://reze.studio", label: "Animation" },
              { href: "https://reze.design", label: "Design" },
            ].map(({ href, label }) => (
              <Button
                key={href}
                variant="ghost"
                size="sm"
                asChild
                className="h-7 rounded-lg px-2 text-xs font-normal text-muted-foreground hover:bg-white/5 hover:text-foreground"
              >
                <Link href={href} target="_blank">
                  {label}
                </Link>
              </Button>
            ))}
          </div>

          <div className={cn(PILL, "pointer-events-auto flex h-10 items-center gap-1 px-1.5")}>
            <Button variant="ghost" size="icon" asChild className={ICON_BUTTON}>
              <Link href="https://github.com/AmyangXYZ/MiKaPo" target="_blank" aria-label="GitHub">
                <Github className="size-4" strokeWidth={1.75} />
              </Link>
            </Button>
            {hasUploads && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className={ICON_BUTTON}
                    onClick={() => void resetToDemo()}
                    aria-label="Back to the demo model and video"
                  >
                    <RotateCcw className="size-4" strokeWidth={1.75} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Back to the demo model and video</TooltipContent>
              </Tooltip>
            )}
            <Button
              type="button"
              size="sm"
              disabled={!engineInited}
              className="h-7 gap-1.5 rounded-lg bg-blue-400 px-2 text-xs font-medium text-white hover:bg-blue-300 disabled:opacity-50 has-[>svg]:px-2"
              onClick={() => pmxFolderInputRef.current?.click()}
            >
              <FolderOpen className="size-3.5" strokeWidth={1.75} />
              <span className="hidden sm:inline">Your Model</span>
            </Button>
          </div>
        </div>
      </div>

      {pmxPickDialogOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <button
            type="button"
            aria-label="Dismiss"
            className="absolute inset-0 bg-scrim backdrop-blur-xs"
            onClick={dismissPmxPickDialog}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="pmx-picker-title"
            className="relative z-[1] w-full max-w-md rounded-surface border border-line-strong bg-surface-raised p-5 text-foreground shadow-float backdrop-blur-xs"
          >
            <div className="mb-1 flex items-start justify-between gap-3">
              <h2 id="pmx-picker-title" className="text-sm font-semibold tracking-tight">
                Multiple .pmx files in folder
              </h2>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="-mr-1 -mt-1 size-7 shrink-0 rounded-lg text-muted-foreground hover:bg-white/5 hover:text-foreground"
                aria-label="Close"
                onClick={dismissPmxPickDialog}
              >
                <X className="size-4" />
              </Button>
            </div>
            <p className="mb-4 text-xs text-muted-foreground">Pick which model to load.</p>
            <select
              className="mb-5 w-full rounded-interior border border-line-strong bg-white/5 px-2.5 py-2 text-sm text-foreground outline-none transition-colors hover:bg-white/10 focus-visible:border-blue-400/50"
              value={pmxPickSelected}
              onChange={(ev) => setPmxPickSelected(ev.target.value)}
            >
              {pmxPickPaths.map((p) => (
                <option key={p} value={p} className="bg-zinc-900 text-foreground">
                  {p}
                </option>
              ))}
            </select>
            <div className="flex flex-row justify-end gap-2">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-8 rounded-lg text-xs text-muted-foreground hover:bg-white/5 hover:text-foreground"
                onClick={dismissPmxPickDialog}
              >
                Cancel
              </Button>
              <Button
                type="button"
                size="sm"
                className="h-8 rounded-lg bg-blue-400 text-xs font-medium text-white hover:bg-blue-300"
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
        restPose={restPose}
        colliders={colliders}
        modelMorphs={modelMorphs}
        exportVmd={exportVmd}
        onUploadStored={() => setHasUploads(true)}
      />

      {/* One message at a time. A failed boot leaves `modelLoaded` false, so the
          loader kept counting dots underneath the error that explained why it
          never would finish — two centred overlays, both unreadable. */}
      {engineError ? (
        <div className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center p-6">
          <div className="max-w-md rounded-surface border border-red-400/30 bg-surface-raised px-5 py-4 text-center text-sm leading-relaxed text-red-400 shadow-float backdrop-blur-xs">
            {engineError}
          </div>
        </div>
      ) : (
        <Loading modelLoaded={modelLoaded} mediaPipeReady={mediaPipeReady} />
      )}
      <canvas ref={canvasRef} className="absolute inset-0 z-0 h-full w-full touch-none outline-none" />
    </main>
  )
}
