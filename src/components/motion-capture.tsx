import { useEffect, useRef, useState, useCallback } from "react"
import Image from "next/image"
import { BoneState, Solver, type BodyCollider } from "@/lib/solver"
import { FaceBlendshapeSolver, FaceSolverResult, FaceMorphWeights } from "@/lib/face-blendshape-solver"
import { buildClip, clipSummary, RecordedFrame } from "@/lib/vmd"
import { smoothTakeZeroPhase } from "@/lib/filters"
import { ASSETS } from "@/lib/assets"
import { loadVideoUpload, saveVideoUpload } from "@/lib/asset-store"
import { Vec3 } from "reze-engine"
import type { PoseWorkerRequest, PoseWorkerResponse, PoseWorkerResult } from "@/lib/pose-worker"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Camera, Image as ImageIcon, Video, Webcam, Pause, Play, Download, Square } from "lucide-react"
import DebugScene from "./debug-scene"

/**
 * One tuning row: name, setting, slider — and for anything with a live signal, a
 * meter under it showing what the solver is currently reading.
 *
 * The meter is the point. A threshold is meaningless on its own; watched against
 * the level it gates, "drag until blinking registers" becomes something you can
 * see rather than guess.
 */

type InputMode = "image" | "video" | "camera" | null

/** Debug skeleton preview refresh (React re-render); the model itself is driven
 * directly from the detection callback and doesn't wait for React. */
const DEBUG_PREVIEW_INTERVAL_MS = 66

/** Copy a solver pose into a reusable buffer. The solver mutates its output
 * array in place every frame, so the display pair must keep its own copies. */
const snapshotPose = (pose: BoneState[], into: BoneState[] | null): BoneState[] => {
  if (!into || into.length !== pose.length) {
    return pose.map((bs) => ({
      name: bs.name,
      rotation: bs.rotation.clone(),
      ...(bs.translation ? { translation: new Vec3(bs.translation.x, bs.translation.y, bs.translation.z) } : {}),
    }))
  }
  for (let i = 0; i < pose.length; i++) {
    into[i].rotation.set(pose[i].rotation)
    const t = pose[i].translation
    const d = into[i].translation
    if (t && d) d.setXYZ(t.x, t.y, t.z)
  }
  return into
}

export const MotionCapture = ({
  applyPose,
  applyFace,
  modelLoaded,
  onMediaPipeReadyChange,
  resetModel,
  restPose,
  colliders,
  modelMorphs,
  exportVmd,
}: {
  applyPose: (boneStates: BoneState[], tweenMs: number) => void
  applyFace: (faceResult: FaceSolverResult, tweenMs: number) => void
  modelLoaded: boolean
  onMediaPipeReadyChange?: (ready: boolean) => void
  resetModel?: () => void
  /** Hands a finished clip to the model, which writes the VMD. The engine owns
   *  the format — the same writer Reze Studio exports through. */
  exportVmd?: (clip: ReturnType<typeof buildClip>) => void
  // MMD rest-pose world bone positions, keyed by Japanese bone name.
  restPose?: Record<string, { x: number; y: number; z: number }> | null
  colliders?: BodyCollider[] | null
  // Morph names present on the loaded model — resolves blendshape mappings.
  modelMorphs?: string[] | null
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const workerRef = useRef<Worker | null>(null)
  const [mediaPipeReady, setMediaPipeReady] = useState(false)
  const [landmarks, setLandmarks] = useState<PoseWorkerResult | null>(null)
  const [inputMode, setInputMode] = useState<InputMode>("video")
  const [isStreamActive, setIsStreamActive] = useState(false)
  // The ?cors suffix skips browser-cache entries saved before these loads went
  // through CORS — the demo assets carry a one-year immutable cache header, and
  // a cached response without access-control-allow-origin fails the CORS check
  // for as long as it lives.
  const [currentImage, setCurrentImage] = useState<string>(`${ASSETS}/4.png?cors`)
  const [videoSrc, setVideoSrc] = useState<string>(`${ASSETS}/Stellar (스텔라) - Vibrato (떨려요)- DANCE COVER.mp4?cors`)
  const [lastMedia, setLastMedia] = useState<"IMAGE" | "VIDEO">("VIDEO")
  // Set the moment the user picks any input this session — a slow IndexedDB
  // restore must never clobber a fresher choice.
  const userPickedMediaRef = useRef(false)
  const solverRef = useRef<Solver>(new Solver())
  const faceBlendshapeSolverRef = useRef<FaceBlendshapeSolver>(new FaceBlendshapeSolver())
  const onMediaPipeReadyChangeRef = useRef(onMediaPipeReadyChange)
  useEffect(() => {
    onMediaPipeReadyChangeRef.current = onMediaPipeReadyChange
  }, [onMediaPipeReadyChange])

  // Tuning. Applied straight to the live solvers — no re-detection needed, so a
  // drag shows its effect on the next frame.
  //
  // Every face control is a SENSITIVITY: 0 needs a deliberate expression, 1
  // triggers readily. The underlying thresholds do not agree on direction — a
  // higher eye-closed ratio triggers a blink sooner, while a higher mouth or
  // smile threshold triggers later — so exposing them raw meant three sliders
  // where "more" meant different things. The mapping lives here; the UI only
  // ever says "more sensitive to the right".
  // Face capture runs with the body. The solver's thresholds are its own
  // business — surfacing them as unlabelled dials asked people to tune what the
  // defaults already handle.
  const faceEnabledRef = useRef(true)

  // A previously uploaded video survives the refresh: restore it as the
  // active source in place of the bundled demo. Absence (or eviction) simply
  // leaves the demo — persistence is a convenience, never a precondition.
  useEffect(() => {
    let cancelled = false
    void loadVideoUpload().then((file) => {
      if (!file || cancelled || userPickedMediaRef.current) return
      setVideoSrc(URL.createObjectURL(file))
      setVideoTime(0)
      setInputMode("video")
      setLastMedia("VIDEO")
    })
    return () => {
      cancelled = true
    }
  }, [])

  // Offline conversion state. Nothing is "recorded" as it plays — the video is
  // stepped frame by frame, so the result does not depend on how fast this
  // machine happens to be.
  const [converting, setConverting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [exported, setExported] = useState<string | null>(null)
  const convertingRef = useRef(false)
  const cancelRef = useRef(false)
  // Set while a conversion frame is in flight; the worker's reply resolves it
  // instead of going down the live path.
  const awaitingRef = useRef<((r: PoseWorkerResult | null) => void) | null>(null)

  // Current pose/face state
  const currentBoneStatesRef = useRef<BoneState[]>([])
  const currentMorphWeightsRef = useRef<FaceMorphWeights | null>(null)

  // Custom video controls — replaces native browser chrome to match the panel style.
  const [videoPlaying, setVideoPlaying] = useState(false)
  const [videoTime, setVideoTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  /** Accept a duration only once it is a real number. When the browser says
   *  Infinity (common for WebM and anything streamed), nudge it into resolving:
   *  seeking far past the end forces the demuxer to find the true end, and the
   *  durationchange that follows carries the answer. Guarded by a flag so the
   *  nudge happens once per file. */
  const durationNudged = useRef(false)
  const resolveDuration = useCallback((video: HTMLVideoElement) => {
    const d = video.duration
    if (Number.isFinite(d) && d > 0) {
      setVideoDuration(d)
      durationNudged.current = false
      return
    }
    if (durationNudged.current) return
    durationNudged.current = true
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked)
      video.currentTime = 0
    }
    video.addEventListener("seeked", onSeeked)
    video.currentTime = 1e9
  }, [])
  // The page is prerendered, so a warm cache can finish loading the video's
  // metadata before React hydrates — loadedmetadata and durationchange then
  // fire with no listener attached and the duration sits at 0:00. On mount,
  // read whatever the element already knows.
  useEffect(() => {
    const v = videoRef.current
    if (v && v.readyState >= 1) resolveDuration(v)
  }, [resolveDuration, inputMode])
  const formatTime = (s: number): string => {
    if (!Number.isFinite(s) || s < 0) return "0:00"
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, "0")}`
  }
  const toggleVideoPlay = () => {
    // The conversion owns the playhead while it steps through the take.
    if (!videoRef.current || convertingRef.current) return
    if (videoRef.current.paused) videoRef.current.play()
    else videoRef.current.pause()
  }

  // Space toggles playback, the way every player does. A focused control keeps
  // its own Space (that is the control's job), and typing is never intercepted.
  const toggleVideoPlayRef = useRef(toggleVideoPlay)
  useEffect(() => {
    toggleVideoPlayRef.current = toggleVideoPlay
  })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space") return
      const t = e.target as HTMLElement | null
      if (t && (["INPUT", "TEXTAREA", "BUTTON", "SELECT"].includes(t.tagName) || t.isContentEditable)) return
      e.preventDefault()
      toggleVideoPlayRef.current()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  // Re-calibrate solver reference directions when a (new) model's rest pose arrives.
  useEffect(() => {
    if (restPose) solverRef.current.calibrate(restPose)
    if (restPose && colliders) solverRef.current.calibrateColliders(colliders, restPose)
  }, [restPose, colliders])

  // Resolve blendshape→morph mappings against the loaded model's morph list.
  useEffect(() => {
    if (modelMorphs && modelMorphs.length > 0) faceBlendshapeSolverRef.current.configure(modelMorphs)
  }, [modelMorphs])

  // Hot path: detection result → solve → apply, all through refs — no React
  // state or effects between a video frame and the model moving.
  const modelLoadedRef = useRef(modelLoaded)
  useEffect(() => {
    modelLoadedRef.current = modelLoaded
  }, [modelLoaded])
  const applyPoseRef = useRef(applyPose)
  const applyFaceRef = useRef(applyFace)
  useEffect(() => {
    applyPoseRef.current = applyPose
    applyFaceRef.current = applyFace
  }, [applyPose, applyFace])

  // The live loop reads this without re-subscribing: a still is solved
  // unfiltered (no time axis to smooth along).
  const inputModeRef = useRef<InputMode>(null)
  useEffect(() => {
    inputModeRef.current = inputMode
  }, [inputMode])

  const lastDebugUpdateRef = useRef(0)
  // Display interpolation. Results arrive at ~30 Hz while the renderer runs at
  // 60: the model shows prev→curr interpolated on the render clock — exact
  // playback, one detection interval behind the newest result. The previous
  // approach tweened from wherever the display currently was toward each new
  // result, which is an exponential chase: measured on a synthetic 45° swing
  // at dance speed it costs ~10ms MORE latency than this and shaves ~7% off
  // the amplitude, because a chase never arrives before its target moves again.
  const displayPrevRef = useRef<BoneState[] | null>(null)
  const displayCurrRef = useRef<BoneState[] | null>(null)
  const displayBufRef = useRef<BoneState[] | null>(null)
  const displayIntervalRef = useRef(33)
  const displayPrevTsRef = useRef(0)
  const displayCurrTsRef = useRef(0)
  /** Playback cursor on the media timeline (see the display loop below). */
  const displayClockRef = useRef(0)
  const lastMediaTsRef = useRef(-1)
  /** Set by the capture loop when the grabbed frame came from a PAUSED video
   *  (fresh upload, scrub): its result is solved as a still. */
  const stillGrabRef = useRef(false)
  const handleResult = useCallback((result: PoseWorkerResult, timestampMs: number) => {
    // Throttled React update — feeds only the debug skeleton preview.
    const now = performance.now()
    if (now - lastDebugUpdateRef.current >= DEBUG_PREVIEW_INTERVAL_MS) {
      lastDebugUpdateRef.current = now
      setLandmarks(result)
    }

    if (!modelLoadedRef.current) return

    // A paused video's frame is a still: one unfiltered solve completes the
    // pose (crossfades snap rather than creep), and a single engine tween
    // eases the model there — instead of re-solving on a timer, which walked
    // the pose in visible 250ms steps.
    const isImage = inputModeRef.current === "image"
    const isStill = isImage || stillGrabRef.current
    const pose = solverRef.current.solve(result, timestampMs, isStill)
    currentBoneStatesRef.current = pose

    let faceTweenMs = 0
    if (isStill) {
      // Shown directly; the interpolation loop idles until playback returns.
      displayPrevRef.current = null
      displayCurrRef.current = null
      lastMediaTsRef.current = -1
      // An image snaps — the model was just reset, there is nothing to ease
      // from. A paused frame eases in from wherever the model stands.
      faceTweenMs = isImage ? 0 : 250
      applyPoseRef.current(pose, faceTweenMs)
    } else {
      // Media-time spacing, not arrival spacing: arrivals jitter with worker
      // load while the frames themselves are evenly spaced. Seeks and stalls
      // fall back to a nominal frame.
      const delta = timestampMs - lastMediaTsRef.current
      lastMediaTsRef.current = timestampMs
      displayIntervalRef.current = delta > 5 && delta < 500 ? delta : 33
      faceTweenMs = displayIntervalRef.current
      const recycled = displayPrevRef.current
      displayPrevRef.current = displayCurrRef.current
      displayPrevTsRef.current = displayCurrTsRef.current
      displayCurrRef.current = snapshotPose(pose, recycled)
      displayCurrTsRef.current = timestampMs
    }

    if (faceEnabledRef.current && result.faceLandmarks?.[0]) {
      const faceResult = faceBlendshapeSolverRef.current.solve(result.faceLandmarks[0], timestampMs)
      currentMorphWeightsRef.current = faceResult.morphWeights
      applyFaceRef.current(faceResult, faceTweenMs)
    }
  }, [])

  // Drive the model every render frame from the display pair, interpolated by
  // a playback cursor that advances along the MEDIA timeline by wall time.
  // Starting each segment on result ARRIVAL instead makes every early arrival
  // a small forward jump — arrival times jitter with worker load, media
  // stamps don't, and that jitter is visible as micro-stutter. The cursor
  // never modulates with arrival timing: it plays, at most one result behind.
  // nlerp per bone (hemisphere-aligned) — over one detection interval the arc
  // is a few degrees, where nlerp and slerp are indistinguishable.
  useEffect(() => {
    let raf = 0
    let lastWall = 0
    const step = () => {
      raf = requestAnimationFrame(step)
      const wall = performance.now()
      const wallDt = lastWall > 0 ? Math.min(100, wall - lastWall) : 0
      lastWall = wall
      // A conversion drives the model itself, stepping through the take.
      if (convertingRef.current || !modelLoadedRef.current) return
      const curr = displayCurrRef.current
      if (!curr) return
      const prev = displayPrevRef.current
      const span = displayCurrTsRef.current - displayPrevTsRef.current
      if (!prev || prev.length !== curr.length || span <= 0 || span > 500) {
        // First result, or a seek/stall: restart playback at the new segment.
        displayClockRef.current = displayCurrTsRef.current
        applyPoseRef.current(curr, 0)
        return
      }
      // Aim the cursor a margin BEHIND the newest result and steer its rate
      // (±15%) toward that point. Riding right at the newest result means
      // every late arrival stalls the cursor at the ceiling for a frame — a
      // visible hiccup, several times a second under real worker-load jitter.
      // The margin absorbs lateness; the rate steer repays it over hundreds
      // of ms, far below what the eye can see as a speed change.
      const target = displayCurrTsRef.current - span * 0.4
      let clock = displayClockRef.current
      const err = clock + wallDt - target
      const rate = 1 - Math.max(-0.15, Math.min(0.15, err / 250))
      clock += wallDt * rate
      // Never extrapolate past the newest result; after a real stall, skip
      // forward rather than replay the backlog.
      if (clock > displayCurrTsRef.current) clock = displayCurrTsRef.current
      if (clock < displayPrevTsRef.current) clock = displayPrevTsRef.current
      displayClockRef.current = clock
      const t = (clock - displayPrevTsRef.current) / span
      let buf = displayBufRef.current
      if (!buf || buf.length !== curr.length) buf = displayBufRef.current = snapshotPose(curr, null)
      for (let i = 0; i < curr.length; i++) {
        const a = prev[i].rotation
        const b = curr[i].rotation
        const o = buf[i].rotation
        const s = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w < 0 ? -1 : 1
        o.setXYZW(
          a.x + (b.x * s - a.x) * t,
          a.y + (b.y * s - a.y) * t,
          a.z + (b.z * s - a.z) * t,
          a.w + (b.w * s - a.w) * t,
        )
        o.normalize()
        const ta = prev[i].translation
        const tb = curr[i].translation
        const to = buf[i].translation
        if (ta && tb && to) to.setXYZ(ta.x + (tb.x - ta.x) * t, ta.y + (tb.y - ta.y) * t, ta.z + (tb.z - ta.z) * t)
      }
      applyPoseRef.current(buf, 0)
    }
    step()
    return () => cancelAnimationFrame(raf)
  }, [])
  const handleResultRef = useRef(handleResult)
  handleResultRef.current = handleResult
  const exportVmdRef = useRef(exportVmd)
  exportVmdRef.current = exportVmd

  // Initialize the MediaPipe detection worker and the frame-feed loop.
  // Detection runs off the main thread so the WebGPU render loop never blocks
  // on it; this loop only snapshots frames (createImageBitmap) and ships them.
  useEffect(() => {
    let rafId = 0
    let ready = false
    // In-flight guard: never queue a second frame while the worker is busy —
    // detection latency then paces capture instead of building a frame backlog.
    let pending = false
    let pendingSince = 0

    const worker = new Worker(new URL("../lib/pose-worker.ts", import.meta.url))
    workerRef.current = worker
    const send = (msg: PoseWorkerRequest, transfer?: Transferable[]) =>
      worker.postMessage(msg, transfer ?? [])

    worker.onmessage = (e: MessageEvent<PoseWorkerResponse>) => {
      const msg = e.data
      if (msg.type === "ready") {
        ready = true
        setMediaPipeReady(true)
        onMediaPipeReadyChangeRef.current?.(true)
      } else if (msg.type === "result") {
        pending = false
        // A conversion owns the worker while it runs; its awaiting frame takes
        // the reply instead of the live path.
        const awaiting = awaitingRef.current
        if (awaiting) {
          awaitingRef.current = null
          awaiting(msg.result)
        } else if (msg.result.poseWorldLandmarks[0]) {
          handleResultRef.current(msg.result, msg.mediaTs)
        }
      } else if (msg.type === "error") {
        pending = false
        const awaiting = awaitingRef.current
        if (awaiting) {
          awaitingRef.current = null
          awaiting(null)
        }
        console.error("Pose worker error:", msg.message)
      }
    }
    worker.onerror = (e) => console.error("Failed to initialize pose worker:", e.message)
    send({ type: "init" })

    let lastVideoTime = -1
    let lastVideoSrc = ""
    let lastImgSrc = ""

    const detect = () => {
      rafId = requestAnimationFrame(detect)
      // A conversion steps the video itself; the opportunistic loop would fight
      // it for the worker and re-solve the same frames.
      if (!ready || convertingRef.current) return
      const now = performance.now()
      if (pending) {
        // Recover if the worker dropped a frame (e.g. mode switch mid-flight).
        if (now - pendingSince > 2000) pending = false
        else return
      }
      const video = videoRef.current
      // A new source must be captured even if it sits at the same media time
      // as the old one — a fresh upload starts at 0, and so did the last.
      if (video && video.currentSrc !== lastVideoSrc) {
        lastVideoSrc = video.currentSrc
        lastVideoTime = -1
      }
      // readyState gates the FIRST frame: before HAVE_CURRENT_DATA the grab
      // below fails, and the time-gate would already be marked consumed — a
      // paused upload then showed nothing until play moved the clock.
      if (video && video.videoWidth > 0 && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        // Pacing: the in-flight guard above (one frame in the worker at a time)
        // plus the new-frame gate (source fps) — no artificial rate floor, so
        // result cadence stays as steady as the worker can deliver.
        lastVideoTime = video.currentTime
        // A frame grabbed while paused (fresh upload, scrub) is a STILL —
        // handleResult solves it unfiltered and eases the model there in one
        // tween. Read at grab time; the in-flight guard keeps it paired with
        // its result.
        stillGrabRef.current = video.paused
        // Media time drives the solver's smoothing filters so pause/seek
        // reset them correctly; detectForVideo gets wall time because it
        // requires a monotonically increasing clock.
        const mediaTs = video.currentTime * 1000
        pending = true
        pendingSince = now
        createImageBitmap(video)
          .then((bitmap) => send({ type: "video", bitmap, ts: performance.now(), mediaTs }, [bitmap]))
          .catch(() => {
            pending = false
          })
      } else if (
        imageRef.current &&
        imageRef.current.src.length > 0 &&
        imageRef.current.src !== lastImgSrc &&
        imageRef.current.complete &&
        imageRef.current.naturalWidth > 0
      ) {
        const img = imageRef.current
        lastImgSrc = img.src
        pending = true
        pendingSince = now
        createImageBitmap(img)
          .then((bitmap) => send({ type: "image", bitmap, mediaTs: performance.now() }, [bitmap]))
          .catch(() => {
            pending = false
          })
      }
    }
    detect()

    return () => {
      cancelAnimationFrame(rafId)
      worker.terminate()
      workerRef.current = null
    }
  }, [])

  // Handle image upload
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.includes("image")) {
      userPickedMediaRef.current = true
      const url = URL.createObjectURL(file)
      resetModel?.()
      solverRef.current.reset()
      faceBlendshapeSolverRef.current.reset()
      // Worker messages are FIFO — the mode switch lands before the next frame.
      workerRef.current?.postMessage({ type: "mode", running: "IMAGE" } satisfies PoseWorkerRequest)
      // ...and the landmarker forgets the previous still, so this one is solved
      // on its own merits rather than tracked from the last.
      workerRef.current?.postMessage({ type: "reset" } satisfies PoseWorkerRequest)
      setCurrentImage(url)
      setVideoSrc("")
      setInputMode("image")
      setLastMedia("IMAGE")
    }
  }

  // Handle video upload
  const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.includes("video")) {
      userPickedMediaRef.current = true
      void saveVideoUpload(file)
      const url = URL.createObjectURL(file)
      // No model reset: the new video's first frame arrives as a still within
      // a few hundred ms and the model eases straight from its current pose —
      // a bind-pose snap in between is exactly the jarring cut this avoids.
      solverRef.current.reset()
      faceBlendshapeSolverRef.current.reset()
      if (lastMedia === "IMAGE") {
        workerRef.current?.postMessage({ type: "mode", running: "VIDEO" } satisfies PoseWorkerRequest)
        setCurrentImage("")
      }
      // The landmarker's tracker still carries the previous video; the new
      // one's first frame — solved immediately, while paused — deserves a
      // clean slate.
      workerRef.current?.postMessage({ type: "reset" } satisfies PoseWorkerRequest)
      setVideoSrc(url)
      setInputMode("video")
      if (videoRef.current) {
        videoRef.current.currentTime = 0
      }
      // Player state reflects the new file at once — duration follows from
      // loadedmetadata, the playhead must not wait for a timeupdate.
      setVideoTime(0)
      setVideoPlaying(false)
      setLastMedia("VIDEO")
    }
  }

  // Stop current input
  const stopCurrentInput = () => {
    if (isStreamActive && videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.src = ""
      videoRef.current.load()
    }
    setIsStreamActive(false)
    setInputMode(null)
  }

  // Camera functions
  const toggleCamera = async () => {
    userPickedMediaRef.current = true
    if (isStreamActive) {
      stopCurrentInput()
    } else {
      try {
        stopCurrentInput()
        resetModel?.()
        solverRef.current.reset()
      faceBlendshapeSolverRef.current.reset()
        setInputMode("camera")
        setIsStreamActive(true)

        if (lastMedia === "IMAGE") {
          workerRef.current?.postMessage({ type: "mode", running: "VIDEO" } satisfies PoseWorkerRequest)
        }

        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
        setLastMedia("VIDEO")
      } catch (error) {
        console.error("Error accessing camera:", error)
        setIsStreamActive(false)
        setInputMode(null)
      }
    }
  }

  /**
   * Video → VMD, offline.
   *
   * The video is STEPPED, not played: seek to each 1/30s mark, detect, solve,
   * keep the pose. Nothing is paced by wall time, so a slow machine produces the
   * same file as a fast one — it only takes longer — and no frame is ever
   * skipped because detection fell behind. Media time drives the solver's
   * filters, which is what they were designed for.
   */
  /**
   * A still → a one-frame VMD.
   *
   * There is no timeline to walk: the live loop has already solved the image and
   * left the pose in the refs, so exporting is a matter of writing down what is
   * already on screen. A single-frame motion is how MMD carries a pose.
   */
  const exportPoseVmd = useCallback(() => {
    const pose = currentBoneStatesRef.current
    if (pose.length === 0) return
    const clip = buildClip([
      {
        time: 0,
        boneStates: pose.map((bs) => ({ name: bs.name, rotation: bs.rotation.clone() })),
        morphWeights: faceEnabledRef.current ? currentMorphWeightsRef.current : null,
      },
    ])
    exportVmdRef.current?.(clip)
    setExported("pose saved")
  }, [])

  const convertVideoToVmd = useCallback(async () => {
    const video = videoRef.current
    const worker = workerRef.current
    // A non-finite duration (unresolved WebM/stream) would make the frame loop
    // below run forever — refuse until it is a real number.
    if (!video || !worker || !Number.isFinite(video.duration) || video.duration <= 0 || convertingRef.current) return

    video.pause()
    cancelRef.current = false
    convertingRef.current = true
    setConverting(true)
    setProgress(0)
    setExported(null)

    const step = 1 / 30
    const frames: RecordedFrame[] = []
    // detectForVideo demands a monotonically rising clock of its own, separate
    // from the media time the solver reads.
    let tick = performance.now()

    const seek = (t: number) =>
      new Promise<void>((resolve) => {
        const done = () => {
          video.removeEventListener("seeked", done)
          resolve()
        }
        video.addEventListener("seeked", done)
        video.currentTime = t
      })

    const detectAt = (bitmap: ImageBitmap, mediaTs: number) =>
      new Promise<PoseWorkerResult | null>((resolve) => {
        awaitingRef.current = resolve
        tick += 33
        const msg: PoseWorkerRequest = { type: "video", bitmap, ts: tick, mediaTs }
        worker.postMessage(msg, [bitmap])
      })

    // Seek and detect are the two expensive steps and neither needs the other's
    // result, so the next frame is decoding while the current one is in the
    // worker. Serially they added up; overlapped, the slower of the two sets the
    // pace. (The worker handles one frame at a time, so one lookahead is all
    // there is to win.)
    const grab = async (t: number): Promise<ImageBitmap | null> => {
      if (t >= video.duration) return null
      await seek(t)
      return createImageBitmap(video)
    }

    let lastPaint = 0
    try {
      let bitmap = await grab(0)
      for (let t = 0; t < video.duration && !cancelRef.current; t += step) {
        if (!bitmap) break
        const ahead = grab(t + step)
        const result = await detectAt(bitmap, t * 1000)
        bitmap = await ahead
        if (cancelRef.current) break
        if (!result?.poseWorldLandmarks[0]) continue

        const pose = solverRef.current.solve(result, t * 1000)
        currentBoneStatesRef.current = pose
        // Applied untweened: the character steps through the take as it converts,
        // which is the progress bar people actually watch.
        applyPoseRef.current(pose, 0)

        let morphWeights: FaceMorphWeights | null = null
        if (faceEnabledRef.current && result.faceLandmarks?.[0]) {
          const face = faceBlendshapeSolverRef.current.solve(result.faceLandmarks[0], t * 1000)
          morphWeights = face.morphWeights
          currentMorphWeightsRef.current = morphWeights
          applyFaceRef.current(face, 0)
        }

        frames.push({
          time: t,
          boneStates: pose.map((bs) => ({ name: bs.name, rotation: bs.rotation.clone() })),
          morphWeights,
        })
        // Throttled: a React render per frame is a render the conversion pays
        // for thousands of times, for a number that changes by a fraction.
        const now = performance.now()
        if (now - lastPaint > 100) {
          lastPaint = now
          setProgress(t / video.duration)
        }
      }
    } finally {
      awaitingRef.current = null
      convertingRef.current = false
      setConverting(false)
      setProgress(0)
      // The take's last frame stays on screen; without this the display loop
      // would snap back to the pair it held before the conversion started.
      displayPrevRef.current = null
      displayCurrRef.current = null
      lastMediaTsRef.current = -1
    }

    if (frames.length === 0) return
    // The live solve was causal — it filtered each frame knowing only the past.
    // The finished take can be read in both directions: a zero-phase polynomial
    // fit removes the residual shake without shifting timing, and hands fast
    // transients (a kick, a snap) back their original amplitude. Export only;
    // the on-screen preview is inherently real-time.
    smoothTakeZeroPhase(frames)
    const clip = buildClip(frames)
    exportVmdRef.current?.(clip)
    const { frames: n, seconds } = clipSummary(clip)
    setExported(`${n}f · ${seconds.toFixed(1)}s`)
  }, [])

  const statusPill =
    inputMode === "camera" ? (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-red-500/40 bg-red-500/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-red-300">
        <span className="size-1.5 animate-pulse rounded-full bg-red-500" />
        Live
      </span>
    ) : inputMode === "video" ? (
      <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white/60">
        Video
      </span>
    ) : inputMode === "image" ? (
      <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white/60">
        Image
      </span>
    ) : null

  return (
    <div className="absolute left-2 top-2 z-10 w-[148px] max-w-[calc(100vw-1rem)] md:left-3 md:top-12 md:w-[300px]">
      <div className="overflow-hidden rounded-xl border border-white/10 bg-zinc-950/35 shadow-2xl shadow-black/40 backdrop-blur-md md:bg-zinc-950/60">
        {/* Toolbar — mode buttons + status + record (status/record desktop-only). */}
        <div className="flex items-center gap-0.5 border-b border-white/5 px-1.5 py-1.5 md:gap-1 md:px-3 md:py-2">
          <TooltipProvider delayDuration={150}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={toggleCamera}
                  variant="ghost"
                  size="icon"
                  className={`size-7 ${
                    isStreamActive
                      ? "bg-white/10 text-white hover:bg-white/15"
                      : "text-white/70 hover:bg-white/10 hover:text-white"
                  }`}
                  disabled={!mediaPipeReady}
                >
                  {isStreamActive ? <Pause className="size-3.5" /> : <Webcam className="size-3.5" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {!mediaPipeReady ? "Loading…" : isStreamActive ? "Stop webcam" : "Start webcam"}
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={() => imageInputRef.current?.click()}
                  variant="ghost"
                  size="icon"
                  className="size-7 text-white/70 hover:bg-white/10 hover:text-white"
                  disabled={!mediaPipeReady}
                >
                  <ImageIcon className="size-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{!mediaPipeReady ? "Loading…" : "Upload image"}</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={() => videoInputRef.current?.click()}
                  variant="ghost"
                  size="icon"
                  className="size-7 text-white/70 hover:bg-white/10 hover:text-white"
                  disabled={!mediaPipeReady}
                >
                  <Video className="size-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{!mediaPipeReady ? "Loading…" : "Upload video"}</TooltipContent>
            </Tooltip>

            <div className="ml-auto hidden items-center gap-1.5 md:flex">
              {converting ? (
                <span className="font-mono text-[10px] tabular-nums text-blue-300/90">
                  {Math.round(progress * 100)}%
                </span>
              ) : exported ? (
                <span className="font-mono text-[10px] tabular-nums text-emerald-300/90">{exported} saved</span>
              ) : (
                statusPill
              )}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={() =>
                      converting
                        ? (cancelRef.current = true)
                        : inputMode === "image"
                          ? exportPoseVmd()
                          : void convertVideoToVmd()
                    }
                    variant="ghost"
                    size="icon"
                    className={`size-7 ${
                      converting
                        ? "bg-blue-500/10 text-blue-300 hover:bg-blue-500/15"
                        : "text-white/70 hover:bg-white/10 hover:text-white"
                    }`}
                    // A webcam has no end, so there is nothing to convert. A
                    // video is walked frame by frame; a still is written as it
                    // stands.
                    disabled={(inputMode !== "video" && inputMode !== "image") || !mediaPipeReady}
                  >
                    {converting ? <Square className="size-3.5 fill-current" /> : <Download className="size-3.5" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {converting
                    ? "Stop converting"
                    : inputMode === "image"
                      ? "Save this pose as a VMD file"
                      : "Convert this video to a VMD file"}
                </TooltipContent>
              </Tooltip>
            </div>
          </TooltipProvider>
        </div>

        <input ref={imageInputRef} type="file" accept="image/*" onChange={handleImageUpload} hidden />
        <input ref={videoInputRef} type="file" accept="video/*" onChange={handleVideoUpload} hidden />

        {/* Media — mobile uses opacity-70 so the model bleeds through; tap/hover restores
            full clarity. Desktop stays opaque. */}
        <div className="group/media relative aspect-video bg-black/30 opacity-70 transition-opacity duration-200 hover:opacity-100 active:opacity-100 md:bg-black/50 md:opacity-100">
          {inputMode === "image" && (
            <div className="flex h-full w-full items-center justify-center">
              <Image
                src={currentImage}
                alt="Motion capture input"
                crossOrigin="anonymous"
                ref={imageRef}
                width={320}
                height={320}
                priority
                className="max-h-full max-w-full object-contain"
              />
            </div>
          )}

          {(inputMode === "video" || inputMode === "camera") && (
            <>
              <video
                ref={videoRef}
                // The demo video is served cross-origin from the assets bucket in
                // production. Without CORS its frames are tainted: createImageBitmap
                // still resolves, but transferring the bitmap to the pose worker
                // throws DataCloneError — swallowed by the send .catch, so mocap
                // silently never starts. Dev never sees this (assets come from
                // public/, same-origin).
                crossOrigin="anonymous"
                className={`h-full w-full object-contain ${inputMode === "camera" ? "scale-x-[-1]" : ""}`}
                playsInline
                autoPlay={inputMode === "camera"}
                disablePictureInPicture
                controlsList="nofullscreen noremoteplayback nodownload"
                src={isStreamActive ? undefined : videoSrc}
                onPlay={() => setVideoPlaying(true)}
                onPause={() => setVideoPlaying(false)}
                onTimeUpdate={(e) => setVideoTime(e.currentTarget.currentTime)}
                onLoadedMetadata={(e) => resolveDuration(e.currentTarget)}
                // Browsers report Infinity at loadedmetadata for WebM/streamed
                // and variable-frame-rate files, then refine it later — so take
                // the update too, not just the first announcement.
                onDurationChange={(e) => resolveDuration(e.currentTarget)}
              />

              {inputMode === "video" && videoSrc && (
                <div className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-black/70 via-black/30 to-transparent px-2 py-1.5">
                  <button
                    type="button"
                    onClick={toggleVideoPlay}
                    disabled={converting}
                    className="flex size-6 shrink-0 items-center justify-center rounded text-white/90 transition-colors hover:bg-white/10 hover:text-white disabled:opacity-40"
                    aria-label={videoPlaying ? "Pause" : "Play"}
                  >
                    {videoPlaying ? <Pause className="size-3.5" /> : <Play className="size-3.5 translate-x-[1px]" />}
                  </button>
                  <input
                    type="range"
                    min={0}
                    max={videoDuration || 1}
                    step={0.01}
                    value={videoTime}
                    disabled={converting}
                    onChange={(e) => {
                      if (videoRef.current && !convertingRef.current) videoRef.current.currentTime = Number(e.target.value)
                    }}
                    // Release focus after a scrub for the same reason.
                    onPointerUp={(e) => e.currentTarget.blur()}
                    className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-white/20 disabled:cursor-not-allowed accent-white outline-none [&::-moz-range-thumb]:size-2.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-white [&::-webkit-slider-thumb]:size-2.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
                  />
                  <span className="hidden font-mono text-[10px] tabular-nums text-white/70 sm:block">
                    {formatTime(videoTime)} / {formatTime(videoDuration)}
                  </span>
                </div>
              )}
            </>
          )}

          {!inputMode && (
            <div className="flex h-full w-full items-center justify-center">
              <Camera className="size-8 text-white/30" />
            </div>
          )}
        </div>

        {/* Skeleton preview — desktop only */}
        <div className="hidden aspect-[16/10] border-t border-white/5 bg-black/50 md:block">
          <DebugScene landmarks={landmarks} />
        </div>
      </div>
    </div>
  )
}
