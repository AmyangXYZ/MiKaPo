import { useEffect, useRef, useState, useCallback, memo } from "react"
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
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Camera, Image as ImageIcon, Video, Webcam, Pause, Play, Download, Square, GripVertical, Upload } from "lucide-react"
import DebugScene from "./debug-scene"
import { FloatingPanel, clampRect, type Rect } from "./floating-panel"
import { useStoredRect } from "@/hooks/use-stored-rect"
import { cn } from "@/lib/utils"
import { SKIN, SEGMENT, SEGMENT_ON, SEGMENT_OFF, SEGMENT_TRACK } from "@/lib/chrome"

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

/** Width the detector is fed. MediaPipe scales the frame down internally
 *  anyway; handing it 1080p means paying for a 2-megapixel bitmap copy and a
 *  2-megapixel upload on every single frame, which is capture rate spent on
 *  pixels the model never sees. 960 keeps enough detail for hands and faces
 *  cropped out of the frame. */
const DEMO_VIDEO = `${ASSETS}/Stellar (스텔라) - Vibrato (떨려요)- DANCE COVER.mp4`

const CAPTURE_WIDTH = 960

const PANEL_RECT_KEY = "mikapo.capture-panel.1"
const PANEL_MIN_W = 240
const PANEL_MIN_H = 300

/** Where the panel sits before anyone moves it: under the header, left edge,
 *  tall enough for the picture, the skeleton and the status strip. Narrow
 *  windows get a narrower panel — a phone has no room for 320px of chrome
 *  beside the character. */
function initialPanelRect(): Rect {
  if (typeof window === "undefined") return { x: 12, y: 56, w: 320, h: 470 }
  const w = Math.min(320, Math.max(PANEL_MIN_W, window.innerWidth * 0.42))
  const h = Math.min(470, Math.max(PANEL_MIN_H, window.innerHeight * 0.62))
  return clampRect({ x: 12, y: 56, w, h }, PANEL_MIN_W, PANEL_MIN_H)
}

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

const MotionCaptureImpl = ({
  applyPose,
  applyFace,
  modelLoaded,
  onMediaPipeReadyChange,
  resetModel,
  restPose,
  colliders,
  modelMorphs,
  exportVmd,
  onUploadStored,
  restoreDemoSignal,
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
  /** Fired once an upload is actually persisted, so the header can offer the
   *  way back to the demo without waiting for a reload to notice. */
  onUploadStored?: () => void
  /** Bumped when the app goes back to its bundled assets. */
  restoreDemoSignal?: number
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const mediaInputRef = useRef<HTMLInputElement>(null)
  const workerRef = useRef<Worker | null>(null)
  const [mediaPipeReady, setMediaPipeReady] = useState(false)
  const [landmarks, setLandmarks] = useState<PoseWorkerResult | null>(null)
  /** Status strip: is a body being found, and how fast frames are arriving. */
  const [tracking, setTracking] = useState(false)
  const [captureHz, setCaptureHz] = useState(0)
  /** Detector time per frame, smoothed. What the capture rate is made of. */
  const [inferenceMs, setInferenceMs] = useState(0)
  const inferenceEmaRef = useRef(0)
  const arrivalEmaRef = useRef(0)
  const lastArrivalRef = useRef(0)
  /** Set when returning to an image the capture loop has already consumed. */
  const redetectImageRef = useRef(false)
  const [inputMode, setInputMode] = useState<InputMode>("video")
  const [isStreamActive, setIsStreamActive] = useState(false)
  // No ?cors suffix. It was once a browser-cache buster, and became the
  // problem: the edge cached THAT url's response from the old domain, without
  // an access-control-allow-origin the new one accepts, and served the stale
  // copy under a one-year immutable header. The bare url varies on Origin and
  // answers each domain correctly.
  //
  // Images have no default. A demo photo is a picture of someone else's pose —
  // the mode is worth entering only with your own.
  const [currentImage, setCurrentImage] = useState<string>("")
  const [videoSrc, setVideoSrc] = useState<string>(DEMO_VIDEO)
  const [lastMedia, setLastMedia] = useState<"IMAGE" | "VIDEO">("VIDEO")
  // Set the moment the user picks any input this session — a slow IndexedDB
  // restore must never clobber a fresher choice.
  const userPickedMediaRef = useRef(false)
  const solverRef = useRef<Solver>(new Solver())
  const faceBlendshapeSolverRef = useRef<FaceBlendshapeSolver>(new FaceBlendshapeSolver())
  const onUploadStoredRef = useRef(onUploadStored)
  onUploadStoredRef.current = onUploadStored
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
  /** False from the moment a new source is chosen until the detector says it
   *  is rebuilt for it. Playing before then spends real frames on a graph that
   *  is still tearing down the last source. */
  const [sourceReady, setSourceReady] = useState(true)
  const sourceGateTimerRef = useRef<number | null>(null)
  /** Shut until the detector produces a pose for the new source. A rebuilt
   *  graph answers in milliseconds; its first inference is far slower, and
   *  that gap is exactly the frames that would play unposed. */
  const closeSourceGate = useCallback(() => {
    setSourceReady(false)
    if (sourceGateTimerRef.current !== null) window.clearTimeout(sourceGateTimerRef.current)
    // Footage with nobody in it still has to be playable.
    sourceGateTimerRef.current = window.setTimeout(() => setSourceReady(true), 4000)
  }, [])
  const openSourceGate = useCallback(() => {
    if (sourceGateTimerRef.current !== null) {
      window.clearTimeout(sourceGateTimerRef.current)
      sourceGateTimerRef.current = null
    }
    setSourceReady(true)
  }, [])
  const openSourceGateRef = useRef(openSourceGate)
  openSourceGateRef.current = openSourceGate
  const [videoTime, setVideoTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  /** Accept a duration only once it is a real number. When the browser says
   *  Infinity (common for WebM and anything streamed), nudge it into resolving:
   *  seeking far past the end forces the demuxer to find the true end, and the
   *  durationchange that follows carries the answer. Guarded by a flag so the
   *  nudge happens once per file. */
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
    // The conversion owns the playhead while it steps through the take, and a
    // detector still rebuilding would miss the opening frames.
    if (!videoRef.current || convertingRef.current || !sourceReady) return
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
  const handleResult = useCallback((result: PoseWorkerResult, timestampMs: number) => {
    const ms = result.inferenceMs ?? 0
    inferenceEmaRef.current = inferenceEmaRef.current > 0 ? inferenceEmaRef.current * 0.8 + ms * 0.2 : ms
    const arrivedAt = performance.now()
    const gap = arrivedAt - lastArrivalRef.current
    lastArrivalRef.current = arrivedAt
    // A gap that long is a stopped source, not a slow one.
    if (gap > 0 && gap < 1000) {
      arrivalEmaRef.current = arrivalEmaRef.current > 0 ? arrivalEmaRef.current * 0.8 + gap * 0.2 : gap
    } else {
      arrivalEmaRef.current = 0
    }
    // Throttled React update — feeds only the debug skeleton preview.
    const now = performance.now()
    // A still produces exactly one result: dropping it to the throttle leaves
    // the preview showing the previous source for good, next to a model that
    // has already moved.
    if (inputModeRef.current === "image" || now - lastDebugUpdateRef.current >= DEBUG_PREVIEW_INTERVAL_MS) {
      lastDebugUpdateRef.current = now
      setLandmarks(result)
      // The status strip rides the same throttle: a readout that re-rendered
      // per detection would cost more than it reports.
      setTracking(true)
      // Measured between ARRIVALS: how many poses a second are actually being
      // produced. Media-time spacing reads 30 Hz off a paused video, which is
      // the frame interval of a source that is not being sampled at all.
      setCaptureHz(arrivalEmaRef.current > 0 ? Math.round(1000 / arrivalEmaRef.current) : 0)
      setInferenceMs(Math.round(inferenceEmaRef.current))
    }

    if (!modelLoadedRef.current) return

    const isImage = inputModeRef.current === "image"
    const pose = solverRef.current.solve(result, timestampMs, isImage)
    currentBoneStatesRef.current = pose

    let faceTweenMs = 0
    if (isImage) {
      // A still is shown as-is; the interpolation loop idles until video returns.
      displayPrevRef.current = null
      displayCurrRef.current = null
      lastMediaTsRef.current = -1
      applyPoseRef.current(pose, 0)
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

    let lastVideoTime = -1
    let lastImgSrc = ""

    worker.onmessage = (e: MessageEvent<PoseWorkerResponse>) => {
      const msg = e.data
      if (msg.type === "ready") {
        ready = true
        setMediaPipeReady(true)
        onMediaPipeReadyChangeRef.current?.(true)
      } else if (msg.type === "prepared") {
        // The graph is rebuilt: take the standing frame now, so the pose is
        // already there when playback is allowed to start.
        lastVideoTime = -1
        detect()
      } else if (msg.type === "result") {
        pending = false
        // A pose for this source exists — that is what "ready" means.
        openSourceGateRef.current()
        // Grab the next frame now rather than on the next animation frame:
        // waiting for rAF spends up to 16ms of every capture cycle idle.
        if (!awaitingRef.current) detect()
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

    const detect = () => {
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
      if (video && video.videoWidth > 0 && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        // Pacing: the in-flight guard above (one frame in the worker at a time)
        // plus the new-frame gate (source fps) — no artificial rate floor, so
        // result cadence stays as steady as the worker can deliver.
        lastVideoTime = video.currentTime
        // Media time drives the solver's smoothing filters so pause/seek
        // reset them correctly; detectForVideo gets wall time because it
        // requires a monotonically increasing clock.
        const mediaTs = video.currentTime * 1000
        pending = true
        pendingSince = now
        createImageBitmap(
          video,
          video.videoWidth > CAPTURE_WIDTH ? { resizeWidth: CAPTURE_WIDTH, resizeQuality: "medium" } : undefined,
        )
          .then((bitmap) => send({ type: "video", bitmap, ts: performance.now(), mediaTs }, [bitmap]))
          .catch(() => {
            pending = false
          })
      } else if (
        imageRef.current &&
        imageRef.current.src.length > 0 &&
        (imageRef.current.src !== lastImgSrc || redetectImageRef.current) &&
        imageRef.current.complete &&
        imageRef.current.naturalWidth > 0
      ) {
        const img = imageRef.current
        lastImgSrc = img.src
        redetectImageRef.current = false
        pending = true
        pendingSince = now
        createImageBitmap(
          img,
          img.naturalWidth > CAPTURE_WIDTH ? { resizeWidth: CAPTURE_WIDTH, resizeQuality: "medium" } : undefined,
        )
          .then((bitmap) => send({ type: "image", bitmap, mediaTs: performance.now() }, [bitmap]))
          .catch(() => {
            pending = false
          })
      }
    }
    const pump = () => {
      rafId = requestAnimationFrame(pump)
      detect()
    }
    pump()

    return () => {
      cancelAnimationFrame(rafId)
      worker.terminate()
      workerRef.current = null
    }
  }, [])

  /** One picker for both: what the file IS decides which mode it opens, so
   *  there is a single obvious way to bring your own footage in. */
  const handleMediaUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    // Cleared so picking the same file twice still fires a change.
    event.target.value = ""
    if (!file) return
    if (file.type.startsWith("video/")) loadVideoFile(file)
    else if (file.type.startsWith("image/")) loadImageFile(file)
  }

  /** Everything that describes the previous source, dropped together: the
   *  solver's history, the preview, the interpolation pair and the readouts.
   *  Leaving any one of them behind is how the panel ends up describing a
   *  source that is no longer playing. */
  const clearCaptureState = () => {
    solverRef.current.reset()
    faceBlendshapeSolverRef.current.reset()
    setLandmarks(null)
    setTracking(false)
    setCaptureHz(0)
    setInferenceMs(0)
    inferenceEmaRef.current = 0
    arrivalEmaRef.current = 0
    lastArrivalRef.current = 0
    displayPrevRef.current = null
    displayCurrRef.current = null
    lastMediaTsRef.current = -1
  }

  // Back to the bundled footage, in place: the uploaded source is dropped and
  // the demo takes its slot without the page going away and coming back.
  const firstDemoSignal = useRef(restoreDemoSignal)
  useEffect(() => {
    if (restoreDemoSignal === firstDemoSignal.current) return
    firstDemoSignal.current = restoreDemoSignal
    userPickedMediaRef.current = false
    clearCaptureState()
    closeSourceGate()
    if (lastMedia === "IMAGE") {
      workerRef.current?.postMessage({ type: "mode", running: "VIDEO" } satisfies PoseWorkerRequest)
    }
    workerRef.current?.postMessage({ type: "reset" } satisfies PoseWorkerRequest)
    setCurrentImage("")
    setVideoSrc(DEMO_VIDEO)
    setInputMode("video")
    setLastMedia("VIDEO")
    setVideoTime(0)
    setVideoPlaying(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restoreDemoSignal])

  const loadImageFile = (file: File) => {
    {
      userPickedMediaRef.current = true
      const url = URL.createObjectURL(file)
      resetModel?.()
      clearCaptureState()
      // Worker messages are FIFO — the mode switch lands before the next frame.
      closeSourceGate()
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

  const loadVideoFile = (file: File) => {
    {
      userPickedMediaRef.current = true
      void saveVideoUpload(file).then((ok) => {
        if (ok) onUploadStoredRef.current?.()
      })
      const url = URL.createObjectURL(file)
      // No model reset: the model holds its pose until the user plays the new
      // video, then transitions to its motion — a bind-pose snap in between
      // is exactly the jarring cut this avoids.
      clearCaptureState()
      if (lastMedia === "IMAGE") {
        closeSourceGate()
        workerRef.current?.postMessage({ type: "mode", running: "VIDEO" } satisfies PoseWorkerRequest)
        setCurrentImage("")
      }
      // The landmarker's tracker still carries the previous video; the new one
      // deserves a clean slate, and playback waits for it.
      closeSourceGate()
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
        clearCaptureState()
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
        // The preview watches the conversion too — a take being stepped is
        // exactly when you want to see what the detector is finding.
        if (performance.now() - lastDebugUpdateRef.current >= DEBUG_PREVIEW_INTERVAL_MS) {
          lastDebugUpdateRef.current = performance.now()
          setLandmarks(result)
        }
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

  // Panel geometry, remembered across reloads and re-clamped into whatever
  // window exists now.
  const [rect, setRect] = useStoredRect(PANEL_RECT_KEY, initialPanelRect, PANEL_MIN_W, PANEL_MIN_H)

  /** Switch the capture source. A source with nothing loaded opens its picker
   *  instead — the segment is where you go for that input, either way. */
  const selectSource = (key: "camera" | "media") => {
    if (key === "camera") {
      void toggleCamera()
      return
    }
    if (isStreamActive) stopCurrentInput()
    userPickedMediaRef.current = true
    // Back to whichever file is loaded; with none, straight to the picker.
    if (lastMedia === "IMAGE" && currentImage) {
      closeSourceGate()
      workerRef.current?.postMessage({ type: "mode", running: "IMAGE" } satisfies PoseWorkerRequest)
      workerRef.current?.postMessage({ type: "reset" } satisfies PoseWorkerRequest)
      // The capture loop skips an image it has already seen; coming back to one
      // has to re-solve it, or the model keeps whatever the video left behind.
      redetectImageRef.current = true
      setInputMode("image")
      return
    }
    if (videoSrc) {
      if (lastMedia === "IMAGE") {
        closeSourceGate()
        workerRef.current?.postMessage({ type: "mode", running: "VIDEO" } satisfies PoseWorkerRequest)
      }
      setLastMedia("VIDEO")
      setInputMode("video")
      return
    }
    mediaInputRef.current?.click()
  }

  // Two sources, because there are two: the camera, and a file. Which KIND of
  // file is the file's business, not a mode the user has to pick first.
  const SOURCES = [
    { key: "camera", label: "Webcam", Icon: Webcam, active: inputMode === "camera" },
    {
      key: "media",
      label: "Media",
      Icon: inputMode === "image" ? ImageIcon : Video,
      active: inputMode === "video" || inputMode === "image",
    },
  ] as const

  const exportDisabled = (inputMode !== "video" && inputMode !== "image") || !mediaPipeReady

  const panel = (
    // The ground is translucent by design: the surface tint lets the character
    // read through the panel while it is only being watched, and solidifies
    // under the pointer, when the controls are what matters.
    <div className={cn("flex h-full flex-col overflow-hidden rounded-surface transition-colors hover:bg-surface-raised", SKIN)}>
      {/* Header — the whole strip drags the panel, except the controls in it. */}
      <header
        data-drag-handle
        className="flex shrink-0 cursor-grab items-center gap-1.5 border-b border-line px-1.5 py-1.5 active:cursor-grabbing"
      >
        <GripVertical className="size-3.5 shrink-0 text-muted-foreground" strokeWidth={1.75} aria-hidden />
        <div className={cn(SEGMENT_TRACK, "min-w-0 flex-1")}>
          {SOURCES.map(({ key, label, Icon, active }) => (
            <button
              key={key}
              type="button"
              disabled={!mediaPipeReady}
              onClick={() => selectSource(key)}
              className={cn(SEGMENT, active ? SEGMENT_ON : SEGMENT_OFF, "min-w-0 px-1.5 disabled:opacity-40")}
            >
              <Icon className="size-3.5 shrink-0" strokeWidth={1.75} />
              <span className="truncate">{label}</span>
            </button>
          ))}
        </div>

        {inputMode === "camera" && (
          <span className="flex shrink-0 items-center gap-1 rounded-chip border border-red-400/40 bg-red-400/15 px-1.5 py-0.5 text-[10px] font-medium text-red-400">
            <span className="size-1.5 animate-pulse rounded-full bg-red-400" />
            Live
          </span>
        )}

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              disabled={!mediaPipeReady}
              onClick={() => mediaInputRef.current?.click()}
              className="size-7 shrink-0 rounded-lg text-muted-foreground hover:bg-white/5 hover:text-foreground"
              aria-label="Upload a video or image"
            >
              <Upload className="size-4" strokeWidth={1.75} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Upload a video or image</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              disabled={exportDisabled}
              onClick={() =>
                converting ? (cancelRef.current = true) : inputMode === "image" ? exportPoseVmd() : void convertVideoToVmd()
              }
              className={cn(
                "size-7 shrink-0 rounded-lg",
                converting
                  ? "text-blue-400 hover:bg-blue-400/10 hover:text-blue-400"
                  : "text-muted-foreground hover:bg-white/5 hover:text-foreground",
              )}
              aria-label={converting ? "Stop converting" : "Export VMD"}
            >
              {converting ? <Square className="size-3.5 fill-current" /> : <Download className="size-4" strokeWidth={1.75} />}
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
      </header>

      {/* The source picture. */}
      <div className="relative min-h-0 flex-1 bg-black">
        {inputMode === "image" && currentImage && (
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
          <video
            ref={videoRef}
            // The demo video is served cross-origin from the assets bucket in
            // production. Without CORS its frames are tainted: createImageBitmap
            // still resolves, but transferring the bitmap to the pose worker
            // throws DataCloneError — swallowed by the send .catch, so mocap
            // silently never starts. Dev never sees this (assets come from
            // public/, same-origin).
            crossOrigin="anonymous"
            className={cn("h-full w-full object-contain", inputMode === "camera" && "scale-x-[-1]")}
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
        )}

        {(!inputMode || (inputMode === "image" && !currentImage)) && (
          <div className="flex h-full w-full items-center justify-center">
            <Camera className="size-8 text-muted-foreground" strokeWidth={1.5} />
          </div>
        )}

      </div>

      {/* Transport — video only; a webcam has no timeline and a still has no time.
          A conversion steps this same playhead, so the scrubber is the progress
          bar and there is no second one to keep in agreement with it. */}
      {inputMode === "video" && videoSrc && (
        <div className="flex shrink-0 items-center gap-2 border-t border-line px-2 py-1.5">
          <button
            type="button"
            onClick={toggleVideoPlay}
            disabled={converting || !sourceReady}
            className="flex size-5 shrink-0 items-center justify-center rounded-chip text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground disabled:opacity-40"
            aria-label={videoPlaying ? "Pause" : "Play"}
          >
            {videoPlaying ? <Pause className="size-3.5" /> : <Play className="size-3.5 translate-x-px" />}
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
            onPointerUp={(e) => e.currentTarget.blur()}
            className="h-1 min-w-0 flex-1 cursor-pointer appearance-none rounded-full bg-white/10 accent-blue-400 outline-none disabled:cursor-not-allowed [&::-moz-range-thumb]:size-2.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-blue-400 [&::-webkit-slider-thumb]:size-2.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-400"
          />
          <span className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground">
            {formatTime(videoTime)} / {formatTime(videoDuration)}
          </span>
        </div>
      )}

      {/* What the detector sees — the reason a bad pose is never a mystery. */}
      <div className="h-[150px] shrink-0 border-t border-line bg-black/40">
        <DebugScene landmarks={landmarks} />
      </div>

      {/* Status: tracking state on the left, what the solver is being fed on the right. */}
      <footer
        className="flex h-6 shrink-0 items-center gap-2 border-t border-line-strong px-2 text-[10px] text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        <span className={cn("truncate", tracking ? "text-foreground" : undefined)}>
          {converting
            ? "Converting to VMD"
            : !sourceReady
              ? "Preparing detector"
              : exported
                ? `Saved ${exported}`
                : tracking
                  ? "Tracking"
                  : "No pose"}
        </span>
        <span className="ml-auto shrink-0 tabular-nums">
          {converting ? (
            // Live rate means nothing here: the take is stepped frame by frame
            // at VMD's own 30 fps and takes exactly as long as it takes.
            `${Math.round(progress * 100)}% · 30 Hz`
          ) : (
            <>
              {captureHz > 0 ? `${captureHz} Hz` : "— Hz"} · {inferenceMs > 0 ? `${inferenceMs} ms` : "— ms"}
            </>
          )}
        </span>
      </footer>
    </div>
  )

  return (
    <>
      <input ref={mediaInputRef} type="file" accept="video/*,image/*" onChange={handleMediaUpload} hidden />
      <FloatingPanel rect={rect} onRectChange={setRect} minW={PANEL_MIN_W} minH={PANEL_MIN_H}>
        {panel}
      </FloatingPanel>
    </>
  )
}

/** The panel re-renders on its own signals (a new preview frame, a status
 *  tick), never because the scene around it drew a frame. */
export const MotionCapture = memo(MotionCaptureImpl)
