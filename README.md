# MiKaPo: Real-time MMD Motion Capture

A web-based tool that drives MikuMikuDance (MMD) models — **full body, both hands, and face** — from a webcam, video, or photo in real time. One shot, no offline preprocessing, no multi-pass.

## Overview

[MiKaPo](https://mikapo.vercel.app) covers all three motion modalities in one pipeline:

- **Body and hands** are driven by MMD **bone rotations** — 3D landmarks from MediaPipe are mapped to per-bone quaternions in each bone's parent-local frame.
- **Face is driven by MMD morphs**, not bone retargeting — face blendshapes from MediaPipe are converted directly into MMD morph weights (`まばたき`, `あ`, `ワ`, `ウィンク`, `ウィンク右`), which is how MMD models are natively rigged for facial expression. Eye direction is the one face channel that does drive bones (`左目` / `右目`).

The hard part isn't detection — it's the transformation. MediaPipe and MMD use different coordinate systems, every MMD model has its own rest-pose reference directions, and the bone hierarchy means each rotation has to be computed in its parent chain's local space.

**MiKaPo 3.0** — solver and capture pipeline rewritten by [Claude (Fable 5)](https://www.anthropic.com/claude): **60 FPS rendering with real-time capture**.

- **Web Worker detection** — MediaPipe holistic runs off the main thread; the WebGPU render loop never blocks on inference and holds 60 FPS during capture
- **Data-driven solver** — one bone-definition table + generic direction/basis/twist solvers replaced ~40 hand-written per-bone functions; parent chains computed once per frame via cached world rotations, matrix inversion replaced by quaternion conjugation, zero allocations per frame (verified bit-equivalent to the 2.0 solver, 1.7× faster)
- **Solver math on [Reze Engine](https://github.com/AmyangXYZ/reze-engine)'s Vec3/Quat** — Babylon.js remains only in the debug skeleton preview (so you can see when a bad pose comes from MediaPipe, not the solver), lazily loaded
- **Roll witnesses** — the forearm/shin direction pins upper-arm and thigh roll, so elbow creases and knee planes orient correctly instead of being left to shortest-arc chance
- **Anatomical finger clamps** — swing-twist decomposition per finger with human flexion/spread ranges; noisy landmark frames can no longer bend fingers backward
- **Visibility gating + hold-last-pose** — off-frame or occluded limbs hold their pose instead of snapping to identity or chasing garbage landmarks
- **Adaptive motion interpolation** — pose tweens are sized to the measured detection interval, upsampling ~30 Hz capture to smooth 60 FPS motion; One-Euro filters run on media time so video seeks don't warp smoothing

**MiKaPo 2.0** rewrote the solver from scratch (hierarchical parent-chain solving, rest-pose auto-calibration, One-Euro filtering, swing-twist forearm), migrated Vite → Next.js, and moved rendering from [babylon-mmd](https://github.com/noname0310/babylon-mmd) to my custom WebGPU MMD renderer [Reze Engine](https://github.com/AmyangXYZ/reze-engine).

![](./screenshots/1.png)
![](./screenshots/2.png)
![](./screenshots/3.png)
![](./screenshots/3.webp)
![](./screenshots/4.webp)

Demo model: 深空之眼 - 裁暗之锋·塞尔凯特

## Features

- **Holistic capture** — body pose, both hands (21 points each), and face all run through one MediaPipe HolisticLandmarker pass
- **Body & hands → MMD bones** — 33-point pose drives upper/lower body, arms, legs, and per-finger phalanges; forearm twist via swing-twist decomposition
- **Face → MMD morphs** — face blendshapes convert directly to native MMD morph weights (`まばたき`, `あ`, `ワ`, `ウィンク`, `ウィンク右`); eye gaze drives `左目` / `右目` bones
- **Per-model calibration** — reference directions derived from each loaded MMD's rest pose at load time, so swapping models works without a config file
- **Three input modes** — webcam (live), uploaded video, single image
- **Custom model upload** — drop a PMX folder to swap the default avatar
- **VMD export** — record live capture to a standard MMD `.vmd` motion file (30fps)
- **WebGPU rendering** via [Reze Engine](https://github.com/AmyangXYZ/reze-engine)

## Stack

- **Detection** — [MediaPipe HolisticLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker), running in a Web Worker
- **Renderer** — [Reze Engine](https://github.com/AmyangXYZ/reze-engine) (custom WebGPU MMD)
- **Framework** — [Next.js 15](https://nextjs.org/)
- **UI** — Tailwind v4 + shadcn/ui

## Run locally

```bash
npm install
npm run dev
```

Then open [http://localhost:4000](http://localhost:4000).

## How the solver works

MediaPipe gives world-space 3D landmark positions per frame. MMD bones rotate in their parent's local frame, with each model defining its own rest orientation. The solver bridges these:

1. **Calibrate (once, on model load)** — read each rest-pose bone world position from the loaded MMD. Since the bone chain is identity at rest, world-space `parent → child` direction equals the parent-local reference direction.
2. **Solve (per frame, per bone)** — each bone is one row in a definition table (parent, landmark pair, optional roll witness / anatomical clamp). World rotations accumulate down the hierarchy in solve order, so every parent chain is computed exactly once; rotating into parent-local space is a quaternion conjugation, no matrices involved.
3. **Smooth** — pass each output quaternion through a [One-Euro filter](https://gery.casiez.net/1euro/) (on media time) to remove jitter without lag, then tween to display rate.

```typescript
// One row of the bone table drives the generic solver:
{ kind: "direction", name: "左ひじ", parent: "左腕", source: "pose",
  from: "left_elbow", to: "left_wrist" }

function solveDirection(def, out: Quat): void {
  const dir = landmarkDelta(def.source, def.from, def.to)     // world-space segment
  rotateVecInv(worlds[def.parent], dir, dir)                  // → parent-local (conjugate, no matrix)
  quatFromUnitVectors(getRef(def.name), dir.normalize(), out) // rest ref → live direction
  // then: optional roll witness (arms/legs), anatomical clamp (fingers)
}
```

### Notable cases

- **Forearm twist** (`左手捩` / `右手捩`) — uses swing-twist decomposition along the elbow's forearm axis. A naive Euler-based approach bleeds wrist roll into pitch/yaw and gimbals.
- **Lower body bend** (`下半身`) — 3-axis Gram-Schmidt basis from hip line + spine direction so the pelvis tilts forward when leaning, instead of staying vertical and kinking the spine at the waist.
- **Head** (`頭`) — single rotation matrix from a Gram-Schmidt basis (ear axis + ear→eye direction) decomposed to a quaternion, instead of two `FromUnitVectors` calls composed (which compounds error).
- **Ankle** (`左足首` / `右足首`) — calibrated from the `足首 → つま先` bone direction; runtime uses `ankle → foot_index` landmarks (not heel) so the rest and runtime measurement frames line up.
