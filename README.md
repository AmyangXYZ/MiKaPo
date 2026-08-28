# REZE MIPO: Real-time MMD Motion Capture

_(previously MiKaPo)_

Drive MikuMikuDance models — **full body, both hands, and face** — from a webcam, video, or photo, in the browser. Real time, one pass, no preprocessing, exporting straight to VMD.

**[Try it](https://mikapo.reze.one)** · Demo model: 深空之眼 - 裁暗之锋·塞尔凯特

One piece of the **Reze MMD family**, covering the whole MMD workflow on the web:

|                                                         |                                                                                |
| ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [reze-engine](https://github.com/AmyangXYZ/reze-engine) | The WebGPU foundation — anime-character rendering and physics, dependency-free |
| [reze-design](https://github.com/AmyangXYZ/reze-design) | Scene design, rendering and sharing platform                                   |
| [reze-studio](https://github.com/AmyangXYZ/reze-studio) | Animation editing on a professional timeline and curve editor                  |
| **Reze MiPo**                                           | This repo — real-time motion capture in the browser, exporting straight to VMD |
| [reze-rig](https://github.com/AmyangXYZ/reze-rig)       | Retarget FBX animations to MMD VMD format, Mixamo and Unity tested             |

## Features

- **Holistic capture** — body, both hands (21 points each) and face in one MediaPipe pass, detection running in a Web Worker so rendering holds 60 FPS
- **Body and hands drive bones** — 33 pose points become per-bone quaternions in each bone's parent-local frame, fingers included
- **Face drives morphs** — blendshapes convert to native MMD morph weights (`まばたき`, `あ`, `ワ`, `ウィンク`), the way MMD models are actually rigged; eye direction drives `左目` / `右目`
- **Any model** — reference directions are calibrated from whatever PMX you load, so swapping characters needs no config file and no T-pose
- **In-place capture** — `センター` carries height, so crouches and level changes read, while the root never wanders off its mark
- **VMD export** — a video is stepped frame by frame at 30 fps, so a slow machine writes the same file as a fast one; a still exports as a one-frame pose
- **Your assets stay** — an uploaded model or video is remembered across a refresh

![](./screenshots/1.png)
![](./screenshots/2.png)
![](./screenshots/3.png)
![](./screenshots/3.webp)
![](./screenshots/4.webp)

## Stack

- **Detection** — [MediaPipe HolisticLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker) in a Web Worker
- **Renderer** — [Reze Engine](https://github.com/AmyangXYZ/reze-engine) (custom WebGPU MMD)
- **Framework** — [Next.js 16](https://nextjs.org/), Tailwind v4, shadcn/ui

## Run locally

```bash
npm install
npm run dev
```

Then open [http://localhost:4000](http://localhost:4000).

---

## How it works

**Detection is a solved problem you can download. The transformation is the product.**

MediaPipe gives 3D landmark positions in a world centred on the subject's hip. MMD wants a **quaternion per bone in that bone's parent-local frame**, on a skeleton whose rest pose belongs to whoever modelled the character. Nothing in the landmark stream says what a shoulder's rotation is; it only says where the elbow ended up. Three facts shape everything:

1. **A direction is not a rotation.** A landmark pair says where a bone points; a bone that points somewhere can still spin about its own axis. Roll is unobservable from a segment, and shortest-arc solving silently invents it.
2. **Depth is a guess, and the hips are the origin.** x/y are measured from the image, z is inferred and far noisier — and because coordinates are hip-centred, the pelvis's own translation has been subtracted out entirely.
3. **The model is not the person.** A rotation correct for a body is wrong for a character unless it is measured against that character's own bind pose.

### The pipeline

1. **Calibrate, once per model** — at rest every parent chain is identity, so the world-space `parent → child` direction _is_ the parent-local reference direction. Read them when the PMX loads.
2. **Solve, per frame** — one generic pass over a bone table; each row names a bone, its parent, and the two landmarks whose difference points it. Bones are solved in hierarchy order so each parent chain is computed exactly once, rotating into parent-local space is a quaternion conjugation rather than a matrix inverse, and the frame allocates nothing.
3. **Smooth** — One-Euro on media time, driven by true angular velocity and bounded by what a limb could physically have done since the last frame.
4. **Clear the body** — the model's own rigid bodies define where an arm may not go; an arm inside the chest is swung out at the shoulder, rotation only.
5. **Place it** — walk both legs from the hips and drop the body until the lower foot rests where it rests in the bind pose.

```typescript
// One row of the table drives the generic solver:
{ kind: "direction", name: "左ひじ", parent: "左腕", source: "pose",
  from: "left_elbow", to: "left_wrist" }

function solveDirection(def, out: Quat): boolean {
  const dir = landmarkDelta(def.source, def.from, def.to)     // world-space segment
  rotateVecInv(worlds[def.parent], dir, dir)                  // → parent-local (conjugate, no matrix)
  quatFromUnitVectors(getRef(def.name), dir.normalize(), out) // rest ref → live direction
  // then: optional roll witness (arms/legs), anatomical clamp (fingers)
}
```

### The tricks

The parts that came out of measuring rather than reasoning:

- **Roll needs a witness.** The forearm's direction is what says how the upper arm is twisted. Each arm and thigh names a child segment as its roll witness, used only to the extent the roll is observable — faded by the sine of the child joint's bend, so a straight limb falls back to shortest-arc instead of to noise. Thighs get the foot as a second witness: with the knee extended, where the toes point is where the femur is rotated.
- **Roll is also the noisiest channel, and invisible in a landmark preview** — it changes no bone _direction_, so the debug skeleton looks clean while the character shimmers. Arms and thighs decompose into swing and twist, and the twist alone gets a much calmer filter.
- **Distrust z as policy.** Every landmark's z channel is low-passed before solving; x and y pass through untouched. Torso wobble went from 6.7°/frame to 0.3°.
- **In-place on purpose.** The deleted pelvis translation _can_ be rebuilt from 2D projective geometry, and both axes are implemented — but they drift over a session, and a capture that artists re-edit is better with no root motion than with root motion that wanders.
- **Placed by its lowest part, not its feet.** Taking the _lower_ foot means a standing split measures against the standing foot; a hip-clearance candidate in the same `max()` means a body rolling on the floor rests on its hips instead of being hauled around by legs in the air.
- **Dropouts crossfade, they don't freeze** — out to rest over 500ms, back in over 250ms. Holding forever is right for one dropped frame and wrong for two seconds. Hands add hysteresis, because MediaPipe hands flicker and a three-frame hand is usually garbage.
- **A collapsed hand still has 21 valid points.** Hand landmarks carry no visibility field, and when tracking fails they don't vanish — they collapse toward the origin, structurally perfect and completely wrong. The hand's own span and the pose model's wrist visibility gate them.
- **The trunk is three joints.** The chest rotation splits evenly across `上半身` and `上半身2` so the spine curves; `下半身` takes a quarter of the mean thigh pitch, which is what makes sitting read; the shoulder-line tilt — which the trunk basis otherwise orthogonalizes away, making a shrug invisible to the solver — is handed to the `肩` bones.
- **The clavicle carries its share.** Anatomy splits elevation roughly 2:1, so the shoulder takes a fraction of the arm's rotation and the arm gives back exactly that much: it still points where the landmarks put it, only the joint that bends changes.
- **穿模 answered with the model's own rigid bodies.** MMD never tests those capsules against each other — they are bone-following statics — which is exactly how a hand ends up inside a chest.
- **An acceleration limit, not a speed limit.** A real limb ramps up over several frames; a detection glitch arrives at full speed from nothing. On the first frame that is the only thing telling them apart.
- **The display plays, it doesn't chase.** A cursor advances along the media timeline and interpolates the two results bracketing it, with a small latency margin — so detection-arrival jitter cannot modulate playback. Against a chase tween: ~10ms less latency, ~7 points more amplitude on fast motion.
- **Exports read the future.** A Savitzky-Golay pass fits a local polynomial rather than averaging, so shake goes without flattening a kick: 59% less jitter, 99% of peak amplitude, zero phase shift.

### Notable cases

- **Forearm twist** (`左手捩` / `右手捩`) — swing-twist decomposition along the elbow's forearm axis; a Euler approach bleeds wrist roll into pitch and gimbals.
- **Head** (`頭`) — one Gram-Schmidt basis (ear axis + ear→eye) decomposed to a quaternion, rather than two `FromUnitVectors` calls composed, which compounds error.
- **Ankle** (`左足首` / `右足首`) — calibrated from `足首 → つま先`, measured at runtime as `ankle → foot_index`, so the rest and live frames line up.

**Where the ceiling is.** The solver adds ~7–17ms of filter latency on top of MediaPipe's ~30 Hz cadence, keeps 100% of a snap's amplitude, and filters jitter to near the landmark-noise floor. For the body, the limit is now the detector's landmarks rather than the transformation. Fingers at full-body framing distance are the honest exception, and the fix is known: crop around the wrists and run a dedicated hand pass.
