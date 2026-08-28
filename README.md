# REZE MIPO: Real-time MMD Motion Capture

_(previously MiKaPo)_

Dance in front of your webcam. Watch your MMD character dance with you, live, in the browser. Full body, both hands, and face, at 60 FPS, and every take saves as a `.vmd` you can open in MMD.

**[Try it](https://mipo.reze.one)** · Demo model: 深空之眼 - 裁暗之锋·塞尔凯特

One piece of the **Reze MMD family**, covering the whole MMD workflow on the web:

|                                                         |                                                                                |
| ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [reze-engine](https://github.com/AmyangXYZ/reze-engine) | The WebGPU foundation — anime-character rendering and physics, dependency-free |
| [reze-design](https://github.com/AmyangXYZ/reze-design) | Scene design, rendering and sharing platform                                   |
| [reze-studio](https://github.com/AmyangXYZ/reze-studio) | Animation editing on a professional timeline and curve editor                  |
| **Reze MiPo**                                           | This repo — real-time motion capture in the browser, exporting straight to VMD |
| [reze-rig](https://github.com/AmyangXYZ/reze-rig)       | Retarget FBX animations to MMD VMD format, Mixamo and Unity tested             |

## What it does

- **Captures everything at once.** Body, both hands at 21 points each, and face, from a single MediaPipe pass running in a Web Worker while WebGPU holds 60 FPS.
- **Drives real MMD bones.** 33 pose points become per-bone quaternions in each bone's parent-local frame, fingers and forearm twist included.
- **Drives real MMD morphs.** Face blendshapes land on `まばたき`, `あ`, `ワ`, `ウィンク` — the morphs your model already has — and gaze drives `左目` / `右目`.
- **Takes your character.** Drop a PMX folder and the solver calibrates to that model's rest pose on load. Any model, no config, no T-pose.
- **Stands on the ground.** `センター` carries height, so crouches, level changes and weight drops all land in the file.
- **Writes MMD files.** A video steps frame by frame at 30 fps for a result that is identical on any machine. A photo saves as a one-frame pose.
- **Remembers your work.** Your model and your footage survive a refresh.

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

MediaPipe hands you 3D points around a hip-centred origin. MMD wants a quaternion per bone in that bone's parent-local frame, on a skeleton whose rest pose belongs to whoever modelled the character. **Everything MiPo does lives in that gap**, and the whole pipeline is one generic pass over a bone table:

```typescript
// One row drives the solver:
{ kind: "direction", name: "左ひじ", parent: "左腕", source: "pose",
  from: "left_elbow", to: "left_wrist" }

function solveDirection(def, out: Quat): boolean {
  const dir = landmarkDelta(def.source, def.from, def.to)     // world-space segment
  rotateVecInv(worlds[def.parent], dir, dir)                  // → parent-local (conjugate, no matrix)
  quatFromUnitVectors(getRef(def.name), dir.normalize(), out) // rest ref → live direction
  // then: roll witness (arms/legs), anatomical clamp (fingers)
}
```

Five steps per frame:

1. **Calibrate, once per model.** At rest every parent chain is identity, so the world-space `parent → child` direction _is_ the parent-local reference direction. Reading them at load is what lets any PMX work.
2. **Solve.** Bones go in hierarchy order, so each parent chain is computed exactly once. Rotating into parent-local space is a quaternion conjugation. The frame allocates nothing.
3. **Smooth.** One-Euro on media time, driven by true angular velocity, bounded by what a limb could physically have done since the last frame.
4. **Clear the body.** The model's own physics capsules define where an arm may go, and an arm inside the chest swings out at the shoulder.
5. **Place it.** Walk both legs from the hips and drop the body until the lower foot rests where it rests in the bind pose.

### The good parts

**Roll comes from a witness.** The forearm's direction reveals how the upper arm is twisted, so every arm and thigh names a child segment and builds full rest and live orthonormal bases from it, weighted by how observable the roll is at that bend angle. Thighs keep a second witness in reserve: with the knee straight, where the toes point is where the femur is rotated.

**Roll also gets its own filter.** Twist about a bone's own axis is where landmark noise concentrates and where a landmark preview can never show it, since roll moves no bone _direction_. Arms and thighs split into swing and twist, the twist rides a much calmer filter, and the character stops shimmering.

**The z channel is filtered alone.** x and y come off the image; z is inferred and carries the noise. Low-passing that one channel before solving took torso wobble from 6.7°/frame to 0.3°.

**Capture stays in place.** `センター` moves vertically, so squats and crouches read while the character holds its mark — exactly the base an animator wants to key over.

**The body rests on its lowest part.** Taking the lower foot means a standing split measures against the standing foot. A hip-clearance candidate in the same `max()` means a body rolling on the floor rests on its hips.

**Limbs that leave frame come home gently.** A bone holds its last measurement for 250ms, eases to rest over 500ms, and eases back in over 250ms. Ordinary detection gaps cost nothing at all.

**Hands earn their trust.** Hand landmarks collapse toward the origin when tracking fails, staying 21 structurally perfect points. The hand's own span and the pose model's wrist visibility gate them, and a hand engages after a full second of continuous tracking.

**The trunk is three joints.** Chest rotation splits evenly across `上半身` and `上半身2` so the spine curves. `下半身` takes a quarter of the mean thigh pitch, which is what makes sitting look like sitting. Shoulder-line tilt reaches the `肩` bones, so a shrug shrugs.

**The clavicle carries its share.** Anatomy splits elevation roughly 2:1, so the shoulder takes a fraction of the arm's rotation and the arm gives back exactly that much. The arm still points where the landmarks put it.

**Arms stay out of the chest.** An MMD model ships its author's own capsules; MiPo reads them as clearance volume and swings the shoulder just past contact, rotation only, so the pose keeps its shape.

**Fast moves land at full amplitude.** An acceleration limit tells a real strike from a detection glitch: a limb ramps up over several frames, a glitch arrives at full speed from nothing. Snaps reach 100% of their peak.

**The display plays.** A cursor advances along the media timeline and interpolates the two results bracketing it, so ~30 Hz capture becomes 60 FPS motion that arrival jitter never touches. Worth ~10ms of latency and 7 points of amplitude against a chase tween.

**Exports read the future.** A Savitzky-Golay pass fits a local polynomial over the finished take: 59% less jitter, 99% of peak amplitude, zero phase shift.

### Notable cases

- **Forearm twist** (`左手捩` / `右手捩`) — swing-twist decomposition along the elbow's forearm axis.
- **Head** (`頭`) — one Gram-Schmidt basis (ear axis + ear→eye) decomposed to a quaternion.
- **Ankle** (`左足首` / `右足首`) — calibrated from `足首 → つま先`, measured live as `ankle → foot_index`, so rest and runtime frames line up.

The solver adds 7–17ms on top of MediaPipe's ~30 Hz cadence and filters jitter to the landmark-noise floor: body capture now runs at the detector's own limit. **Next up:** wrist-cropped hand detection, for finger detail at full-body framing distance.
