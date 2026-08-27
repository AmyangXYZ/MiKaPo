# Solver improvement roadmap

Research handoff, 2026-08-27. Source: a full read of SystemAnimatorOnline /
XR Animator (github.com/ButzYung/SystemAnimatorOnline) — the most battle-tested
browser MediaPipe→MMD mocap, same input (holistic landmarks) and output (MMD
bones) as this app. Its solver ships minified but was beautified and fully
readable; findings below are mechanisms, with our adoption ranked.

**Licensing: that repo has NO license file (all rights reserved; the author
monetizes early access). Reimplement the ideas — never copy code.**

Our solver: `src/lib/solver.ts`, filters `src/lib/filters.ts`, face
`src/lib/face-blendshape-solver.ts`. Detection: `src/lib/pose-worker.ts`
(HolisticLandmarker in a worker, model self-hosted at
`${ASSETS}/holistic_landmarker.task`).

## Ranked adoptions

1. **Distrust MediaPipe z as policy.** XRA filters ONLY the z channel of every
   landmark (OneEuro 30Hz, minCut 1, beta 1, dCut 2), trusting MediaPipe's own
   x/y smoothing. Depth is then rebuilt from 2D projective geometry wherever a
   length is known: hip depth (センター z) from projected 2D spine length vs the
   model's known spine length, foreshortening-corrected by torso pitch
   (`/|cos(pitch)|`); hand depth from palm size vs shoulder width. We already
   calibrate all segment lengths from the PMX — the ingredients exist.
   Expected: less front-back torso wobble, correct toward-camera motion.

2. **Per-bone dropout crossfade, not freeze-forever.** A bone entering tracking
   blends in over 250ms; a bone losing tracking blends to rest over 500ms,
   cosine-eased, per bone. Hands add hysteresis: the arm follows only after the
   hand has been visible ~1s continuously; on re-acquisition, filter cutoffs
   start clamped low and ramp up over ~1s. We currently hold the last rotation
   on visibility loss — right for one dropped frame, wrong for two seconds.
   Expected: no frozen limbs, no re-acquisition pops.

3. **Trunk decomposition.** Measured shoulder-vs-hip yaw is split half into
   上半身, half into 上半身2 (we leave 上半身2 unsolved). Shoulder-line roll is
   shared with the 肩 bones (scaled, clamped ±15°, damped on whichever arm is
   raised). 下半身 pitch = trunk pitch + 0.25 × mean thigh pitch, so sitting
   and crouching read. Expected: natural torso twist in dance, believable
   sitting — our named "wobbly shoulders/hips" weakness.

4. **Dedicated hand pipeline: crops.** XRA crops regions around the pose wrists
   (radius from shoulder width), upscales, and runs a separate HandLandmarker
   on the crops — optionally in a second worker. Gating: discard hands whose
   handedness contradicts their crop (or flip them), reject hands farther than
   ~shoulder-width² from the pose wrist, two detector instances at confidence
   0.5/0.1 switched by subject size, 3-frame grace before declaring hands gone.
   The single biggest finger-quality lever at full-body framing distance.

5. **Hand geometry repair before solving.** Palm aspect ratio outside
   [1.25, 1.75] ⇒ solve an analytic z-rescale returning it to 1.5. Per finger,
   a phalanx shorter than its reference length (foreshortened = pointing at
   camera) gets the missing length restored as z displacement propagated to
   distal joints. Hand SHAPE is then filtered palm-relative, separately from
   hand position. Expected: fingers pointing at the camera stop collapsing.

6. **Foot basis.** Build a full 3-axis foot basis from ankle→heel and
   heel→foot_index, apply a fixed −π/8 pitch-bias correction (MediaPipe's foot
   plane is systematically tilted), fade to pelvis-yaw-only as the body turns
   away, fade to identity as legs leave frame. We use a single direction — no
   roll, no bias, no fades.

7. **Adaptive per-signal filtering.** ~25 named OneEuro instances, each on one
   semantic signal (spine, hip, camera-depth, per-leg segment directions…),
   cutoffs adapted by subject size / distance / hand-visibility state. Leg
   segment directions are filtered normalized-then-length-restored so filtering
   never changes bone length. We run one global tuning for every bone quat.

8. **Wrist-twist unwrap into 手捩.** Express the wrist target relative to the
   chest (上半身2), not the forearm — decouples hand orientation from
   elbow-depth noise. Extract forearm twist in T-pose space; if the angle jumps
   >π within 250ms, unwrap by ±2π (allow 1.5π in the natural direction). Half
   the twist onto 手捩 about its PMX fixedAxis, half stays on 手首. Kills the
   360° twist-flip glitch class.

9. **Camera-tilt dial + lean cue.** A user-set camera pitch premultiplied into
   head/trunk/arm targets (laptop webcams look up at people). Upper-body mode
   estimates lean from running-average face width. Also a nonlinear
   forward-bend suppressor.

10. **Hardening.** NaN guards restoring last-good on every write; a hip
    teleport beyond threshold triggers a physics reset (we have filter reseed
    but no physics reset — matters for skirts/hair).

## Where we are already equal or better — do not regress

- Quaternion OneEuro **with velocity/acceleration plausibility clamps** — XRA
  has no clamps at all. Keep ours.
- Offline zero-phase smoothing (`smoothTakeZeroPhase`) for conversions — XRA
  has nothing comparable.
- Thigh-roll witness with observability smoothstep; body clearance derived from
  the model's own rigid bodies (XRA uses user-percent spheres).
- Neither app does true stance-foot locking; XRA does not solve it either.

## Loose thread worth one experiment

XRA obtains ARKit-52 blendshapes from HolisticLandmarker with
`outputFaceBlendshapes: true` on the GPU delegate. Re-test our assumption that
the blendshape subgraph doesn't run under holistic — it may be stale.
