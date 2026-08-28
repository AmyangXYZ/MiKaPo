# Harness

Record what MediaPipe actually says about a real video, then replay it through
the real solver as many times as an experiment needs.

The split exists because detection needs a browser and analysis does not.
MediaPipe's vision tasks build a WebGL canvas through `document` before they
will look at a frame, so detection runs once in a headless Chrome and its
output is kept. Everything after that — solving, cleanup, filtering, export —
runs in plain Node against that recording, in seconds, deterministically.

## Record

```bash
node harness/record.mjs flash.mp4
node harness/record.mjs "some dance.mp4" --start 20 --seconds 25 --name dance
```

Reads from `public/`, writes `harness/fixtures/<name>.json`.

Frames are decoded by ffmpeg up front and served to the page as files. A
`<video>` was the obvious source and the wrong one: seeking it in a headless
browser resolves `seeked` before the new picture exists, so the detector is
handed the same frame repeatedly and every number taken from it describes a
still. Exact files cannot do that — and a recording is reproducible.

## Replay

```bash
npm run harness -- dance
```

Solves the fixture twice, once the way live capture does and once the way an
export does, and reports per bone:

- **°/frame** — how far the bone's world direction moves between frames.
  Measured in world space on purpose: a local rotation reports shake that the
  posed body does not have, because parents and children counter-rotate.
- **worst** — the largest single-frame step. This is where flips and pops show.
- **pops** — steps over 20°, which no limb does at 30fps.
- **range** — how far the bone travels around its own mean direction. Without
  it a still clip reads exactly like a well-solved one.

Above that it prints what the detector actually found: pose, face, and each
hand, as a band across the take. The face row matters more than it looks —
it is the only signal that says which way a body faces, and how often it goes
missing is what any use of it has to survive.

## What this is for

Synthetic input tests the hypothesis that produced it. A facing correction
shipped from this repo passed every synthetic turn it was built against and
flipped a body on real footage the first time it ran, because the synthetic
face signal was clean and MediaPipe's is not. Record the clip that breaks, and
measure the fix against it.
