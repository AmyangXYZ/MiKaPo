#!/usr/bin/env node
// Record a video's landmarks to a fixture, headless.
//
//   node harness/record.mjs flash.mp4 [--fps 30] [--width 960]
//                            [--start 12 --seconds 20] [--name turns]
//
// Chrome is launched with no window, pointed at a page that runs the real
// MediaPipe graph over the real video, and the landmarks it produces are
// written to harness/fixtures/. Detection is the only part of the pipeline
// that needs a browser; everything downstream replays from the fixture in
// plain Node, as many times as an experiment needs.

import { spawn, execFileSync } from "node:child_process"
import { createServer } from "node:http"
import { readFile, writeFile, mkdir, readdir } from "node:fs/promises"
import { existsSync } from "node:fs"
import { extname, join, basename } from "node:path"
import { fileURLToPath } from "node:url"

const HERE = fileURLToPath(new URL(".", import.meta.url))
const ROOT = join(HERE, "..")
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
const PORT = 8137

const args = process.argv.slice(2)
const video = args.find((a) => !a.startsWith("--")) ?? "flash.mp4"
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 ? args[i + 1] : fallback
}
const fps = flag("fps", "30")
const width = flag("width", "960")
const start = flag("start", "0")
const seconds = flag("seconds", "0")
const name = flag("name", "")
const presence = flag("presence", "0.7")
const detection = flag("detection", "0.7")
const hands = flag("hands", "0.95")
const face = flag("face", "0.4")
const model = flag("model", "holistic")

if (!existsSync(CHROME)) {
  console.error("Chrome not found at", CHROME)
  process.exit(1)
}

// Frames are decoded up front by ffmpeg, at the rate and size the recorder
// will use. Handing the detector exact files is what makes a recording
// reproducible — and it is the only way to be sure every frame is a different
// frame, which scrubbing a video element in a headless browser is not.
const FRAMES_DIR = "/tmp/mipo-harness-frames"
console.log("decoding frames…")
execFileSync("rm", ["-rf", FRAMES_DIR])
execFileSync("mkdir", ["-p", FRAMES_DIR])
execFileSync("ffmpeg", [
  "-v", "error",
  ...(Number(start) > 0 ? ["-ss", String(start)] : []),
  "-i", join(ROOT, "public", video),
  ...(Number(seconds) > 0 ? ["-t", String(seconds)] : []),
  "-vf", `fps=${fps},scale=${width}:-2`,
  "-q:v", "2",
  join(FRAMES_DIR, "%05d.jpg"),
])
const frameFiles = (await readdir(FRAMES_DIR)).filter((f) => f.endsWith(".jpg")).sort()
const probe = execFileSync("ffprobe", [
  "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
  "-of", "csv=p=0", join(FRAMES_DIR, frameFiles[0]),
]).toString().trim().split(",")
console.log(`  ${frameFiles.length} frames at ${probe[0]}×${probe[1]}`)

// The recorder is bundled once per run, so editing it needs no build step.
console.log("bundling recorder…")
const bundle = execFileSync(
  "npx",
  ["esbuild", join(HERE, "recorder.js"), "--bundle", "--format=esm", "--log-level=warning"],
  { cwd: ROOT, maxBuffer: 1 << 28 },
)

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".mp4": "video/mp4",
  ".task": "application/octet-stream",
  ".json": "application/json",
}

let resolveResult
const finished = new Promise((res) => (resolveResult = res))

const page = `<!doctype html><meta charset="utf-8"><title>recorder</title>
<body style="background:#111"><script type="module" src="/recorder.js"></script>`

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`)
  const path = decodeURIComponent(url.pathname)

  if (req.method === "POST") {
    const chunks = []
    for await (const c of req) chunks.push(c)
    const body = Buffer.concat(chunks)
    if (path === "/log") console.log("  ·", body.toString())
    else if (path === "/fail") {
      console.error("\npage failed:\n" + body.toString())
      res.end("ok")
      server.close()
      chrome.kill()
      process.exit(1)
    } else if (path === "/result") resolveResult(JSON.parse(body.toString()))
    res.end("ok")
    return
  }

  const serve = async (file) => {
    try {
      const buf = await readFile(file)
      res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" })
      res.end(buf)
    } catch {
      res.writeHead(404).end("not found")
    }
  }

  if (path === "/") {
    res.writeHead(200, { "content-type": "text/html" }).end(page)
  } else if (path === "/recorder.js") {
    res.writeHead(200, { "content-type": "text/javascript" }).end(bundle)
  } else if (path.startsWith("/wasm/")) {
    await serve(join(ROOT, "node_modules/@mediapipe/tasks-vision/wasm", basename(path)))
  } else if (path === "/frames.json") {
    res.writeHead(200, { "content-type": "application/json" }).end(
      JSON.stringify({ count: frameFiles.length, width: Number(probe[0]), height: Number(probe[1]), start: Number(start) }),
    )
  } else if (path.startsWith("/frame/")) {
    const n = Number(path.slice("/frame/".length))
    await serve(join(FRAMES_DIR, frameFiles[n]))
  } else if (path.startsWith("/media/")) {
    await serve(join(ROOT, "public", path.slice("/media/".length)))
  } else {
    res.writeHead(404).end("not found")
  }
})

server.listen(PORT)
const target =
  `http://127.0.0.1:${PORT}/?video=${encodeURIComponent(video)}&fps=${fps}&width=${width}` +
  `&start=${start}&seconds=${seconds}` +
  `&presence=${presence}&detection=${detection}&hands=${hands}&face=${face}&model=${model}`
console.log("recording", video, "→", target)

const chrome = spawn(
  CHROME,
  [
    "--headless=new",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--autoplay-policy=no-user-gesture-required",
    // MediaPipe wants a GL context; software rendering is fine and portable.
    "--use-gl=swiftshader",
    "--enable-unsafe-swiftshader",
    `--user-data-dir=/tmp/mipo-harness-chrome`,
    target,
  ],
  { stdio: ["ignore", "ignore", "pipe"] },
)
chrome.stderr.on("data", (d) => {
  const line = String(d).trim()
  // Chrome is noisy about things that do not matter here.
  if (/ERROR|FATAL/.test(line) && !/GPU|gl_|Vulkan|dawn|voice/i.test(line)) console.error("  chrome:", line)
})

const timeout = setTimeout(() => {
  console.error("timed out with no result")
  chrome.kill()
  process.exit(1)
}, 10 * 60_000)

const result = await finished
clearTimeout(timeout)
chrome.kill()
server.close()

await mkdir(join(HERE, "fixtures"), { recursive: true })
const out = join(HERE, "fixtures", `${name || basename(video, extname(video))}.json`)
await writeFile(out, JSON.stringify(result))
const withPose = result.frames.filter((f) => f.pose).length
const withFace = result.frames.filter((f) => f.faceSeen).length
console.log(
  `\nwrote ${out}\n  ${result.frames.length} frames · pose in ${withPose} · face in ${withFace} · ${result.delegate}`,
)
process.exit(0)
