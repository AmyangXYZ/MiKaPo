# MiKaPo: AI Pose Picker for MikuMikuDance

MiKaPo is a **Web-based tool** that poses MMD models from video input in real-time.

## Tech Stack

- 3D key points detection: [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js)
- 3D scene: [Babylon.js](https://www.babylonjs.com/)
- MMD model viewer: [babylon-mmd](https://github.com/noname0310/babylon-mmd)
- Web framework: [Vite+React](https://vitejs.dev/)

## Todo

- [x] Pose detection
- [x] Face detection
- [ ] Hand detection
- [x] Video upload
- [x] Camera input
- [ ] Model selection
- [ ] Ollama support (electron version)

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Type-Check, Compile and Minify for Production

```sh
npm run build
```

### Lint with [ESLint](https://eslint.org/)

```sh
npm run lint
```
