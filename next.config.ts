import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  reactStrictMode: false,
  devIndicators: false,
  // Cross-origin isolation unlocks multithreaded wasm for the ONNX worker's
  // CPU fallback. "credentialless" (not "require-corp") so cross-origin
  // subresources — CDN wasm, HF model, MediaPipe assets — keep loading
  // without needing CORP headers of their own.
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
    ]
  },
}

export default nextConfig
