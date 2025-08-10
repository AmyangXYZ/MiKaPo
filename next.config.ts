import type { NextConfig } from "next"

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {
  reactStrictMode: false,
  devIndicators: false,
  webpack: (config) => {
    // Handle babylon-mmd WASM files
    config.module.rules.push({
      test: /\.wasm$/,
      type: "asset/resource",
    })

    // Fix the wasm-bindgen-rayon workerHelpers issue
    config.module.rules.push({
      test: /workerHelpers\.js$/,
      use: {
        loader: "string-replace-loader",
        options: {
          search: /self/g,
          replace: 'typeof self !== "undefined" ? self : globalThis',
        },
      },
    })

    // Ensure proper module resolution for .js extensions
    config.resolve.extensions = [...(config.resolve.extensions || []), ".js", ".mjs"]

    return config
  },
  // Ensure proper headers for SharedArrayBuffer if needed
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Cross-Origin-Opener-Policy",
            value: "same-origin",
          },
          {
            key: "Cross-Origin-Embedder-Policy",
            value: "require-corp",
          },
        ],
      },
    ]
  },
}

module.exports = nextConfig
