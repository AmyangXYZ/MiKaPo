import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { VitePWA } from "vite-plugin-pwa"

// Custom plugin to handle the problematic wasm-bindgen-rayon file
const wasmBindgenPlugin = () => {
  return {
    name: "wasm-bindgen-fix",
    transform(_: string, id: string) {
      if (id.includes("wasm-bindgen-rayon") && id.includes("workerHelpers.js")) {
        // Provide a dummy startWorkers export
        return {
          code: "export function startWorkers() { /* dummy */ }",
          map: null,
        }
      }
      return null
    },
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    wasmBindgenPlugin(),
    VitePWA({
      registerType: "autoUpdate",
      devOptions: {
        enabled: true,
      },
      manifest: {
        name: "Mikapo",
        short_name: "Mikapo",
        description: "AI Pose Picker For MMD",
        theme_color: "#000000",
        start_url: "/",
        display: "standalone",
        icons: [
          {
            src: "/logo.png",
            sizes: "192x192",
            type: "image/png",
          },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,ico,png,svg}"],
        maximumFileSizeToCacheInBytes: 10 * 1024 * 1024, // 10 MB
      },
    }),
  ],
  optimizeDeps: {
    exclude: ["babylon-mmd", "pose_solver"],
  },
  build: {
    target: "esnext",
    rollupOptions: {
      output: {
        format: "es",
        entryFileNames: "index.js",
        chunkFileNames: "index.js",
        assetFileNames: "[name].[ext]",
      },
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
})
