import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

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
  plugins: [react(), wasmBindgenPlugin()],
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
