import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { VitePWA } from "vite-plugin-pwa"

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      workbox: {
        maximumFileSizeToCacheInBytes: 10 * 1024 * 1024, // 5 MB
      },
    }),
  ],
  optimizeDeps: {
    exclude: ["babylon-mmd"],
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
})
