import type { NextConfig } from "next"
import { version } from "./package.json"
// import { join } from "path"

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {
  // The app stamps its own version in the header; read it from the one place
  // it is already declared.
  env: { NEXT_PUBLIC_APP_VERSION: version },
  // outputFileTracingRoot: join(__dirname, ".."),
  reactStrictMode: false,
  devIndicators: false,
  images: {
    remotePatterns: [{ protocol: "https", hostname: "assets.reze.one" }],
  },
}

module.exports = nextConfig
