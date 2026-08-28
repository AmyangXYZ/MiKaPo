import type { NextConfig } from "next"
// import { join } from "path"

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {
  // outputFileTracingRoot: join(__dirname, ".."),
  reactStrictMode: false,
  devIndicators: false,
  images: {
    remotePatterns: [{ protocol: "https", hostname: "assets.reze.one" }],
  },
}

module.exports = nextConfig
