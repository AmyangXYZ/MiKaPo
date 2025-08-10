import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"
import { Analytics } from "@vercel/analytics/next"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "MiKaPo - MMD Motion Capture",
  description:
    "Transform video input into real-time MMD model poses using MediaPipe 3D landmarks and hierarchical bone transformations. Web-based motion capture for MikuMikuDance with face, hand, and body tracking.",
  keywords: [
    "MMD",
    "MikuMikuDance",
    "motion capture",
    "real-time",
    "MediaPipe",
    "3D landmarks",
    "pose estimation",
    "bone rotations",
    "quaternions",
    "Babylon.js",
    "web-based",
    "computer vision",
    "animation",
    "character posing",
  ],
  authors: [{ name: "MiKaPo Team" }],
  openGraph: {
    title: "MiKaPo - Real-time MMD Motion Capture",
    description:
      "Transform video input into real-time MMD model poses using advanced 3D landmark detection and hierarchical bone transformations.",
    type: "website",
    url: "https://mikapo.vercel.app",
  },
  twitter: {
    card: "summary_large_image",
    title: "MiKaPo - Real-time MMD Motion Capture",
    description: "Web-based motion capture for MMD models using MediaPipe and hierarchical bone transformations.",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="select-none outline-none">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>{children}</body>
      <Analytics />
    </html>
  )
}
