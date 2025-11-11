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
  description: "Real-time motion capture for MMD models.",
  keywords: ["MMD", "MikuMikuDance", "motion capture", "mediapipe", "landmarks", "pose estimation"],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="select-none outline-none">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
        style={{ backgroundColor: "#4a044e" }}
      >
        {children}
      </body>
      <Analytics />
    </html>
  )
}
