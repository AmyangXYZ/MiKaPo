import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"
import { Analytics } from "@vercel/analytics/next"
import { TooltipProvider } from "@/components/ui/tooltip"
import { NoStickyFocus } from "@/components/no-sticky-focus"
import { NoNativeContextMenu } from "@/components/no-native-context-menu"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "Reze MiPo - MMD Motion Capture",
  description: "Real-time motion capture for MMD models.",
  keywords: ["MMD", "MikuMikuDance", "motion capture", "mediapipe", "landmarks", "pose estimation"],
  // Machine translators rewrite text nodes React still holds a handle on,
  // which crashes the reconciler on the next removeChild.
  other: { google: "notranslate" },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      translate="no"
      className={`dark notranslate h-full select-none antialiased ${geistSans.variable} ${geistMono.variable}`}
    >
      {/* The ground plane is magenta; the page behind the viewport is its
          deep end, so a slow load and a loaded scene are the same colour. */}
      <body className="h-full text-foreground" style={{ backgroundColor: "#4a044e" }}>
        <NoStickyFocus />
        <NoNativeContextMenu />
        <TooltipProvider delayDuration={200}>{children}</TooltipProvider>
      </body>
      <Analytics />
    </html>
  )
}
