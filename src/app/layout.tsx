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
  title: "MiKaPo - MMD Motion Capture",
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
      <body className="h-full bg-black text-foreground">
        <NoStickyFocus />
        <NoNativeContextMenu />
        <TooltipProvider delayDuration={200}>{children}</TooltipProvider>
      </body>
      <Analytics />
    </html>
  )
}
