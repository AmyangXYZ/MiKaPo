"use client"

import dynamic from "next/dynamic"

const MainScene = dynamic(() => import("@/components/main-scene"), {
  ssr: false,
})

export default function Home() {
  return (
    <div className="w-full h-screen">
      <MainScene />
    </div>
  )
}
