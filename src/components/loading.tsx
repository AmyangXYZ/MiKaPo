"use client"

interface LoadingProps {
  modelLoaded: boolean
  mediaPipeReady: boolean
}

export default function Loading({ modelLoaded, mediaPipeReady }: LoadingProps) {
  if (modelLoaded && mediaPipeReady) return null

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-4 p-6 text-white">
      <div className="text-md font-medium flex items-baseline">
        <span>Loading MediaPipe vision model and MMD character</span>
        <Dot delay="0s" />
        <Dot delay="0.2s" />
        <Dot delay="0.4s" />
      </div>
    </div>
  )
}

function Dot({ delay }: { delay: string }) {
  return (
    <span
      className="inline-block"
      style={{
        animation: `loading-dot 1.4s ease-in-out infinite`,
        animationDelay: delay,
        willChange: "opacity",
      }}
    >
      .
    </span>
  )
}
