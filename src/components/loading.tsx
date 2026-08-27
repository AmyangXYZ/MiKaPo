"use client"

// The app's one loading indicator: a pill over the viewport, naming whichever
// half is still missing. Two things load independently — the character and the
// vision model — and saying which one is still out is the difference between
// waiting and wondering.

interface LoadingProps {
  modelLoaded: boolean
  mediaPipeReady: boolean
}

export default function Loading({ modelLoaded, mediaPipeReady }: LoadingProps) {
  if (modelLoaded && mediaPipeReady) return null

  const label = !modelLoaded && !mediaPipeReady
    ? "Loading the character and the vision model"
    : !modelLoaded
      ? "Loading the character"
      : "Loading the vision model"

  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center">
      <div className="flex max-w-[90vw] items-center gap-2.5 rounded-full border border-line-strong bg-surface-raised px-4 py-2 text-xs text-muted-foreground tabular-nums backdrop-blur-xs">
        <span className="size-2 shrink-0 animate-pulse rounded-full bg-blue-400" />
        <span className="truncate">{label}</span>
      </div>
    </div>
  )
}
