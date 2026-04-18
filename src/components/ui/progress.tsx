import * as React from "react"

import { cn } from "@/lib/utils"

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number
  indeterminate?: boolean
}

function Progress({ className, value = 0, indeterminate = false, ...props }: ProgressProps) {
  const clamped = Math.min(100, Math.max(0, value))
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : clamped}
      className={cn("relative h-2 w-full overflow-hidden rounded-full bg-white/20", className)}
      style={{ isolation: "isolate", contain: "paint" }}
      {...props}
    >
      {indeterminate ? (
        <div
          className="absolute inset-y-0 left-0 h-full w-1/3 bg-white"
          style={{
            animation: "progress-sweep 1.5s linear infinite",
            willChange: "transform",
            transform: "translate3d(-100%, 0, 0)",
            backfaceVisibility: "hidden",
          }}
        />
      ) : (
        <div
          className="h-full bg-white transition-[width] duration-100 ease-linear"
          style={{ width: `${clamped}%`, willChange: "width" }}
        />
      )}
    </div>
  )
}

export { Progress }
