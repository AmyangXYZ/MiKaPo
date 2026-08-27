"use client"

// A floating panel's geometry, remembered across reloads.
//
// The window it was placed in may be gone by the time it comes back — a
// smaller laptop screen, a different monitor — so every read is re-clamped
// into the viewport that actually exists now, and so is every resize.

import { useCallback, useState } from "react"
import { clampRect, type Rect } from "@/components/floating-panel"

export function useStoredRect(
  key: string,
  fallback: () => Rect,
  minW: number,
  minH: number,
): [Rect, (r: Rect) => void] {
  // Read at first render rather than in an effect: the panel is client-only
  // (it portals to document.body), so there is no server HTML for a restored
  // position to disagree with, and no first frame at the wrong place.
  const [rect, setRect] = useState<Rect>(() => {
    if (typeof window === "undefined") return fallback()
    try {
      const raw = localStorage.getItem(key)
      if (raw) {
        const p = JSON.parse(raw) as Partial<Rect>
        if (
          typeof p?.x === "number" &&
          typeof p?.y === "number" &&
          typeof p?.w === "number" &&
          typeof p?.h === "number"
        ) {
          return clampRect(p as Rect, minW, minH)
        }
      }
    } catch {
      // Unparseable or unavailable (private mode) — the fallback stands.
    }
    return fallback()
  })

  const commit = useCallback(
    (r: Rect) => {
      setRect(r)
      try {
        localStorage.setItem(key, JSON.stringify(r))
      } catch {
        // Full or unavailable: the panel still moves, it just won't be
        // remembered. Never a reason to fail the drag.
      }
    },
    [key],
  )

  return [rect, commit]
}
