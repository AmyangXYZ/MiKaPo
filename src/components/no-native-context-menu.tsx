"use client"

// Suppresses the browser's native context menu site-wide: a right-click on a
// viewport that orbits should not offer "Save image as…". Text entry keeps its
// menu, where copy and paste are the point.

import { useEffect } from "react"

export function NoNativeContextMenu() {
  useEffect(() => {
    const onCtx = (e: MouseEvent) => {
      const t = e.target as HTMLElement | null
      if (t?.closest('input, textarea, [contenteditable]:not([contenteditable="false"])')) return
      e.preventDefault()
    }
    document.addEventListener("contextmenu", onCtx)
    return () => document.removeEventListener("contextmenu", onCtx)
  }, [])
  return null
}
