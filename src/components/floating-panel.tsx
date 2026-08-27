"use client"

/** A free-floating window: drag it by any descendant marked [data-drag-handle],
 *  resize it from any edge or corner.
 *
 *  Shared with reze-design and reze-studio, minus the fullscreen mode and the
 *  z-order stack — MiKaPo has exactly one floating surface, so a single fixed
 *  layer is the whole stacking story. That layer sits BELOW the z-50 Radix
 *  uses for dialogs and menus: a modal is modal, and a capture panel that
 *  covered the PMX picker would be in the way of the thing that put it on
 *  screen. */

import { useEffect, useRef, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { cn } from "@/lib/utils"

export type Rect = { x: number; y: number; w: number; h: number }
type Mode = "move" | "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw"

/** Keep this many px between the panel and the viewport edge. */
const PAD = 8

/** Under Radix's z-50 overlay layer, over everything the app draws itself. */
const LAYER = 45

/** Clamp a rect back inside a viewport that may have shrunk under it. */
/** Drop the panel's backdrop blur for the duration of a drag. Compositing a
 *  blur over a live WebGPU canvas costs real frames, and a drag is exactly
 *  when the panel must keep up with the pointer. */
function setPerf(el: HTMLElement, active: boolean) {
  if (active) {
    el.style.setProperty("backdrop-filter", "none")
    el.style.setProperty("-webkit-backdrop-filter", "none")
    el.style.willChange = "transform"
  } else {
    el.style.removeProperty("backdrop-filter")
    el.style.removeProperty("-webkit-backdrop-filter")
    el.style.willChange = ""
  }
}

export function clampRect(r: Rect, minW: number, minH: number): Rect {
  if (typeof window === "undefined") return r
  const w = Math.max(minW, Math.min(r.w, window.innerWidth - PAD * 2))
  const h = Math.max(minH, Math.min(r.h, window.innerHeight - PAD * 2))
  return {
    w,
    h,
    x: Math.min(Math.max(PAD, r.x), Math.max(PAD, window.innerWidth - w - PAD)),
    y: Math.min(Math.max(PAD, r.y), Math.max(PAD, window.innerHeight - h - PAD)),
  }
}

export function FloatingPanel({
  rect,
  onRectChange,
  minW = 240,
  minH = 180,
  className,
  children,
}: {
  rect: Rect
  onRectChange: (r: Rect) => void
  minW?: number
  minH?: number
  className?: string
  children: ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const drag = useRef<{ mode: Mode; sx: number; sy: number; start: Rect } | null>(null)
  const latest = useRef<Rect>(rect)

  const onMove = (e: PointerEvent) => {
    const d = drag.current
    const el = ref.current
    if (!d || !el) return
    const dx = e.clientX - d.sx
    const dy = e.clientY - d.sy
    const vw = window.innerWidth
    const vh = window.innerHeight
    if (d.mode === "move") {
      // Move via a compositor-only transform: no layout on any of the ~60
      // pointer events a drag emits, and — the part that matters here — no
      // reflow of the <video> underneath, which would hitch the picture.
      const x = Math.min(Math.max(PAD, d.start.x + dx), vw - d.start.w - PAD)
      const y = Math.min(Math.max(PAD, d.start.y + dy), vh - d.start.h - PAD)
      latest.current = { x, y, w: d.start.w, h: d.start.h }
      el.style.transform = `translate3d(${x - d.start.x}px, ${y - d.start.y}px, 0)`
    } else {
      let { x, y, w, h } = d.start
      if (d.mode.includes("e")) w = Math.min(Math.max(minW, d.start.w + dx), vw - x - PAD)
      if (d.mode.includes("s")) h = Math.min(Math.max(minH, d.start.h + dy), vh - y - PAD)
      if (d.mode.includes("w")) {
        const nw = Math.min(Math.max(minW, d.start.w - dx), d.start.x + d.start.w - PAD)
        x = d.start.x + d.start.w - nw
        w = nw
      }
      if (d.mode.includes("n")) {
        const nh = Math.min(Math.max(minH, d.start.h - dy), d.start.y + d.start.h - PAD)
        y = d.start.y + d.start.h - nh
        h = nh
      }
      latest.current = { x, y, w, h }
      el.style.left = `${x}px`
      el.style.top = `${y}px`
      el.style.width = `${w}px`
      el.style.height = `${h}px`
    }
  }

  const onUp = () => {
    window.removeEventListener("pointermove", onMove)
    window.removeEventListener("pointerup", onUp)
    const el = ref.current
    const d = drag.current
    if (el && d) {
      if (d.mode === "move") {
        // Bake the transform into left/top synchronously, so there is no flash
        // between releasing the pointer and the state re-render landing.
        el.style.transform = ""
        el.style.left = `${latest.current.x}px`
        el.style.top = `${latest.current.y}px`
      }
      setPerf(el, false)
    }
    if (d) onRectChange(latest.current)
    drag.current = null
  }

  const begin = (mode: Mode) => (e: React.PointerEvent) => {
    e.preventDefault()
    drag.current = { mode, sx: e.clientX, sy: e.clientY, start: { ...rect } }
    latest.current = { ...rect }
    if (ref.current) setPerf(ref.current, true)
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
  }

  const onContainerDown = (e: React.PointerEvent) => {
    const t = e.target as HTMLElement
    // Drag from anywhere inside a [data-drag-handle] region — the whole header,
    // not a token grip — except the controls that live in it.
    if (!t.closest("[data-drag-handle]")) return
    if (t.closest("button, a, input, textarea, select, [role='button'], [data-no-drag]")) return
    begin("move")(e)
  }

  // A window that shrinks under the panel must not strand it off screen.
  useEffect(() => {
    const onResize = () => {
      const next = clampRect(latest.current, minW, minH)
      if (next.x !== latest.current.x || next.y !== latest.current.y || next.w !== latest.current.w || next.h !== latest.current.h) {
        latest.current = next
        onRectChange(next)
      }
    }
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [minW, minH, onRectChange])

  // Mirror committed rect into the drag scratchpad (never mid-drag: the pointer
  // handlers own it then, and the state behind `rect` is a render behind them).
  useEffect(() => {
    if (!drag.current) latest.current = rect
  }, [rect])

  const edge = "absolute touch-none"

  if (typeof document === "undefined") return null
  // Portal to <body> so the panel's stacking compares directly against the
  // other body-level portals (menus, dialogs) rather than against whatever
  // stacking context the left panel happens to be in.
  return createPortal(
    <div
      ref={ref}
      className={cn("fixed", className)}
      style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h, zIndex: LAYER }}
      onPointerDown={onContainerDown}
    >
      {children}
      <div className={cn(edge, "inset-x-0 top-0 h-1.5 cursor-ns-resize")} onPointerDown={begin("n")} />
      <div className={cn(edge, "inset-x-0 bottom-0 h-1.5 cursor-ns-resize")} onPointerDown={begin("s")} />
      <div className={cn(edge, "inset-y-0 left-0 w-1.5 cursor-ew-resize")} onPointerDown={begin("w")} />
      <div className={cn(edge, "inset-y-0 right-0 w-1.5 cursor-ew-resize")} onPointerDown={begin("e")} />
      <div className={cn(edge, "top-0 left-0 size-3 cursor-nwse-resize")} onPointerDown={begin("nw")} />
      <div className={cn(edge, "top-0 right-0 size-3 cursor-nesw-resize")} onPointerDown={begin("ne")} />
      <div className={cn(edge, "bottom-0 left-0 size-3 cursor-nesw-resize")} onPointerDown={begin("sw")} />
      <div className={cn(edge, "bottom-0 right-0 size-3 cursor-nwse-resize")} onPointerDown={begin("se")} />
    </div>,
    document.body,
  )
}
