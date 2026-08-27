"use client"

// Drops focus that a POINTER left behind, app-wide.
//
// Clicking a button focuses it, and the ring then sits there until something else
// takes focus — worst after a popover or dialog closes, where the trigger is left
// ringed with nothing open. Keyboard focus is untouched: Tab and arrow keys never
// fire pointerup, so only mouse-driven focus is dropped, and the ring still does
// its job for keyboard users.
//
// `Button` already does this per-instance (`e.detail > 0` → blur); this is the
// same rule for the many plain <button>s, popover triggers and colour swatches.

import { useEffect } from "react"

/** Text entry keeps focus — blurring mid-typing would be absurd. */
const KEEPS_FOCUS = "input,textarea,select,[contenteditable=true]"
/** Things whose focus ring is noise once the pointer is done with them. */
const DROPS_FOCUS = 'button,[role="button"],[role="menuitem"],[role="option"],[role="tab"],a[href],summary'

export function NoStickyFocus() {
  useEffect(() => {
    const drop = () => {
      const el = document.activeElement
      if (!(el instanceof HTMLElement) || el === document.body) return
      if (el.closest(KEEPS_FOCUS)) return
      if (el.matches(DROPS_FOCUS)) el.blur()
    }
    // After the click handler, so anything that deliberately moves focus wins.
    const onPointerUp = () => queueMicrotask(drop)
    // Tab-away / app switch: drop whatever was focused so returning is clean.
    const onWindowBlur = () => {
      const el = document.activeElement
      if (el instanceof HTMLElement && el !== document.body) el.blur()
    }
    document.addEventListener("pointerup", onPointerUp, true)
    window.addEventListener("blur", onWindowBlur)
    return () => {
      document.removeEventListener("pointerup", onPointerUp, true)
      window.removeEventListener("blur", onWindowBlur)
    }
  }, [])
  return null
}
