// The chrome vocabulary, in one place. Tokens live in globals.css; these are
// the recipes built from them, shared with reze.design and reze.studio so the
// three apps read as one product.

import { cn } from "@/lib/utils"

/** The canonical panel skin: bounded, floating, blurred. */
export const SKIN = "border border-line-strong bg-surface shadow-float backdrop-blur-xs"

/** A floating top-bar pill. Height is always declared by the caller (h-10):
 *  derived heights only agree while every pill holds the same children. */
export const PILL = cn("rounded-xl", SKIN)

/** Chrome icon button — top bar and panel headers. */
export const ICON_BUTTON = "size-7 shrink-0 rounded-lg text-muted-foreground hover:bg-white/5 hover:text-foreground"

/** A segment in a segmented control (input source, and anything like it). */
export const SEGMENT = "flex h-6 flex-1 items-center justify-center gap-1.5 rounded-chip text-[11px] font-medium transition-colors"
export const SEGMENT_ON = "bg-white/10 text-foreground"
export const SEGMENT_OFF = "text-muted-foreground hover:bg-white/5 hover:text-foreground"

/** The track a segmented control sits in. */
export const SEGMENT_TRACK = "flex items-center gap-0.5 rounded-interior bg-white/[0.06] p-0.5"

/** Micro label: all-caps, one step down, spaced so small caps don't clot. */
export const MICRO_LABEL = "text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground"
