/**
 * Where the demo model and the reference video come from.
 *
 * A deployed build reads them from R2, whose egress is free, so the ~38MB a
 * visitor downloads never touches the deployment's transfer budget — one pool
 * shared across every project on the account. `next dev` reads the same files
 * out of `public/`, which keeps a checkout self-contained: swap the video,
 * reload, no round trip through a bucket.
 *
 * Keys there are versioned by path, which is what lets them carry a one-year
 * immutable cache header: rename, never overwrite in place. They are also
 * spelled NFC, the way git stores them and source code writes them — macOS
 * hands filenames back decomposed, and an R2 key is matched byte for byte.
 */
export const ASSETS = process.env.NODE_ENV === "production" ? "https://assets.reze.one/demo/mikapo" : ""
