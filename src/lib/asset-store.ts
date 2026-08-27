// Uploaded assets, in IndexedDB — the same pattern reze-studio and
// reze-design use: the model folder's files and the capture video are real
// bytes, often tens of megabytes, and localStorage tops out around five for
// the whole origin.
//
// ONE record per kind, deliberately: MiKaPo drives one model from one video.
// Uploading another replaces it.
//
// Every failure path resolves null/false rather than throwing: persistence is
// a convenience, never a precondition. Browsers evict IndexedDB under storage
// pressure, so a caller must treat "gone" as normal — the app boots the
// bundled defaults and the user re-uploads.

const DB_NAME = "mikapo"
const DB_VERSION = 1
const MODEL_STORE = "uploaded-model"
const VIDEO_STORE = "uploaded-video"

// Keys carry the version that wrote them, so a future shape change simply
// stops seeing old data instead of learning to migrate every past shape.
const STORAGE_VERSION = "1"
const key = (name: string) => `mikapo.${name}.${STORAGE_VERSION}`

type StoredEntry = { path: string; file: File }
type StoredModel = { pmxPath: string; stem: string; files: StoredEntry[] }
type StoredVideo = { name: string; file: File }

function open(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null)
  return new Promise((resolve) => {
    let req: IDBOpenDBRequest
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION)
    } catch {
      return resolve(null) // private mode in some browsers
    }
    req.onupgradeneeded = () => {
      for (const store of [MODEL_STORE, VIDEO_STORE]) {
        if (!req.result.objectStoreNames.contains(store)) req.result.createObjectStore(store)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => resolve(null)
    req.onblocked = () => resolve(null)
  })
}

function put(store: string, k: string, record: unknown): Promise<boolean> {
  return open().then((db) => {
    if (!db) return false
    return new Promise<boolean>((resolve) => {
      try {
        const tx = db.transaction(store, "readwrite")
        tx.objectStore(store).put(record, k)
        tx.oncomplete = () => resolve(true)
        tx.onerror = () => resolve(false)
        tx.onabort = () => resolve(false)
      } catch {
        resolve(false)
      } finally {
        db.close()
      }
    })
  })
}

function get<T>(store: string, k: string): Promise<T | null> {
  return open().then((db) => {
    if (!db) return null
    return new Promise<T | null>((resolve) => {
      try {
        const req = db.transaction(store, "readonly").objectStore(store).get(k)
        req.onsuccess = () => resolve((req.result as T | undefined) ?? null)
        req.onerror = () => resolve(null)
      } catch {
        resolve(null)
      } finally {
        db.close()
      }
    })
  })
}

/**
 * True only once the bytes are actually down — quota is the expected failure
 * mode for a large model in a tight browser.
 *
 * `path` is captured per file as `webkitRelativePath || name` HERE, while the
 * File objects are still live from the picker — that property does not
 * reliably survive an IndexedDB structured-clone round trip, so it must be
 * read now and stored explicitly rather than re-read after restore.
 */
export function saveModelUpload(files: File[], pmxFile: File, stem: string): Promise<boolean> {
  const record: StoredModel = {
    pmxPath: pmxFile.webkitRelativePath || pmxFile.name,
    stem,
    files: files.map((f) => ({ path: f.webkitRelativePath || f.name, file: f })),
  }
  return put(MODEL_STORE, key("uploaded-model"), record)
}

/**
 * The stored model as `{ files, pmxFile }` — the exact shape
 * `engine.loadModel(name, { files, pmxFile })` takes. Null when absent or
 * evicted.
 *
 * Re-wrapped so `.name` IS the stored path: a restored File's own
 * `webkitRelativePath` is always `""` (browsers only set it for a live
 * folder pick), and reze-engine's file-map resolver falls back to `.name`
 * for exactly this case.
 */
export async function loadModelUpload(): Promise<{ files: File[]; pmxFile: File; stem: string } | null> {
  const rec = await get<StoredModel>(MODEL_STORE, key("uploaded-model"))
  if (!rec) return null
  const files = rec.files.map((e) => new File([e.file], e.path, { type: e.file.type }))
  const pmxFile = files.find((f) => f.name === rec.pmxPath)
  if (!pmxFile) return null
  return { files, pmxFile, stem: rec.stem }
}

export function saveVideoUpload(file: File): Promise<boolean> {
  const record: StoredVideo = { name: file.name, file }
  return put(VIDEO_STORE, key("uploaded-video"), record)
}

export async function loadVideoUpload(): Promise<File | null> {
  const rec = await get<StoredVideo>(VIDEO_STORE, key("uploaded-video"))
  return rec?.file ?? null
}
