// What the model ends up in, not what the solver said.
//
// Everything measured so far stops at the solver's output. Between that and
// the character on screen sit the helper bones an MMD rig is built from —
// 肩P and 肩C, 腕捩 and its three followers, 腰キャンセル — each carrying an
// append of some fraction of another bone. A rotation written to 肩 that a
// helper quietly cancels is indistinguishable, in every number reported up to
// now, from a rotation that arrived.
//
// So: load the real PMX, pose it with a real solved frame, and read back where
// the engine says the bones actually point.
import { readFileSync } from "node:fs"
import { PmxLoader } from "reze-engine"

type Pt = { x: number; y: number; z: number; visibility: number }
const MODEL = "public/models/塞尔凯特/塞尔凯特.pmx"
const buf = readFileSync(MODEL)
const model = await PmxLoader.loadFromBuffer(
  buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  "塞尔凯特.pmx",
)
const skeleton = model.skeleton ?? model.getSkeleton?.()
const bones = skeleton.bones as { name: string; parentIndex: number }[]
const nameIndex: Record<string, number> = {}
bones.forEach((b, i) => (nameIndex[b.name] = i))

console.log(`${bones.length} bones · has rotateBones: ${typeof model.rotateBones === "function"} · has update: ${typeof model.update === "function"}`)

// The chain each of our targets really hangs from, as the FILE describes it.
const chainOf = (name: string): string[] => {
  const out: string[] = []
  let i = nameIndex[name]
  while (i !== undefined && i >= 0) {
    out.push(bones[i].name)
    i = bones[i].parentIndex
  }
  return out.reverse()
}
for (const n of ["上半身", "上半身2", "首", "頭", "左腕", "左ひじ", "左手首", "左足", "左ひざ"]) {
  console.log(`  ${n.padEnd(6)} ← ${chainOf(n).join(" → ")}`)
}

// Which of the solver's targets have a helper hanging off them that the engine
// will drive by append — the ones whose rotation is not the last word.
const appends = (skeleton.bones as { name: string; appendParentIndex?: number; appendRatio?: number }[])
  .filter((b) => b.appendParentIndex !== undefined && b.appendParentIndex >= 0)
console.log(`\n${appends.length} bones take an append from another bone:`)
for (const b of appends.slice(0, 14)) {
  const src = bones[(b as { appendParentIndex: number }).appendParentIndex]?.name
  console.log(`  ${b.name.padEnd(10)} ← ${((b as { appendRatio?: number }).appendRatio ?? 1).toFixed(2)} × ${src}`)
}
