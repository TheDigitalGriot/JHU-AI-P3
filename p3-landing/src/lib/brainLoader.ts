// Loader for the bioticus-style brain region JSONs at public/brain/*.json.
// Each file contains: { model: { vertices: number[], faces: number[], normals: number[], uvs: ... } }
// We only use `vertices` (a flat XYZ list) — point cloud morph targets don't need faces.

interface BrainModel {
  model: {
    name: string
    vertices: number[]
    normals?: number[]
    faces?: number[]
    transform?: number[]
  }
}

export interface SampledRegion {
  name: string
  positions: Float32Array // length N*3, centered + normalized
  vertexCount: number
}

const cache = new Map<string, Promise<SampledRegion>>()

// Resample a vertex pool of size V (flat XYZ) into exactly N points by random repetition.
// Cheap and good enough for ambient particle morphs.
function resample(verts: number[], N: number): Float32Array {
  const out = new Float32Array(N * 3)
  const V = Math.floor(verts.length / 3)
  if (V === 0) return out
  for (let i = 0; i < N; i++) {
    const j = Math.floor(Math.random() * V)
    out[3 * i + 0] = verts[3 * j + 0]
    out[3 * i + 1] = verts[3 * j + 1]
    out[3 * i + 2] = verts[3 * j + 2]
  }
  return out
}

// Compute centroid + max-extent scale, normalize positions into a [-targetRadius, targetRadius] cube.
function normalize(arr: Float32Array, targetRadius = 1.4) {
  const N = arr.length / 3
  let cx = 0, cy = 0, cz = 0
  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
  for (let i = 0; i < N; i++) {
    const x = arr[3 * i], y = arr[3 * i + 1], z = arr[3 * i + 2]
    cx += x; cy += y; cz += z
    if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z
    if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z
  }
  cx /= N; cy /= N; cz /= N
  const longest = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1
  const norm = (targetRadius * 2) / longest
  for (let i = 0; i < N; i++) {
    arr[3 * i]     = (arr[3 * i]     - cx) * norm
    arr[3 * i + 1] = (arr[3 * i + 1] - cy) * norm
    arr[3 * i + 2] = (arr[3 * i + 2] - cz) * norm
  }
  return arr
}

export async function loadRegion(file: string, N: number, targetRadius = 1.4): Promise<SampledRegion> {
  const cacheKey = `${file}@${N}@${targetRadius}`
  let p = cache.get(cacheKey)
  if (p) return p
  p = (async () => {
    const url = `${import.meta.env.BASE_URL || '/'}brain/${file}`.replace(/\/+/g, '/')
    const res = await fetch(url)
    if (!res.ok) throw new Error(`brain region fetch failed: ${url} (${res.status})`)
    const data: BrainModel = await res.json()
    const positions = normalize(resample(data.model.vertices, N), targetRadius)
    return {
      name: data.model.name,
      positions,
      vertexCount: Math.floor(data.model.vertices.length / 3),
    }
  })()
  cache.set(cacheKey, p)
  return p
}

// Combine multiple region files into one point cloud (positions concatenated then normalized as a whole).
export async function loadCombined(files: string[], N: number, targetRadius = 1.4): Promise<SampledRegion> {
  const cacheKey = `combined:${files.join(',')}@${N}@${targetRadius}`
  let p = cache.get(cacheKey)
  if (p) return p
  p = (async () => {
    const datas = await Promise.all(files.map(async (f) => {
      const url = `${import.meta.env.BASE_URL || '/'}brain/${f}`.replace(/\/+/g, '/')
      const res = await fetch(url)
      if (!res.ok) throw new Error(`brain region fetch failed: ${url} (${res.status})`)
      const d: BrainModel = await res.json()
      return d.model.vertices
    }))
    const allVerts: number[] = []
    datas.forEach((v) => { for (const n of v) allVerts.push(n) })
    const positions = normalize(resample(allVerts, N), targetRadius)
    return {
      name: files.join('+'),
      positions,
      vertexCount: Math.floor(allVerts.length / 3),
    }
  })()
  cache.set(cacheKey, p)
  return p
}
