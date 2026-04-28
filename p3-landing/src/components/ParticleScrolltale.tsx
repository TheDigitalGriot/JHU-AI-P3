import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { loadCombined, loadRegion, type SampledRegion } from '../lib/brainLoader'

const N = 18000

// ── Procedural targets (used for tumor masses + cluster + confusion) ─────────
function pointsTumorMass(n: number, kind: 'glioma' | 'meningioma' | 'pituitary', envelope: Float32Array | null) {
  const arr = new Float32Array(n * 3)
  let cx = 0, cy = 0, cz = 0, rr = 0.35, irregular = 1.0
  if (kind === 'glioma')      { cx = -0.35; cy =  0.05; cz =  0.25; rr = 0.45; irregular = 1.2 }
  if (kind === 'meningioma')  { cx =  0.55; cy =  0.55; cz =  0.0;  rr = 0.32; irregular = 0.5 }
  if (kind === 'pituitary')   { cx =  0.0;  cy = -0.25; cz =  0.15; rr = 0.20; irregular = 0.3 }

  // Reuse envelope (real brain points) for the background 65% of particles, generate tumor blob for the rest.
  const envCount = Math.floor(n * 0.65)
  if (envelope && envelope.length >= envCount * 3) {
    for (let i = 0; i < envCount; i++) {
      const j = Math.floor(Math.random() * (envelope.length / 3))
      arr[3 * i]     = envelope[3 * j]
      arr[3 * i + 1] = envelope[3 * j + 1]
      arr[3 * i + 2] = envelope[3 * j + 2]
    }
  } else {
    // procedural fallback envelope
    for (let i = 0; i < envCount; i++) {
      const u = Math.random(), v = Math.random()
      const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1)
      arr[3 * i]     = 1.1 * Math.sin(phi) * Math.cos(theta)
      arr[3 * i + 1] = 1.1 * Math.sin(phi) * Math.sin(theta)
      arr[3 * i + 2] = 1.1 * Math.cos(phi)
    }
  }
  for (let i = envCount; i < n; i++) {
    const a = Math.random() * Math.PI * 2
    const z = (Math.random() - 0.5)
    const r = (1 - Math.pow(Math.random(), 0.5)) * rr * (1 + 0.5 * irregular * Math.sin(a * 3))
    arr[3 * i]     = cx + Math.cos(a) * r
    arr[3 * i + 1] = cy + Math.sin(a) * r
    arr[3 * i + 2] = cz + z * rr * 0.6
  }
  return arr
}

function pointsCluster4(n: number) {
  const centers = [
    [-1.1,  0.7, 0],
    [ 1.1,  0.7, 0],
    [-1.1, -0.7, 0],
    [ 1.1, -0.7, 0],
  ]
  const arr = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    const c = centers[i % 4]
    const r = Math.pow(Math.random(), 0.6) * 0.45
    const a = Math.random() * Math.PI * 2
    const z = (Math.random() - 0.5) * 0.4
    arr[3 * i]     = c[0] + Math.cos(a) * r
    arr[3 * i + 1] = c[1] + Math.sin(a) * r
    arr[3 * i + 2] = c[2] + z
  }
  return arr
}

function pointsConfusionGrid(n: number) {
  const arr = new Float32Array(n * 3)
  const cellSize = 0.75
  const gap = 0.12
  for (let i = 0; i < n; i++) {
    const cell = i % 16
    const r = Math.floor(cell / 4) - 1.5
    const c = (cell % 4) - 1.5
    const cx = c * (cellSize + gap)
    const cy = -r * (cellSize + gap)
    const px = (Math.random() - 0.5) * cellSize * 0.85
    const py = (Math.random() - 0.5) * cellSize * 0.85
    arr[3 * i]     = cx + px
    arr[3 * i + 1] = cy + py
    arr[3 * i + 2] = (Math.random() - 0.5) * 0.04
  }
  return arr
}

const STAGES = [
  'cerebrum',
  'cerebellum',
  'brainstem',
  'thalamus',
  'pituitary',
  'glioma',
  'meningioma',
  'cluster',
  'confusion',
] as const
type Stage = typeof STAGES[number]

const STAGE_COLORS: Record<Stage, [number, number, number]> = {
  cerebrum:    [0.30, 1.00, 0.82],
  cerebellum:  [0.30, 1.00, 0.82],
  brainstem:   [0.30, 1.00, 0.82],
  thalamus:    [0.60, 1.00, 0.76],
  pituitary:   [1.00, 0.43, 0.82],
  glioma:      [0.30, 1.00, 0.82],
  meningioma:  [1.00, 0.72, 0.30],
  cluster:     [0.60, 1.00, 0.76],
  confusion:   [0.30, 1.00, 0.82],
}

const STAGE_HEAD: Record<Stage, { kicker: string; title: string; body: string }> = {
  cerebrum:    { kicker: 'CEREBRUM',     title: 'The cerebrum.',         body: 'Left and right hemispheres, sampled as a particle cloud from a 600k-vertex anatomical scan.' },
  cerebellum:  { kicker: 'CEREBELLUM',   title: 'Below the occipital lobe.', body: 'Coordinates motor learning. Distinct sulcal pattern — a useful landmark for orientation in MRI slices.' },
  brainstem:   { kicker: 'BRAIN STEM',   title: 'Midbrain · pons.',       body: 'The bridge between cerebrum and spinal cord. Where the autonomic system lives — and where surgical risk runs highest.' },
  thalamus:    { kicker: 'THALAMUS',     title: 'The relay.',             body: 'Routes nearly every sensory signal to the cortex. Sits dead-center in the brain — a clue for the model when the lesion encroaches.' },
  pituitary:   { kicker: 'PITUITARY',    title: 'A 0.5g gland.',          body: 'The class with the most distinctive location. Hormonal symptoms drive most discoveries; the model gets a strong prior from where the bright spot sits.' },
  glioma:      { kicker: 'GLIOMA',       title: 'Irregular · infiltrative.', body: 'High-grade tumors that spread through brain tissue. The boundary blurs — and so does the model\'s confidence. The class the rubric weights heaviest.' },
  meningioma:  { kicker: 'MENINGIOMA',   title: 'Round · encapsulated.',  body: 'Arises from the meninges. Well-defined boundary — but on a single slice, the same intensity profile as a glioma. The most common confusion the model makes.' },
  cluster:     { kicker: '2,895 SCANS',  title: 'Four classes, one scan each.', body: 'Each particle becomes a single MRI in the validation set. They group by their predicted class — and the spread between groups is the model\'s recall.' },
  confusion:   { kicker: 'CONFUSION',    title: 'The 4×4 matrix.',        body: 'Where the model agrees with itself, and where it doesn\'t. The diagonal is recall. The off-diagonal is the work to do next.' },
}

const VERT = /* glsl */ `
  attribute vec3 positionA;
  attribute vec3 positionB;
  attribute float aRnd;
  uniform float uMix;
  uniform float uTime;
  uniform float uSize;
  uniform float uPixelRatio;
  varying float vAlpha;
  varying float vBulge;

  // Cheap pseudo-noise for transit displacement
  vec3 transitDisp(vec3 p, float t) {
    return vec3(
      sin(p.y * 1.7 + t * 0.7) + cos(p.z * 1.3 + t * 0.5),
      sin(p.z * 1.5 + t * 0.6) + cos(p.x * 1.4 + t * 0.4),
      sin(p.x * 1.6 + t * 0.5) + cos(p.y * 1.5 + t * 0.6)
    );
  }

  void main() {
    // Per-particle delay 0..0.55, duration 0.45 → fluid wave-like morph
    float delay = aRnd * 0.55;
    float dur   = 0.45;
    float local = clamp((uMix - delay) / dur, 0.0, 1.0);
    float t = local * local * (3.0 - 2.0 * local); // smoothstep

    vec3 p = mix(positionA, positionB, t);

    // Bulge displacement during the transit — peaks at t=0.5
    float bulge = sin(t * 3.14159);
    vBulge = bulge;
    p += transitDisp(p * 1.2 + uTime * 0.18, uTime * 0.18) * bulge * 0.13;

    // Gentle ambient float so the cloud never feels frozen
    p.x += sin(uTime * 0.6 + aRnd * 6.28) * 0.006;
    p.y += cos(uTime * 0.5 + aRnd * 6.28) * 0.006;

    vec4 mvPos = modelViewMatrix * vec4(p, 1.0);

    // Small, near-constant pixel size with subtle perspective scaling
    float persp = 1.6 / max(1.0, -mvPos.z * 0.3);
    gl_PointSize = uSize * uPixelRatio * persp;
    gl_PointSize *= 0.7 + 0.6 * aRnd; // size variance for organic feel

    vAlpha = 0.45 + 0.55 * aRnd;
    // Particles fade slightly during transit (when they're spread out)
    vAlpha *= 1.0 - bulge * 0.35;

    gl_Position = projectionMatrix * mvPos;
  }
`

const FRAG = /* glsl */ `
  precision mediump float;
  uniform vec3 uColor;
  uniform vec3 uTransitColor;
  varying float vAlpha;
  varying float vBulge;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    // Hard centre, soft halo
    float a = 1.0 - smoothstep(0.05, 0.5, d);
    // Tint slightly towards transit color when bulging
    vec3 col = mix(uColor, uTransitColor, vBulge * 0.5);
    gl_FragColor = vec4(col, a * vAlpha * 0.85);
  }
`

export function ParticleScrolltale() {
  const sectionRef = useRef<HTMLElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)
  const [active, setActive] = useState<Stage>('cerebrum')
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const container = canvasRef.current
    if (!container) return

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    renderer.setSize(container.clientWidth, container.clientHeight, false)
    renderer.setClearColor(0x000000, 0)
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(35, container.clientWidth / container.clientHeight, 0.1, 100)
    camera.position.set(0, 0, 5.0)

    const rnd = new Float32Array(N)
    for (let i = 0; i < N; i++) rnd[i] = Math.random()

    // Targets — start with an identical placeholder, replaced as soon as brain regions load
    const placeholder = (() => {
      const arr = new Float32Array(N * 3)
      for (let i = 0; i < N; i++) {
        const u = Math.random(), v = Math.random()
        const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1)
        arr[3 * i]     = 1.1 * Math.sin(phi) * Math.cos(theta)
        arr[3 * i + 1] = 1.1 * Math.sin(phi) * Math.sin(theta)
        arr[3 * i + 2] = 1.1 * Math.cos(phi)
      }
      return arr
    })()
    const targets: Record<Stage, Float32Array> = {
      cerebrum:    placeholder.slice(),
      cerebellum:  placeholder.slice(),
      brainstem:   placeholder.slice(),
      thalamus:    placeholder.slice(),
      pituitary:   placeholder.slice(),
      glioma:      placeholder.slice(),
      meningioma:  placeholder.slice(),
      cluster:     pointsCluster4(N),
      confusion:   pointsConfusionGrid(N),
    }

    const geom = new THREE.BufferGeometry()
    const initA = targets.cerebrum.slice()
    const initB = targets.cerebellum.slice()
    geom.setAttribute('position', new THREE.BufferAttribute(initA.slice(), 3))
    geom.setAttribute('positionA', new THREE.BufferAttribute(initA, 3))
    geom.setAttribute('positionB', new THREE.BufferAttribute(initB, 3))
    geom.setAttribute('aRnd', new THREE.BufferAttribute(rnd, 1))

    const uniforms = {
      uMix: { value: 0 },
      uTime: { value: 0 },
      uSize: { value: 2.2 },
      uPixelRatio: { value: Math.min(devicePixelRatio, 2) },
      uColor: { value: new THREE.Color(...STAGE_COLORS.cerebrum) },
      uTransitColor: { value: new THREE.Color(0xffffff) },
    }
    const mat = new THREE.ShaderMaterial({
      uniforms, vertexShader: VERT, fragmentShader: FRAG,
      transparent: true, depthWrite: false, blending: THREE.NormalBlending,
    })

    const points = new THREE.Points(geom, mat)
    scene.add(points)

    let lastA = -1, lastB = -1
    const swapTo = (a: number, b: number) => {
      if (a === lastA && b === lastB) return
      const aAttr = geom.getAttribute('positionA') as THREE.BufferAttribute
      const bAttr = geom.getAttribute('positionB') as THREE.BufferAttribute
      ;(aAttr.array as Float32Array).set(targets[STAGES[a]])
      ;(bAttr.array as Float32Array).set(targets[STAGES[b]])
      aAttr.needsUpdate = true
      bAttr.needsUpdate = true
      lastA = a; lastB = b
    }

    // Load brain regions and replace targets in-place
    let cancelled = false
    ;(async () => {
      try {
        const [cerebrum, cerebellum, brainstem, thalamus, pituitary]: SampledRegion[] = await Promise.all([
          loadCombined(['cerebral_hemisphere-left.json', 'cerebral_hemisphere-right.json'], N, 1.5),
          loadRegion('cerebellum.json', N, 1.4),
          loadCombined(['brain_stem-midbrain.json', 'brain_stem-pons.json'], N, 1.4),
          loadRegion('thalamus.json', N, 1.4),
          loadRegion('pituitary_gland.json', N, 1.4),
        ])
        if (cancelled) return
        targets.cerebrum   = cerebrum.positions
        targets.cerebellum = cerebellum.positions
        targets.brainstem  = brainstem.positions
        targets.thalamus   = thalamus.positions
        targets.pituitary  = pituitary.positions
        targets.glioma     = pointsTumorMass(N, 'glioma',     cerebrum.positions)
        targets.meningioma = pointsTumorMass(N, 'meningioma', cerebrum.positions)
        // Force a re-swap so the currently-displayed pair refreshes
        const a = lastA, b = lastB
        lastA = -1; lastB = -1
        swapTo(a < 0 ? 0 : a, b < 0 ? 1 : b)
        setReady(true)
      } catch (err) {
        console.error('[ParticleScrolltale] brain region load failed:', err)
      }
    })()

    const onScroll = () => {
      const el = sectionRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const total = el.offsetHeight - window.innerHeight
      const p = Math.max(0, Math.min(1, -rect.top / Math.max(1, total)))
      const segCount = STAGES.length - 1
      const exact = p * segCount
      const stageIdx = Math.min(segCount - 1, Math.floor(exact))
      const inStage = exact - stageIdx
      swapTo(stageIdx, stageIdx + 1)
      uniforms.uMix.value = inStage

      const ca = STAGE_COLORS[STAGES[stageIdx]]
      const cb = STAGE_COLORS[STAGES[stageIdx + 1]]
      const t = inStage
      uniforms.uColor.value.setRGB(
        ca[0] * (1 - t) + cb[0] * t,
        ca[1] * (1 - t) + cb[1] * t,
        ca[2] * (1 - t) + cb[2] * t,
      )

      const labelStage = inStage > 0.5 ? STAGES[stageIdx + 1] : STAGES[stageIdx]
      setActive(labelStage)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })

    let mx = 0, my = 0
    const onMove = (e: MouseEvent) => {
      mx = (e.clientX / window.innerWidth) * 2 - 1
      my = -((e.clientY / window.innerHeight) * 2 - 1)
    }
    window.addEventListener('mousemove', onMove)

    const onResize = () => {
      const w = container.clientWidth, h = container.clientHeight
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', onResize)

    const clock = new THREE.Clock()
    let raf = 0
    const animate = () => {
      const t = clock.getElapsedTime()
      uniforms.uTime.value = t
      points.rotation.y += (mx * 0.4 - points.rotation.y) * 0.04
      points.rotation.x += (my * 0.2 - points.rotation.x) * 0.04
      points.rotation.z = Math.sin(t * 0.05) * 0.04
      renderer.render(scene, camera)
      raf = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      geom.dispose()
      mat.dispose()
      if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement)
    }
  }, [])

  const head = STAGE_HEAD[active]
  return (
    <section ref={sectionRef} className="particle-scrolltale" data-screen-label="01b Particle Brain">
      <div className="particle-scrolltale-stage">
        <div className="particle-scrolltale-canvas" ref={canvasRef} aria-hidden="true" />
        <div className="particle-scrolltale-overlay">
          <div className="particle-scrolltale-text">
            <div className="kicker">{head.kicker}</div>
            <h2 className="serif">{head.title}</h2>
            <p>{head.body}</p>
            {!ready && (
              <p className="particle-scrolltale-load mono">Loading anatomical regions…</p>
            )}
          </div>
          <div className="particle-scrolltale-pager mono">
            {STAGES.map((s, i) => (
              <span key={s} className={s === active ? 'on' : ''}>
                {String(i + 1).padStart(2, '0')} · {s.toUpperCase()}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
