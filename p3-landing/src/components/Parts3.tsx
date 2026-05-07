import { useEffect, useRef, useState } from 'react'
import { DATA, SNIPPETS } from '../data/data'
import { Section } from './Section'
import { CodeBlock } from './CodeBlock'
import { ModelScoreCard } from './ModelScoreCard'

const D = DATA
const SNIP = SNIPPETS

export function ModelANN() {
  return (
    <Section id="08 Simple ANN"
      kicker="07 · Model 1"
      title={<>Flatten everything. <em>See what survives.</em></>}
      lede="The simplest neural network we can build: take 49,152 pixel intensities, ignore where any of them are, and learn weights. Spatial structure dies at the Flatten step. The remarkable thing is how far you still get."
    >
      <div className="split" style={{ marginTop: '3rem' }}>
        <div>
          <CodeBlock label="MODEL · SIMPLE ANN" code={SNIP.ann} />
          <FlattenAnimation />
        </div>
        <div>
          <ModelScoreCard model={D.models[0]} per={D.perClass.ann as any} />
          <p className="reveal" style={{ marginTop: '1.4rem' }}>The ANN learns global brightness statistics — exactly the signal flagged in the leakage audit. Notumor recall is <span style={{ color: D.classColors.notumor }}>0.98</span> because notumor scans are systematically brighter; glioma recall is <span style={{ color: D.classColors.glioma }}>0.70</span> because gliomas need <em>shape</em>, and shape is gone.</p>
          <div className="callout reveal"><em>Train F1 1.00 vs val F1 0.88</em> — a 12-point overfitting gap. The model is memorizing.</div>
        </div>
      </div>
    </Section>
  )
}

function FlattenAnimation() {
  const ref = useRef<HTMLDivElement>(null)
  const [t, setT] = useState(0)
  useEffect(() => {
    let id: number | undefined
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) {
        let v = 0
        id = window.setInterval(() => {
          v += 0.02
          setT(Math.min(1, v))
          if (v >= 1) clearInterval(id)
        }, 32)
      }
    }), { threshold: 0.4 })
    if (ref.current) io.observe(ref.current)
    return () => { io.disconnect(); if (id) clearInterval(id) }
  }, [])
  const G = 12, px = 14
  return (
    <div className="chart reveal" ref={ref} style={{ marginTop: '1.4rem' }}>
      <div className="title"><span>FLATTEN OPERATION</span><span>{Math.round(t * 100)}%</span></div>
      <svg viewBox="0 0 520 220" style={{ width: '100%', height: 'auto' }}>
        {Array.from({ length: G * G }).map((_, i) => {
          const r = Math.floor(i / G), c = i % G
          const gx = 30 + c * px, gy = 30 + r * px
          const lx = 30 + i * ((520 - 60) / (G * G)), ly = 180
          const x = gx + (lx - gx) * t, y = gy + (ly - gy) * t
          const intensity = 0.3 + 0.7 * (((i * 9301 + 49297) % 233280) / 233280)
          return <rect key={i} x={x} y={y} width={px - 2} height={px - 2} fill="var(--cyan)" opacity={intensity * (0.4 + 0.6 * (1 - t * 0.4))} />
        })}
        <text x="30" y="20" fill="var(--dim)" fontSize="9" fontFamily="var(--mono)" letterSpacing="0.18em">128×128×3 IMAGE</text>
        <text x="30" y="210" fill="var(--cyan)" fontSize="9" fontFamily="var(--mono)" letterSpacing="0.18em">→ 49,152-VECTOR (SPATIAL INFO LOST)</text>
      </svg>
    </div>
  )
}

export function ModelOpt() {
  return (
    <Section id="09 Optimized ANN"
      kicker="08 · Model 2"
      title={<>Adding the regularizers.</>}
      lede="BatchNorm to stabilize, Dropout to prevent memorization, a learning-rate schedule to slow down at the bottom of the loss bowl. The training/val gap stays wide (~10 points) and the headline lands flat with the simple ANN — fixing one problem revealed another."
    >
      <div className="split" style={{ marginTop: '3rem' }}>
        <div>
          <CodeBlock label="MODEL · OPTIMIZED ANN" code={SNIP.opt} />
          <p className="reveal" style={{ marginTop: '1.4rem' }}>Glioma recall lifts to <span style={{ color: D.classColors.glioma }}>0.76</span>, but meningioma stays flat at 0.83. Regularization without spatial features is a blunt instrument — and 5-fold CV later reveals the optimized variant is actually <em>weaker</em> than the simple ANN on average across folds.</p>
        </div>
        <ModelScoreCard model={D.models[1]} per={D.perClass.opt as any} />
      </div>
    </Section>
  )
}

export function ModelVGG16() {
  return (
    <Section id="10 VGG16 Base"
      kicker="09 · Model 3 · Transfer learning"
      title={<>Borrowing 14 million parameters from <em>ImageNet</em>.</>}
      lede="VGG-16 has been staring at cats and cars and street signs for years. We freeze its convolutional eyes, swap the head for a 4-class softmax, and ask: do those features transfer to brain MRIs? Yes, and the leap is immediate."
    >
      <div className="split" style={{ marginTop: '3rem' }}>
        <div>
          <CodeBlock label="MODEL · VGG-16 BASE (FROZEN)" code={SNIP.vgg16} />
          <ConvKernelSweep />
        </div>
        <div>
          <ModelScoreCard model={D.models[2]} per={D.perClass.vgg16 as any} />
          <p className="reveal" style={{ marginTop: '1.4rem' }}>Only <span className="mono" style={{ color: 'var(--cyan)' }}>2,052</span> trainable parameters out of <span className="mono">14.7M</span>. Everything else stays frozen — we're just learning a 4-way decision over features that already exist.</p>
          <div className="callout reveal"><em>Glioma recall lifts</em> 0.70 → 0.77 in val; the train–val gap also tightens from 12 points to 7. Spatial features matter.</div>
        </div>
      </div>
    </Section>
  )
}

function ConvKernelSweep() {
  const [pos, setPos] = useState({ x: 0, y: 0 })
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let id: number | undefined
    const start = () => {
      let t = 0
      id = window.setInterval(() => {
        t += 0.018
        const x = (Math.sin(t * 0.7) * 0.4 + 0.5) * 9
        const y = (Math.cos(t * 0.5) * 0.4 + 0.5) * 9
        setPos({ x, y })
      }, 32)
    }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) start() }), { threshold: 0.4 })
    if (ref.current) io.observe(ref.current)
    return () => { io.disconnect(); if (id) clearInterval(id) }
  }, [])
  const G = 12, cell = 22, pad = 16
  return (
    <div className="chart reveal" ref={ref} style={{ marginTop: '1.4rem' }}>
      <div className="title"><span>3×3 CONVOLUTION · FEATURE EXTRACTION</span><span>STRIDE 1</span></div>
      <svg viewBox="0 0 560 320" style={{ width: '100%', height: 'auto' }}>
        <text x={pad} y="14" fill="var(--dim)" fontSize="9" fontFamily="var(--mono)" letterSpacing="0.18em">INPUT PATCH</text>
        {Array.from({ length: G * G }).map((_, i) => {
          const r = Math.floor(i / G), c = i % G
          const cx = G / 2, cy = G / 2
          const dist = Math.hypot(c - cx, r - cy)
          const v = Math.max(0, 1 - dist / (G * 0.6)) * (0.6 + 0.4 * Math.sin(c * 1.2 + r * 0.8))
          return <rect key={i} x={pad + c * cell} y={20 + r * cell} width={cell - 1} height={cell - 1} fill="rgba(255,255,255,0.85)" opacity={v} />
        })}
        <rect x={pad + Math.floor(pos.x) * cell - 2} y={20 + Math.floor(pos.y) * cell - 2}
          width={cell * 3 + 2} height={cell * 3 + 2}
          fill="none" stroke="var(--cyan)" strokeWidth="2" style={{ filter: 'drop-shadow(0 0 6px var(--cyan))' }} />
        <line x1={pad + 12 * cell + 18} y1="160" x2="340" y2="160" stroke="var(--cyan)" strokeWidth="1" markerEnd="url(#arr)" />
        <defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--cyan)" /></marker></defs>
        <text x="370" y="14" fill="var(--dim)" fontSize="9" fontFamily="var(--mono)" letterSpacing="0.18em">FEATURE MAP</text>
        {Array.from({ length: 10 * 10 }).map((_, i) => {
          const r = Math.floor(i / 10), c = i % 10
          const dx = c - Math.floor(pos.x), dy = r - Math.floor(pos.y)
          const active = Math.abs(dx) <= 1 && Math.abs(dy) <= 1 && dx >= 0 && dy >= 0
          const v = active ? 0.85 : 0.18
          return <rect key={i} x={370 + c * 15} y={20 + r * 15} width="14" height="14" fill="var(--cyan)" opacity={v} />
        })}
      </svg>
    </div>
  )
}
