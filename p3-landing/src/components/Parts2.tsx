import { useEffect, useRef, useState } from 'react'
import { DATA, SNIPPETS, type ClassKey } from '../data/data'
import { Section } from './Section'
import { CodeBlock } from './CodeBlock'
import { NumTween } from './NumTween'

const D = DATA
const SNIP = SNIPPETS

export function WhyHard() {
  return (
    <Section id="05 Why Hard"
      kicker="04 · Why this is genuinely hard"
      title={<>Two tumors. <em style={{ fontStyle: 'italic', color: 'var(--cyan)' }}>One signal.</em></>}
      lede="Glioma and meningioma both present as bright, irregular masses inside brain tissue. On a single 2D slice, intensity histograms overlap so much that even trained radiologists disagree on borderline cases."
    >
      <div className="split" style={{ marginTop: '3rem' }}>
        <div>
          <p className="reveal">A model staring at flattened pixels has two ways to be right:</p>
          <ul className="reveal" style={{ listStyle: 'none', padding: 0, fontFamily: 'var(--mono)', fontSize: 13, lineHeight: 1.8 }}>
            <li>→ Learn the <em style={{ color: 'var(--cyan)' }}>tumor morphology</em> — the shape, the boundary, the spatial signature.</li>
            <li>→ Learn the <em style={{ color: 'var(--magenta)' }}>scanner fingerprint</em> — the contrast and noise that mark which machine took the scan.</li>
          </ul>
          <p className="reveal" style={{ marginTop: '1.4rem' }}>Both routes give high accuracy on this dataset. Only one transfers to a different hospital. We need to know which one we taught.</p>
        </div>
        <div className="reveal">
          <OverlapDiagram />
        </div>
      </div>
    </Section>
  )
}

function OverlapDiagram() {
  return (
    <div className="chart">
      <div className="title"><span>VISUAL OVERLAP · WHAT THE MODEL SEES</span><span>T1 · AXIAL</span></div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <div className="mri glioma" style={{ aspectRatio: '1' }}><div className="crosshair-mri" /><div className="label">GLIOMA · IRREGULAR</div></div>
        <div className="mri meningioma" style={{ aspectRatio: '1' }}><div className="crosshair-mri" /><div className="label">MENINGIOMA · ROUND</div></div>
      </div>
      <p style={{ marginTop: 14, fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '0.06em', color: 'var(--dim)' }}>
        Same slice plane. Similar tissue contrast. The boundary tells the story — but only at full spatial resolution.
      </p>
    </div>
  )
}

export function Preprocess() {
  return (
    <Section id="06 Preprocessing"
      kicker="05 · Preparing the pixels"
      title={<>Resize. <em>Normalize.</em> One-hot.</>}
      lede="Three steps that look unglamorous and decide whether the model has any chance. Resize collapses scanner-specific dimensions; normalization stabilizes gradients; one-hot turns four words into four targets."
    >
      <div className="split" style={{ marginTop: '3rem', alignItems: 'start' }}>
        <div>
          <h3 className="reveal serif" style={{ marginBottom: '1rem' }}>1 · Resize → 128 × 128</h3>
          <p className="reveal">Every image becomes a 128×128×3 tensor. We lose detail; we gain a tractable input. <span className="mono" style={{ color: 'var(--cyan)' }}>49,152</span> features per image — enough for an ANN, plenty for a CNN.</p>
          <h3 className="reveal serif" style={{ marginTop: '2.4rem', marginBottom: '1rem' }}>2 · Normalize → [0, 1]</h3>
          <CodeBlock label="NORMALIZE" code={SNIP.normalize} />
          <h3 className="reveal serif" style={{ marginTop: '2.4rem', marginBottom: '1rem' }}>3 · One-hot encode</h3>
          <p className="reveal mono" style={{ fontSize: 12.5, lineHeight: 1.9, color: 'var(--dim)' }}>
            glioma     → [<span style={{ color: D.classColors.glioma }}>1</span>, 0, 0, 0]<br />
            meningioma → [0, <span style={{ color: D.classColors.meningioma }}>1</span>, 0, 0]<br />
            notumor    → [0, 0, <span style={{ color: D.classColors.notumor }}>1</span>, 0]<br />
            pituitary  → [0, 0, 0, <span style={{ color: D.classColors.pituitary }}>1</span>]
          </p>
        </div>
        <div>
          <h3 className="reveal serif" style={{ marginBottom: '1rem' }}>4 · Stratified split</h3>
          <SplitDiagram />
          <CodeBlock label="STRATIFIED SPLIT" code={SNIP.splits} />
        </div>
      </div>
    </Section>
  )
}

function SplitDiagram() {
  const segs = [
    { name: 'TRAIN', n: D.splits.train, pct: 70, color: 'var(--cyan)' },
    { name: 'VAL',   n: D.splits.val,   pct: 15, color: 'var(--magenta)' },
    { name: 'TEST',  n: D.splits.test,  pct: 15, color: 'var(--amber)' },
  ]
  return (
    <div className="chart reveal" style={{ marginBottom: '1.4rem' }}>
      <div className="title"><span>70 / 15 / 15 · n = 2,895</span><span>STRATIFIED</span></div>
      <div style={{ display: 'flex', height: 42, border: '1px solid var(--line)', overflow: 'hidden' }}>
        {segs.map((s) => (
          <div key={s.name} style={{ flex: s.pct, background: `linear-gradient(90deg, ${s.color}1f, ${s.color}05)`, borderRight: '1px solid var(--line)', position: 'relative' }}>
            <div style={{ position: 'absolute', inset: 0, background: s.color, opacity: 0.18 }} />
            <div style={{ position: 'absolute', left: 10, top: 8, fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: '0.18em', color: s.color }}>{s.name}</div>
            <div style={{ position: 'absolute', right: 10, bottom: 6, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink)' }}>{s.n}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function Leakage() {
  const [animated, setAnimated] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) setAnimated(true) }), { threshold: 0.35 })
    io.observe(ref.current)
    return () => io.disconnect()
  }, [])
  return (
    <Section id="07 Leakage Audit"
      kicker="06 · The audit"
      title={<>Can you predict the class <em>without</em> looking at the tumor?</>}
      lede="A logistic regression sees only five numbers per image — mean, std, median, skew, kurtosis. No spatial information. No tumor shape. If it guesses better than chance, the dataset is leaking scanner identity through brightness."
    >
      <div ref={ref}>
        <div className="eq reveal" style={{ marginTop: '2.4rem' }}>
          <span style={{ color: 'var(--dim)' }}>P(class | image) ≈ softmax(</span>
          <span style={{ color: 'var(--cyan)' }}>w</span><sub>1</sub> · μ +
          <span style={{ color: 'var(--cyan)' }}> w</span><sub>2</sub> · σ +
          <span style={{ color: 'var(--cyan)' }}> w</span><sub>3</sub> · skew +
          <span style={{ color: 'var(--cyan)' }}> w</span><sub>4</sub> · kurt +
          <span style={{ color: 'var(--cyan)' }}> w</span><sub>5</sub> · median
          <span style={{ color: 'var(--dim)' }}>)</span>
        </div>
        <CodeBlock label="LEAKAGE AUDIT · 14 LINES OF SKLEARN" code={SNIP.leakage} />
        <div className="split reveal" style={{ marginTop: '2.4rem', alignItems: 'center' }}>
          <div>
            <div className="kicker">VERDICT</div>
            <div style={{ fontFamily: 'var(--serif)', fontSize: 'clamp(38px, 5vw, 64px)', lineHeight: 1, marginTop: '0.6rem', letterSpacing: '-0.02em' }}>
              {animated ? <NumTween to={D.leakage.testAcc * 100} decimals={2} suffix="%" /> : '0.00%'}
            </div>
            <p style={{ marginTop: '1.2rem' }}>Test accuracy on a held-out 15% of images, with no spatial information. Random guessing would give <strong style={{ color: 'var(--dim)' }}>25.00%</strong>.</p>
            <p>The model that knows nothing about tumors still gets <strong style={{ color: 'var(--magenta)' }}>2.6× better than chance</strong>. That's the scanner fingerprint, baked into the dataset.</p>
            <div className="callout"><em>Probable leakage.</em> Some fraction of any model's recall on this corpus is the model recognizing where the image came from — not what's inside it.</div>
          </div>
          <LeakageMeter pct={D.leakage.testAcc} animated={animated} />
        </div>
      </div>
    </Section>
  )
}

function LeakageMeter({ pct, animated }: { pct: number; animated: boolean }) {
  const W = 480, H = 280
  return (
    <div className="chart">
      <div className="title"><span>SCANNER-FINGERPRINT METER</span><span>4-CLASS · INTENSITY ONLY</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <defs>
          <linearGradient id="leakgrad" x1="0" x2="1">
            <stop offset="0%" stopColor="var(--notumor)" />
            <stop offset="40%" stopColor="var(--amber)" />
            <stop offset="100%" stopColor="var(--magenta)" />
          </linearGradient>
        </defs>
        <path d={`M 60 220 A 180 180 0 0 1 420 220`} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="14" />
        <path d={`M 60 220 A 180 180 0 0 1 420 220`} fill="none" stroke="url(#leakgrad)" strokeWidth="14"
          strokeDasharray={animated ? `${pct * 565} 565` : '0 565'} style={{ transition: 'stroke-dasharray 1.6s cubic-bezier(.2,.7,.2,1)' }} />
        {[
          { v: 0.25, label: 'CHANCE 25%', color: 'var(--notumor)' },
          { v: 0.50, label: 'INTERMEDIATE 50%', color: 'var(--amber)' },
          { v: 0.70, label: 'PROBABLE 70%', color: 'var(--magenta)' },
        ].map((t) => {
          const a = Math.PI * (1 - t.v)
          const x1 = 240 + Math.cos(a) * 165, y1 = 220 - Math.sin(a) * 165
          const x2 = 240 + Math.cos(a) * 195, y2 = 220 - Math.sin(a) * 195
          return (
            <g key={t.v}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={t.color} strokeWidth="2" />
              <text x={x2 + (x2 - x1) * 0.4} y={y2 + (y2 - y1) * 0.4 + 4} fill={t.color} fontSize="9" fontFamily="var(--mono)" textAnchor={x2 < 240 ? 'end' : 'start'} letterSpacing="0.12em">{t.label}</text>
            </g>
          )
        })}
        {(() => {
          const a = Math.PI * (1 - (animated ? pct : 0))
          const x = 240 + Math.cos(a) * 175, y = 220 - Math.sin(a) * 175
          return <line x1="240" y1="220" x2={x} y2={y} stroke="var(--ink)" strokeWidth="3" style={{ transition: 'all 1.6s cubic-bezier(.2,.7,.2,1)' }} />
        })()}
        <circle cx="240" cy="220" r="6" fill="var(--ink)" />
        <text x="240" y="260" fill="var(--ink)" fontSize="32" fontFamily="var(--serif)" textAnchor="middle">{(pct * 100).toFixed(2)}%</text>
      </svg>
    </div>
  )
}

// Re-export for convenience
export type { ClassKey }
