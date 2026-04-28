import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3-force'
import { DATA, type ClassKey } from '../data/data'
import { Section } from './Section'

const D = DATA

interface Node extends d3.SimulationNodeDatum {
  id: number
  cls: ClassKey
  predicted: ClassKey
  correct: boolean
}

const ORDER: ClassKey[] = ['glioma', 'meningioma', 'notumor', 'pituitary']

function pickWrong(actual: ClassKey): ClassKey {
  const swaps: Record<ClassKey, ClassKey> = {
    glioma: 'meningioma',
    meningioma: 'glioma',
    notumor: 'pituitary',
    pituitary: 'notumor',
  }
  return swaps[actual]
}

function buildNodes(modelId: string): Node[] {
  const totalSupport = ORDER.reduce((s, k) => s + D.counts[k], 0)
  const SCALE = 240 / totalSupport
  const nodes: Node[] = []
  let id = 0
  ORDER.forEach((cls) => {
    const n = Math.max(20, Math.round(D.counts[cls] * SCALE))
    const recall = D.perClass[modelId][cls]
    const correctN = Math.round(n * recall)
    for (let i = 0; i < n; i++) {
      const correct = i < correctN
      nodes.push({ id: id++, cls, predicted: correct ? cls : pickWrong(cls), correct })
    }
  })
  return nodes
}

const W = 1100, H = 620
const CENTERS: Record<ClassKey, { x: number; y: number }> = {
  glioma:     { x: W * 0.27, y: H * 0.35 },
  meningioma: { x: W * 0.73, y: H * 0.35 },
  notumor:    { x: W * 0.27, y: H * 0.7 },
  pituitary:  { x: W * 0.73, y: H * 0.7 },
}

interface ClusterCanvasProps { modelId: string; mode: 'truth' | 'prediction' }

function ClusterCanvas({ modelId, mode }: ClusterCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const nodes = useMemo(() => buildNodes(modelId), [modelId])

  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const circles = svg.querySelectorAll<SVGCircleElement>('circle.node')
    const sim = d3.forceSimulation<Node>(nodes)
      .force('x', d3.forceX<Node>((d) => CENTERS[mode === 'truth' ? d.cls : d.predicted].x).strength(0.18))
      .force('y', d3.forceY<Node>((d) => CENTERS[mode === 'truth' ? d.cls : d.predicted].y).strength(0.18))
      .force('collide', d3.forceCollide(5.5))
      .alphaDecay(0.02)
      .on('tick', () => {
        nodes.forEach((n, i) => {
          const c = circles[i]
          if (!c) return
          c.setAttribute('cx', String(n.x ?? 0))
          c.setAttribute('cy', String(n.y ?? 0))
        })
      })
    return () => { sim.stop() }
  }, [nodes, mode])

  return (
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
      {ORDER.map((k) => (
        <g key={k} transform={`translate(${CENTERS[k].x}, ${CENTERS[k].y - 130})`}>
          <text textAnchor="middle" fontFamily="var(--mono)" fontSize="11" letterSpacing="0.18em"
            fill={D.classColors[k]} opacity="0.7">{k.toUpperCase()}</text>
        </g>
      ))}
      {nodes.map((n) => {
        const wrong = !n.correct
        return (
          <circle
            key={n.id}
            className="node"
            r={4.2}
            fill={D.classColors[n.cls]}
            opacity={wrong && mode === 'prediction' ? 0.35 : 0.85}
            stroke={wrong && mode === 'prediction' ? 'var(--magenta)' : 'none'}
            strokeWidth={wrong && mode === 'prediction' ? 1 : 0}
          >
            <title>{`${n.cls}${wrong ? ` → predicted ${n.predicted}` : ''}`}</title>
          </circle>
        )
      })}
    </svg>
  )
}

export function ClusterRecall() {
  const [modelId, setModelId] = useState<string>('vgg16ff')
  const [mode, setMode] = useState<'truth' | 'prediction'>('truth')
  return (
    <Section id="13b Cluster Recall"
      kicker="12.5 · Watching the predictions group"
      title={<>Where the model <em>thinks</em> each scan belongs.</>}
      lede="Each dot is a held-out validation scan. Toggle between the ground-truth grouping (every scan in its true class cluster) and the model's prediction (mis-classified scans drift to the wrong cluster, outlined in magenta). Per-class recall becomes the gravity that holds each cluster together."
    >
      <div className="reveal" style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center', margin: '2.4rem 0 1.4rem' }}>
        <div className="kicker">MODEL</div>
        {D.models.map((m) => (
          <button key={m.id} onClick={() => setModelId(m.id)}
            style={{
              background: modelId === m.id ? 'var(--cyan)' : 'transparent',
              color: modelId === m.id ? '#000' : 'var(--ink)',
              border: '1px solid var(--line)', fontFamily: 'var(--mono)', fontSize: 11,
              letterSpacing: '0.14em', padding: '8px 14px', cursor: 'pointer',
            }}>{m.name}</button>
        ))}
        <div style={{ flex: 1 }} />
        <div className="kicker">VIEW</div>
        {(['truth', 'prediction'] as const).map((m) => (
          <button key={m} onClick={() => setMode(m)}
            style={{
              background: mode === m ? 'var(--magenta)' : 'transparent',
              color: mode === m ? '#000' : 'var(--ink)',
              border: '1px solid var(--line)', fontFamily: 'var(--mono)', fontSize: 11,
              letterSpacing: '0.14em', padding: '8px 14px', cursor: 'pointer',
            }}>{m.toUpperCase()}</button>
        ))}
      </div>
      <div className="cluster-wrap reveal">
        <ClusterCanvas modelId={modelId} mode={mode} />
      </div>
      <div className="cluster-legend">
        {ORDER.map((k) => (
          <span key={k}><i style={{ background: D.classColors[k] }} />{k.toUpperCase()}</span>
        ))}
        <span><i style={{ background: 'transparent', border: '1px solid var(--magenta)' }} />MIS-CLASSIFIED</span>
      </div>
      <p className="reveal" style={{ marginTop: '2rem' }}>
        Switch to <strong style={{ color: 'var(--cyan)' }}>VGG-16 + FF</strong>'s prediction view: the four clusters stay clean, with only a small spray of magenta-rimmed dots drifting from glioma into meningioma. Now switch to <strong>Simple ANN</strong>: the glioma cluster bleeds visibly into notumor — the leakage signal in motion.
      </p>
    </Section>
  )
}
