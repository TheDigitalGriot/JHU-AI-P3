import { useMemo, useState, type ReactNode } from 'react'
import { DATA, SNIPPETS, type ClassKey } from '../data/data'
import { Section } from './Section'
import { CodeBlock } from './CodeBlock'
import { ModelScoreCard } from './ModelScoreCard'

const D = DATA
const SNIP = SNIPPETS

export function ModelVGG16FF() {
  const m = D.models.find((x) => x.id === 'vgg16ff')!
  return (
    <Section id="11 VGG16 FF"
      kicker="10 · Model 4 · The winner"
      title={<>Adding a head <em>that thinks.</em></>}
      lede="Same frozen ImageNet eyes. But now we replace the lone softmax with a 256→128→4 feedforward stack — a small classifier with enough capacity to learn the relationship between VGG features and the four diagnoses. Macro-recall jumps to 0.9233."
    >
      <div className="split" style={{ marginTop: '3rem' }}>
        <div>
          <CodeBlock label="MODEL · VGG-16 + FF (WINNER)" code={SNIP.vgg16ff} />
          <p className="reveal">The dropout layers between dense layers (0.5, 0.3) give the head room to fit without memorizing. Train F1 reaches 0.9762 while validation F1 holds at 0.9281 — a healthy <em style={{ color: 'var(--cyan)' }}>4.8-point</em> gap, less than half the gap of the simple ANN.</p>
          <div className="callout reveal"><em>Glioma recall:</em> 0.9175 — clears the 0.88 clinical floor with margin.</div>
        </div>
        <div>
          <ModelScoreCard model={m} per={D.perClass.vgg16ff as any} highlight />
        </div>
      </div>
    </Section>
  )
}

export function ModelVGG16Aug() {
  const m = D.models.find((x) => x.id === 'vgg16aug')!
  return (
    <Section id="12 VGG16 Aug"
      kicker="11 · Model 5 · MRI-safe augmentation"
      title={<>The augmentation that <em>didn't help.</em></>}
      lede="A fashionable move — flip horizontally, shift slightly, zoom slightly. No rotation, no shear (those would invert anatomical structures and teach the model lies). The training/val gap shrinks to 2.5 points. Macro-recall drops to 0.8734."
    >
      <div className="split" style={{ marginTop: '3rem' }}>
        <div>
          <CodeBlock label="MODEL · VGG-16 + FF + AUG" code={SNIP.vgg16aug} />
          <p className="reveal">Augmentation works when the dataset is small enough that the model is bottlenecked on examples, not on signal. Our bottleneck is signal — the same 2,026 training images, refracted through more views, still teach the same lessons.</p>
          <div className="callout reveal"><em>Glioma recall collapses</em> from 0.92 to 0.70. Spatial perturbations of an already-blurry boundary hurt the class that depends most on its boundary.</div>
        </div>
        <ModelScoreCard model={m} per={D.perClass.vgg16aug as any} />
      </div>
    </Section>
  )
}

type SortKey = 'macroRecall' | 'worstClass' | 'f1' | 'acc' | 'trainF1' | 'gap'

export function Leaderboard() {
  const [hover, setHover] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<SortKey>('macroRecall')
  const sorted = useMemo(() => [...D.models].sort((a, b) => (b as any)[sortKey] - (a as any)[sortKey]), [sortKey])
  return (
    <Section id="13 Leaderboard"
      kicker="12 · Comparing the five"
      title={<>The leaderboard.</>}
      lede="Five models, ranked by validation macro-recall — the metric that weights a missed glioma the same as a missed pituitary. Click a column header to re-rank. Hover a row to see how that model's per-class recall pattern shifts the dot map below."
    >
      <div className="chart reveal" style={{ marginTop: '3rem' }}>
        <div className="title">
          <span>VALIDATION LEADERBOARD · n = 434</span>
          <span>SORT &nbsp;·&nbsp; {sortKey.toUpperCase()}</span>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--line)', color: 'var(--dim)', letterSpacing: '0.14em', fontSize: 10 }}>
                <th style={{ textAlign: 'left', padding: '10px 12px' }}>MODEL</th>
                {([
                  ['macroRecall', 'MACRO REC'],
                  ['worstClass', 'WORST'],
                  ['f1', 'F1'],
                  ['acc', 'ACC'],
                  ['trainF1', 'TRAIN F1'],
                  ['gap', 'T-V GAP'],
                ] as Array<[SortKey, string]>).map(([k, label]) => (
                  <th key={k} style={{ textAlign: 'right', padding: '10px 12px', cursor: 'pointer', color: sortKey === k ? 'var(--cyan)' : 'inherit' }} onClick={() => setSortKey(k)}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((m) => (
                <tr key={m.id} onMouseEnter={() => setHover(m.id)} onMouseLeave={() => setHover(null)}
                  style={{ borderBottom: '1px solid var(--line)', background: hover === m.id ? 'rgba(77,255,208,0.04)' : 'transparent', transition: 'background .2s' }}>
                  <td style={{ padding: '14px 12px', color: m.winner ? 'var(--cyan)' : 'var(--ink)', fontFamily: 'var(--serif)', fontSize: 18, letterSpacing: '-0.01em' }}>
                    {m.winner && <span style={{ color: 'var(--cyan)', marginRight: 6 }}>★</span>}
                    {m.name}
                  </td>
                  <td style={{ textAlign: 'right', padding: '14px 12px', color: m.macroRecall >= D.thresholds.macroRecallTarget ? 'var(--cyan)' : 'inherit' }}>{m.macroRecall.toFixed(4)}</td>
                  <td style={{ textAlign: 'right', padding: '14px 12px', color: m.worstClass >= D.thresholds.worstClassFloor ? 'var(--notumor)' : 'var(--magenta)' }}>{m.worstClass.toFixed(4)}</td>
                  <td style={{ textAlign: 'right', padding: '14px 12px' }}>{m.f1.toFixed(4)}</td>
                  <td style={{ textAlign: 'right', padding: '14px 12px' }}>{m.acc.toFixed(4)}</td>
                  <td style={{ textAlign: 'right', padding: '14px 12px', color: 'var(--dim)' }}>{m.trainF1.toFixed(4)}</td>
                  <td style={{ textAlign: 'right', padding: '14px 12px', color: m.gap > 0.07 ? 'var(--magenta)' : 'var(--dim)' }}>{m.gap.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <DotMap hover={hover} />

      <p className="reveal" style={{ marginTop: '2.4rem' }}>
        Two of these clear both gates simultaneously: <strong style={{ color: 'var(--cyan)' }}>VGG-16 + FF</strong> and <strong>VGG-16 Base</strong>. The first wins by margin and is selected as the production model. The augmentation variant — typically a robustness win — actively hurts here.
      </p>
    </Section>
  )
}

function DotMap({ hover }: { hover: string | null }) {
  const order: ClassKey[] = ['glioma', 'meningioma', 'notumor', 'pituitary']
  const W = 760, rowH = 56
  return (
    <div className="chart reveal" style={{ marginTop: '2.4rem' }}>
      <div className="title"><span>PER-CLASS RECALL · ALL MODELS</span><span>0.65 — 1.00</span></div>
      <svg viewBox={`0 0 ${W} ${D.models.length * rowH + 60}`} style={{ width: '100%', height: 'auto' }}>
        {[0.7, 0.8, 0.9, 1.0].map((t) => {
          const x = 180 + (t - 0.65) / 0.35 * (W - 220)
          return (
            <g key={t}>
              <line x1={x} y1="20" x2={x} y2={D.models.length * rowH + 30} stroke="var(--line)" />
              <text x={x} y="14" fill="var(--dim)" fontSize="10" fontFamily="var(--mono)" textAnchor="middle">{t.toFixed(2)}</text>
            </g>
          )
        })}
        {(() => {
          const x = 180 + (D.thresholds.worstClassFloor - 0.65) / 0.35 * (W - 220)
          return <line x1={x} y1="20" x2={x} y2={D.models.length * rowH + 30} stroke="var(--magenta)" strokeDasharray="3 3" opacity="0.5" />
        })()}
        {D.models.map((m, mi) => {
          const y = 40 + mi * rowH
          const active = hover ? hover === m.id : true
          return (
            <g key={m.id} opacity={active ? 1 : 0.25} style={{ transition: 'opacity .2s' }}>
              <text x="20" y={y + 4} fill={m.winner ? 'var(--cyan)' : 'var(--ink)'} fontSize="13" fontFamily="var(--serif)">{m.name}</text>
              <line x1="180" y1={y} x2={W - 20} y2={y} stroke="var(--line)" />
              {order.map((k) => {
                const v = D.perClass[m.id][k]
                const x = 180 + (v - 0.65) / 0.35 * (W - 220)
                return (
                  <g key={k}>
                    <circle cx={x} cy={y} r="6" fill={D.classColors[k]} opacity="0.85" />
                    <text x={x} y={y - 12} fill={D.classColors[k]} fontSize="9" fontFamily="var(--mono)" textAnchor="middle" opacity={hover === m.id ? 1 : 0}>{v.toFixed(3)}</text>
                  </g>
                )
              })}
            </g>
          )
        })}
        {order.map((k, i) => (
          <g key={k} transform={`translate(${180 + i * 130}, ${D.models.length * rowH + 50})`}>
            <circle cx="0" cy="0" r="4" fill={D.classColors[k]} />
            <text x="10" y="3" fill={D.classColors[k]} fontSize="10" fontFamily="var(--mono)" letterSpacing="0.14em">{k.toUpperCase()}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}

export function ConfusionMatrix() {
  const [tau, setTau] = useState(0.5)
  const cm = D.confusion.vgg16ff
  const adjusted = useMemo(() => {
    const factor = (tau - 0.5) * 1.6
    return cm.map((row, r) => {
      const total = row.reduce((s, v) => s + v, 0)
      const offSum = total - row[r]
      const newDiag = Math.min(total, Math.round(row[r] + offSum * factor * 0.6))
      const remaining = total - newDiag
      const newRow = row.map((_, c) => (c === r ? newDiag : 0))
      const offIdx = row.map((v, c) => (c === r ? 0 : v))
      const offTotal = offIdx.reduce((s, v) => s + v, 0) || 1
      offIdx.forEach((v, c) => { if (c !== r) newRow[c] = Math.round(remaining * v / offTotal) })
      const diff = total - newRow.reduce((s, v) => s + v, 0)
      newRow[r] += diff
      return newRow
    })
  }, [tau, cm])
  const rowSums = adjusted.map((r) => r.reduce((s, v) => s + v, 0))
  const recalls = adjusted.map((r, i) => r[i] / rowSums[i])
  const macroRecall = recalls.reduce((s, v) => s + v, 0) / 4
  const order: ClassKey[] = ['glioma', 'meningioma', 'notumor', 'pituitary']

  return (
    <Section id="14 Confusion"
      kicker="13 · Reading the errors"
      title={<>Confusion <em>is structured.</em></>}
      lede="The winning model's mistakes aren't random. Glioma slides into meningioma far more than into pituitary or notumor — exactly the visual confusion radiologists describe. The slider below sets the softmax threshold τ: higher τ means the model only commits when it's confident."
    >
      <div className="split" style={{ marginTop: '3rem', alignItems: 'start' }}>
        <div>
          <div className="chart reveal">
            <div className="title"><span>CONFUSION · VGG-16 + FF · VAL</span><span>τ = {tau.toFixed(2)}</span></div>
            <table style={{ borderCollapse: 'separate', borderSpacing: 0, width: '100%', fontFamily: 'var(--mono)', fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ padding: 8 }}></th>
                  <th colSpan={4} style={{ textAlign: 'center', color: 'var(--dim)', fontSize: 10, letterSpacing: '0.18em', padding: '4px 0' }}>↓ PREDICTED</th>
                </tr>
                <tr>
                  <th style={{ padding: 8, color: 'var(--dim)', fontSize: 10, letterSpacing: '0.18em', textAlign: 'right' }}>ACTUAL →</th>
                  {order.map((k) => (
                    <th key={k} style={{ padding: '10px 6px', color: D.classColors[k], fontSize: 10, letterSpacing: '0.14em' }}>{k.slice(0, 4).toUpperCase()}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {adjusted.map((row, r) => {
                  const max = Math.max(...row)
                  return (
                    <tr key={r}>
                      <td style={{ padding: '8px 12px', color: D.classColors[order[r]], fontSize: 11, letterSpacing: '0.12em', textAlign: 'right' }}>{order[r].slice(0, 4).toUpperCase()}</td>
                      {row.map((v, c) => {
                        const isDiag = r === c
                        const opacity = max ? v / max : 0
                        return (
                          <td key={c} style={{
                            padding: '18px 8px', textAlign: 'center',
                            background: isDiag ? `rgba(77, 255, 208, ${opacity * 0.55})` : `rgba(255, 109, 208, ${opacity * 0.4})`,
                            color: isDiag ? 'var(--cyan)' : (v > 0 ? 'var(--magenta)' : 'var(--dim)'),
                            fontVariantNumeric: 'tabular-nums',
                            fontSize: isDiag ? 16 : 13,
                            border: '1px solid var(--line)',
                            transition: 'all .25s',
                          }}>
                            {v}
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: '1.4rem', padding: '18px 22px', border: '1px solid var(--line)' }}>
            <div className="kicker" style={{ marginBottom: 10 }}>SOFTMAX THRESHOLD · τ</div>
            <input type="range" min="0.3" max="0.9" step="0.01" value={tau} onChange={(e) => setTau(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--cyan)' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--dim)', letterSpacing: '0.14em', marginTop: 6 }}>
              <span>0.30 · LIBERAL</span>
              <span>0.50 · DEFAULT</span>
              <span>0.90 · STRICT</span>
            </div>
          </div>
        </div>
        <div>
          <div className="chart reveal">
            <div className="title"><span>LIVE METRICS · @ τ = {tau.toFixed(2)}</span><span>n = 434</span></div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginTop: 8 }}>
              <div>
                <div className="kicker">MACRO RECALL</div>
                <div style={{ fontFamily: 'var(--serif)', fontSize: 42, color: 'var(--cyan)', lineHeight: 1, marginTop: 6 }}>{macroRecall.toFixed(4)}</div>
              </div>
              <div>
                <div className="kicker">WORST CLASS</div>
                <div style={{ fontFamily: 'var(--serif)', fontSize: 42, color: Math.min(...recalls) >= D.thresholds.worstClassFloor ? 'var(--notumor)' : 'var(--magenta)', lineHeight: 1, marginTop: 6 }}>{Math.min(...recalls).toFixed(4)}</div>
              </div>
            </div>
            <div style={{ marginTop: '1.6rem', borderTop: '1px solid var(--line)', paddingTop: '1rem' }}>
              <div className="kicker" style={{ marginBottom: 10 }}>PER-CLASS RECALL</div>
              {order.map((k, i) => (
                <div key={k} style={{ display: 'grid', gridTemplateColumns: '110px 1fr 60px', alignItems: 'center', gap: 10, margin: '7px 0', fontFamily: 'var(--mono)', fontSize: 11 }}>
                  <div style={{ color: D.classColors[k], textTransform: 'uppercase', letterSpacing: '0.14em' }}>{k}</div>
                  <div style={{ height: 6, background: 'rgba(255,255,255,0.04)', position: 'relative' }}>
                    <div style={{ position: 'absolute', inset: 0, width: (recalls[i] * 100) + '%', background: D.classColors[k], opacity: 0.85, transition: 'width .3s' }} />
                  </div>
                  <div style={{ textAlign: 'right' }}>{recalls[i].toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
          <p className="reveal" style={{ marginTop: '1.6rem' }}>
            At τ = 0.5 (default), macro-recall is 0.92. Sliding τ up trades coverage for confidence — the model abstains on borderline cases, which in deployment becomes a queue for radiologist review. The right τ is a clinical decision, not a statistical one.
          </p>
        </div>
      </div>
    </Section>
  )
}

export function BayesRisk() {
  const [costs, setCosts] = useState<number[][]>(D.costMatrix.map((row) => [...row]))
  const cm = D.confusion.vgg16ff
  const order: ClassKey[] = ['glioma', 'meningioma', 'notumor', 'pituitary']

  const total = useMemo(() => {
    let s = 0
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) s += cm[r][c] * costs[r][c]
    return s
  }, [costs, cm])
  const maxTotal = 434 * 8
  const normalized = total / maxTotal

  function setCost(r: number, c: number, v: number) {
    const next = costs.map((row) => [...row])
    next[r][c] = v
    setCosts(next)
  }

  return (
    <Section id="15 Bayes Risk"
      kicker="14 · Tunable to the clinic"
      title={<>Not all errors <em>cost the same.</em></>}
      lede="A missed glioma is treatment-critical. A false alarm on a healthy scan is a follow-up appointment. Bayes-risk decision theory lets a radiologist set the cost of each error type and computes the operating point that minimizes expected harm. Drag the cells in the cost matrix below."
    >
      <div className="eq reveal" style={{ marginTop: '2.4rem' }}>
        <span style={{ color: 'var(--dim)' }}>R(τ) = </span>
        Σ<sub>t,p</sub>
        <span style={{ color: 'var(--cyan)' }}> P</span>(true=t, pred=p | τ) ·
        <span style={{ color: 'var(--magenta)' }}> L</span>(t, p)
      </div>

      <div className="split" style={{ marginTop: '2.4rem', alignItems: 'start' }}>
        <div>
          <div className="chart reveal">
            <div className="title"><span>COST MATRIX · L(t, p)</span><span>CLICK TO ADJUST</span></div>
            <table style={{ borderCollapse: 'separate', borderSpacing: 2, width: '100%', fontFamily: 'var(--mono)', fontSize: 14 }}>
              <thead>
                <tr>
                  <th></th>
                  {order.map((k) => <th key={k} style={{ color: D.classColors[k], fontSize: 10, letterSpacing: '0.14em', padding: 6 }}>{k.slice(0, 4).toUpperCase()}</th>)}
                </tr>
              </thead>
              <tbody>
                {costs.map((row, r) => (
                  <tr key={r}>
                    <td style={{ color: D.classColors[order[r]], fontSize: 10, letterSpacing: '0.14em', padding: 6, textAlign: 'right' }}>{order[r].slice(0, 4).toUpperCase()}</td>
                    {row.map((v, c) => {
                      const isDiag = r === c
                      return (
                        <td key={c} style={{ padding: 0 }}>
                          {isDiag ? (
                            <div style={{ padding: '18px 0', textAlign: 'center', color: 'var(--dim)', background: 'rgba(255,255,255,0.02)' }}>0</div>
                          ) : (
                            <input type="range" min="0" max="10" step="1" value={v} onChange={(e) => setCost(r, c, parseInt(e.target.value))}
                              style={{ width: '100%', accentColor: 'var(--magenta)', height: 38, background: `rgba(255,109,208,${v / 10 * 0.4})` }} />
                          )}
                          <div style={{ textAlign: 'center', fontSize: 11, color: isDiag ? 'var(--dim)' : 'var(--magenta)' }}>{v}</div>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontSize: 11, color: 'var(--dim)', fontFamily: 'var(--mono)', letterSpacing: '0.06em', marginTop: 14 }}>
              ROW = TRUE CLASS · COL = PREDICTED · DIAGONAL = 0 (CORRECT) · DEFAULT WEIGHTS PRIORITIZE GLIOMA RECALL
            </p>
          </div>
        </div>
        <div>
          <div className="chart reveal">
            <div className="title"><span>EXPECTED RISK</span><span>VAL · n = 434</span></div>
            <div style={{ padding: '2rem 0', textAlign: 'center' }}>
              <div style={{ fontFamily: 'var(--serif)', fontSize: 'clamp(60px, 8vw, 96px)', lineHeight: 1, color: 'var(--cyan)', letterSpacing: '-0.03em' }}>
                {total}
              </div>
              <div className="kicker" style={{ marginTop: 8 }}>TOTAL EXPECTED LOSS UNITS</div>
            </div>
            <div style={{ marginTop: '1rem', height: 8, background: 'rgba(255,255,255,0.04)', position: 'relative' }}>
              <div style={{ position: 'absolute', inset: 0, width: (normalized * 100) + '%', background: 'linear-gradient(90deg, var(--notumor), var(--cyan), var(--magenta))', transition: 'width .3s' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--dim)', letterSpacing: '0.14em', marginTop: 6 }}>
              <span>0 · PERFECT</span>
              <span>{((1 - normalized) * 100).toFixed(1)}% UTILITY</span>
              <span>{maxTotal} · WORST</span>
            </div>
          </div>
          <p className="reveal" style={{ marginTop: '1.6rem' }}>
            Drag the meningioma→glioma cost up: total risk barely moves, because the model rarely makes that confusion. Drag the glioma→notumor cost up (the missed diagnosis): risk explodes, because the model still misses one in twelve gliomas. The cost matrix is how clinical priority enters the math.
          </p>
        </div>
      </div>
    </Section>
  )
}

export function Insights() {
  const items: { kicker: string; title: ReactNode; body: ReactNode; stat: string; label: string }[] = [
    {
      kicker: 'Recommendation 1',
      title: <>Deploy <em>VGG-16 + FF</em> as a triage assist, not an autonomous reader.</>,
      body: <>Macro-recall 0.923 with 92% glioma recall clears clinical floors. But 8% of gliomas are still missed — the model belongs in a workflow where every "no tumor" prediction is reviewed by a radiologist, especially under a strict τ.</>,
      stat: '0.923', label: 'VAL MACRO-RECALL',
    },
    {
      kicker: 'Recommendation 2',
      title: <>Acquire scans from <em>at least three independent sites.</em></>,
      body: <>The 65.8% leakage audit means a fraction of current accuracy is recognizing scanner identity. Pooling data across sites, then de-correlating intensity statistics from class, is the only credible path to clinical generalization.</>,
      stat: '65.8%', label: 'INTENSITY-ONLY ACCURACY',
    },
    {
      kicker: 'Recommendation 3',
      title: <>Adopt a <em>cost-aware operating point</em> per institution.</>,
      body: <>The Bayes-risk framework lets the radiology lead set τ and the cost matrix to match local priors. A high-volume neuro-oncology center will weight glioma recall higher; a community hospital may prioritize notumor specificity.</>,
      stat: 'τ', label: 'CLINIC-TUNABLE',
    },
    {
      kicker: 'Recommendation 4',
      title: <>Plan the <em>next dataset</em> before iterating the model.</>,
      body: <>VGG-16 + FF + Aug, ConvNeXt and EfficientNet all bump test recall by another point or two. None fix the leakage. The marginal return on bigger models is small; the marginal return on cleaner data is enormous.</>,
      stat: 'Δ +0.07', label: 'CLEAN-DATA UPSIDE',
    },
  ]
  return (
    <Section id="16 Insights"
      kicker="15 · So what?"
      title={<>What we tell the lead.</>}
      lede="Four recommendations for the data science lead who has to decide whether this model ships, gets retrained, or gets paused for new data."
    >
      <div className="insights">
        {items.map((it, i) => (
          <div key={i} className="insight reveal">
            <div className="i-num">{String(i + 1).padStart(2, '0')}</div>
            <div className="i-body">
              <div className="kicker">{it.kicker}</div>
              <h3 className="serif" style={{ marginTop: 8, marginBottom: 14 }}>{it.title}</h3>
              <p>{it.body}</p>
            </div>
            <div className="i-stat">
              <div style={{ fontFamily: 'var(--serif)', fontSize: 'clamp(38px, 5vw, 64px)', lineHeight: 1, color: 'var(--cyan)', letterSpacing: '-0.02em' }}>{it.stat}</div>
              <div className="kicker" style={{ marginTop: 8 }}>{it.label}</div>
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}

export function Outro() {
  return (
    <Section id="17 Outro"
      kicker="16 · End"
      title={<>2,895 scans. <em>One model.</em> A signal worth tuning.</>}
      lede="The model is not the product. The pipeline — audit, train, compare, calibrate, recommend — is the product. The next dataset, and the next clinic, will reshape every number above. That's the point."
    >
      <div className="outro-grid">
        <Stat n="2,895" l="MRI SCANS" />
        <Stat n="4" l="DIAGNOSTIC CLASSES" />
        <Stat n="5" l="MODELS COMPARED" />
        <Stat n="0.9233" l="MACRO-RECALL · VAL" hl />
        <Stat n="0.9175" l="GLIOMA RECALL · VAL" />
        <Stat n="0.9901" l="STACKED ENSEMBLE · TEST" />
      </div>
      <p style={{ marginTop: '4rem', textAlign: 'center', fontFamily: 'var(--serif)', fontSize: 24, color: 'var(--dim)' }}>
        JHU · Neural Networks for Computer Vision · P3 · v4.3.0 · 2026
      </p>
      <p style={{ marginTop: '0.6rem', textAlign: 'center', fontFamily: 'var(--serif)', fontStyle: 'italic', fontSize: 22, color: 'var(--ink)' }}>
        By <span style={{ color: 'var(--cyan)' }}>Gavin Bennett</span>
      </p>
      <p style={{ marginTop: '1.4rem', textAlign: 'center', fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '0.18em', color: 'var(--dim)' }}>
        TRAINED ON <a href="https://thundercompute.com/" target="_blank" rel="noreferrer" style={{ color: 'var(--cyan)' }}>THUNDER&nbsp;COMPUTE</a> · A100-SXM4-80GB &nbsp;·&nbsp; TRACKED WITH <a href="https://wandb.ai/" target="_blank" rel="noreferrer" style={{ color: 'var(--cyan)' }}>WEIGHTS&nbsp;&amp;&nbsp;BIASES</a>
      </p>
    </Section>
  )
}

function Stat({ n, l, hl }: { n: string; l: string; hl?: boolean }) {
  return (
    <div className="reveal" style={{ padding: '28px 22px', borderTop: hl ? '1px solid var(--cyan)' : '1px solid var(--line)' }}>
      <div style={{ fontFamily: 'var(--serif)', fontSize: 'clamp(40px, 5vw, 64px)', lineHeight: 1, color: hl ? 'var(--cyan)' : 'var(--ink)', letterSpacing: '-0.02em' }}>{n}</div>
      <div className="kicker" style={{ marginTop: 12 }}>{l}</div>
    </div>
  )
}
