import { useEffect, useRef } from 'react'
import { DATA, type ClassKey } from '../data/data'
import { Section } from './Section'
import { NumTween } from './NumTween'

const D = DATA

export function FourClasses() {
  const cards = [
    { id: 'glioma' as ClassKey, name: 'Glioma', latin: 'Aggressive · Infiltrative', n: D.counts.glioma,
      blurb: 'High-grade tumors that invade surrounding brain tissue with diffuse, irregular borders. Treatment is aggressive — surgery, radiation, chemo. Misclassifying a glioma is the costliest error.',
      tag: 'IRREGULAR MASS · BLURRED MARGINS', color: D.classColors.glioma },
    { id: 'meningioma' as ClassKey, name: 'Meningioma', latin: 'Slow · Encapsulated', n: D.counts.meningioma,
      blurb: 'Usually benign tumors that arise from the meninges, the brain’s outer membranes. Well-defined boundaries — but on a single MRI slice they can mimic gliomas in intensity and shape.',
      tag: 'ROUND MASS · DURAL ATTACHMENT', color: D.classColors.meningioma },
    { id: 'pituitary' as ClassKey, name: 'Pituitary', latin: 'Hormonal · Localized', n: D.counts.pituitary,
      blurb: 'Tumors of the pituitary gland — small, central, often discovered through hormonal symptoms. Distinctive location helps the model, but the lesion barely shifts global image statistics.',
      tag: 'CENTRAL · SUB-CRANIAL', color: D.classColors.pituitary },
    { id: 'notumor' as ClassKey, name: 'No Tumor', latin: 'Healthy · Reference', n: D.counts.notumor,
      blurb: 'Healthy brain scans. The reference class — except in this dataset the healthy scans were aggregated from many sources, so they carry a different scanner signature than the tumor classes.',
      tag: 'REFERENCE · MIXED ACQUISITION', color: D.classColors.notumor },
  ]
  return (
    <Section id="02 Four Classes"
      kicker="01 · The clinical question"
      title={<>Four diagnoses. <em style={{ fontStyle: 'italic', color: 'var(--cyan)' }}>One scan.</em></>}
      lede="A neuroradiologist looks at an axial slice and has to commit. The model has to learn the same commitment from pixels alone — and the four answers don't always look as different as they sound."
    >
      <div className="class-grid">
        {cards.map((c) => (
          <div key={c.id} className="class-card reveal" style={{ ['--accent' as any]: c.color }}>
            <div className="latin">{c.latin}</div>
            <div className="name serif">{c.name}</div>
            <div className={`mri ${c.id}`} style={{ margin: '18px 0', height: 'auto' }}>
              <div className="crosshair-mri" />
              <div className="label">T1 · {c.id}</div>
            </div>
            <div className="blurb">{c.blurb}</div>
            <div className="count">{c.n}<small>SCANS</small></div>
            <div className="tag">{c.tag}</div>
          </div>
        ))}
      </div>
    </Section>
  )
}

export function Dataset() {
  return (
    <Section id="03 Dataset"
      kicker="02 · The corpus"
      title={<>Two thousand, eight hundred and ninety-five MRIs.</>}
      lede="A modest dataset by ImageNet standards — and it shows. Class balance is mild but real, and the resolution distribution hides the first hint that something is off."
    >
      <div className="big-num reveal serif" style={{ margin: '2.4rem 0 1rem' }}>
        <NumTween to={D.total} suffix="" decimals={0} />
      </div>
      <div className="stat-row reveal">
        <div><div className="label">GLIOMA</div><div className="val" style={{ color: D.classColors.glioma }}>{D.counts.glioma}</div></div>
        <div><div className="label">MENINGIOMA</div><div className="val" style={{ color: D.classColors.meningioma }}>{D.counts.meningioma}</div></div>
        <div><div className="label">NO TUMOR</div><div className="val" style={{ color: D.classColors.notumor }}>{D.counts.notumor}</div></div>
        <div><div className="label">PITUITARY</div><div className="val" style={{ color: D.classColors.pituitary }}>{D.counts.pituitary}</div></div>
      </div>
      <ClassBalanceChart />
      <p className="reveal" style={{ marginTop: '2rem' }}>
        The imbalance ratio is <strong style={{ color: 'var(--cyan)' }}>1.30 : 1</strong> between the largest and smallest class — mild enough to keep training honest, but enough that we lead the comparison with <em>macro-recall</em> rather than accuracy.
      </p>
    </Section>
  )
}

function ClassBalanceChart() {
  const max = Math.max(...Object.values(D.counts))
  const order: ClassKey[] = ['glioma', 'meningioma', 'notumor', 'pituitary']
  return (
    <div className="chart reveal" style={{ marginTop: '2.4rem' }}>
      <div className="title"><span>CLASS BALANCE · n = 2,895</span><span>↑ COUNT</span></div>
      {order.map((k) => {
        const w = D.counts[k] / max * 100
        return (
          <div key={k} style={{ display: 'grid', gridTemplateColumns: '160px 1fr 80px', alignItems: 'center', gap: 14, margin: '10px 0', fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '0.12em' }}>
            <div style={{ textTransform: 'uppercase', color: D.classColors[k] }}>{k}</div>
            <div style={{ height: 14, background: 'rgba(255,255,255,0.04)', position: 'relative' }}>
              <div className="reveal-bar" style={{ position: 'absolute', inset: 0, width: w + '%', background: D.classColors[k], opacity: 0.85 }} />
            </div>
            <div style={{ textAlign: 'right' }}>{D.counts[k]}</div>
          </div>
        )
      })}
    </div>
  )
}

export function EDA() {
  const ref = useRef<HTMLElement>(null)
  return (
    <section className="beat" id="04 EDA" ref={ref} data-screen-label="04 EDA">
      <div className="container">
        <div className="kicker reveal">03 · Looking at the pixels</div>
        <h2 className="reveal serif" style={{ marginTop: '0.4rem', maxWidth: '18ch' }}>The dataset gives away its source.</h2>
        <div className="split" style={{ marginTop: '3rem' }}>
          <div>
            <p className="reveal">The three <em>tumor</em> classes are tightly distributed at 512×512. The <strong>no-tumor</strong> scans are not — widths and heights swing from 150 to 1,920 pixels, with non-square aspect ratios scattered in.</p>
            <p className="reveal">That isn't a tumor signal. It's a <em>scanner signal</em>. Healthy scans were aggregated from many imaging centers; tumor scans came from a curated pipeline.</p>
            <p className="reveal">Resizing to 128×128 erases the dimension-as-feature shortcut. But the intensity distribution — the average brightness of every pixel — still differs by class:</p>
            <div className="stat-row reveal">
              <div><div className="label">GLIOMA µ</div><div className="val">37.16</div></div>
              <div><div className="label">MENING. µ</div><div className="val">40.79</div></div>
              <div><div className="label">PITUIT. µ</div><div className="val">48.07</div></div>
              <div><div className="label">NO-TUMOR µ</div><div className="val" style={{ color: D.classColors.notumor }}>61.04</div></div>
            </div>
            <div className="callout reveal serif">
              <em>The healthy class is systematically brighter</em> — a class-correlated artifact baked into the data before any model sees it.
            </div>
          </div>
          <div className="reveal">
            <IntensityDistChart />
          </div>
        </div>
      </div>
    </section>
  )
}

function IntensityDistChart() {
  const W = 520, H = 320, P = 28
  const xMin = 0, xMax = 130
  const xs = (v: number) => P + (v - xMin) / (xMax - xMin) * (W - 2 * P)
  const order: ClassKey[] = ['glioma', 'meningioma', 'pituitary', 'notumor']
  const curves = order.map((k) => {
    const { mean, std } = D.intensityStats[k]
    const pts: [number, number][] = []
    for (let x = xMin; x <= xMax; x += 1.2) {
      const z = (x - mean) / std
      const y = Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI))
      pts.push([x, y])
    }
    return { k, pts }
  })
  const maxY = Math.max(...curves.flatMap((c) => c.pts.map((p) => p[1])))
  const ys = (v: number) => H - P - (v / maxY) * (H - 2 * P)

  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    const paths = ref.current.querySelectorAll<SVGPathElement>('path.curve')
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (e.isIntersecting) {
          paths.forEach((p) => {
            const len = p.getTotalLength()
            p.style.strokeDasharray = String(len)
            p.style.strokeDashoffset = String(len)
            p.getBoundingClientRect()
            p.style.transition = 'stroke-dashoffset 1.6s cubic-bezier(.2,.7,.2,1) var(--d, 0ms)'
            p.style.strokeDashoffset = '0'
          })
        }
      })
    }, { threshold: 0.4 })
    io.observe(ref.current)
    return () => io.disconnect()
  }, [])

  return (
    <div className="chart" ref={ref}>
      <div className="title"><span>MEAN PIXEL INTENSITY · BY CLASS</span><span>0—255</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="var(--line)" />
        <line x1={P} y1={P} x2={P} y2={H - P} stroke="var(--line)" />
        {[0, 25, 50, 75, 100, 125].map((t) => (
          <g key={t}>
            <line x1={xs(t)} y1={H - P} x2={xs(t)} y2={H - P + 4} stroke="var(--dim)" />
            <text x={xs(t)} y={H - P + 18} fill="var(--dim)" fontSize="10" fontFamily="var(--mono)" textAnchor="middle">{t}</text>
          </g>
        ))}
        {curves.map((c, i) => {
          const dStr = 'M' + c.pts.map(([x, y], j) => (j === 0 ? '' : 'L') + xs(x) + ',' + ys(y)).join(' ')
          const fillD = dStr + ` L${xs(c.pts[c.pts.length - 1][0])},${H - P} L${xs(c.pts[0][0])},${H - P} Z`
          return (
            <g key={c.k}>
              <path d={fillD} fill={D.classColors[c.k]} opacity="0.08" />
              <path className="curve" d={dStr} fill="none" stroke={D.classColors[c.k]} strokeWidth="2" style={{ ['--d' as any]: (i * 180) + 'ms' }} />
              <line x1={xs(D.intensityStats[c.k].mean)} y1={H - P} x2={xs(D.intensityStats[c.k].mean)} y2={ys(maxY * 0.92)} stroke={D.classColors[c.k]} strokeDasharray="3 3" opacity="0.5" />
              <text x={xs(D.intensityStats[c.k].mean)} y={ys(maxY * 0.92) - 6} fill={D.classColors[c.k]} fontSize="10" fontFamily="var(--mono)" textAnchor="middle" letterSpacing="0.15em">{c.k.toUpperCase()}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
