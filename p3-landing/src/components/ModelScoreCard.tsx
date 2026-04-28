import { DATA, type ClassKey, type ModelRow } from '../data/data'

const D = DATA

interface Props {
  model: ModelRow
  per: Record<ClassKey, number>
  highlight?: boolean
}

export function ModelScoreCard({ model, per, highlight }: Props) {
  const targets = D.thresholds
  return (
    <div className={'chart reveal' + (highlight ? ' winner' : '')} style={highlight ? { borderColor: 'var(--cyan)', boxShadow: '0 0 0 1px var(--cyan-soft)' } : {}}>
      <div className="title">
        <span style={{ color: highlight ? 'var(--cyan)' : 'inherit' }}>{model.name}{model.winner ? ' · WINNER' : ''}</span>
        <span>VAL · n=434</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <Metric label="MACRO RECALL" value={model.macroRecall} target={targets.macroRecallTarget} highlight />
        <Metric label="WORST CLASS" value={model.worstClass} target={targets.worstClassFloor} />
        <Metric label="F1 (W)" value={model.f1} />
        <Metric label="ACCURACY" value={model.acc} />
      </div>
      <div style={{ marginTop: '1.4rem', borderTop: '1px solid var(--line)', paddingTop: '1rem' }}>
        <div className="kicker" style={{ marginBottom: '0.8rem' }}>PER-CLASS RECALL</div>
        {(['glioma', 'meningioma', 'notumor', 'pituitary'] as ClassKey[]).map((k) => (
          <div key={k} style={{ display: 'grid', gridTemplateColumns: '110px 1fr 60px', alignItems: 'center', gap: 10, margin: '7px 0', fontFamily: 'var(--mono)', fontSize: 11 }}>
            <div style={{ color: D.classColors[k], textTransform: 'uppercase', letterSpacing: '0.14em' }}>{k}</div>
            <div style={{ height: 6, background: 'rgba(255,255,255,0.04)', position: 'relative' }}>
              <div style={{ position: 'absolute', inset: 0, width: (per[k] * 100) + '%', background: D.classColors[k], opacity: 0.85, transition: 'width 1.4s cubic-bezier(.2,.7,.2,1)' }} />
              <div style={{ position: 'absolute', left: (targets.worstClassFloor * 100) + '%', top: -3, bottom: -3, width: 1, background: 'var(--ink)', opacity: 0.4 }} />
            </div>
            <div style={{ textAlign: 'right' }}>{per[k].toFixed(3)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

interface MetricProps {
  label: string
  value: number
  target?: number
  highlight?: boolean
}

function Metric({ label, value, target, highlight }: MetricProps) {
  const pass = target ? value >= target : true
  return (
    <div style={{ padding: '14px 0' }}>
      <div className="kicker" style={{ marginBottom: 6 }}>{label}</div>
      <div style={{ fontFamily: 'var(--serif)', fontSize: 32, color: highlight ? 'var(--cyan)' : 'var(--ink)' }}>{value.toFixed(4)}</div>
      {target && (
        <div style={{ fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: '0.14em', color: pass ? 'var(--notumor)' : 'var(--magenta)', marginTop: 2 }}>
          {pass ? '✓' : '✗'} TARGET {target.toFixed(3)}
        </div>
      )}
    </div>
  )
}
