import { useEffect, useRef, useState } from 'react'
import Three from '../three-skull/core/Three.js'

export function Hero() {
  const stageRef = useRef<HTMLDivElement>(null)
  const threeRef = useRef<any>(null)
  const [supported, setSupported] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    const hasGPU = typeof navigator !== 'undefined' && 'gpu' in navigator
    if (!hasGPU) {
      setSupported(false)
      document.body.classList.remove('loading')
      return
    }
    setSupported(true)
    ;(async () => {
      try {
        const t: any = new Three(stageRef.current)
        threeRef.current = t
        await t.run()
        await t.scene?.ready
        if (cancelled) { t.dispose?.(); return }
        document.body.classList.remove('loading')
      } catch (err) {
        console.error('[Hero] WebGPU init failed', err)
        setSupported(false)
        document.body.classList.remove('loading')
      }
    })()
    return () => {
      cancelled = true
      threeRef.current?.dispose?.()
      threeRef.current = null
    }
  }, [])

  return (
    <div className="hero" data-screen-label="01 Hero">
      <div className="hero-stage" ref={stageRef}>
        <div className="crosshair tl" />
        <div className="crosshair tr" />
        <div className="crosshair bl" />
        <div className="crosshair br" />
        <div className="hero-overlay">
          <div className="hero-mark">
            <span className="pulse" />
            <span>JHU&nbsp;·&nbsp;NEURAL&nbsp;NETWORKS&nbsp;FOR&nbsp;COMPUTER&nbsp;VISION</span>
            <span style={{ marginLeft: 'auto' }}>P3 ·  v6.0.2  ·  2026</span>
          </div>
          <div className="hero-title">
            <h1 className="serif">
              Reading<br />the brain<br />in <em>2,895</em><br />scans.
            </h1>
            <div className="hero-byline">
              <span>BY</span>
              <em>Gavin Bennett</em>
            </div>
          </div>
          <div className="hero-bottom">
            <div>BRAIN&nbsp;TUMOR&nbsp;CLASSIFICATION  ·  GLIOMA / MENINGIOMA / PITUITARY / NO-TUMOR</div>
            <div className="scroll-hint">
              <span>SCROLL</span><span className="line" />
            </div>
            <div>RUBRIC VGG-16 + FF  ·  0.8452 test  &nbsp;|&nbsp;  PYTORCH&nbsp;ENSEMBLE  ·  0.9700 test  &nbsp;|&nbsp;  THUNDER&nbsp;A100&nbsp;·&nbsp;W&amp;B</div>
          </div>
        </div>
        {supported === false && (
          <div style={{
            position: 'absolute', inset: 0, display: 'grid', placeItems: 'center',
            color: 'var(--dim)', fontFamily: 'var(--mono)', fontSize: 12, letterSpacing: '0.18em',
            textAlign: 'center', padding: '4rem',
          }}>
            <div>
              WEBGPU UNAVAILABLE — HERO SCENE DISABLED.<br />
              <span style={{ color: 'var(--cyan)' }}>OPEN IN CHROME 121+ OR EDGE 121+ WITH WEBGPU ENABLED.</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
