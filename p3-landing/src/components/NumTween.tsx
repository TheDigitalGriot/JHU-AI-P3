import { useEffect, useRef, useState } from 'react'

interface Props {
  to: number
  decimals?: number
  suffix?: string
  prefix?: string
  dur?: number
}

export function NumTween({ to, decimals = 0, suffix = '', prefix = '', dur = 1400 }: Props) {
  const [v, setV] = useState(0)
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    let raf = 0, started = false, t0 = 0
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (e.isIntersecting && !started) {
          started = true
          t0 = performance.now()
          const tick = (now: number) => {
            const t = Math.min(1, (now - t0) / dur)
            const eased = 1 - Math.pow(1 - t, 3)
            setV(to * eased)
            if (t < 1) raf = requestAnimationFrame(tick)
          }
          raf = requestAnimationFrame(tick)
        }
      })
    }, { threshold: 0.5 })
    if (ref.current) io.observe(ref.current)
    return () => { io.disconnect(); cancelAnimationFrame(raf) }
  }, [to, dur])
  return <span ref={ref}>{prefix}{v.toFixed(decimals)}{suffix}</span>
}
