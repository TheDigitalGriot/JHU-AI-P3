import { ReactNode, useRef } from 'react'
import { useReveal } from '../hooks/useReveal'

interface Props {
  id: string
  kicker?: ReactNode
  title?: ReactNode
  lede?: ReactNode
  children?: ReactNode
  className?: string
}

export function Section({ id, kicker, title, lede, children, className = '' }: Props) {
  const ref = useRef<HTMLElement>(null)
  useReveal(ref)
  return (
    <section id={id} ref={ref} className={`beat ${className}`} data-screen-label={id}>
      <div className="container">
        {kicker && <div className="kicker reveal">{kicker}</div>}
        {title && <h2 className="reveal serif" style={{ marginTop: '0.4rem' }}>{title}</h2>}
        {lede && <p className="lede reveal" style={{ marginTop: '1.2rem' }}>{lede}</p>}
        {children}
      </div>
    </section>
  )
}
