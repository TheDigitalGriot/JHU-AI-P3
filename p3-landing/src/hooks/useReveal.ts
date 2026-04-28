import { useEffect, RefObject } from 'react'

export function useReveal(ref: RefObject<HTMLElement>) {
  useEffect(() => {
    const root = ref.current
    if (!root) return
    const els = root.querySelectorAll<HTMLElement>('.reveal')
    const checkNow = () => {
      els.forEach((el) => {
        const r = el.getBoundingClientRect()
        if (r.top < window.innerHeight * 0.9 && r.bottom > 0) el.classList.add('in')
      })
    }
    checkNow()
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add('in') }),
      { threshold: 0.05, rootMargin: '0px 0px -10% 0px' },
    )
    els.forEach((el) => io.observe(el))
    const onScroll = () => checkNow()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => { io.disconnect(); window.removeEventListener('scroll', onScroll) }
  }, [ref])
}
