import { useEffect, useState } from 'react'
import { Pre, highlight, type HighlightedCode } from 'codehike/code'

interface Props {
  label?: string
  code: string
  lang?: string
}

export function CodeBlock({ label, code, lang = 'python' }: Props) {
  const [hl, setHl] = useState<HighlightedCode | null>(null)
  useEffect(() => {
    let alive = true
    highlight({ value: code, lang, meta: '' }, 'github-dark').then((h) => { if (alive) setHl(h) })
    return () => { alive = false }
  }, [code, lang])
  return (
    <div className="code-wrap" data-label={label || lang.toUpperCase()}>
      {hl ? <Pre code={hl} className="code" /> : <pre className="code"><code>{code}</code></pre>}
    </div>
  )
}
