import { useEffect, useMemo, useState } from 'react'
import { z } from 'zod'
import { Block, CodeBlock as CHCodeBlock, parseRoot } from 'codehike/blocks'
import { highlight, type HighlightedCode } from 'codehike/code'
import { Selection, Selectable, SelectionProvider } from 'codehike/utils/selection'
import { Section } from './Section'
import Content from '../content/pipeline.mdx'
import { SmoothPre } from './SmoothPre'

const Schema = Block.extend({
  intro: Block,
  steps: z.array(Block.extend({ code: CHCodeBlock })),
  outro: Block,
})

export function PipelineCodehike() {
  const { intro, steps, outro } = useMemo(() => parseRoot(Content as any, Schema), [])
  const [highlighted, setHighlighted] = useState<HighlightedCode[] | null>(null)

  useEffect(() => {
    let alive = true
    Promise.all(steps.map((s) => highlight(s.code, 'github-dark'))).then((hs) => {
      if (alive) setHighlighted(hs)
    })
    return () => { alive = false }
  }, [steps])

  return (
    <Section
      id="06b Pipeline"
      kicker="05.5 · The pipeline, scroll-by-scroll"
      title={<>{intro.title}</>}
      lede={<>{intro.children}</>}
    >
      <SelectionProvider className="ch-scrolly">
        <div className="ch-steps">
          {steps.map((step, i) => (
            <Selectable
              key={i}
              index={i}
              selectOn={['click', 'scroll']}
              className="ch-step"
            >
              <div className="ch-step-badge">
                <span>STEP &middot; {String(i + 1).padStart(2, '0')} / {String(steps.length).padStart(2, '0')}</span>
              </div>
              <h3 className="serif">{step.title}</h3>
              <div className="ch-step-body">{step.children}</div>
            </Selectable>
          ))}
        </div>
        <div className="ch-sticky">
          <div className="ch-sticky-inner">
            {highlighted ? (
              <Selection
                from={highlighted.map((h) => (
                  // eslint-disable-next-line react/jsx-key
                  <SmoothPre code={h} className="ch-pre" />
                ))}
              />
            ) : (
              <pre className="ch-pre" />
            )}
          </div>
        </div>
      </SelectionProvider>

      <div className="ch-outro reveal">
        {outro.children}
      </div>
    </Section>
  )
}
