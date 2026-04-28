import { useEffect, useState } from 'react'
import { Hero } from './components/Hero'
import { FourClasses, Dataset, EDA } from './components/Parts1'
import { WhyHard, Preprocess, Leakage } from './components/Parts2'
import { ModelANN, ModelOpt, ModelVGG16 } from './components/Parts3'
import { ModelVGG16FF, ModelVGG16Aug, Leaderboard, ConfusionMatrix, BayesRisk, Insights, Outro } from './components/Parts4'
import { ClusterRecall } from './components/ClusterViz'
import { PipelineCodehike } from './components/PipelineCodehike'
import { ParticleScrolltale } from './components/ParticleScrolltale'

const SECTIONS = [
  'Hero', 'Particle Brain', 'Classes', 'Dataset', 'EDA', 'Why Hard', 'Preprocess', 'Pipeline',
  'Leakage', 'M1 ANN', 'M2 Opt', 'M3 VGG', 'M4 VGG+FF', 'M5 Aug',
  'Board', 'Clusters', 'Conf', 'Bayes', 'Insights', 'End',
]

export default function App() {
  const [progress, setProgress] = useState(0)
  const [active, setActive] = useState(0)

  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement
      const p = (h.scrollTop || document.body.scrollTop) / ((h.scrollHeight - h.clientHeight) || 1)
      setProgress(p)
      const sections = document.querySelectorAll('section.beat, .hero, .particle-scrolltale')
      let a = 0
      sections.forEach((s, i) => {
        const r = s.getBoundingClientRect()
        if (r.top < window.innerHeight * 0.4) a = i
      })
      setActive(a)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Safety: drop loading class after 6s in case WebGPU fails silently.
  useEffect(() => {
    const id = setTimeout(() => document.body.classList.remove('loading'), 6000)
    return () => clearTimeout(id)
  }, [])

  return (
    <>
      <div className="progress"><div style={{ width: progress * 100 + '%' }} /></div>
      <div className="toc">
        {SECTIONS.map((s, i) => (
          <button key={i} className={i === active ? 'active' : ''} title={s} onClick={() => {
            const el = document.querySelectorAll('section.beat, .hero, .particle-scrolltale')[i]
            if (el) (el as HTMLElement).scrollIntoView({ behavior: 'smooth', block: 'start' })
          }} />
        ))}
      </div>

      <main>
        <Hero />
        <ParticleScrolltale />
        <FourClasses />
        <Dataset />
        <EDA />
        <WhyHard />
        <Preprocess />
        <PipelineCodehike />
        <Leakage />
        <ModelANN />
        <ModelOpt />
        <ModelVGG16 />
        <ModelVGG16FF />
        <ModelVGG16Aug />
        <Leaderboard />
        <ClusterRecall />
        <ConfusionMatrix />
        <BayesRisk />
        <Insights />
        <Outro />
      </main>

      <footer>
        <span>JHU · NN-CV · P3 · v4.3.0 · 2026</span>
        <span>BRAIN TUMOR DETECTION · GLIOMA / MENINGIOMA / NOTUMOR / PITUITARY</span>
        <span>THUNDER COMPUTE · A100-SXM4-80GB · W&amp;B</span>
        <span><a href="https://github.com/TheDigitalGriot/JHU-AI-P3" target="_blank" rel="noreferrer">SOURCE →</a></span>
      </footer>
    </>
  )
}
