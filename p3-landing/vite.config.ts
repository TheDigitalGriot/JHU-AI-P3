import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import mdx from '@mdx-js/rollup'
import { remarkCodeHike, recmaCodeHike, type CodeHikeConfig } from 'codehike/mdx'

const chConfig: CodeHikeConfig = {
  components: { code: 'Code' },
  syntaxHighlighting: { theme: 'github-dark' },
}

export default defineConfig(({ command }) => ({
  plugins: [
    {
      enforce: 'pre',
      ...mdx({
        remarkPlugins: [[remarkCodeHike, chConfig]],
        recmaPlugins: [[recmaCodeHike, chConfig]],
      }),
    },
    react(),
  ],
  base: command === 'build' ? '/JHU-AI-P3/' : '/',
  server: { host: true, port: 5173 },
  optimizeDeps: {
    exclude: ['three/webgpu', 'three/tsl'],
  },
}))
