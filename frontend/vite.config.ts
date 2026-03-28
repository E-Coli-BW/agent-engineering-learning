import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': __dirname + 'src',
    },
  },
  server: {
    port: 3000,
    proxy: {
      // RAG API Server (port 8000)
      '/api/rag': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/rag/, ''),
      },
      // A2A Expert Agent (port 5001)
      '/api/a2a': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/a2a/, ''),
      },
      // ReAct Agent (port 5002)
      '/api/react': {
        target: 'http://localhost:5002',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/react/, ''),
      },
    },
  },
})
