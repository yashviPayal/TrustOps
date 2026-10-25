import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/model-info': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/monitoring': 'http://127.0.0.1:8000',
      '/replay': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
      '/transactions': 'http://127.0.0.1:8000',
      '/scam': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
    },
  },
});
