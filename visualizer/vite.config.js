import { defineConfig } from 'vite';
import path from 'node:path';

// Dev-only: serves this folder as the Vite root.
// Run from repo root:
//   npx vite --config visualizer/vite.config.js
export default defineConfig({
  root: path.resolve(__dirname),
  server: {
    port: 5173,
    strictPort: true
  }
});
