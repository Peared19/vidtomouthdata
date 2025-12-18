# Modular Visualizer (New)

This folder contains a **new** visualizer implementation that keeps your existing `visualizer.html` intact.

## How to run (using your existing Python server)

1. Start the phoneme server:
   - `python server_phoneme.py`
2. Open the new visualizer:
   - http://localhost:8000/visualizer/index.html

## Dev workflow (optional, with Vite)

Because the code uses ES modules (native `import`/`export`), you can also run a frontend dev server:

- From the repo root:
  - `npx vite --config visualizer/vite.config.js`
- Then open the URL Vite prints (usually http://localhost:5173).

The visualizer still talks to your Python server at `http://localhost:8000/animate`, so keep `server_phoneme.py` running.

## What changed vs the old visualizer

- Split JS into small ES modules under `visualizer/js/`
- Precomputes morph-target indices once (less per-frame overhead)
- Sync uses `audio.currentTime * fps` consistently, with clamping
- Shows status messages instead of `alert()`

No server/backend changes were made.
