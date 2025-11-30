# Quick Start

## 1. Prerequisites
- **Python:** 3.8.10 (system interpreter works—no virtualenv needed, but recommended).
- **Node.js + npm:** required for the Three.js frontend assets (`node --version` / `npm --version`).
- **FFmpeg:** `pydub` needs `ffmpeg`/`ffprobe` on your `PATH`. Download it from https://ffmpeg.org/download.html and add the `bin/` directory to your system `PATH`, then restart PowerShell.
- **GPU drivers:** Optional. PyTorch will fall back to CPU if CUDA isn’t available.

## 2. Install dependencies
### Python dependencies
```powershell
pip install -r requirements.txt
```
`requirements.txt` already pins the exact versions we currently import (`torch`, `numpy`, `pandas`, `mediapipe`, `opencv-python`, `requests`, `tqdm`, `gTTS`, `pydub`).

### Frontend assets
```powershell
npm install
```
This downloads `three` (and `vite`, if you use it) into `node_modules/`, matching the setup the visualizer expects.

## 3. Prepare the dataset (optional)
If you need to re-generate GRID data:
1. Run `python grid_downloader.py` to download GRID speakers (audio/video/align files).
2. Run `python dataset_processor_multithread.py` to produce `gridcorpus/mouth_data_context.csv` and split the data. Make sure `gridcorpus/` contains the expected structure.
3. Generate the vocabulary.
```powershell
python vocabulary_generator.py --csv gridcorpus/mouth_data_context.csv
```












## 4. Training (optional)
```powershell
python train_simple.py --epochs 3 --batch-size 32
```
This will save checkpoints to `checkpoints_simple/`. The best model is copied to `checkpoints_simple/best_model.pt`.

## 5. Evaluation (optional)
```powershell
python evaluate_simple.py
```
Uses the saved model and the same dataset split to report MSE on the test set.

## 6. Run the visualization + TTS pipeline
```powershell
python server.py
```
- Opens an HTTP server on port `8000` serving `visualizer.html`.
- The `/animate` endpoint handles text input, generates per-word audio with `gTTS`, and produces blend shape frames.
- Visit `http://localhost:8000/visualizer.html` to type sentences and see / hear the result.

> Tip: if you modify `requirements.txt` or `package.json`, re-run the matching install command and restart the server.
