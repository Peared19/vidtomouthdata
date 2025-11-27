# Quick Start

## 1. Prerequisites
- **Python:** 3.8.10 (system interpreter works—no virtualenv needed, but recommended).
- **FFmpeg:** required for `pydub`. Install it and make sure `ffmpeg` is on your `PATH`.
- **GPU drivers:** Optional. PyTorch will fall back to CPU if a CUDA-enabled GPU is not available.

## 2. Install Python dependencies
```powershell
pip install -r requirements.txt
```

`requirements.txt` already pins the exact versions we currently import (`torch`, `numpy`, `pandas`, `mediapipe`, `opencv-python`, `requests`, `tqdm`, `gTTS`, `pydub`).

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

> Tip: if you modify `requirements.txt`, re-run `pip install -r requirements.txt` and restart the server.
