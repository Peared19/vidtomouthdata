# 🎬 Mouth Animation Synthesis with PyTorch

A complete pipeline for synthesizing realistic mouth animations from text input using deep learning.

## 📋 Project Overview

```
Text Input
    ↓
[1] Vocabulary Lookup (word → ID)
    ↓
[2] Neural Network (Transformer)
    - Input: 3 words (prev, curr, next) + frame position + duration
    - Output: 52 blend shapes (ARKit compatible)
    ↓
[3] Blend Shape Sequence (animation)
    ↓
[4] TTS Audio Generation + Synchronization
    ↓
Output: Talking head video with synchronized speech
```

## 🏗️ Architecture

### Data Pipeline
- **Input:** GRID corpus videos (10 speakers, 400+ videos each)
- **Processing:** MediaPipe Face Landmarker (52 blend shapes per frame)
- **Storage:** mouth_data_context.csv (400K+ frames)
- **Context:** prev_word + curr_word + next_word + frame_pos + duration

### Neural Network
- **Model:** Transformer-based regression
- **Input:** Word embeddings (128D) + continuous features (2D)
- **Output:** 52 blend shapes (0-1 normalized)
- **Parameters:** ~500K trainable parameters

## 📁 Files Structure

```
word_tomoutmap/
├── gridcorpus/
│   ├── video/                    # Downloaded GRID corpus videos
│   ├── align/                    # Alignment files (word timing)
│   └── mouth_data_context.csv    # Processed dataset (400K+ frames)
│
├── dataloader.py                 # PyTorch Dataset + DataLoader
├── model.py                      # Transformer/LSTM models
├── train.py                      # Training pipeline
├── vocabulary_generator.py       # Generate vocabulary.json
├── inference.py                  # [TODO] Inference pipeline
├── tts_integration.py           # [TODO] TTS + audio sync
│
├── vocabulary.json               # Word ↔ ID mapping
├── checkpoints/
│   └── best_model.pt            # Best trained model
├── logs/
│   └── run_*/                   # TensorBoard logs
│
└── docs/
    ├── DATA_ANALYSIS.md
    ├── PYTORCH_TRAINING_GUIDE.md
    └── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install PyTorch (CUDA enabled recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install pandas numpy tqdm tensorboard opencv-python mediapipe
```

### Step 1: Verify Dataset
```bash
# Check if mouth_data_context.csv exists
ls -lh gridcorpus/mouth_data_context.csv

# Expected: ~500 MB, 400K+ lines
```

### Step 2: Generate Vocabulary
```bash
python vocabulary_generator.py \
    --csv gridcorpus/mouth_data_context.csv \
    --output vocabulary.json \
    --validate
```

Output:
```
Loaded 400000 samples
Found 53 unique words
Vocabulary saved to vocabulary.json
✓ All 53 words in CSV are in vocabulary
```

### Step 3: Train Model
```bash
# Basic training (uses defaults)
python train.py

# Or with custom parameters
python train.py \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --model-type transformer \
    --embedding-dim 128 \
    --num-heads 8 \
    --num-layers 4
```

### Step 4: Monitor Training
```bash
# In another terminal, start TensorBoard
tensorboard --logdir logs

# Open browser to http://localhost:6006
```

## 📊 Model Architecture Details

### Transformer Model
```
Input Batch (64 samples)
    ↓
Word Embeddings:
  - prev_word: (64, 128)
  - curr_word: (64, 128)
  - next_word: (64, 128)
    ↓
Continuous Feature Projection:
  - frame_pos + duration → (64, 64)
    ↓
Concatenate: (64, 64 + 384) = (64, 448)
    ↓
Linear Projection: (64, 448) → (64, 128)
    ↓
Transformer Encoder (4 layers, 8 heads):
  - Multi-head attention
  - Position-wise FFN
    ↓
Output Head:
  - Dense: 128 → 512
  - GELU activation
  - Dense: 512 → 256
  - Dense: 256 → 52
  - Sigmoid: 52 → [0, 1]
    ↓
Output: (64, 52) blend shapes
```

### Loss Function
```
Total Loss = MSE(predicted, target) + λ × Smoothness(predicted)

Where:
- MSE: Main regression objective
- Smoothness: Regularization (λ = 0.1)
- Encourages smooth, natural animations
```

## 📈 Training Details

### Hyperparameters (Recommended)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 50 | Early stopping at patience=5 |
| Batch Size | 64 | Good GPU utilization |
| Learning Rate | 1e-3 | Adam optimizer |
| Scheduler | Cosine Annealing | With warmup |
| Gradient Clip | 1.0 | Prevent exploding gradients |
| Dropout | 0.1 | Regularization |
| Train/Val/Test | 80/10/10 | Random split |

### Expected Results
```
Epoch 1:   Train Loss: 0.0845 | Val Loss: 0.0782
Epoch 10:  Train Loss: 0.0125 | Val Loss: 0.0098
Epoch 20:  Train Loss: 0.0045 | Val Loss: 0.0052
Epoch 30:  Train Loss: 0.0028 | Val Loss: 0.0031
Epoch 40:  Train Loss: 0.0020 | Val Loss: 0.0024
Epoch 50:  Train Loss: 0.0016 | Val Loss: 0.0020  ← Best model
```

### Training Time
- **RTX 3080/4080:** ~2-3 hours
- **RTX 3070/3090:** ~4-6 hours
- **RTX 3060/2080:** ~8-12 hours
- **CPU:** ~24-48 hours (not recommended)

## 🎯 Key Features

### Data Loading
- ✅ Efficient parallel data loading (num_workers=4)
- ✅ Automatic vocabulary mapping
- ✅ Normalization and preprocessing
- ✅ Custom collate function for batching

### Model Architecture
- ✅ Transformer encoder (8 heads, 4 layers)
- ✅ Word embeddings + continuous features
- ✅ Proper initialization and regularization
- ✅ Output constrained to [0, 1]

### Training Pipeline
- ✅ Adam optimizer with learning rate scheduling
- ✅ Gradient clipping and early stopping
- ✅ Checkpoint saving (best + periodic)
- ✅ TensorBoard logging
- ✅ Comprehensive validation metrics

## 📚 Usage Examples

### Train and Save Model
```bash
python train.py --epochs 50 --batch-size 64
# Best model saved to: checkpoints/best_model.pt
```

### Load and Evaluate
```python
import torch
from model import create_model
import json

# Load vocabulary
with open('vocabulary.json') as f:
    vocab = json.load(f)

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
model = create_model(
    model_type='transformer',
    vocab_size=checkpoint['vocab_size']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare batch
batch = {
    'prev_word': torch.tensor([0]),              # <START>
    'curr_word': torch.tensor([vocab['word_to_id']['bin']]),
    'next_word': torch.tensor([vocab['word_to_id']['blue']]),
    'frame_pos': torch.tensor([0.5]),
    'word_duration': torch.tensor([0.5])
}

# Predict blend shapes
with torch.no_grad():
    blend_shapes = model(batch)
    print(blend_shapes.shape)  # (1, 52)
    print(blend_shapes)         # 52 values in [0, 1]
```

## 🔧 Configuration

### Modify Training Defaults
Edit `train.py` → `TrainingConfig` class:

```python
class TrainingConfig:
    def __init__(self):
        self.epochs = 50              # Change here
        self.batch_size = 64          # Or here
        self.learning_rate = 1e-3     # Or here
        # ... etc
```

### GPU/Device Selection
```python
# Automatically detects GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Or force CPU
device = torch.device('cpu')
```

## 🐛 Troubleshooting

### Out of Memory (OOM) Error
```bash
# Reduce batch size
python train.py --batch-size 32

# Reduce model size
python train.py --embedding-dim 64 --num-layers 2
```

### Training Loss Not Decreasing
```bash
# Reduce learning rate
python train.py --lr 5e-4

# Use smaller model
python train.py --num-layers 2
```

### Validation Loss Much Higher Than Train Loss
```python
# In model.py, increase dropout:
dropout: 0.3  # Instead of 0.1
```

## 📊 Monitoring

### TensorBoard
```bash
tensorboard --logdir logs
# Then open http://localhost:6006
```

Metrics tracked:
- `train/loss_step` - Loss at each training step
- `train/loss_epoch` - Average training loss per epoch
- `val/loss` - Validation loss per epoch
- `val/mae` - Mean absolute error per epoch
- `lr` - Learning rate schedule

## 🎬 Next Steps

### Phase 1: Training (Current)
- ✅ Data preparation
- ✅ Model architecture
- ✅ Training pipeline
- 🔄 Train model on full dataset

### Phase 2: Inference
- 📝 Build inference.py
- 📝 Create animation rendering
- 📝 Integrate with Three.js viewer

### Phase 3: Audio Synthesis
- 📝 Integrate TTS (gTTS, XTTS, etc.)
- 📝 Implement audio time-stretching
- 📝 Synchronize audio + animation

### Phase 4: Final System
- 📝 Web interface
- 📝 Real-time inference
- 📝 Video export

## 📖 References

- **Transformer:** [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- **PyTorch:** https://pytorch.org/
- **GRID Corpus:** https://www.researchgate.net/publication/228629248_GRID_A_High_Quality_and_Large_Lexicon_Audio-Visual_Corpus
- **ARKit Blend Shapes:** https://developer.apple.com/documentation/arkit/arfacearchor/blendshapelocation

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

Speech Animation Synthesis Project

---

**Questions or Issues?** Check the documentation files:
- `DATA_ANALYSIS.md` - Dataset analysis
- `PYTORCH_TRAINING_GUIDE.md` - Detailed training guide
