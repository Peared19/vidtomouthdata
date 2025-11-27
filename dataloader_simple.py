"""
BEGINNER-FRIENDLY: Data Loading for Blend Shape Training

This loads mouth animation data from CSV and prepares it for training.

What it does:
  1. Read mouth_data_context.csv file
  2. Convert words to word IDs using vocabulary
  3. Split data into training/validation/testing sets
  4. Create batches for neural network training

Simplified features:
  - No parallel loading (num_workers=0)
  - Simple list-based operations
  - Easy to understand and debug
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from typing import Tuple, List
import numpy as np


# ============================================================================
# PART 1: DATASET CLASS
# ============================================================================

class SimpleBlendShapeDataset(Dataset):
    """
    Dataset class to load mouth animation data with lazy loading.
    
    Loads data from CSV in segments to avoid memory issues with large files.
    """
    
    def __init__(self, 
                 data_file: str,
                 vocab_file: str,
                 segment_size: int = 50000):
        """
        Initialize the dataset with lazy loading.
        
        Args:
            data_file: Path to CSV file
            vocab_file: Path to vocabulary JSON
            segment_size: Number of rows to keep in memory at once
        """
        print(f"Loading vocabulary from {vocab_file}...")
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
            self.word_to_id = vocab_data['word_to_id']
            self.vocab_size = vocab_data['vocab_size']
        print(f"  [OK] Loaded {self.vocab_size} words")
        
        # Blend shape column names (52 total)
        self.blend_shape_cols = [
            '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp',
            'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
            'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
            'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
            'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
            'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward',
            'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
            'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
            'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
            'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
            'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
            'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
            'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'
        ]
        
        # Store file path
        self.data_file = data_file
        self.segment_size = segment_size
        self.total_rows = 742079  # Known from vocabulary_generator
        
        # Cache the full dataframe (we'll load it all once for simplicity)
        print(f"Pre-loading CSV into memory (this may take a minute)...")
        self.df = None
        self._load_full_csv()
    
    def _load_full_csv(self):
        """Load the entire CSV file in chunks and concatenate."""
        chunks = []
        chunk_count = 0
        for chunk in pd.read_csv(self.data_file, delimiter=';', chunksize=50000, engine='c'):
            chunks.append(chunk)
            chunk_count += 1
            print(f"  Loaded chunk {chunk_count}...")
        
        self.df = pd.concat(chunks, ignore_index=True)
        print(f"  [OK] CSV loaded: {len(self.df)} rows")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.df)
    
    def _load_full_csv(self):
        """Load the entire CSV file in chunks and concatenate."""
        chunks = []
        chunk_count = 0
        for chunk in pd.read_csv(self.data_file, delimiter=';', chunksize=50000, engine='c'):
            chunks.append(chunk)
            chunk_count += 1
            print(f"  Loaded chunk {chunk_count}...")
        
        self.df = pd.concat(chunks, ignore_index=True)
        print(f"  [OK] CSV loaded: {len(self.df)} rows")
    
    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor]:
        """Get one sample: returns (input_dict, blend_shapes_tensor)."""
        row = self.df.iloc[idx]
        
        # Get word IDs
        prev_word_id = self.word_to_id.get(row['prev_word'], 0)
        curr_word_id = self.word_to_id.get(row['curr_word'], 0)
        next_word_id = self.word_to_id.get(row['next_word'], 0)
        
        # Get continuous features
        frame_pos = float(row['frame_pos'])
        word_duration = float(row['word_duration_frames']) / 30.0
        word_duration = max(0.0, min(1.0, word_duration))
        
        # Parse blend shapes from JSON
        import json as json_module
        blend_shapes_dict = json_module.loads(row['blend_shapes'])
        
        blend_shapes = []
        for col in self.blend_shape_cols:
            value = float(blend_shapes_dict.get(col, 0.0))
            value = max(0.0, min(1.0, value))
            blend_shapes.append(value)
        
        blend_shapes_tensor = torch.tensor(blend_shapes, dtype=torch.float32)
        
        input_dict = {
            'prev_word': torch.tensor(prev_word_id, dtype=torch.long),
            'curr_word': torch.tensor(curr_word_id, dtype=torch.long),
            'next_word': torch.tensor(next_word_id, dtype=torch.long),
            'frame_pos': torch.tensor(frame_pos, dtype=torch.float32),
            'word_duration': torch.tensor(word_duration, dtype=torch.float32)
        }
        
        return input_dict, blend_shapes_tensor


# ============================================================================
# PART 2: BATCH COLLATION FUNCTION
# ============================================================================

def simple_collate_fn(batch: List[Tuple[dict, torch.Tensor]]) -> Tuple[dict, torch.Tensor]:
    """
    Combine multiple samples into a batch.
    
    When DataLoader gets samples from dataset, it calls this function
    to combine them into a batch suitable for neural network training.
    
    Args:
        batch: List of (input_dict, output_tensor) tuples
               Example: 64 samples for batch_size=64
    
    Returns:
        Tuple of (batched_input_dict, batched_output_tensor)
        
    Example:
    ────────
    Input: [
        ({'prev_word': 1, 'curr_word': 5, ..., 'frame_pos': 0.3}, tensor([0.01, 0.45, ...])),
        ({'prev_word': 5, 'curr_word': 2, ..., 'frame_pos': 0.7}, tensor([0.02, 0.40, ...])),
        ...
    ]
    
    Output: (
        {
            'prev_word': tensor([1, 5, ...]),      shape (64,)
            'curr_word': tensor([5, 2, ...]),      shape (64,)
            'next_word': tensor([12, 8, ...]),     shape (64,)
            'frame_pos': tensor([0.3, 0.7, ...]),  shape (64,)
            'word_duration': tensor([0.4, 0.6, ...])  shape (64,)
        },
        tensor([...])  shape (64, 52)
    )
    """
    
    # Separate inputs and outputs
    inputs = [sample[0] for sample in batch]
    outputs = [sample[1] for sample in batch]
    
    # Stack words into single tensor for each position
    batched_input = {
        'prev_word': torch.stack([inp['prev_word'] for inp in inputs]),
        'curr_word': torch.stack([inp['curr_word'] for inp in inputs]),
        'next_word': torch.stack([inp['next_word'] for inp in inputs]),
        'frame_pos': torch.stack([inp['frame_pos'] for inp in inputs]),
        'word_duration': torch.stack([inp['word_duration'] for inp in inputs])
    }
    
    # Stack all blend shapes into single tensor
    batched_output = torch.stack(outputs)  # shape (batch_size, 52)
    
    return batched_input, batched_output


# ============================================================================
# PART 3: CREATE DATALOADERS
# ============================================================================

def create_simple_dataloaders(
    data_file: str,
    vocab_file: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    This function:
      1. Loads dataset
      2. Splits into train/val/test (80/10/10)
      3. Creates DataLoader for each split
    
    Args:
        data_file: Path to CSV data file
        vocab_file: Path to vocabulary JSON
        batch_size: Number of samples per batch (default 32)
        train_split: Fraction for training (default 0.8 = 80%)
        val_split: Fraction for validation (default 0.1 = 10%)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Example:
    ────────
    train_loader, val_loader, test_loader = create_simple_dataloaders(
        'mouth_data_context.csv',
        'vocabulary.json',
        batch_size=64
    )
    """
    
    print("\n" + "=" * 80)
    print("Creating Dataloaders")
    print("=" * 80)
    
    # Step 1: Load full dataset
    dataset = SimpleBlendShapeDataset(data_file, vocab_file)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples:,}")
    
    # Step 2: Calculate split indices
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = total_samples - train_size - val_size
    
    print(f"Train: {train_size:,} ({train_split*100:.0f}%)")
    print(f"Val:   {val_size:,} ({val_split*100:.0f}%)")
    print(f"Test:  {test_size:,} ({(1-train_split-val_split)*100:.0f}%)")
    
    # Step 3: Split dataset using random indices
    # Generate random indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)  # Randomize order
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"\n[OK] Datasets created")
    
    # Step 4: Create DataLoaders
    # DataLoader handles batching, shuffling, etc.
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data each epoch
        collate_fn=simple_collate_fn,
        num_workers=0          # No parallel loading (simpler)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # Don't shuffle validation data
        collate_fn=simple_collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # Don't shuffle test data
        collate_fn=simple_collate_fn,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)} (each has {batch_size} samples)")
    print(f"Val batches: {len(val_loader)} (each has {batch_size} samples)")
    print(f"Test batches: {len(test_loader)} (each has {batch_size} samples)")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# PART 4: TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SIMPLE DATALOADER - TEST")
    print("=" * 80)
    
    # Note: This test assumes you have the data files
    # If you don't, it will fail (which is okay for now)
    
    try:
        # Try to create dataloaders
        train_loader, val_loader, test_loader = create_simple_dataloaders(
            data_file='mouth_data_context.csv',
            vocab_file='vocabulary.json',
            batch_size=32
        )
        
        # Test by loading one batch
        print("\n" + "=" * 80)
        print("Loading one batch from training data...")
        print("=" * 80)
        
        for batch_input, batch_output in train_loader:
            print(f"\nBatch Input:")
            print(f"  prev_word shape: {batch_input['prev_word'].shape}")
            print(f"  curr_word shape: {batch_input['curr_word'].shape}")
            print(f"  next_word shape: {batch_input['next_word'].shape}")
            print(f"  frame_pos shape: {batch_input['frame_pos'].shape}")
            print(f"  word_duration shape: {batch_input['word_duration'].shape}")
            
            print(f"\nBatch Output (blend shapes):")
            print(f"  Shape: {batch_output.shape}")
            print(f"  Values (first sample): {batch_output[0]}")
            
            print(f"\n[OK] Dataloader working correctly!")
            break
        
    except FileNotFoundError as e:
        print(f"\n⚠ Test skipped (data files not found): {e}")
        print("This is expected if you haven't generated the data yet.")
        print("Run vocabulary_generator.py and dataset_processor_multithread.py first.")
