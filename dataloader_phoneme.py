import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from typing import Tuple, List
import numpy as np

class PhonemeBlendShapeDataset(Dataset):
    """
    Dataset class to load mouth animation data with lazy loading.
    Adapted for PHONEME-based data.
    """
    
    def __init__(self, 
                 data_file: str,
                 vocab_file: str,
                 segment_size: int = 50000):
        """
        Initialize the dataset with lazy loading.
        """
        print(f"Loading vocabulary from {vocab_file}...")
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
            # Changed from word_to_id to phoneme_to_id
            self.phoneme_to_id = vocab_data['phoneme_to_id']
            # Vocab size is just the length of the map
            self.vocab_size = len(self.phoneme_to_id)
        print(f"  [OK] Loaded {self.vocab_size} phonemes")
        
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
        
        self.data_file = data_file
        self.segment_size = segment_size
        
        print(f"Pre-loading CSV into memory (this may take a minute)...")
        self.df = None
        self._load_full_csv()
    
    def _load_full_csv(self):
        """Load the entire CSV file in chunks and concatenate."""
        chunks = []
        chunk_count = 0
        # Using 'c' engine for speed
        for chunk in pd.read_csv(self.data_file, delimiter=';', chunksize=50000, engine='c'):
            chunks.append(chunk)
            chunk_count += 1
            print(f"  Loaded chunk {chunk_count}...")
        
        self.df = pd.concat(chunks, ignore_index=True)
        print(f"  [OK] CSV loaded: {len(self.df)} rows")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor]:
        """Get one sample: returns (input_dict, blend_shapes_tensor)."""
        row = self.df.iloc[idx]
        
        # Get phoneme IDs (using phoneme columns)
        prev_ph_id = self.phoneme_to_id.get(row['prev_phoneme'], 0)
        curr_ph_id = self.phoneme_to_id.get(row['curr_phoneme'], 0)
        next_ph_id = self.phoneme_to_id.get(row['next_phoneme'], 0)
        
        # Get continuous features
        frame_pos = float(row['frame_pos'])
        # Phoneme duration in frames / 30.0 (assuming 30fps max normalization or just as float)
        # Note: In dataset_processor, we calculated duration in frames.
        # Normalizing by 30.0 is a reasonable heuristic if most phonemes are short.
        ph_duration = float(row['phoneme_duration_frames']) / 30.0
        ph_duration = max(0.0, min(1.0, ph_duration))
        
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
            'prev_phoneme': torch.tensor(prev_ph_id, dtype=torch.long),
            'curr_phoneme': torch.tensor(curr_ph_id, dtype=torch.long),
            'next_phoneme': torch.tensor(next_ph_id, dtype=torch.long),
            'frame_pos': torch.tensor(frame_pos, dtype=torch.float32),
            'phoneme_duration': torch.tensor(ph_duration, dtype=torch.float32)
        }
        
        return input_dict, blend_shapes_tensor

def phoneme_collate_fn(batch: List[Tuple[dict, torch.Tensor]]) -> Tuple[dict, torch.Tensor]:
    """Combine multiple samples into a batch."""
    inputs = [sample[0] for sample in batch]
    outputs = [sample[1] for sample in batch]
    
    batched_input = {
        'prev_phoneme': torch.stack([inp['prev_phoneme'] for inp in inputs]),
        'curr_phoneme': torch.stack([inp['curr_phoneme'] for inp in inputs]),
        'next_phoneme': torch.stack([inp['next_phoneme'] for inp in inputs]),
        'frame_pos': torch.stack([inp['frame_pos'] for inp in inputs]),
        'phoneme_duration': torch.stack([inp['phoneme_duration'] for inp in inputs])
    }
    
    batched_output = torch.stack(outputs)
    
    return batched_input, batched_output

def create_phoneme_dataloaders(
    data_file: str,
    vocab_file: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    print("\n" + "=" * 80)
    print("Creating Phoneme Dataloaders")
    print("=" * 80)
    
    dataset = PhonemeBlendShapeDataset(data_file, vocab_file)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples:,}")
    
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = total_samples - train_size - val_size
    
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=phoneme_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=phoneme_collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=phoneme_collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader
