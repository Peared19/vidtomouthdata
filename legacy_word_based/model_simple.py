"""
BEGINNER-FRIENDLY: Blend Shape Prediction Model

This model takes text and predicts mouth animation (52 blend shapes).

Architecture (SIMPLE VERSION):
  Input: 3 word IDs (prev, current, next) + 2 numbers (frame position, word duration)
  ↓
  Convert words to vectors (embeddings)
  ↓
  Feed through neural network layers
  ↓
  Output: 52 blend shapes (numbers between 0-1)

Key Differences from Advanced Version:
- No TransformerEncoder (simpler than Transformers)
- Just basic Dense layers (Linear layers)
- Easier to understand and modify
- Slower but more readable
"""

import torch
import torch.nn as nn
from typing import Dict


# ============================================================================
# PART 1: WORD EMBEDDINGS
# ============================================================================

class SimpleWordEmbedding(nn.Module):
    """
    Converts word IDs (like 0, 1, 2, ..., 54) into vectors (128 numbers each).
    
    Think of it like a dictionary:
      Word 0 (START)  → [0.02, -0.01, 0.03, ..., -0.02]  (128 numbers)
      Word 1 (the)    → [-0.01, 0.04, -0.02, ..., 0.01]  (128 numbers)
      Word 2 (at)     → [0.03, -0.02, 0.01, ..., 0.02]   (128 numbers)
      ... etc
    
    These vectors are learned during training.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Args:
            vocab_size: Number of words in vocabulary (55 = 53 words + <START> + <END>)
            embedding_dim: Size of each word vector (128)
        """
        super().__init__()
        
        # Create embedding table: vocab_size × embedding_dim matrix
        # Each row is the embedding for one word
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        # This helps training start from a good point
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
    
    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert word IDs to embeddings.
        
        Input shape:  (batch_size,)  = (64,)
                      Example: [1, 5, 12, ...]  (64 word IDs)
        
        Output shape: (batch_size, embedding_dim) = (64, 128)
                      Example: [[0.02, -0.01, ..., 0.04],    ← embedding for word ID 1
                                [-0.01, 0.04, ..., 0.03],     ← embedding for word ID 5
                                [0.03, -0.02, ..., 0.01],     ← embedding for word ID 12
                                ...]
        
        Args:
            word_ids: LongTensor of shape (batch_size,)
        
        Returns:
            embeddings: FloatTensor of shape (batch_size, embedding_dim)
        """
        return self.embedding(word_ids)


# ============================================================================
# PART 2: SIMPLE NEURAL NETWORK MODEL
# ============================================================================

class SimpleBlendShapeModel(nn.Module):
    """
    A SIMPLE neural network for predicting blend shapes.
    
    Architecture Diagram:
    ═══════════════════════════════════════════════════════════════
    
    INPUTS:
    ──────
    prev_word_id (0-54)         ──→  Word Embedding (128 values)
    curr_word_id (0-54)         ──→  Word Embedding (128 values)
    next_word_id (0-54)         ──→  Word Embedding (128 values)
    frame_pos (0.0-1.0)         ──→  No embedding needed (already a number)
    word_duration (0.0-1.0)     ──→  No embedding needed (already a number)
    
    CONCATENATE ALL (stack them together):
    ──────────────────────────────────────
    128 + 128 + 128 + 1 + 1 = 386 values
    
    NEURAL NETWORK LAYERS:
    ──────────────────────
    Layer 1: 386 values → Dense layer → 256 values → ReLU activation
    Layer 2: 256 values → Dense layer → 128 values → ReLU activation
    Layer 3: 128 values → Dense layer → 52 values (blend shapes)
    
    OUTPUT ACTIVATION:
    ──────────────────
    Apply Sigmoid to squeeze values to range [0, 1]
    (because blend shapes are 0-1: 0=not active, 1=fully active)
    
    OUTPUT:
    ──────
    52 blend shape values (all between 0 and 1)
    """
    
    def __init__(self, vocab_size: int = 55, embedding_dim: int = 128):
        """
        Initialize the model.
        
        Args:
            vocab_size: Number of words (default 55: 53 words + START + END)
            embedding_dim: Size of each word embedding (default 128)
        """
        super().__init__()
        
        # Step 1: Create word embeddings
        # Converts word IDs to 128-dimensional vectors
        self.word_embedding = SimpleWordEmbedding(vocab_size, embedding_dim)
        
        # Step 2: Create neural network layers
        # Input will be: 3 embeddings (128 each) + 2 continuous values
        # Total input size = 128*3 + 2 = 386
        input_size = embedding_dim * 3 + 2
        
        # Layer 1: 386 → 256
        # This layer learns to combine all input features
        self.layer1 = nn.Linear(input_size, 256)
        
        # Layer 2: 256 → 128
        # This layer learns patterns in the 256-dimensional representation
        self.layer2 = nn.Linear(256, 128)
        
        # Layer 3: 128 → 52 (blend shapes)
        # This layer outputs the final blend shapes
        self.layer3 = nn.Linear(128, 52)
        
        # Step 3: Activation function to squeeze outputs to [0, 1]
        self.sigmoid = nn.Sigmoid()
        
        # Step 4: Activation for hidden layers (ReLU = max(0, x))
        self.relu = nn.ReLU()
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process a batch of data and return blend shapes.
        
        Args:
            batch: Dictionary containing:
                - 'prev_word': shape (batch_size,)  e.g., [1, 5, 12, ...]
                - 'curr_word': shape (batch_size,)  e.g., [5, 2, 7, ...]
                - 'next_word': shape (batch_size,)  e.g., [3, 8, 1, ...]
                - 'frame_pos': shape (batch_size,)  e.g., [0.3, 0.7, 0.1, ...]
                - 'word_duration': shape (batch_size,)  e.g., [0.4, 0.6, 0.2, ...]
        
        Returns:
            blend_shapes: FloatTensor of shape (batch_size, 52)
                         All values between 0 and 1
        
        Example:
        ────────
        For batch_size=64:
          Input: 5 values per sample
          Output: 52 blend shapes per sample
        """
        
        # ──────────────────────────────────────────────────────────────
        # STEP 1: EMBED THE WORDS
        # ──────────────────────────────────────────────────────────────
        # Convert word IDs to 128-dimensional vectors
        
        prev_embedding = self.word_embedding(batch['prev_word'])
        # Shape: (batch_size, 128) = (64, 128)
        # Example: [[0.02, -0.01, 0.03, ..., -0.02],  ← prev word for sample 1
        #           [-0.01, 0.04, -0.02, ..., 0.01],  ← prev word for sample 2
        #           ...]
        
        curr_embedding = self.word_embedding(batch['curr_word'])
        # Shape: (batch_size, 128) = (64, 128)
        
        next_embedding = self.word_embedding(batch['next_word'])
        # Shape: (batch_size, 128) = (64, 128)
        
        # ──────────────────────────────────────────────────────────────
        # STEP 2: GET CONTINUOUS FEATURES (frame_pos, duration)
        # ──────────────────────────────────────────────────────────────
        # These are already numbers, no embedding needed
        # But we need to reshape them for concatenation
        
        frame_pos = batch['frame_pos'].unsqueeze(1)
        # .unsqueeze(1) adds a dimension: (64,) → (64, 1)
        # Needed because we concatenate horizontally
        
        word_duration = batch['word_duration'].unsqueeze(1)
        # Shape: (64, 1)
        
        # ──────────────────────────────────────────────────────────────
        # STEP 3: CONCATENATE ALL FEATURES
        # ──────────────────────────────────────────────────────────────
        # Stack all features side-by-side
        
        all_features = torch.cat([
            prev_embedding,      # (64, 128)
            curr_embedding,      # (64, 128)
            next_embedding,      # (64, 128)
            frame_pos,           # (64, 1)
            word_duration        # (64, 1)
        ], dim=1)               # Concatenate along dimension 1 (columns)
        # Result shape: (64, 128+128+128+1+1) = (64, 386)
        
        # ──────────────────────────────────────────────────────────────
        # STEP 4: PASS THROUGH NEURAL NETWORK LAYERS
        # ──────────────────────────────────────────────────────────────
        
        # Layer 1: 386 → 256
        x = self.layer1(all_features)  # (64, 386) → (64, 256)
        # This layer learns to combine features
        
        # Apply ReLU activation: negative values → 0, positive values → same
        x = self.relu(x)  # (64, 256) → (64, 256)
        # ReLU helps the network learn non-linear patterns
        
        # Layer 2: 256 → 128
        x = self.layer2(x)  # (64, 256) → (64, 128)
        
        # Apply ReLU activation again
        x = self.relu(x)  # (64, 128) → (64, 128)
        
        # Layer 3: 128 → 52 (final blend shapes)
        x = self.layer3(x)  # (64, 128) → (64, 52)
        
        # ──────────────────────────────────────────────────────────────
        # STEP 5: APPLY SIGMOID TO CONSTRAIN TO [0, 1]
        # ──────────────────────────────────────────────────────────────
        # Sigmoid formula: sigmoid(x) = 1 / (1 + e^-x)
        # This converts any value to range [0, 1]
        
        blend_shapes = self.sigmoid(x)  # (64, 52) with values in [0, 1]
        
        # ──────────────────────────────────────────────────────────────
        # Return final result
        # ──────────────────────────────────────────────────────────────
        return blend_shapes


# ============================================================================
# PART 3: HELPER FUNCTION TO CREATE MODEL
# ============================================================================

def create_simple_model(vocab_size: int = 55, embedding_dim: int = 128):
    """
    Create a simple model.
    
    This is just a helper function to make model creation easier.
    
    Args:
        vocab_size: Number of words (default 55)
        embedding_dim: Embedding dimension (default 128)
    
    Returns:
        SimpleBlendShapeModel instance
    
    Usage:
    ──────
    model = create_simple_model(vocab_size=55, embedding_dim=128)
    """
    model = SimpleBlendShapeModel(vocab_size, embedding_dim)
    return model


# ============================================================================
# PART 4: TESTING AND EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SIMPLE BLEND SHAPE MODEL - TEST")
    print("=" * 80)
    
    # Create model
    model = create_simple_model(vocab_size=55, embedding_dim=128)
    print("\n✓ Model created successfully!")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total trainable parameters: {total_params:,}")
    
    # Create dummy batch (random test data)
    print("\n" + "=" * 80)
    print("Creating test batch...")
    print("=" * 80)
    
    batch_size = 32
    vocab_size = 55
    
    batch = {
        'prev_word': torch.randint(0, vocab_size, (batch_size,)),
        'curr_word': torch.randint(0, vocab_size, (batch_size,)),
        'next_word': torch.randint(0, vocab_size, (batch_size,)),
        'frame_pos': torch.rand(batch_size),
        'word_duration': torch.rand(batch_size)
    }
    
    print(f"Batch size: {batch_size}")
    print(f"  prev_word shape: {batch['prev_word'].shape}  values: {batch['prev_word'][:5]}")
    print(f"  curr_word shape: {batch['curr_word'].shape}  values: {batch['curr_word'][:5]}")
    print(f"  next_word shape: {batch['next_word'].shape}  values: {batch['next_word'][:5]}")
    print(f"  frame_pos shape: {batch['frame_pos'].shape}  values: {batch['frame_pos'][:5]}")
    print(f"  word_duration shape: {batch['word_duration'].shape}  values: {batch['word_duration'][:5]}")
    
    # Forward pass (process batch through model)
    print("\n" + "=" * 80)
    print("Running forward pass...")
    print("=" * 80)
    
    output = model(batch)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"  Expected: (batch_size, 52) = ({batch_size}, 52)")
    print(f"✓ Output value range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Expected: [0.0000, 1.0000] (due to sigmoid)")
    
    # Show example output
    print(f"\nExample output for first sample (52 blend shapes):")
    print(f"  {output[0][:10]}  ... (showing first 10 of 52)")
    print(f"\nAll 52 blend shapes for first sample:")
    for i, val in enumerate(output[0].detach().numpy()):
        print(f"  Blend shape {i:2d}: {val:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ TEST PASSED! Model is working correctly.")
    print("=" * 80)
