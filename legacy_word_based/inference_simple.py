"""
BEGINNER-FRIENDLY: Inference Script for Blend Shape Prediction

This script demonstrates how to use a trained model to predict blend shapes
given text input (words) and frame information.

What it does:
  1. Load a trained model and vocabulary
  2. Convert text to word IDs
  3. Make predictions on blend shapes
  4. Display results

How to use:
  python inference_simple.py
"""

import torch
import json
import numpy as np
import os
from model_simple import create_simple_model


def load_model(model_path: str, vocab_size: int = 55, device: str = 'cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint (.pt file)
        vocab_size: Size of vocabulary (should match training vocab)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model on the specified device
    """
    print(f"Loading model from {model_path}...")
    
    # Create model architecture
    model = create_simple_model(vocab_size=vocab_size)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("  Loading from checkpoint dictionary...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  Loading state dict directly...")
        model.load_state_dict(checkpoint)
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()  # Important! Set to evaluation mode
    
    print("  [OK] Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def load_vocabulary(vocab_path: str):
    """
    Load vocabulary mapping.
    
    Args:
        vocab_path: Path to vocabulary JSON file
    
    Returns:
        word_to_id: Dictionary mapping words to IDs
    """
    print(f"Loading vocabulary from {vocab_path}...")
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        word_to_id = vocab_data['word_to_id']
    
    print(f"  [OK] Loaded {len(word_to_id)} words")
    
    return word_to_id


def predict_blend_shapes(model: torch.nn.Module, 
                        prev_word_id: int,
                        curr_word_id: int,
                        next_word_id: int,
                        frame_pos: float,
                        word_duration: float,
                        device: str = 'cuda') -> np.ndarray:
    """
    Make a prediction for blend shapes.
    
    Args:
        model: Loaded trained model
        prev_word_id: ID of previous word (0-54)
        curr_word_id: ID of current word (0-54)
        next_word_id: ID of next word (0-54)
        frame_pos: Frame position within word (0.0-1.0)
        word_duration: Word duration normalized (0.0-1.0)
        device: Device model is on
    
    Returns:
        blend_shapes: NumPy array of 52 blend shape values (0.0-1.0)
    """
    
    # Create input batch (add batch dimension)
    # Usually we process batches, but here we process a single sample
    batch_input = {
        'prev_word': torch.tensor([prev_word_id], dtype=torch.long, device=device),
        'curr_word': torch.tensor([curr_word_id], dtype=torch.long, device=device),
        'next_word': torch.tensor([next_word_id], dtype=torch.long, device=device),
        'frame_pos': torch.tensor([frame_pos], dtype=torch.float32, device=device),
        'word_duration': torch.tensor([word_duration], dtype=torch.float32, device=device)
    }
    
    # Make prediction (no gradient computation needed for inference)
    with torch.no_grad():
        output = model(batch_input)  # shape: (1, 52)
    
    # Extract first (and only) sample and convert to numpy
    blend_shapes = output[0].cpu().numpy()  # shape: (52,)
    
    return blend_shapes


def main():
    """
    Main inference script.
    Demonstrates how to predict blend shapes for given words.
    """
    
    # ========================================================================
    # SETUP: Load model and vocabulary
    # ========================================================================
    
    print("=" * 80)
    print("SIMPLE BLEND SHAPE INFERENCE")
    print("=" * 80 + "\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    model_path = 'checkpoints_simple/best_model.pt'
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found, trying blend_shape_model.pt")
        model_path = 'blend_shape_model.pt'
        
    model = load_model(model_path, vocab_size=55, device=device)
    print()
    
    # Load vocabulary
    word_to_id = load_vocabulary('vocabulary.json')
    print()
    
    # Define blend shape names (52 total)
    blend_shape_names = [
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
    
    # ========================================================================
    # EXAMPLE 1: Simple prediction
    # ========================================================================
    
    print("=" * 80)
    print("EXAMPLE 1: Predict blend shapes for a word sequence")
    print("=" * 80)
    print()
    
    # Example: Words at start of sentence "Blue"
    prev_word = '<START>'
    curr_word = 'blue'
    next_word = 'green'
    frame_pos = 0.5  # Middle of the word
    word_duration = 0.8  # Normalized duration
    
    # Convert words to IDs
    prev_word_id = word_to_id.get(prev_word, 0)
    curr_word_id = word_to_id.get(curr_word, 0)
    next_word_id = word_to_id.get(next_word, 0)
    
    print(f"Word sequence: {prev_word} -> {curr_word} -> {next_word}")
    print(f"Word IDs:      {prev_word_id}   ->  {curr_word_id}  ->  {next_word_id}")
    print(f"Frame position: {frame_pos:.2f} (0.0=start, 1.0=end)")
    print(f"Word duration: {word_duration:.2f}")
    print()
    
    # Make prediction
    blend_shapes = predict_blend_shapes(
        model, prev_word_id, curr_word_id, next_word_id,
        frame_pos, word_duration, device
    )
    
    print("Predicted blend shapes (showing most active):")
    print()
    
    # Sort by activation strength and show top 10
    sorted_indices = np.argsort(blend_shapes)[::-1]  # Sort descending
    for rank, idx in enumerate(sorted_indices[:10], 1):
        value = blend_shapes[idx]
        name = blend_shape_names[idx]
        bar = '█' * int(value * 30)  # Visual bar
        print(f"  {rank:2d}. {name:25s}: {value:.4f}  {bar}")
    
    print()
    print(f"Average activation: {blend_shapes.mean():.4f}")
    print(f"Min/Max: {blend_shapes.min():.4f} / {blend_shapes.max():.4f}")
    
    # ========================================================================
    # EXAMPLE 2: Compare different frame positions
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: How blend shapes change across the word")
    print("=" * 80)
    print()
    
    print(f"Predicting for word: '{curr_word}'")
    print(f"Showing top 5 blend shapes at different frame positions:")
    print()
    
    for frame_pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blend_shapes = predict_blend_shapes(
            model, prev_word_id, curr_word_id, next_word_id,
            frame_pos, word_duration, device
        )
        
        # Get top 3
        sorted_indices = np.argsort(blend_shapes)[::-1]
        top_3 = [(blend_shape_names[i], blend_shapes[i]) for i in sorted_indices[:3]]
        
        top_3_str = ", ".join([f"{name}:{val:.3f}" for name, val in top_3])
        print(f"  Frame {frame_pos:.2f}: {top_3_str}")
    
    # ========================================================================
    # EXAMPLE 3: Interactive prediction
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Interactive prediction")
    print("=" * 80)
    print()
    print("Available words:", list(word_to_id.keys())[:20], "...")
    print()
    
    # Try a few combinations
    test_cases = [
        ('<START>', 'blue', 'green', 0.5, 0.8),
        ('<START>', 'green', 'place', 0.3, 0.6),
        ('blue', 'green', 'place', 0.7, 0.9),
    ]
    
    for prev_w, curr_w, next_w, fp, wd in test_cases:
        prev_id = word_to_id.get(prev_w, 0)
        curr_id = word_to_id.get(curr_w, 0)
        next_id = word_to_id.get(next_w, 0)
        
        blend_shapes = predict_blend_shapes(
            model, prev_id, curr_id, next_id, fp, wd, device
        )
        
        max_idx = np.argmax(blend_shapes)
        max_name = blend_shape_names[max_idx]
        max_val = blend_shapes[max_idx]
        
        print(f"{prev_w:8s} -> {curr_w:8s} -> {next_w:8s}  " +
              f"[pos:{fp:.2f}, dur:{wd:.2f}] -> Top: {max_name}={max_val:.4f}")
    
    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
