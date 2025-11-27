"""
Evaluation Script for Blend Shape Model

This script evaluates the trained model on the test dataset to calculate
the Mean Squared Error (MSE) loss.

It helps determine how well the model generalizes to unseen data.
"""

import torch
import torch.nn as nn
import os
from model_simple import create_simple_model
from dataloader_simple import create_simple_dataloaders

def evaluate_model():
    print("=" * 80)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 80)

    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Data
    # We need to load the data to get the test set
    # NOTE: This might take a moment as it loads the CSV
    print("\nLoading dataset (this may take a minute)...")
    try:
        # Use a smaller batch size for evaluation to be safe with memory
        train_loader, val_loader, test_loader = create_simple_dataloaders(
            data_file='gridcorpus/mouth_data_context.csv',
            vocab_file='vocabulary.json',
            batch_size=16  # Smaller batch size to be safe
        )
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. Load Model
    print("\nLoading model...")
    vocab_size = 55  # Must match training
    model = create_simple_model(vocab_size=vocab_size)
    
    model_path = 'checkpoints_simple/best_model.pt'
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found, trying blend_shape_model.pt")
        model_path = 'blend_shape_model.pt'

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval() # Set to evaluation mode
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Evaluation Loop
    print(f"\nEvaluating on {len(test_loader)} batches...")
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(): # No gradients needed for evaluation
        for i, (inputs, targets) in enumerate(test_loader):
            # Move batch to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} batches...")

    # 5. Results
    avg_loss = total_loss / num_batches
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS")
    print("=" * 80)
    print(f"Test Set MSE Loss: {avg_loss:.6f}")
    print("-" * 80)
    
    # Interpretation
    print("Interpretation:")
    if avg_loss < 0.01:
        print("  Excellent! The model is predicting very accurately.")
    elif avg_loss < 0.02:
        print("  Good. The model has learned the general patterns.")
    else:
        print("  High Loss. The model might be untrained or not learning well.")
        print("  (Note: An untrained model usually has loss around 0.08 - 0.10)")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_model()
