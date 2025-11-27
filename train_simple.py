"""
BEGINNER-FRIENDLY: Training Loop

This script trains the blend shape prediction model.

What it does:
  1. Load data using dataloaders
  2. Create model
  3. Train for multiple epochs:
     - Process batches
     - Calculate loss
     - Update model weights (backpropagation)
     - Validate on validation set
  4. Save best model

Simplified features:
  - No TensorBoard (simpler than advanced logging)
  - No gradient clipping (less complex)
  - No learning rate scheduling (just constant learning rate)
  - Clear comments explaining every step
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import time



# Import simplified versions
from model_simple import create_simple_model
from dataloader_simple import create_simple_dataloaders


# ============================================================================
# PART 1: CONFIGURATION
# ============================================================================

class SimpleConfig:
    """Configuration for training - all settings in one place."""
    
    # File paths
    data_file = 'gridcorpus/mouth_data_context.csv'
    vocab_file = 'vocabulary.json'
    checkpoint_dir = 'checkpoints_simple'
    
    # Model hyperparameters
    vocab_size = 55
    embedding_dim = 128
    
    # Training hyperparameters
    batch_size = 32
    learning_rate = 0.001  # 1e-3
    num_epochs = 50
    
    # Device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self):
        """Create checkpoint directory if it doesn't exist."""
        Path(self.checkpoint_dir).mkdir(exist_ok=True)


# ============================================================================
# PART 2: TRAINER CLASS
# ============================================================================

class SimpleTrainer:
    """
    Trainer class that handles the training loop.
    
    Think of this as the "teacher" that:
      1. Shows samples to the model
      2. Checks if predictions are correct
      3. Tells the model where it went wrong
      4. Model learns and improves
    """
    
    def __init__(self, config: SimpleConfig):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object with all settings
        """
        self.config = config
        self.device = config.device
        
        print(f"Using device: {self.device}")
        
        # Create model
        print("\nCreating model...")
        self.model = create_simple_model(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim
        )
        self.model.to(self.device)
        print(f"✓ Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Create optimizer (tells model how to update weights)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        print(f"✓ Using Adam optimizer with learning rate {config.learning_rate}")
        
        # Loss function (measures how wrong predictions are)
        self.loss_fn = nn.MSELoss()  # Mean Squared Error
        print(f"✓ Using MSELoss for training")
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        self.patience = 10  # Stop if no improvement for 10 epochs
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        An epoch = one complete pass through all training data.
        
        Steps:
          1. Load batch of samples
          2. Feed through model (forward pass)
          3. Calculate loss (how wrong are we?)
          4. Calculate gradients (how to improve?)
          5. Update weights (backpropagation)
          6. Repeat for all batches
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Average loss for the epoch
        """
        
        self.model.train()  # Set model to training mode
        total_loss = 0.0
        num_batches = 0
        
        # Loop through all batches
        for batch_num, (batch_input, batch_target) in enumerate(train_loader):
            # Move data to device (GPU or CPU)
            batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
            batch_target = batch_target.to(self.device)
            
            # ──────────────────────────────────────────────────────────────
            # FORWARD PASS: Feed input through model
            # ──────────────────────────────────────────────────────────────
            # Model learns to predict blend shapes from input
            
            predictions = self.model(batch_input)
            # predictions shape: (batch_size, 52) with values [0, 1]
            
            # ──────────────────────────────────────────────────────────────
            # CALCULATE LOSS: How wrong are the predictions?
            # ──────────────────────────────────────────────────────────────
            # Loss = average squared difference between prediction and ground truth
            
            loss = self.loss_fn(predictions, batch_target)
            # loss is a single number representing total error
            
            # ──────────────────────────────────────────────────────────────
            # BACKWARD PASS: Calculate gradients
            # ──────────────────────────────────────────────────────────────
            # Gradients tell us how to change weights to reduce loss
            
            self.optimizer.zero_grad()  # Clear old gradients
            loss.backward()  # Calculate new gradients
            
            # ──────────────────────────────────────────────────────────────
            # UPDATE WEIGHTS: Learn from this batch
            # ──────────────────────────────────────────────────────────────
            
            self.optimizer.step()  # Update model weights
            
            # ──────────────────────────────────────────────────────────────
            # Track loss for reporting
            # ──────────────────────────────────────────────────────────────
            
            total_loss += loss.item()  # Add to running total
            num_batches += 1
            
            # Print progress every 10 batches
            if (batch_num + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_num+1}: Loss = {avg_loss:.4f}")
        
        # Calculate average loss for entire epoch
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate on validation set (no learning, just evaluation).
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        
        self.model.eval()  # Set model to evaluation mode (disables dropout)
        total_loss = 0.0
        num_batches = 0
        
        # Don't calculate gradients during validation (faster, saves memory)
        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                # Move data to device
                batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
                batch_target = batch_target.to(self.device)
                
                # Forward pass (no gradients)
                predictions = self.model(batch_input)
                
                # Calculate loss
                loss = self.loss_fn(predictions, batch_target)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Average loss
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader):
        """
        Complete training loop for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*80}")
            
            # ──────────────────────────────────────────────────────────────
            # TRAINING
            # ──────────────────────────────────────────────────────────────
            
            train_loss = self.train_epoch(train_loader)
            print(f"\nTrain Loss: {train_loss:.4f}")
            
            # ──────────────────────────────────────────────────────────────
            # VALIDATION
            # ──────────────────────────────────────────────────────────────
            
            val_loss = self.validate(val_loader)
            print(f"Val Loss:   {val_loss:.4f}")
            
            # ──────────────────────────────────────────────────────────────
            # CHECK FOR IMPROVEMENT
            # ──────────────────────────────────────────────────────────────
            
            if val_loss < self.best_val_loss:
                print(f"✓ Validation improved! ({self.best_val_loss:.4f} → {val_loss:.4f})")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                print(f"⚠ No improvement ({self.patience_counter}/{self.patience})")
                
                # Early stopping: Stop if no improvement for many epochs
                if self.patience_counter >= self.patience:
                    print(f"\n✗ Early stopping: No improvement for {self.patience} epochs")
                    break
            
            # Save checkpoint every epoch
            self.save_checkpoint(epoch, is_best=False)
        
        # Training finished
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✓ TRAINING FINISHED in {elapsed/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        A checkpoint contains:
          - Model weights
          - Optimizer state
          - Epoch number
          - Loss values
        
        This allows us to resume training or use the model later.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'vocab_size': self.config.vocab_size,
                'embedding_dim': self.config.embedding_dim,
            }
        }
        
        if is_best:
            path = f"{self.config.checkpoint_dir}/best_model.pt"
            print(f"  → Saved best model to {path}")
        else:
            path = f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a saved checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        epoch = checkpoint['epoch']
        print(f"✓ Loaded checkpoint from epoch {epoch+1}")
        print(f"  Best val loss at checkpoint: {self.best_val_loss:.4f}")


# ============================================================================
# PART 3: MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for training."""
    
    # Create configuration
    config = SimpleConfig()
    
    print("=" * 80)
    print("SIMPLE BLEND SHAPE TRAINING")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    try:
        train_loader, val_loader, test_loader = create_simple_dataloaders(
            data_file=config.data_file,
            vocab_file=config.vocab_file,
            batch_size=config.batch_size
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Generated vocabulary.json (run vocabulary_generator.py)")
        print("  2. Generated mouth_data_context.csv (run dataset_processor_multithread.py)")
        return
    
    # Create trainer
    print("\n" + "=" * 80)
    print("CREATING TRAINER")
    print("=" * 80)
    
    trainer = SimpleTrainer(config)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Test the model (optional)
    print("\n" + "=" * 80)
    print("TESTING ON TEST SET")
    print("=" * 80)
    
    test_loss = trainer.validate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best model saved to: {config.checkpoint_dir}/best_model.pt")
    print(f"You can now use this model for inference!")


# ============================================================================
# PART 4: COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Simple command-line argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description="Train blend shape model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data', default='mouth_data_context.csv', help='Data file')
    parser.add_argument('--vocab', default='vocabulary.json', help='Vocabulary file')
    
    args = parser.parse_args()
    
    # Update config with arguments
    config = SimpleConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.data_file = args.data
    config.vocab_file = args.vocab
    
    main()
