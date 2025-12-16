import torch
import torch.nn as nn
import json
from pathlib import Path
import time

# Import PHONEME versions
from model_phoneme import create_phoneme_model
from dataloader_phoneme import create_phoneme_dataloaders

class PhonemeConfig:
    """Configuration for phoneme-based training."""
    
    # File paths
    data_file = 'gridcorpus/mouth_data_phoneme.csv'
    vocab_file = 'phonemes.json'
    checkpoint_dir = 'checkpoints_phoneme'
    
    # Model hyperparameters
    # Will be updated from vocab file
    vocab_size = 45 
    embedding_dim = 128
    
    # Training hyperparameters
    batch_size = 64  # Increased batch size slightly
    learning_rate = 0.001
    num_epochs = 50
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self):
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        # Load vocab size dynamically
        try:
            with open(self.vocab_file, 'r') as f:
                data = json.load(f)
                self.vocab_size = len(data['phoneme_to_id'])
                print(f"Config: Set vocab_size to {self.vocab_size}")
        except FileNotFoundError:
            print("Config: Vocab file not found yet, using default size.")

class PhonemeTrainer:
    def __init__(self, config: PhonemeConfig):
        self.config = config
        self.device = config.device
        
        print(f"Using device: {self.device}")
        
        print("\nCreating model...")
        self.model = create_phoneme_model(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim
        )
        self.model.to(self.device)
        print(f"✓ Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.loss_fn = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_num, (batch_input, batch_target) in enumerate(train_loader):
            batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
            batch_target = batch_target.to(self.device)
            
            predictions = self.model(batch_input)
            loss = self.loss_fn(predictions, batch_target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_num + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_num+1}: Loss = {avg_loss:.4f}")
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
                batch_target = batch_target.to(self.device)
                
                predictions = self.model(batch_input)
                loss = self.loss_fn(predictions, batch_target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader):
        print("\n" + "=" * 80)
        print("STARTING PHONEME TRAINING")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*80}")
            
            train_loss = self.train_epoch(train_loader)
            print(f"\nTrain Loss: {train_loss:.4f}")
            
            val_loss = self.validate(val_loader)
            print(f"Val Loss:   {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                print(f"✓ Validation improved! ({self.best_val_loss:.4f} → {val_loss:.4f})")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                print(f"⚠ No improvement ({self.patience_counter}/{self.patience})")
                
                if self.patience_counter >= self.patience:
                    print(f"\n✗ Early stopping: No improvement for {self.patience} epochs")
                    break
            
            self.save_checkpoint(epoch, is_best=False)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✓ TRAINING FINISHED in {elapsed/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
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

def main():
    config = PhonemeConfig()
    
    print("=" * 80)
    print("PHONEME BLEND SHAPE TRAINING")
    print("=" * 80)
    
    try:
        train_loader, val_loader, test_loader = create_phoneme_dataloaders(
            data_file=config.data_file,
            vocab_file=config.vocab_file,
            batch_size=config.batch_size
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return
    
    trainer = PhonemeTrainer(config)
    trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 80)
    print("TESTING ON TEST SET")
    print("=" * 80)
    
    test_loss = trainer.validate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
