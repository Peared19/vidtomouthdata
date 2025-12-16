import torch
import torch.nn as nn
from typing import Dict

class SimplePhonemeEmbedding(nn.Module):
    """
    Converts phoneme IDs into vectors.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
    
    def forward(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(phoneme_ids)

class PhonemeBlendShapeModel(nn.Module):
    """
    Neural network for predicting blend shapes from PHONEMES.
    
    Inputs:
    - prev_phoneme_id
    - curr_phoneme_id
    - next_phoneme_id
    - frame_pos
    - phoneme_duration
    """
    
    def __init__(self, vocab_size: int = 45, embedding_dim: int = 128):
        super().__init__()
        
        # Phoneme embeddings
        self.phoneme_embedding = SimplePhonemeEmbedding(vocab_size, embedding_dim)
        
        # Input size: 3 embeddings + 2 continuous values
        input_size = embedding_dim * 3 + 2
        
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 52)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process a batch of data.
        Expects keys: 'prev_phoneme', 'curr_phoneme', 'next_phoneme', 'frame_pos', 'phoneme_duration'
        """
        
        prev_embedding = self.phoneme_embedding(batch['prev_phoneme'])
        curr_embedding = self.phoneme_embedding(batch['curr_phoneme'])
        next_embedding = self.phoneme_embedding(batch['next_phoneme'])
        
        frame_pos = batch['frame_pos'].unsqueeze(1)
        ph_duration = batch['phoneme_duration'].unsqueeze(1)
        
        all_features = torch.cat([
            prev_embedding,
            curr_embedding,
            next_embedding,
            frame_pos,
            ph_duration
        ], dim=1)
        
        x = self.layer1(all_features)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        
        x = self.layer3(x)
        blend_shapes = self.sigmoid(x)
        
        return blend_shapes

def create_phoneme_model(vocab_size: int = 45, embedding_dim: int = 128):
    model = PhonemeBlendShapeModel(vocab_size, embedding_dim)
    return model
