import torch
from torch import nn

import math

def sinusoidal_time_embedding(t, dim):
    """
    t: (batch,)  or (batch, 1)
    dim: embedding dimension (must be even)
    return: (batch, dim)
    """
    half_dim = dim // 2
    device = t.device

    # compute frequencies
    exponent = torch.arange(half_dim, device=device) / half_dim
    freqs = torch.exp(-math.log(10000) * exponent)

    t = t.view(-1, 1)

    angles = t * freqs.view(1, -1)

    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb

def sinusoidal_position_encoding(seq_len, dim):
    """
    seq_len: length of sequence (e.g., 200)
    dim: embedding dimension (must be even)
    return: (seq_len, dim)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    position = torch.arange(seq_len, device=device).float().unsqueeze(1)
    half_dim = dim // 2

    exponent = torch.arange(half_dim, device=device) / half_dim
    div_term = torch.exp(-math.log(10000) * exponent)

    angles = position * div_term.unsqueeze(0)

    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return pe

class ConditionEmbedding(nn.Module):
    def __init__(self, pep_dim=128, time_dim=32, charge_dim=32):
        super().__init__()
        # self.charge_embedding = nn.Linear(1, charge_dim)
        self.charge_dim = charge_dim
        self.time_dim = time_dim
        
        self.pep_embedding = nn.Embedding(22, pep_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pep_dim, 
            nhead=8, 
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    def forward(self, seq: torch.Tensor, charge: torch.Tensor, time: torch.Tensor):
        pep_emb = self.pep_embedding(seq) + sinusoidal_position_encoding(seq.size(1), self.pep_embedding.embedding_dim).unsqueeze(0)
        pep_c = self.transformer(pep_emb).mean(dim=1)
        charge_emb = sinusoidal_time_embedding(charge, self.charge_dim)
        time_emb = sinusoidal_time_embedding(time, self.time_dim)
        # print(pep_c.shape, charge_emb.shape, time_emb.shape)
        return torch.cat([pep_c, charge_emb, time_emb], dim=-1)
        