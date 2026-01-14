import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import MultiHeadAttention
from .ffn import FeedForwardNetwork

class EncodeBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, drop_out: float = 0.2):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, drop_out)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, x: torch.Tensor, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        return self.norm2(x + self.dropout(ff_output))
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            EncodeBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        return x