import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import MultiHeadAttention
from .ffn import FeedForwardNetwork

class EncodeBlock(nn.Module):
    def __init__(self, d_in: int, num_heads: int, d_ff: int, drop_out: float = 0.2):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_in)
        self.ffn = FeedForwardNetwork(d_in, d_ff, drop_out)
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, x: torch.Tensor, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        return self.norm2(x + self.dropout(ff_output))