import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    d_k = q.size(-1) # batch_size, num_heads, seq_len, d_k
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_in:int):
        super().__init__()
        assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_in = d_in
        self.d_k = d_in // num_heads
        
        self.W_q = nn.Linear(d_in, d_in)
        self.W_k = nn.Linear(d_in, d_in)
        self.W_v = nn.Linear(d_in, d_in)
        self.W_o = nn.Linear(d_in, d_in)
        
    def split_heads(self, x: torch.Tensor):
        x = x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        return x
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # q: batchsize, seqlen, d_in
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k)) # batch size, num_heads, seq_len, d_k
        v = self.split_heads(self.W_v(v))        
        
        output, attn = scaled_dot_product_attention(q, k ,v ,mask=mask)
        
        # concat
        B, h, l, d_k = output.size()
        output = output.transpose(1, 2)
        output = output.view(B, l, -1)
        
        return output, attn