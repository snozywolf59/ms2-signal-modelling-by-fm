import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_in: int, d_ff:int, drop_out: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_ff)
        self.drop_out = nn.Dropout(drop_out)
        self.linear2 = nn.Linear(d_ff, d_in)
        
    def forward(self, x: torch.Tensor):
        return self.linear2(self.drop_out(F.silu(self.linear1(x))))