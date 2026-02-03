import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) in [0, 1]
        return: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb
    
class FlowMLP(nn.Module):
    def __init__(
        self,
        x_dim=256,
        c_dim=480,
        time_dim=128,
        hidden_dim=1024,
        num_layers=4
    ):
        super().__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim

        self.time_embed = TimeEmbedding(time_dim)

        input_dim = x_dim + time_dim + c_dim

        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, c=None):
        """
        x: (B, 256)
        t: (B,)
        c: (B, 480) or None
        """
        t_emb = self.time_embed(t)

        if c is None:
            c = torch.zeros(x.shape[0], self.c_dim, device=x.device)

        h = torch.cat([x, t_emb, c], dim=-1)
        v = self.net(h)
        return v