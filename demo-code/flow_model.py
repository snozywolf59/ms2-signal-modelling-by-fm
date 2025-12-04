import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dim=8, layers=4):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(3, hidden_dim))
        self.net.add_module('input_act', nn.ReLU())
        for i in range(layers - 2):
            self.net.add_module(f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim))
            self.net.add_module(f'hidden_act_{i}', nn.ReLU())
        self.net.add_module('output', nn.Linear(hidden_dim, 2))
        
    def forward(self, x, t):
        # x: (B, 2), t: (B, 1)
        xt = torch.cat([x, t], dim=1)  # (B, 3)
        return self.net(xt)  # (B, 2)
    
    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor):
        t_start = t_start.view(1,1 ).expand(x_t.shape[0], 1)
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end - t_start) / 2, t_start + (t_end - t_start) / 2)