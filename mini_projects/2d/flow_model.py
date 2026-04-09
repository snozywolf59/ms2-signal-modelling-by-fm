import torch
import torch.nn as nn

# TO HUU BANG
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=2, layers=4):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(input_dim + 1, hidden_dim))
        self.net.add_module('input_act', nn.SiLU())
        for i in range(layers - 2):
            self.net.add_module(f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim))
            self.net.add_module(f'hidden_act_{i}', nn.SiLU())
        self.net.add_module('output', nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x, t):
        # x: (B, 2), t: (B, 1)
        xt = torch.cat([x, t], dim=1)  # (B, 3)
        return self.net(xt)  # (B, 2)
    
    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor):
        t_start = t_start.view(1,1 ).expand(x_t.shape[0], 1)
        # x + deltat * f(x, t_start + deltat / 2)
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end - t_start) / 2, t_start + (t_end - t_start) / 2)