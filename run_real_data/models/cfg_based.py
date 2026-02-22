import torch
from torch import nn
#
class CFGFlow(nn.Module):
    def __init__(self, noise_dim, *cond_dim):
        super().__init__()
        self.noise_dim = noise_dim
        # self.cond_dim = cond_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, **cond):
        raise NotImplementedError("CFG Model is not implemented")
    
    def step(self, x_t: torch.Tensor, cond:torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor):
        raise NotImplementedError("CFG Model is not implemented")
    
    def sample(self, **cond):
        raise NotImplementedError("CFG Model is not implemented")
    