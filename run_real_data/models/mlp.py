import torch
from torch import nn
from typing import Literal

from .cfg_based import CFGFlow
from .embedding import TfmEmbedding, ConcatEmbedding, PretrainEmbedding

# simple MLP: concat eveything
## Model for flow
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.SiLU())
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
   
## apply above MLP for flow model, with conditioning on peptide sequence, charge and time    
class HCDFlow(CFGFlow):
    def __init__(self, noise_dim, embed_type: Literal["concat","pretrain","tfm"] = "tfm", pep_dim=128, time_dim=32, charge_dim=32):
        super().__init__(noise_dim)
        if embed_type == "tfm": 
            self.cond_embedding = TfmEmbedding(pep_dim, time_dim, charge_dim)
        elif embed_type == "pretrain":
            self.cond_embedding = PretrainEmbedding()
        elif embed_type == "concat":
            self.cond_embedding = ConcatEmbedding()
        else:
            raise RuntimeError(f"Unimplement error: Embedding type {embed_type} not found.")
        total_cond_dim = pep_dim + charge_dim + time_dim + noise_dim
        self.net = MLP(input_dim=total_cond_dim, output_dim=noise_dim, hidden_dim=1024, layers=4)
    
    @property
    def condition_embedding(self):
        return self.cond_embedding
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, pep_seq, charge):
        cond = self.cond_embedding(pep_seq, charge=charge, time=t)
        # print(x.shape, cond.shape)
        return self.net(torch.cat([x, cond], dim=-1))
    
    def step(self, x_t: torch.Tensor, pep_seq, charge, t_start: torch.Tensor, t_end: torch.Tensor):
        # Use midpoint method for better accuracy
        t_mid = (t_start + t_end) / 2
        # x_next = x + f(x+ f(x, t_start) * (t_end - t_start) / 2, t_mid)
        v_x = self(x_t + self(x_t, t_start, pep_seq, charge) * (t_end - t_start) / 2, t_mid, pep_seq, charge)
        x_next = x_t + v_x * (t_end - t_start)
        return x_next
    
    def sample(self,noise, pep_seq, charge, step:int = 10):
        x_t = noise
        t = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt = 1.0 / step
        for _ in range(step):
            t_start = t
            t_end = t + dt
            x_t = self.step(x_t, pep_seq, charge, t_start, t_end)
            t = t_end
        return x_t

