import torch
from torch import nn
from typing import Literal

from .cfg_based import CFGFlow
from .embedding import TfmEmbedding, ConcatEmbedding, PretrainEmbedding

# Deep MLP with residual block and FiLM conditioning

## ResBlock with FiLM conditioning
class ResBlock(nn.Module):
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 4, dim)

        self.film_fc = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.film_fc.weight)
        nn.init.zeros_(self.film_fc.bias)

    def forward(self, x, condition=None):
        h = self.norm(x)

        if condition is not None:
            gamma_beta = self.film_fc(condition)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = h * (1 + gamma) + beta
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + h

## ResMLP with FiLM conditioning
class ResMLPWithConditioning(nn.Module):
    def __init__(self, dim, cond_dim, num_blocks=8):
        super().__init__()
        self.blocks = nn.ModuleList(
           [ ResBlock(dim, cond_dim) for _ in range(num_blocks)]
        )
        self.norm = nn.RMSNorm(dim)

    def forward(self, x, condition):
        for block in self.blocks:
            x = block(x, condition)
        return self.norm(x)

## Model for flow with ResMLP and FiLM conditioning
class HCDFlowResMLP(CFGFlow):
    def __init__(self, noise_dim, embed_type: Literal["tfm", "pretrain", "concat"]="tfm", pep_dim=128, time_dim=64, charge_dim=64, min_charge=2, max_charge=6, num_blocks=8, num_blocks_pep=6):
        super().__init__(noise_dim)
        if embed_type == "tfm":
            self.cond_embedding = TfmEmbedding(pep_dim, time_dim, charge_dim, min_charge, max_charge, num_blocks_pep)
        elif embed_type == "pretrain":
            self.cond_embedding = PretrainEmbedding()
        elif embed_type == "concat":
            self.cond_embedding = ConcatEmbedding(pep_dim, 2* pep_dim)
        else:
            raise RuntimeError(f"Unimplement error: Embedding type {embed_type} not found.")
        if embed_type == "tfm":
            total_cond_dim = pep_dim + charge_dim + time_dim
        elif embed_type == "concat":
            total_cond_dim = 2 * pep_dim
        else:
            raise RuntimeError(f"Unimplement error: Embedding type {embed_type} not found.")
        self.mlp = ResMLPWithConditioning(noise_dim, total_cond_dim ,num_blocks=num_blocks)

    @property
    def condition_embedding(self):
        return self.cond_embedding

    def forward(self, x: torch.Tensor, t: torch.Tensor, pep_seq, charge):
        cond_emb = self.cond_embedding(pep_seq, charge=charge, time=t)
        return self.mlp(x, cond_emb)

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
            # t_start = t
            t_end = t + dt
            x_t = self.step(x_t, pep_seq, charge, t, t_end)
            t = t_end
        return x_t
