import torch
from torch import nn

from embedding import ConditionEmbedding

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
    
    def sample(self, cond: torch.Tensor, batch: int = 32, step:int = 10):
        x_t = torch.randn(batch, self.noise_dim, device=cond.device)
        t = torch.zeros(batch, device=cond.device)
        dt = 1.0 / step
        for _ in range(step):
            t_start = t
            t_end = t + dt
            x_t = self.step(x_t, cond, t_start, t_end)
            t = t_end
        return x_t

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
    pep_dim = 128
    time_dim = 32
    charge_dim = 32
    def __init__(self, noise_dim, pep_dim=128, time_dim=32, charge_dim=32):
        super().__init__(noise_dim)
        self.noise_dim = noise_dim
        self.pep_dim = pep_dim
        self.time_dim = time_dim
        self.charge_dim = charge_dim
        self.pep_embedding = nn.Embedding(22, self.pep_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, pep_dim))
        self.charge_embedding = nn.Linear(1, self.charge_dim)
        self.time_embedding = nn.Linear(1, self.time_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pep_dim, 
            nhead=8, 
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        total_cond_dim = self.pep_dim + self.charge_dim + self.time_dim
        self.net = MLP(input_dim=total_cond_dim + noise_dim, output_dim=noise_dim, hidden_dim=1024, layers=4)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, pep_seq, charge):
        time_embs = self.time_embedding(t)
        charge_embs = self.charge_embedding(charge)
        
        pep_embs = self.pep_embedding(pep_seq) + self.pos_embedding
        
        p_out = self.transformer(pep_embs)
        p_cond = p_out.mean(dim=1)
        cond = torch.cat([time_embs, p_cond, charge_embs], dim=-1)
        
        return self.net(torch.cat([x, cond], dim=-1))
    
    def step(self, x_t: torch.Tensor, pep_seq, charge, t_start: torch.Tensor, t_end: torch.Tensor):
        # Use midpoint method for better accuracy
        t_mid = (t_start + t_end) / 2
        # x_next = x + f(x+ f(x, t_start) * (t_end - t_start) / 2, t_mid)
        v_x = self(x_t + self(x_t, t_start, pep_seq, charge) * (t_end - t_start) / 2, t_mid, pep_seq, charge)
        x_next = x_t + v_x * (t_end - t_start)
        return x_next

    
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
    def __init__(self, noise_dim, pep_dim=128, time_dim=64, charge_dim=64):
        super().__init__(noise_dim)
        self.cond_embedding = ConditionEmbedding(pep_dim, time_dim, charge_dim)
        self.mlp = ResMLPWithConditioning(noise_dim, pep_dim + time_dim + charge_dim ,num_blocks=8) 
    
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