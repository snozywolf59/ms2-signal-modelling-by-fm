import torch
from torch import nn

import math


def sinusoidal_time_embedding(t, dim):
    """
    t: (batch,)  or (batch, 1)
    dim: embedding dimension (must be even)
    return: (batch, dim)
    """
    assert dim % 2 == 0, "Sinusoidal embedding dimension must be division by 2"
    half_dim = dim // 2
    device = t.device

    # compute frequencies
    exponent = torch.arange(half_dim, device=device) / half_dim
    freqs = torch.exp(-math.log(10000) * exponent)

    t = t.view(-1, 1)

    angles = t * freqs.view(1, -1)

    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb


def sinusoidal_position_encoding(seq_len, dim):
    """
    seq_len: length of sequence (e.g., 200)
    dim: embedding dimension (must be even)
    return: (seq_len, dim)
    """
    device = torch.get_default_device()

    position = torch.arange(seq_len, device=device).unsqueeze(1)
    half_dim = dim // 2

    exponent = torch.arange(half_dim, device=device) / half_dim
    div_term = torch.exp(-math.log(10000) * exponent)

    angles = position * div_term.unsqueeze(0)

    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return pe


class ConcatEmbedding(nn.Module):
    def __init__(self, pep_dim, out_dim: 512):
        super().__init__()
        self.pep_embedding = nn.Embedding(22, pep_dim, padding_idx=0)
        self.fc = nn.Linear(30 * pep_dim + 2, out_dim)
        self.act = nn.SiLU()

    def forward(
        self, seq: torch.Tensor, charge: torch.Tensor, time: torch.Tensor
    ):
        seq_embs = self.pep_embedding(seq)
        # print(seq_embs.shape)
        concat_seq = seq_embs.view(seq.size(0), -1)
        # print(concat_seq.shape, charge.shape)
        cond = torch.cat([concat_seq, charge, time], dim=-1)
        return self.act(self.fc(cond))


class ChargeEmbedding(nn.Module):
    def __init__(
        self,
        min_charge: int = 2,
        max_charge: int = 6,
        emb_dim: int = 4,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.min_charge = min_charge
        self.max_charge = max_charge
        self.num_charge = max_charge - min_charge + 1

        self.emb = nn.Embedding(self.num_charge, emb_dim - 1)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(emb_dim)

    def forward(self, charge: torch.Tensor):
        # print(f"Charge shape: {charge.shape}")
        charge_idx = charge - self.min_charge

        emb = self.emb(charge_idx).squeeze(1)
        # print(f"Emb shape: {emb.shape}")
        scalar = (charge.float() - self.min_charge) / (
            self.max_charge - self.min_charge
        )

        # print(f"Scalar shape: {scalar.shape}")
        # scalar = scalar.unsqueeze(-1)
        # print(f"Scalar shape: {scalar.shape}")
        out = torch.cat([scalar, emb], dim=-1)

        if self.use_layernorm:
            out = self.ln(out)

        return out


class TfmEmbedding(nn.Module):
    def __init__(
        self,
        pep_dim=128,
        time_dim=32,
        charge_dim=32,
        min_charge=2,
        max_charge=6,
        num_blocks_pep=6,
    ):
        super().__init__()
        # self.charge_embedding = nn.Linear(1, charge_dim)
        self.charge_dim = charge_dim
        self.time_dim = time_dim
        self.pep_dim = pep_dim

        self.pep_embedding = nn.Embedding(23, pep_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pep_dim + charge_dim, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_blocks_pep
        )
        self.charge_embedding = ChargeEmbedding(
            min_charge, max_charge, charge_dim
        )

    def forward(
        self, seq: torch.Tensor, charge: torch.Tensor, time: torch.Tensor
    ):
        # add cls token
        # cls token id = 22
        seq = torch.cat(
            [torch.full((seq.size(0), 1), 22, dtype=torch.long, device=seq.device), seq], dim=1
        )  # B, L+1
        mask = (seq == 0) # B, L
        charge_emb = self.charge_embedding(charge).unsqueeze(1).expand(-1, seq.size(1), -1)
        x = torch.cat([self.pep_embedding(seq), charge_emb], dim=-1) + sinusoidal_position_encoding(
            seq.size(1), self.pep_dim + self.charge_dim
        ).unsqueeze(0) # B, L, d_model
        x = x.masked_fill(
            mask.unsqueeze(-1),
            0.0,
        )
        x = self.transformer(x, src_key_padding_mask=mask)

        pep_c = x[:, 0, :]
        # charge_emb = sinusoidal_time_embedding(charge, self.charge_dim)
        
        time_emb = sinusoidal_time_embedding(time, self.time_dim)
        # print(pep_c.shape, charge_emb.shape, time_emb.shape)
        return torch.cat([pep_c, time_emb], dim=-1)



class PretrainEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
