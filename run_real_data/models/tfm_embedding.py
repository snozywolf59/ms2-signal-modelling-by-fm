import torch
from torch import nn

from .embedding import (
    sinusoidal_position_encoding,
    sinusoidal_time_embedding,
    ChargeEmbedding,
)


class TimeEmbedding(nn.Module):
    def __init__(self, d_out: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(d_out, 16), nn.GELU(), nn.Linear(16, d_out)
        )
        self.d_out = d_out

    def forward(self, t: torch.Tensor):
        t_emb = sinusoidal_time_embedding(t, self.d_out)
        return self.embedding(t_emb)


class TfmConditionEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers,
        max_len=30,
        min_charge=1,
        max_charge=6,
        charge_dim=8,
    ):
        super().__init__()

        self.pep_embedding = nn.Embedding(
            22, d_model - charge_dim, padding_idx=0
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.tfm = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.register_buffer(
            "pos_encoding",
            sinusoidal_position_encoding(max_len, d_model),
            persistent=False,
        )
        self.charge_embedding = ChargeEmbedding(
            min_charge=min_charge, max_charge=max_charge, emb_dim=charge_dim
        )

    def forward(self, pep: torch.Tensor, charge: torch.Tensor):
        charge_emb = self.charge_embedding(charge)
        pep_tokens = self.pep_embedding(pep)
        B, L, _ = pep_tokens.shape
        charge_tokens = charge_emb.unsqueeze(1).expand(B, L, -1)
        x = torch.cat([pep_tokens, charge_tokens], dim=-1)
        pos = self.pos_encoding[: pep.size(1)].to(x.device)
        x = x + pos.unsqueeze(0)
        # mask = pep == 0

        return self.tfm(x)
