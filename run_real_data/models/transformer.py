import torch
from torch import nn

from .cfg_based import CFGFlow
from .embedding import (
    PepEmbedding,
    ChargeEmbedding,
    sinusoidal_time_embedding,
    sinusoidal_position_encoding,
)


class NoiseProjection(nn.Module):
    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, noise: torch.Tensor):
        return self.projection(noise)


class ReverseNoiseProjection(nn.Module):
    def __init__(self, d_model: int, d_out: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )

    def forward(self, x: torch.Tensor):
        return self.projection(x)


class TimeEmbedding(nn.Module):
    def __init__(self, d_out: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(d_out, 8), nn.GELU(), nn.Linear(8, d_out)
        )
        self.d_out = d_out

    def forward(self, t: torch.Tensor):
        t_emb = sinusoidal_time_embedding(t, self.d_out)
        return self.embedding(t_emb)


class AdaLayerNorm(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x, cond):
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FluxLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, cond_dim):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm1 = AdaLayerNorm(d_model, cond_dim)
        self.norm2 = AdaLayerNorm(d_model, cond_dim)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(0.1)
        self.act = nn.SiLU()

    def forward(self, noise_tokens, cond_tokens, cond):

        tokens = torch.cat([cond_tokens, noise_tokens], dim=1)

        h = self.attn(
            self.norm1(tokens, cond),
            self.norm1(tokens, cond),
            self.norm1(tokens, cond),
        )[0]

        tokens = tokens + h

        h = self.linear2(
            self.dropout(self.act(self.linear1(self.norm2(tokens, cond))))
        )

        tokens = tokens + h

        text_len = cond_tokens.shape[1]

        cond_tokens = tokens[:, :text_len]
        noise_tokens = tokens[:, text_len:]

        return noise_tokens, cond_tokens


class NoiseDiffusionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, cond_dim):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, kdim=cond_dim, vdim=cond_dim
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, src_tokens, condition_tokens, pep_padding_mask=None):
        # self attention
        x = src_tokens + self.dropout(
            self.self_attn(
                self.norm1(src_tokens),
                self.norm1(src_tokens),
                self.norm1(src_tokens),
            )[0]
        )
        # cross attention
        x = x + self.dropout(
            self.cross_attn(
                self.norm2(x),
                condition_tokens,
                condition_tokens,
                key_padding_mask=pep_padding_mask,
            )[0]
        )
        # FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x


class DiffusionFlow(nn.Module):

    def __init__(
        self, d_noise, d_model, nhead=8, num_layers=6, num_pep_layers=6
    ):
        super().__init__()

        self.pep_embedding = PepEmbedding(d_model, num_layers=num_pep_layers)
        self.noise_projection = NoiseProjection(d_noise, d_model)

        self.charge_embedding = ChargeEmbedding(1, 6, use_layernorm=False)
        self.time_embedding = TimeEmbedding(16)

        cond_dim = d_model + 4 + 16

        self.blocks = nn.ModuleList(
            [
                NoiseDiffusionEncoderLayer(
                    d_model,
                    nhead,
                    d_model,
                    cond_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.output_proj = ReverseNoiseProjection(d_model, d_noise)

    @property
    def peptide_embedding(self):
        return self.pep_embedding

    def forward(
        self,
        noise: torch.Tensor,  # B, 29, 6
        pep: torch.Tensor,  # B, L
        charge: torch.Tensor,  # B
        time: torch.Tensor,  # B
    ):

        noise_tokens = self.noise_projection(noise)  # B, 29, d_model
        pep_tokens = self.pep_embedding(pep)  # B, L, d_model
        pep_padding_mask = (pep == 0)

        charge_emb = self.charge_embedding(charge)  # B, 4
        time_emb = self.time_embedding(time)  # B, 16

        # expand condition embeddings
        B, L, _ = pep_tokens.shape

        charge_tokens = charge_emb.unsqueeze(1).expand(B, L, -1)
        time_tokens = time_emb.unsqueeze(1).expand(B, L, -1)

        # concat condition tokens
        condition_tokens = torch.cat(
            [pep_tokens, charge_tokens, time_tokens], dim=-1
        )  # B, L, d_model + 4 + 16

        x = noise_tokens

        for block in self.blocks:
            x = block(x, condition_tokens, pep_padding_mask=pep_padding_mask)

        x = self.final_norm(x)

        out = self.output_proj(x)  # B, 29, d_noise

        return out

    def step(
        self,
        x_t: torch.Tensor,  # (B,29,6)
        pep_seq: torch.Tensor,  # (B,L)
        charge: torch.Tensor,  # (B, 1)
        t_start: torch.Tensor,  # (B,1)
        t_end: torch.Tensor,  # (B,1)
    ):
        t_mid = (t_start + t_end) / 2
        dt = (t_end - t_start).view(-1, 1, 1)

        v_start = self(x_t, pep_seq, charge, t_start)
        x_mid = x_t + v_start * dt / 2
        v_mid = self(x_mid, pep_seq, charge, t_mid)
        x_next = x_t + v_mid * dt
        return x_next

    def sample(self, noise, pep_seq, charge, step: int = 10):
        x_t = noise  # (B,29,6)
        B = noise.shape[0]
        t = torch.zeros(B, 1, device=noise.device)
        dt = 1.0 / step
        for _ in range(step):
            t_end = t + dt
            x_t = self.step(x_t, pep_seq, charge, t, t_end)
            t = t_end

        return x_t
