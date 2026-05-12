import torch
from torch import nn

from .cfg_based import CFGFlow
from .tfm_embedding import (
    TfmConditionEncoder,
    TimeEmbedding,
    sinusoidal_position_encoding,
)

# class NoiseProjection(nn.Module):
#     def __init__(self, d_in: int, d_model: int):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(d_in, 64),
#             nn.GELU(),
#             nn.Linear(64, d_model),
#             nn.LayerNorm(d_model),
#         )

#     def forward(self, noise: torch.Tensor):
#         return self.projection(noise)


# class ReverseNoiseProjection(nn.Module):
#     def __init__(self, d_model: int, d_out: int):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(d_model, 64),
#             nn.GELU(),
#             nn.Linear(64, d_out),
#         )

#     def forward(self, x: torch.Tensor):
#         return self.projection(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(cond_dim, 2 * d_model)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        cond = self.act(cond)
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Modulation(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.GELU(),
            nn.Linear(128, 6 * d_model),
        )
        self.act = nn.SiLU()

    def forward(self, cond):
        cond = self.act(cond)
        a, b, c, d, e, f = self.mod(cond).chunk(6, dim=-1)
        return a, b, c, d, e, f  # 6 x [B, d_model]


class AdaLnLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, cond_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mod = Modulation(d_model, cond_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x, y_emb, mask_x=None):
        a, b, c, d, e, f = self.mod(y_emb)

        old_x = x
        # first modulation
        x = self.norm1(x)
        x = x * (1 + a.unsqueeze(1)) + b.unsqueeze(1)

        # Self-attention on x
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask_x)

        # scale and residual connection
        x = old_x + x * c.unsqueeze(1)
        old_x = x

        # second modulation
        x = self.norm2(old_x)
        x = x * (1 + d.unsqueeze(1)) + e.unsqueeze(1)

        # Feedforward
        x = self.ff(x)

        x = old_x + x * f.unsqueeze(1)

        return x


class DiffusionFlow(nn.Module):

    def __init__(
        self,
        d_noise,
        d_model,
        nhead=2,
        num_layers=6,
        num_pep_layers=6,
        charge_dim=8,
    ):
        super().__init__()

        self.time_dim = 128
        cond_dim = d_model + self.time_dim  # + charge_dim
        self.time_embedding = TimeEmbedding(d_out=self.time_dim)

        self.pep_embedding = TfmConditionEncoder(
            d_model, num_pep_layers, charge_dim=charge_dim
        )

        self.blocks = nn.ModuleList(
            [
                AdaLnLayer(d_noise, nhead, d_noise * 4, cond_dim=cond_dim)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_noise)
        self.line_out = nn.Linear(d_noise, d_noise)

        self.register_buffer(
            "pos_encoding",
            sinusoidal_position_encoding(29, d_noise),
            persistent=False,
        )

    @property
    def peptide_embedding(self):
        return self.pep_embedding

    def forward(
        self,
        noise_tokens: torch.Tensor,  # B, 29, 6
        pep: torch.Tensor,  # B, L
        charge: torch.Tensor,  # B
        time: torch.Tensor,  # B
    ):

        # noise_tokens = self.noise_projection(noise_tokens)

        pep_tokens = self.pep_embedding(pep, charge)  # B, L, d_model
        pep_padding_mask = pep == 0

        mask = (~pep_padding_mask).unsqueeze(-1)  # B, L, 1
        
        # init intensity padding mask - > B, L - 1, 1
        intensity_masked = pep_padding_mask[:,1:]

        pep_sum = (pep_tokens * mask).sum(dim=1)  # B, d_model
        pep_len = mask.sum(dim=1)  # B, 1

        pep_mean = pep_sum / pep_len.clamp(min=1)

        time_emb = self.time_embedding(time)  # B, time dim

        y_emb = torch.cat([time_emb, pep_mean], dim=-1)  # B, cond_dim

        x = noise_tokens

        pos = self.pos_encoding[: x.size(1), :].unsqueeze(0)  # 1, 29, d_model
        x = x + pos  # pos encoding

        for block in self.blocks:
            x = block(x, y_emb,mask_x=intensity_masked)

        x = self.final_norm(x)
        x = self.line_out(x)
        return x

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
