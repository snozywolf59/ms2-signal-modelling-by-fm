import torch
from torch import nn

from .cfg_based import CFGFlow
from .tfm_embedding import TfmConditionEncoder, TimeEmbedding


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
        self.mod = nn.Linear(cond_dim, 6 * d_model)
        self.act = nn.SiLU()

    def forward(self, cond):
        cond = self.act(cond)
        a, b, c, d, e, f = self.mod(cond).chunk(6, dim=-1)
        return a, b, c, d, e, f  # 6 x [B, d_model]


class NoiseDiffusionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, cond_dim):
        super().__init__()

        self.joint_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )

        self.norm1_x = nn.LayerNorm(d_model)
        self.norm1_c = nn.LayerNorm(d_model)
        self.norm2_x = nn.LayerNorm(d_model)
        self.norm2_c = nn.LayerNorm(d_model)
        self.mod_x = Modulation(d_model, cond_dim)
        self.mod_c = Modulation(d_model, cond_dim)

        self.before_attn_linear_x = nn.Linear(d_model, d_model)
        self.before_attn_linear_y = nn.Linear(d_model, d_model)

        self.after_attn_linear_x = nn.Linear(d_model, d_model)
        self.after_attn_linear_y = nn.Linear(d_model, d_model)

        self.ffn_x = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_c = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x, c, y_emb, x_mask=None, c_mask=None):
        # calculate modulation parameters
        ax, bx, cx, dx, ex, fx = self.mod_x(y_emb)

        ac, bc, cc, dc, ec, fc = self.mod_c(y_emb)

        ax, bx, cx, dx, ex, fx = [
            t.unsqueeze(1) for t in (ax, bx, cx, dx, ex, fx)
        ]
        ac, bc, cc, dc, ec, fc = [
            t.unsqueeze(1) for t in (ac, bc, cc, dc, ec, fc)
        ]

        res_x = x
        res_c = c

        x = self.norm1_x(x) * ax + bx
        c = self.norm1_c(c) * ac + bc

        x = self.before_attn_linear_x(x)
        c = self.before_attn_linear_y(c)

        # 2. Joint Attention
        batch_size, len_x, _ = x.shape
        _, len_c, _ = c.shape

        # Gộp token noise (x) và peptide (c)
        joint_tokens = torch.cat([x, c], dim=1)  # [B, len_x + len_c, d_model]

        # Nếu có mask, cũng cần gộp mask (chú ý: MultiheadAttention batch_first dùng key_padding_mask)
        # joint_mask = torch.cat([x_mask, c_mask], dim=1) if x_mask is not None else None

        # Self-attention
        attn_out, _ = self.joint_attn(joint_tokens, joint_tokens, joint_tokens)

        # Tách c và x từ đầu
        x = attn_out[:, :len_x, :]
        c = attn_out[:, len_x:, :]

        # Residual
        x = res_x + self.after_attn_linear_x(x) * cx
        c = res_c + self.after_attn_linear_y(c) * cc

        x = self.norm2_x(x) * dx + ex
        c = self.norm2_c(c) * dc + ec

        # 3. Feed Forward
        x = res_x + self.ffn_x(x) * fx
        c = res_c + self.ffn_c(c) * fc

        return x, c


class DiffusionFlow(nn.Module):

    def __init__(
        self,
        d_noise,
        d_model,
        nhead=8,
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
        self.noise_projection = NoiseProjection(d_noise, d_model)

        self.blocks = nn.ModuleList(
            [
                NoiseDiffusionEncoderLayer(
                    d_model, nhead, d_model, cond_dim=cond_dim
                )
                for _ in range(num_layers)
            ]
        )

        # self.cond_proj = nn.Linear(time_dim + charge_dim, d_model)

        self.final_norm = nn.LayerNorm(d_noise)

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

        noise_tokens = self.noise_projection(noise)

        pep_tokens = self.pep_embedding(pep, charge)  # B, L, d_model
        # pep_padding_mask = pep == 0

        time_emb = self.time_embedding(time)  # B, time dim
        mean_pep_emb = pep_tokens.mean(dim=1)  # B, d_model
        y_emb = torch.cat([time_emb, mean_pep_emb], dim=-1)  # B, cond_dim

        x = noise_tokens
        c = pep_tokens

        for block in self.blocks:
            x, c = block(x, c, y_emb)

        return self.final_norm(self.output_proj(x))

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


class FluxFlow(nn.Module):
    pass
