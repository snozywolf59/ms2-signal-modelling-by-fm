"""
preprocessing.py
────────────────
Ba chế độ preprocessing cho intensity vector:

  PreprocessMode.RAW    — [0,1], không transform
  PreprocessMode.LOGIT  — logit(x) đưa về R (Euclidean flow matching)
  PreprocessMode.SPHERE — L2-normalize lên S^(d-1) (Riemannian flow matching)

Mỗi mode có cặp (encode, decode) để có thể invert khi inference.
"""

from enum import Enum
from typing import Tuple

import torch
from torch import Tensor


# ────────────────────────────────────────────────────────────
class PreprocessMode(str, Enum):
    RAW = "raw"
    LOGIT = "logit"
    SPHERE = "sphere"


# ────────────────────────────────────────────────────────────
class Preprocessor:
    """
    Wrapper stateless: encode() và decode() tương ứng nhau.

    Parameters
    ----------
    mode : str | PreprocessMode
        "raw", "logit", hoặc "sphere"
    logit_eps : float
        Clamp epsilon cho logit mode, tránh log(0)
    sphere_eps : float
        Epsilon tránh chia cho 0 khi normalize
    """

    def __init__(
        self,
        mode: str = "logit",
        logit_eps: float = 1e-4,
        sphere_eps: float = 1e-8,
    ):
        self.mode = PreprocessMode(mode)
        self.logit_eps = logit_eps
        self.sphere_eps = sphere_eps

    # ── Encode: raw intensity → latent space ────────────────
    def encode(self, x: Tensor) -> Tensor:
        """
        x: Tensor shape (..., d), giá trị trong [0, 1]
        returns: latent tensor cùng shape (trừ sphere xem note)
        """
        if self.mode == PreprocessMode.RAW:
            return x.clone()

        elif self.mode == PreprocessMode.LOGIT:
            x_clamped = x.clamp(self.logit_eps, 1 - self.logit_eps)
            return torch.logit(x_clamped)

        elif self.mode == PreprocessMode.SPHERE:
            # Đảm bảo x không âm (intensity ≥ 0)
            x_pos = x.clamp(min=0.0)
            norm = x_pos.norm(dim=-1, keepdim=True).clamp(min=self.sphere_eps)
            return x_pos / norm

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ── Decode: latent space → intensity ────────────────────
    def decode(self, z: Tensor) -> Tensor:
        """
        Inverse của encode().
        z: latent tensor
        returns: intensity trong [0,1]
        """
        if self.mode == PreprocessMode.RAW:
            return z.clamp(0.0, 1.0)

        elif self.mode == PreprocessMode.LOGIT:
            return torch.sigmoid(z)

        elif self.mode == PreprocessMode.SPHERE:
            # Project về [0,1]: sigmoid từng chiều sau khi unnorm
            # (sphere → R^d → [0,1] qua sigmoid)
            return torch.sigmoid(z)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ── Sphere-specific helpers ──────────────────────────────
    def sphere_log_map(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Logarithmic map trên S^(d-1): Log_x(u)
        Trả về vector tiếp tuyến v tại x sao cho Exp_x(v) = u.

        Dùng công thức:
            v = (u - <u,x> x) / sin(θ) * θ
        với θ = arccos(<u, x>)
        """
        dot = (x * u).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)  # geodesic distance
        u_perp = u - dot * x     # phần vuông góc với x
        norm_perp = u_perp.norm(dim=-1, keepdim=True).clamp(min=self.sphere_eps)
        return theta * u_perp / norm_perp

    def sphere_exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map trên S^(d-1): Exp_x(v)
        Dịch chuyển từ điểm x dọc theo geodesic theo hướng v.

            Exp_x(v) = cos(||v||) * x + sin(||v||) * (v / ||v||)
        """
        norm_v = v.norm(dim=-1, keepdim=True).clamp(min=self.sphere_eps)
        return torch.cos(norm_v) * x + torch.sin(norm_v) * (v / norm_v)

    def sphere_interpolate(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Spherical linear interpolation (SLERP) giữa x0 và x1.
        t: scalar hoặc (..., 1)

        x_t = sin((1-t)θ)/sin(θ) * x0 + sin(tθ)/sin(θ) * x1
        """
        dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta).clamp(min=self.sphere_eps)
        coeff0 = torch.sin((1 - t) * theta) / sin_theta
        coeff1 = torch.sin(t * theta) / sin_theta
        return coeff0 * x0 + coeff1 * x1

    def sphere_target_vector(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Vector trường (conditional velocity field) trên sphere tại x_t.

        Với SLERP path: x_t = SLERP(x0, x1, t)
        dx_t/dt = Log_{x_t}(x1) / (1 - t)   [conditional flow]

        Đây là target mà model cần predict (thay cho u_pred - noise
        trong Euclidean case).
        """
        x_t = self.sphere_interpolate(x0, x1, t)
        # Log map từ x_t đến x1
        v = self.sphere_log_map(x_t, x1)
        # Scale theo (1-t) để có constant-speed geodesic
        scale = 1.0 / (1 - t).clamp(min=1e-6)
        return x_t, v * scale

    def __repr__(self) -> str:
        return f"Preprocessor(mode={self.mode.value})"
