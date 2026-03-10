import torch


def get_x0(x_1: torch.Tensor, max_scale=False):
    # Sample x_0 from a distribution and scale based on the element
    # that has the largest absolute value (but keep its sign)

    x_0 = torch.randn_like(x_1)

    if max_scale:
        if x_1.dim() == 1:
            idx = torch.argmax(x_0.abs())
            max_val = x_0[idx]
            max_val = torch.clamp(max_val, min=1e-8) if max_val >= 0 else torch.clamp(max_val, max=-1e-8)
            x_0 = x_0 / max_val

        elif x_1.dim() == 2:  # (B, D)
            idx = torch.argmax(x_0.abs(), dim=1, keepdim=True)
            max_val = torch.gather(x_0, 1, idx)
            max_val = torch.where(
                max_val >= 0,
                torch.clamp(max_val, min=1e-8),
                torch.clamp(max_val, max=-1e-8)
            )
            x_0 = x_0 / max_val

        elif x_1.dim() == 3:  # (B, L, D)
            B = x_0.shape[0]
            flat = x_0.view(B, -1)

            idx = torch.argmax(flat.abs(), dim=1, keepdim=True)
            max_val = torch.gather(flat, 1, idx).view(B, 1, 1)

            max_val = torch.where(
                max_val >= 0,
                torch.clamp(max_val, min=1e-8),
                torch.clamp(max_val, max=-1e-8)
            )

            x_0 = x_0 / max_val

    return x_0


def get_xt(x_0, x_1, t, sigma=0.001):
    noise = torch.randn_like(x_0)
    if noise.dim() == 3:
        t = t.view(-1, 1, 1)
    elif noise.dim() == 2:
        t = t.view(-1, 1)
    x_t = (1 - t) * x_0 + t * x_1 + sigma * noise
    return x_t
