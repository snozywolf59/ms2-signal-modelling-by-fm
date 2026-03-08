import torch


def get_x0(x_1: torch.Tensor, max_scale=False):
    # Sample x_0 from a distribution and scale based on the max value of itself
    # x_1 shape: (batch_size, length, 6)
    x_0 = torch.randn_like(x_1)
    if max_scale:
        if x_1.dim() == 1:
            max_val = x_0.abs().max(keepdim=True)
            max_val = torch.clamp(max_val, min=1e-8)
            x_0 = x_0 / max_val
        elif x_1.dim() == 2:  # (B, D)
            max_val = x_0.abs().max(dim=1, keepdim=True)[0]

            max_val = torch.clamp(max_val, min=1e-8)
            x_0 = x_0 / max_val

        elif x_1.dim() == 3:  # (B, L, D)
            max_val = x_0.abs().amax(dim=(1, 2), keepdim=True)
            max_val = torch.clamp(max_val, min=1e-8)
            x_0 = x_0 / max_val
    return x_0


def get_xt(x_0, x_1, t, sigma=0.001):
    noise = torch.randn_like(x_0)
    t = t.view(-1, 1, 1)
    x_t = (1 - t) * x_0 + t * x_1 + sigma * noise
    return x_t
