import torch

def get_x0(x_1, max_scale=False):
    # Sample x_0 from a distribution and scale based on the max value of itself
    # x_1 shape: (batch_size, length, 6)
    x_0 = torch.randn_like(x_1)
    if max_scale:
        max_val = torch.max(torch.abs(x_0), dim=1, keepdim=True)[0]
        x_0 = x_0 / (max_val + 1e-8)
    return x_0

def get_xt(x_0, x_1, t, sigma=0.001):
    noise = torch.randn_like(x_0)
    t = t.view(-1, 1, 1)
    x_t = (1 - t) * x_0 + t * x_1 + sigma * noise
    return x_t
