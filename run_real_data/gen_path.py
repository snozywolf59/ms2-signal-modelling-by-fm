import torch


def get_xt(x_0, x_1, t, sigma=0.001):
    noise = torch.randn_like(x_0)
    t = t.view(-1, 1, 1)
    x_t = (1 - t) * x_0 + t * x_1 + sigma * noise
    return x_t
