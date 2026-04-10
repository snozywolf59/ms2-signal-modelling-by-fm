import torch


def logit_transform(x: torch.Tensor, eps: float = 1e-4):
    x = eps + (1 - 2 * eps) * x
    return torch.logit(x)


def random_logit_transform(
    x: torch.Tensor, min_eps: float = 1e-4, max_eps: float = 4e-4
):
    eps = torch.rand_like(x) * (max_eps - min_eps) + min_eps
    return torch.logit(x, eps)


def sigmoid_transform(y: torch.Tensor, eps: float = 1e-4):
    x = torch.sigmoid(y)
    return (x - eps) / (1 - 2 * eps)
