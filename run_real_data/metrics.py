import torch

"""
There are implementations of some metrics for flow matching model evaluation
Have 3 metrics:
    - pcc:  Pearson Correlation Coefficient
    - l1:   L1
    - l2:   L2
    - sa:   SA
Params:
    - intensity_1: (B, D)
    - intensity_2: (B, D)

Output:
    - (mean, max)

"""


def pcc(intensity_1, intensity_2, mask=None):
    eps = 1e-8
    dims = tuple(range(1, intensity_1.dim()))

    x, y = intensity_1, intensity_2

    if mask is not None:
        x = x * mask
        y = y * mask
        n = mask.sum(dim=dims, keepdim=True) + eps
        x_mean = x.sum(dim=dims, keepdim=True) / n
        y_mean = y.sum(dim=dims, keepdim=True) / n
        x_centered = (x - x_mean) * mask
        y_centered = (y - y_mean) * mask
    else:
        x_mean = x.mean(dim=dims, keepdim=True)
        y_mean = y.mean(dim=dims, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean

    numerator = (x_centered * y_centered).sum(dim=dims)
    denominator = (
        torch.sqrt(
            (x_centered**2).sum(dim=dims) * (y_centered**2).sum(dim=dims)
        )
        + eps
    )

    pcc_values = numerator / denominator  # Shape: (B,)
    return pcc_values.mean().item(), pcc_values.min().item()


def l1(intensity_1: torch.Tensor, intensity_2: torch.Tensor):
    intensity_1 = (intensity_1 - intensity_2).abs()
    return (
        intensity_1.sum(dim=1).mean().item(),
        intensity_1.sum(dim=1).max().item(),
    )


def l2(intensity_1: torch.Tensor, intensity_2: torch.Tensor):
    intensity_1 = (intensity_1 - intensity_2) ** 2
    return intensity_1.mean().item(), intensity_1.sum(dim=1).max().item()


def sa(intensity_1, intensity_2, mask=None):
    eps = 1e-8
    dims = tuple(range(1, intensity_1.dim()))

    if mask is not None:
        intensity_1 = intensity_1 * mask
        intensity_2 = intensity_2 * mask

    # Dot product tổng hợp trên tất cả các chiều của mẫu
    dot_product = (intensity_1 * intensity_2).sum(dim=dims)

    # Norm tổng hợp
    norm_1 = torch.sqrt((intensity_1**2).sum(dim=dims))
    norm_2 = torch.sqrt((intensity_2**2).sum(dim=dims))

    sa_values = dot_product / (norm_1 * norm_2 + eps) # Shape: (B,)

    return sa_values.mean().item(), sa_values.min().item()
