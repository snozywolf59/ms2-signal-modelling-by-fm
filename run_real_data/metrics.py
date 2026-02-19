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


def pcc(intensity_1: torch.Tensor, intensity_2: torch.Tensor):
    eps = 1e-8
    
    x = intensity_1
    y = intensity_2
    
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    
    cov = (x * y).sum(dim=1)
    
    std_x = torch.sqrt((x ** 2).sum(dim=1))
    std_y = torch.sqrt((y ** 2).sum(dim=1))
    pcc = cov / (std_x * std_y + eps)
    
    return pcc.mean().item(), pcc.max().item()

def l1(intensity_1: torch.Tensor, intensity_2: torch.Tensor):
    intensity_1 = (intensity_1 - intensity_2).abs()
    return intensity_1.mean().item(), intensity_1.sum(dim=1).max().item()

def l2(intensity_1: torch.Tensor, intensity_2: torch.Tensor):
    intensity_1 = (intensity_1 - intensity_2) ** 2
    return intensity_1.mean().item(), intensity_1.sum(dim=1).max().item()

def sa(intensity_1: torch.Tensor, intensity_2: torch.Tensor):
    dot_p = (intensity_1 * intensity_2).sum(dim=1)
    dist_1 = intensity_1.pow(2).sum(dim=1).sqrt()
    dist_2 = intensity_2.pow(2).sum(dim=1).sqrt()
    cos = dot_p / (dist_1 * dist_2 + 1e-8)
    cos = cos.clamp(-1.0, 1.0)
    
    sa_values = torch.acos(cos)
    return sa_values.mean().item(), sa_values.max().item()
    