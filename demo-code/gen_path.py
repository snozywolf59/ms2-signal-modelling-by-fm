import torch

def get_xt(x_0, x_1, t, sigma=0.001):
    """
    Tính x_t = (1 - t) * x_0 + t * x_1 + sigma * noise
    
    x_0: tensor (B, 2) - điểm nguồn
    x_1: tensor (B, 2) - điểm đích
    t  : tensor (B, 1) - thời gian
    sigma: độ lệch chuẩn của nhiễu Gaussian thêm vào
    """
    noise = torch.randn_like(x_0)
    x_t = (1 - t) * x_0 + t * x_1 + sigma * noise
    return x_t