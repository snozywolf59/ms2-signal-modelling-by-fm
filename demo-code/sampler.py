import torch
import math

class GaussSampler:
    @staticmethod
    def sample_single_gauss(batch_size, center=(0.0, 0.0), var=1.0):
        """
        Sinh mẫu 2D Gaussian quanh một điểm.
        
        batch_size : số mẫu cần sinh
        center     : tuple (x_center, y_center) - tọa độ trung tâm
        var        : variance chung cho cả x và y
        """
        std = var**0.5
        center = torch.tensor(center, dtype=torch.float32)
        samples = center + std * torch.randn(batch_size, 2)
        return samples

    @staticmethod
    def sample_symmetric_gauss(batch_size, n_gauss=4, distance=3.0, var=0.5, center=[0.0, 0.0]):
        """
        Sinh mẫu từ n Gaussian 2D đối xứng quanh gốc.
        
        batch_size : số mẫu cần sinh
        n_gauss    : số Gaussian đối xứng
        mean       : khoảng cách trung bình từ gốc đến mỗi Gaussian (bán kính)
        var        : variance chung cho tất cả Gaussian
        """
        std = var**0.5
        # chọn Gaussian cho từng sample
        comp_ids = torch.randint(0, n_gauss, (batch_size,))
        
        # đặt mean trên vòng tròn bán kính 'mean'
        angles = torch.linspace(0, 2*math.pi * (1.0 - 1.0/(n_gauss)), steps=n_gauss, dtype=torch.float32)
        gauss_means = torch.stack([ torch.cos(angles + math.pi/2), torch.sin(angles + math.pi/2)], dim=1) * distance + torch.Tensor(center).to(device=angles.device)
        
        # sinh mẫu
        samples = gauss_means[comp_ids] + std * torch.randn(batch_size, 2)
        return samples

class CheckerBoardSampler:
    @staticmethod
    def sample_checkerboard(batch_size):
        x1 = torch.rand(batch_size) * 4 - 2
    
        x2_ = torch.rand(batch_size) - torch.randint(0, 2, (batch_size,)) * 2
        x2 = (x2_ + (torch.floor(x1) % 2))
        
        data = (torch.stack([x1, x2], dim=1) * 2.0 )
        return data