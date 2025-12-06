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
    def sample_checkerboard_region(batch_size, 
                               min_x=-2, min_y=-2, 
                               max_x=2, max_y=2, 
                               tiles=4):
        # Kích thước mỗi ô
        cell_w = (max_x - min_x) / tiles
        cell_h = (max_y - min_y) / tiles
        
        # Chọn ô (i, j) ngẫu nhiên nhưng chỉ lấy ô đen
        # (i + j) % 2 == 0: ô đen
        i_list = torch.randint(0, tiles, (batch_size,))
        j_list = torch.randint(0, tiles, (batch_size,))
        
        # Lọc: chỉ giữ ô đen
        mask = ((i_list + j_list) % 2 == 0)
        
        # Nếu thiếu điểm thì lấy thêm
        while mask.sum() < batch_size:
            new_i = torch.randint(0, tiles, (batch_size,))
            new_j = torch.randint(0, tiles, (batch_size,))
            new_mask = ((new_i + new_j) % 2 == 0)
            
            i_list = torch.where(mask, i_list, new_i)
            j_list = torch.where(mask, j_list, new_j)
            mask = ((i_list + j_list) % 2 == 0)
        
        # Lấy random trong ô
        x = min_x + i_list * cell_w + torch.rand(batch_size) * cell_w
        y = min_y + j_list * cell_h + torch.rand(batch_size) * cell_h
        
        return torch.stack([x, y], dim=1)
