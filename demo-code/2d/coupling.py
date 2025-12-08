from scipy.optimize import linear_sum_assignment
import torch


def mini_batch_coupling(src: torch.Tensor, des: torch.Tensor, mini_batch_size=256):
    indices = []
    for i in range(0, src.shape[0], mini_batch_size):
        cost_matrix = torch.cdist(src[i : i + mini_batch_size], des[i:i + mini_batch_size]).cpu().numpy()
        _, col_ind = linear_sum_assignment(cost_matrix)
        indices.extend(col_ind + i)
    return indices

def greedy_coupling(src: torch.Tensor, des: torch.Tensor):
    distances = torch.cdist(src, des)
    indices = distances.argmin(dim=1)
    return indices