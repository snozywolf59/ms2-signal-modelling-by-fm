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


def sinkhorm_coupling(src: torch.Tensor, des: torch.Tensor, epsilon=0.1, n_iters=100):
    n, m = src.shape[0], des.shape[0]
    a, b = torch.ones(n) / n, torch.ones(m) / m
    C = torch.cdist(src, des)

    K = torch.exp(-C / epsilon)
    u = torch.ones(n)
    for _ in range(n_iters):
        u = a / (K @ (b / (K.t() @ u)))
    v = b / (K.t() @ u)
    P = torch.diag(u) @ K @ torch.diag(v)
    indices = P.argmax(dim=1)
    return indices