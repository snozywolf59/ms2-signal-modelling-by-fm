from scipy.optimize import linear_sum_assignment
import torch

def ot_coupling(x, y):
    """Compute the optimal transport coupling between two batch 2-D points."""
    C = torch.cdist(x, y, p=2).pow(2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(C)
    coupling = torch.zeros(x.size(0), y.size(0))
    coupling[row_ind, col_ind] = 1.0
    return coupling
    