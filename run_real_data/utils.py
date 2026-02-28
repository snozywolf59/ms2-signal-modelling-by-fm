import matplotlib.pyplot as plt
from time import time

PROSIT_ALHABET = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "M(ox)": 21,
}
PROSIT_INDEXED_ALPHABET = {i: c for c, i in PROSIT_ALHABET.items()}

def get_peptide_seq(integer_seq):
    return "".join(PROSIT_INDEXED_ALPHABET[int(i)] for i in integer_seq if i != 0)

def plot_loss_history(loss_history, smooth_window=None):
    """
    Plot training loss history.
    
    Args:
        loss_history (list or array): Danh sách loss theo từng step/epoch.
        smooth_window (int, optional): Nếu truyền vào, sẽ vẽ thêm đường smooth
                                       bằng moving average với window này.
    """
    plt.figure()
    plt.plot(loss_history)
    
    if smooth_window is not None and smooth_window > 1:
        import numpy as np
        loss_array = np.array(loss_history)
        kernel = np.ones(smooth_window) / smooth_window
        smooth_loss = np.convolve(loss_array, kernel, mode='valid')
        plt.plot(range(smooth_window - 1, len(loss_history)), smooth_loss)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.savefig(f"Loss_History_{time()}.jpg")
    plt.show()



    
    
# def plot_intensity(intensity: list):
    