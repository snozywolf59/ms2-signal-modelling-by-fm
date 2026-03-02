import matplotlib.pyplot as plt
from time import time
import numpy as np
import torch

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

def to_numpy(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)

def plot_intensity(intensity_vector: torch.Tensor | np.ndarray, allow_invalid: bool = True):
    """Plot a Prosit-like intensity

    Args:
        intensity_vector (torch.Tensor): _description_
        allow_invalid (bool, optional): _description_. Defaults to True.

    Raises:
        TypeError: _description_
        ValueError: _description_
    """
    if not isinstance(intensity_vector, torch.Tensor) and not isinstance(intensity_vector, np.ndarray):
        raise TypeError("intensity_vector must be torch.Tensor")

    # convert safely
    if isinstance(intensity_vector, torch.Tensor):
        x = intensity_vector.detach().cpu().numpy()
    else:
        x = intensity_vector

    if x.ndim != 1:
        raise ValueError("intensity_vector must be 1D")

    indices = np.arange(len(x))
    valid_mask = x >= 0
    invalid_mask = x < 0

    plt.figure(figsize=(12, 4))

    if allow_invalid:
        # plot valid
        plt.scatter(indices[valid_mask], x[valid_mask], s=15, label="Valid")

        # plot invalid
        plt.scatter(indices[invalid_mask], x[invalid_mask],
                    marker='x', s=20, label="Invalid")

        plt.ylim(-1.1, 1.05)
    else:
        plt.scatter(indices[valid_mask], x[valid_mask], s=15)
        plt.ylim(0, 1.05)

    plt.xlabel("Fragment Index")
    plt.ylabel("Normalized Intensity")
    plt.title("Fragment Intensity Vector")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    import h5py
    file_path = r"E:\Dai hoc\2526I\dacn\flow-matching\data\traintest_hcd.hdf5"
    with h5py.File(file_path, "r") as f:
        print("Keys:", list(f.keys()))
        intensities_raw = f["intensities_raw"][:1]
    
    print(type(intensities_raw[0]))
    plot_intensity(intensities_raw[0], allow_invalid=False)