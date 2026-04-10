import os
import argparse
import h5py
import csv
import math
from tqdm.auto import tqdm
import torch
import numpy as np

# local imports (assumes same package layout as train_tfm.py)
from run_real_data.utils.gen_path import get_xt
from models import DiffusionFlow
from run_real_data.utils.utils import (
    process_intensity_vector,
    create_batch_fragment_mask_from_peptide,
)


def pcc_per_sample(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
    eps = 1e-8
    dims = tuple(range(1, x.dim()))
    if mask is not None:
        x = x * mask
        y = y * mask
        n = mask.sum(dim=dims, keepdim=True) + eps
        x_mean = x.sum(dim=dims, keepdim=True) / n
        y_mean = y.sum(dim=dims, keepdim=True) / n
        x_centered = (x - x_mean) * mask
        y_centered = (y - y_mean) * mask
    else:
        x_centered = x - x.mean(dim=dims, keepdim=True)
        y_centered = y - y.mean(dim=dims, keepdim=True)
    num = (x_centered * y_centered).sum(dim=dims)
    den = (
        torch.sqrt(
            (x_centered**2).sum(dim=dims) * (y_centered**2).sum(dim=dims)
        )
        + eps
    )
    return (num / den).cpu().numpy()


def sa_per_sample(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
    eps = 1e-8
    dims = tuple(range(1, x.dim()))
    if mask is not None:
        x = x * mask
        y = y * mask
    dot = (x * y).sum(dim=dims)
    n1 = torch.sqrt((x**2).sum(dim=dims))
    n2 = torch.sqrt((y**2).sum(dim=dims))
    return (dot / (n1 * n2 + eps)).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", default=os.getenv("TEST_PATH"), help="Path to dataset h5"
    )
    parser.add_argument(
        "--model", required=True, help="Path to saved model .pth"
    )
    parser.add_argument(
        "--out", default="test_scores.csv", help="Output CSV file"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    # parser.add_argument(
    #     "--holdout-fraction",
    #     type=float,
    #     default=0.1,
    #     help="Fraction taken as holdout from end if no explicit split",
    # )
    parser.add_argument("--sample-step", type=int, default=6)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    # build model (match train config if needed)
    model = DiffusionFlow(
        d_noise=6, d_model=256, num_layers=4, num_pep_layers=4
    )
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with h5py.File(args.h5, "r") as f:
        # For a separate test file, use the whole file as the test set
        num_samples = f["intensities_raw"].shape[0]
        indices = list(range(num_samples))

        # prepare CSV
        with open(args.out, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["index", "pcc", "sa"])

            for start in range(0, len(indices), args.batch_size):
                end = min(start + args.batch_size, len(indices))
                batch_idx = indices[start:end]

                # use slice indexing on the HDF5 datasets (faster and safe for last smaller batch)
                batch_np_int = f["intensities_raw"][start:end]
                batch_np_seq = f["sequence_integer"][start:end]
                batch_np_charges = (
                    np.argmax(f["precursor_charge_onehot"][start:end], axis=1)
                    + 1
                )
                batch_np_mask = create_batch_fragment_mask_from_peptide(
                    batch_np_seq, batch_np_charges
                )

                batch_int = torch.tensor(
                    process_intensity_vector(batch_np_int),
                    dtype=torch.float32,
                    device=device,
                )
                batch_seq = torch.tensor(
                    batch_np_seq, dtype=torch.long, device=device
                )
                batch_charge = torch.tensor(
                    batch_np_charges, dtype=torch.long, device=device
                ).unsqueeze(1)
                batch_mask = torch.tensor(
                    batch_np_mask, dtype=torch.bool, device=device
                )

                # sampling: start from noise as in training sampling call
                noise = torch.randn_like(batch_int, device=device)
                with torch.no_grad():
                    generated = model.sample(
                        noise, batch_seq, batch_charge, step=args.sample_step
                    )

                # compute per-sample metrics
                pcc_vals = pcc_per_sample(generated, batch_int, batch_mask)
                sa_vals = sa_per_sample(generated, batch_int, batch_mask)

                for idx_rel, idx in enumerate(batch_idx):
                    writer.writerow(
                        [idx, float(pcc_vals[idx_rel]), float(sa_vals[idx_rel])]
                    )

    print(f"Saved test scores to {args.out}")


if __name__ == "__main__":
    main()
