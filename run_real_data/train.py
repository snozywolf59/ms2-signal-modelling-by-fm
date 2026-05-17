"""
train.py
────────
Training script chính. Mọi hyperparameter và preprocessing mode
đều đặt trong config.py — không cần sửa file này.

Chạy:
    python train.py
"""

import math
import os
from datetime import datetime
from time import time

import h5py
import numpy as np
import torch
from rich import print
from tqdm.auto import tqdm

import config as C
from preprocess import Preprocessor, PreprocessMode
from models import DiffusionFlow, HCDFlowResMLP, HCDFlow
from utils.gen_path import get_xt
from utils.metrics import pcc, sa
from utils.utils import (
    create_batch_fragment_mask_from_peptide,
    masked_mse_loss,
    plot_loss_history,
    process_intensity_vector,
)

# ────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

preprocessor = Preprocessor(
    mode=C.PREPROCESS_MODE,
    logit_eps=C.LOGIT_EPS,
    sphere_eps=C.SPHERE_EPS,
)
print(f"[bold cyan]Preprocessor:[/bold cyan] {preprocessor}")
print(f"[bold cyan]Device:[/bold cyan] {device}")


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def load_charges(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if C.TRAIN_SAMPLE_SIZE and C.TRAIN_SAMPLE_SIZE > 0:
            charges_oh = f["precursor_charge_onehot"][:C.TRAIN_SAMPLE_SIZE]
        else:
            charges_oh = f["precursor_charge_onehot"][:]
    return np.argmax(charges_oh, axis=1) + 1


def build_batch(
    f: h5py.File,
    charges: np.ndarray,
    start: int,
    end: int,
    reshape: bool = True,
) -> dict:
    """Load một batch từ HDF5, trả về dict tensor đã ở trên device."""
    raw_intensities = f["intensities_raw"][start:end]
    raw_seqs = f["sequence_integer"][start:end]
    batch_charges = charges[start:end]

    mask_np = create_batch_fragment_mask_from_peptide(raw_seqs, batch_charges, reshape=reshape)

    # intensity: [0,1] → encode theo PREPROCESS_MODE
    intensity_01 = torch.tensor(
        process_intensity_vector(raw_intensities, reshape), dtype=torch.float64
    )
    intensity_latent = preprocessor.encode(intensity_01)

    return {
        "intensity_latent": intensity_latent,
        "intensity_01": intensity_01,
        "pep_seq": torch.tensor(raw_seqs, dtype=torch.long),
        "charge": torch.tensor(batch_charges, dtype=torch.long).unsqueeze(1),
        "mask": torch.tensor(mask_np, dtype=torch.bool),
    }


def compute_flow_target(
    noise: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tính x_t và velocity target tuỳ theo preprocessing mode.

    Euclidean (raw/logit):
        x_t  = (1 - t) * noise + t * x1   [linear interpolation]
        u*   = x1 - noise                  [constant vector field]

    Sphere:
        x_t  = SLERP(noise, x1, t)        [geodesic interpolation]
        u*   = Log_{x_t}(x1) / (1-t)      [tangent vector field]
    """
    if preprocessor.mode == PreprocessMode.SPHERE:
        x_t, target = preprocessor.sphere_target_vector(noise, x1, t)
    else:
        x_t = get_xt(noise, x1, t, sigma=C.SIGMA)
        target = x1 - noise
    return x_t, target


# ────────────────────────────────────────────────────────────
# Model & Optimizer
# ────────────────────────────────────────────────────────────
# model = HCDFlowResMLP(
#     noise_dim=174,
#     pep_dim=256,
#     time_dim=128,
#     charge_dim=9,
#     num_blocks=C.MODEL_LAYERS,
#     num_blocks_pep=C.PEP_LAYERS,
#     min_charge=1,
#     max_charge=6,
# )

model = HCDFlow(
    noise_dim=174,
    pep_dim=C.D_MODEL,
    time_dim=128, charge_dim=8)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=C.LR,
    eps=C.ADAM_EPS,
    weight_decay=C.WEIGHT_DECAY,
)

print(
    f"[bold]Model params:[/bold] {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)
print(
    f"[bold]Pep embedding params:[/bold] "
    f"{sum(p.numel() for p in model.condition_embedding.parameters() if p.requires_grad):,}"
)


# ────────────────────────────────────────────────────────────
# Tracking
# ────────────────────────────────────────────────────────────
loss_history: list[float] = []
rolling_buffer: list[float] = []

metrics: dict[str, list[float]] = {
    "pcc_mask": [],
    "sa_mask": [],
    "pcc_raw": [],
    "sa_raw": [],
}

# ────────────────────────────────────────────────────────────
# Validation (tách hàm để train loop gọn)
# ────────────────────────────────────────────────────────────
def _run_validation(
    model: torch.nn.Module,
    last_batch: dict,
    prep: Preprocessor,
    metrics: dict,
    n_logs: int,
):
    model.eval()
    with torch.no_grad():
        n = C.VALIDATE_BATCH_SIZE
        intensity_01 = last_batch["intensity_01"][:n]
        pep_seq = last_batch["pep_seq"][:n]
        charge = last_batch["charge"][:n]
        mask = last_batch["mask"][:n]

        # Noise phù hợp với mode
        noise = torch.randn_like(intensity_01)
        if prep.mode == PreprocessMode.SPHERE:
            noise = prep.encode(torch.sigmoid(noise))

        # Sample từ model (trả về latent)
        gen_latent = model.sample(noise, pep_seq, charge, step=C.ODE_STEPS)
        gen_01 = prep.decode(gen_latent)

        score_pcc_mask = pcc(gen_01, intensity_01, mask)
        score_sa_mask = sa(gen_01, intensity_01, mask)
        # score_pcc_raw = pcc(gen_01, intensity_01)
        # score_sa_raw = sa(gen_01, intensity_01)

        metrics["pcc_mask"].append(score_pcc_mask[0])
        metrics["sa_mask"].append(score_sa_mask[0])
        # metrics["pcc_raw"].append(score_pcc_raw[0])
        # metrics["sa_raw"].append(score_sa_raw[0])

        if n_logs % C.PRINT_SCORE_EVERY_N_LOGS == 0:
            print(
                f"\n[cyan]── Validation (log #{n_logs}) ──[/cyan]\n"
                f"  PCC Mask : {score_pcc_mask[0]:.4f}\n"
                f"  SA  Mask : {score_sa_mask[0]:.4f}\n"
                # f"  PCC Raw  : {score_pcc_raw[0]:.4f}\n"
                # f"  SA  Raw  : {score_sa_raw[0]:.4f}"
            )
    model.train()


# ────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────
charges = load_charges(C.TRAIN_PATH)
num_samples = len(charges)
num_batches = math.ceil(num_samples / C.BATCH_SIZE)

print(
    f"[bold]Dataset:[/bold] {num_samples:,} samples | "
    f"{num_batches} batches/epoch | mode=[yellow]{C.PREPROCESS_MODE}[/yellow]"
)

pbar = tqdm(range(C.EPOCHS), desc="Epoch")
start_time = time()

with h5py.File(C.TRAIN_PATH, "r") as f:
    for ep in pbar:
        model.train()

        for b in range(num_batches):
            optimizer.zero_grad()

            start = b * C.BATCH_SIZE
            end = min((b + 1) * C.BATCH_SIZE, num_samples)

            batch = build_batch(f, charges, start, end, False)
            x1 = batch["intensity_latent"]
            noise = torch.randn_like(x1)

            # # Với sphere mode, noise phải nằm trên S^(d-1)
            # if preprocessor.mode == PreprocessMode.SPHERE:
            #     noise = preprocessor.encode(torch.sigmoid(noise))

            t = torch.rand(end - start, 1)
            x_t, target = compute_flow_target(noise, x1, t, batch)

            u_pred = model(
                x_t,
                t,
                batch["pep_seq"],
                batch["charge"],
            )
            loss = masked_mse_loss(u_pred, target, batch["mask"])
            loss.backward()
            optimizer.step()

            rolling_buffer.append(loss.item())

            # ── Log mỗi LOG_EVERY_N_BATCHES ──────────────────
            if len(rolling_buffer) >= C.LOG_EVERY_N_BATCHES:
                mean_loss = sum(rolling_buffer) / len(rolling_buffer)
                rolling_buffer.clear()
                loss_history.append(mean_loss)

                avg_loss = sum(loss_history) / len(loss_history)
                pbar.set_postfix(
                    {"rolling": f"{mean_loss:.4f}", "avg": f"{avg_loss:.4f}"}
                )

                # ── Validate ──────────────────────────────────
                n_logs = len(loss_history)
                if n_logs % C.VALIDATE_EVERY_N_LOGS == 0:
                    _run_validation(model, batch, preprocessor, metrics, n_logs)

# ────────────────────────────────────────────────────────────
# Save & Plot
# ────────────────────────────────────────────────────────────
end_time = time()
print(f"[bold green]Training time: {end_time - start_time:.1f}s[/bold green]")

ckpt_name = (
    f"MLP_{datetime.fromtimestamp(end_time)}_"
    f"fm_{C.PREPROCESS_MODE}_{C.MODEL_LAYERS}l_"
    f"bs{C.BATCH_SIZE}_{C.EPOCHS}e.pth"
)
torch.save(model.state_dict(), ckpt_name)
print(f"[bold]Saved:[/bold] {ckpt_name}")

plot_loss_history(loss_history)
for key, vals in metrics.items():
    if vals:
        plot_loss_history(vals, f"{key.upper()}", f"MLP_{key.upper()}_{C.PREPROCESS_MODE}")
