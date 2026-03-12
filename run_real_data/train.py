import sys, os

# sys.path.append(r"E
# sys.path.append(r"E:\Dai hoc\2526I\dacn\flow-matching\demo-code\2d")
import h5py
from collections import defaultdict, Counter
import numpy as np
from rich import print
import torch

torch.set_default_device("cuda")
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
from tqdm.auto import tqdm
import random

from gen_path import get_xt, get_x0
from metrics import pcc, sa
from models import HCDFlowResMLP, HCDFlow
from utils import (
    plot_loss_history,
    create_batch_fragment_mask_from_peptide,
    masked_mse_loss,
)

from time import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

MAX_BATCH = 8096

train_path = os.getenv("TRAIN_PATH")
with h5py.File(train_path, "r") as f:
    print("Keys:", list(f.keys()))
    print("Start loading training data")
    seqs = f["sequence_integer"][:MAX_BATCH]
    charges_oh = f["precursor_charge_onehot"][:MAX_BATCH]
    intensities = f["intensities_raw"][:MAX_BATCH]
    print("Load data successfully")

print("Formatting Charges...")
charges = np.argmax(charges_oh, axis=1) + 1
del charges_oh

min_charge = 10
max_charge = 0

for charge in charges:
    min_charge = min(min_charge, charge)
    max_charge = max(charge, max_charge)

print(f"Min charge: {min_charge}")
print(f"Max charge: {max_charge}")
print("Formatting Charges successfully")

epoch = 100
batch_size = 256
model_layer = 4
pep_layer = 4

# model_path = r"E:\Dai hoc\2526I\dacn\flow-matching\run_real_data\checkpoints\tfmemb_adaln6_8e.pth"
model = HCDFlowResMLP(
    noise_dim=174,
    pep_dim=256,
    time_dim=128,
    charge_dim=9,
    num_blocks=model_layer,
    num_blocks_pep=pep_layer,
    min_charge=min_charge,
    max_charge=max_charge,
)
optimizer = torch.optim.AdamW(
    model.parameters(), eps=1e-8, lr=2e-4, weight_decay=2e-3
)

print(f"Train with: {epoch} epochs with batch size: {batch_size}")

print(
    f"Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

print(
    f"Num params of pep embedding: {
        sum(p.numel()
        for p in model.condition_embedding.parameters()
        if p.requires_grad)
        }"
)

loss_history = []
last_100_loss = []
validate_pcc_mask = []
validate_sa_mask = []
validate_pcc_nor = []
validate_sa_nor = []
validate_sa_rev_mask = []
validate_pcc_rev_mask = []

pbar = tqdm(range(int(epoch)), desc="Training")
num_samples = len(seqs)
num_batches = math.ceil(num_samples / batch_size)

print(f"Total batch for 1 epoch is: {num_batches}")
for ep in pbar:
    model.train()

    for b in range(num_batches):
        optimizer.zero_grad()

        start = b * batch_size
        end = min((b + 1) * batch_size, num_samples)

        batch_np_mask = create_batch_fragment_mask_from_peptide(
            seqs[start:end], charges[start:end], reshape=False
        )
        batch_mask = torch.tensor(batch_np_mask, dtype=torch.bool)
        batch_intensities = torch.tensor(
            intensities[start:end], dtype=torch.float64
        )

        batch_intensities[batch_intensities == -1] = 0
        batch_intensities = torch.logit(batch_intensities, eps=1e-3)

        batch_pep_seq = torch.tensor(seqs[start:end], dtype=torch.long)
        batch_charge = torch.tensor(
            charges[start:end], dtype=torch.long
        ).unsqueeze(1)

        noise = get_x0(x_1=batch_intensities)
        # print(noise.shape)
        t = torch.rand(end - start, 1)

        x_t = get_xt(x_0=noise, x_1=batch_intensities, t=t, sigma=1e-5)
        # print(x_t.shape)
        u_pred = model(x_t, t=t, pep_seq=batch_pep_seq, charge=batch_charge)

        loss = masked_mse_loss(u_pred, batch_intensities - noise, batch_mask)
        # loss = nn.MSELoss()(u_pred, batch_intensities - noise)

        loss.backward()
        optimizer.step()

        last_100_loss.append(loss.item())

        if len(last_100_loss) == 100:
            mean_last_100 = sum(last_100_loss) / 100
            last_100_loss.clear()
            loss_history.append(mean_last_100)

            pbar.set_postfix(
                {
                    "Last100": f"{mean_last_100:.4f}",
                    "Avg": f"{(sum(loss_history)/len(loss_history)):.4f}",
                }
            )
            if len(loss_history) % 10 == 0:
                print(
                    f"Avg loss from last 1000 batch: {(sum(loss_history[-10:-1])/10):.4f}"
                )

            # validate batch
            if len(loss_history) > 20 == 0:
                with torch.no_grad():  # validate batch
                    model.eval()

                    batch_mask = torch.tensor(batch_np_mask, dtype=torch.bool)
                    batch_intensities = torch.sigmoid(batch_intensities[0:32])
                    batch_pep_seq = batch_pep_seq[0:32]
                    batch_charge = batch_charge[0:32]
                    batch_mask = batch_mask[0:32]
                    noise = torch.randn_like(batch_intensities)

                    generated_batch = torch.sigmoid(
                        model.sample(noise, batch_pep_seq, batch_charge, step=6)
                    )
                    score_pcc_mask = pcc(
                        generated_batch, batch_intensities, batch_mask
                    )
                    score_sa_mask = sa(
                        generated_batch, batch_intensities, batch_mask
                    )
                    score_pcc_raw = pcc(generated_batch, batch_intensities)

                    score_sa_raw = sa(generated_batch, batch_intensities)

                    if len(loss_history) % 100 == 0:
                        print(
                            f"Score PCC Mask: {score_pcc_mask[0]:.4f},\n"
                            f"Score SA Mask: {score_sa_mask[0]:.4f},\n"
                            f"Score PCC Raw: {score_pcc_raw[0]:.4f},\n"
                            f"Score SA Raw: {score_sa_raw[0]:.4f},\n"
                        )

                    validate_pcc_nor.append(score_pcc_raw[0])
                    validate_sa_nor.append(score_sa_raw[0])
                    validate_pcc_mask.append(score_pcc_mask[0])
                    validate_sa_mask.append(score_sa_mask[0])
                model.train()


torch.save(
    model.state_dict(),
    f"{datetime.fromtimestamp(time()).isoformat(timespec="seconds")}_tfmemb_adalm_{model_layer}_{pep_layer}_{batch_size}_8e.pth",
)

plot_loss_history(loss_history)
plot_loss_history(validate_pcc_mask, "PCC_SCORE_MASK_mlp")
plot_loss_history(validate_sa_mask, "SA_SCORE_MASK_mlp")
plot_loss_history(validate_pcc_nor, "PCC_SCORE_NOR_mlp")
plot_loss_history(validate_sa_nor, "SA_SCORE_NOR_mlp")
