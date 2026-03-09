import sys, os

# sys.path.append(r"E
# sys.path.append(r"E:\Dai hoc\2526I\dacn\flow-matching\demo-code\2d")
import h5py
from collections import defaultdict, Counter
import numpy as np
from rich import print
import torch
from dotenv import load_dotenv

torch.set_default_device("cuda")
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
from tqdm.auto import tqdm
from time import time
from datetime import datetime

from gen_path import get_xt
from metrics import pcc, sa
from models import DiffusionFlow
from utils import (
    plot_loss_history,
    create_batch_fragment_mask_from_peptide,
    masked_mse_loss,
    process_intensity_vector,
)

load_dotenv()

train_path = os.getenv("TRAIN_PATH")
with h5py.File(train_path, "r") as f:
    print("Keys:", list(f.keys()))

    # seqs = f["sequence_integer"][:]
    # intensities = f["intensities_raw"][:]
    charges_oh = f["precursor_charge_onehot"][:]


charges = np.argmax(charges_oh, axis=1) + 1
del charges_oh

min_charge = 10
max_charge = 0

for charge in charges:
    min_charge = min(min_charge, charge)
    max_charge = max(charge, max_charge)

print(f"Min charge: {min_charge}")
print(f"Max charge: {max_charge}")

epoch = 6
batch_size = 512
model_layer = 4
pep_layer = 4

# model_path = r"E:\Dai hoc\2526I\dacn\flow-matching\run_real_data\checkpoints\tfmemb_adaln6_8e.pth"
model = DiffusionFlow(
    d_noise=6, d_model=256, num_layers=model_layer, num_pep_layers=model_layer
)
optimizer = torch.optim.AdamW(
    model.parameters(), eps=1e-8, lr=3e-4, weight_decay=5e-3
)
# model.load_state_dict(torch.load(model_path))

print(
    f"Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

print(
    f"Num params of pep embedding: {
        sum(p.numel()
        for p in model.peptide_embedding.parameters()
        if p.requires_grad)
        }"
)

loss_history = []
last_100_loss = []


pbar = tqdm(range(int(epoch)), desc="Training")
num_samples = len(charges)
num_batches = math.ceil(num_samples / batch_size)

validate_pcc = []
validate_sa = []
start_time = time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with h5py.File(train_path, "r") as f:
    for ep in pbar:
        model.train()

        for b in range(num_batches):
            optimizer.zero_grad()
            start = b * batch_size
            end = min((b + 1) * batch_size, num_samples)

            batch_np_intensities = f["intensities_raw"][start:end]
            batch_np_seqs = f["sequence_integer"][start:end]
            batch_np_charges = charges[start:end]

            batch_np_mask = create_batch_fragment_mask_from_peptide(
                batch_np_seqs, batch_np_charges
            )

            batch_intensities = torch.tensor(
                process_intensity_vector(batch_np_intensities),
                dtype=torch.float32,
            )
            batch_pep_seq = torch.tensor(batch_np_seqs, dtype=torch.long)
            batch_charge = torch.tensor(
                batch_np_charges, dtype=torch.long
            ).unsqueeze(1)

            batch_mask = torch.tensor(batch_np_mask, dtype=torch.bool)

            noise = torch.randn_like(batch_intensities)
            t = torch.rand(end - start, 1)

            x_t = get_xt(noise, batch_intensities, t, sigma=1e-4)
            u_pred = model(
                noise=x_t, time=t, pep=batch_pep_seq, charge=batch_charge
            )

            loss = masked_mse_loss(
                u_pred, batch_intensities - noise, batch_mask
            )
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
                    with torch.no_grad():  # validate batch
                        model.eval()
                        batch_intensities = batch_intensities[0:32]
                        batch_pep_seq = batch_pep_seq[0:32]
                        batch_charge = batch_charge[0:32]
                        batch_mask = batch_mask[0:32]
                        noise = torch.randn_like(batch_intensities)

                        generated_batch = model.sample(
                            noise, batch_pep_seq, batch_charge, step=6
                        )
                        score_pcc = pcc(generated_batch, batch_intensities, batch_mask)
                        score_sa = sa(generated_batch, batch_intensities, batch_mask)
                        if len(loss_history ) % 100 == 0:
                            print(score_pcc)
                        validate_pcc.append(score_pcc[0])
                        validate_sa.append(score_sa[0])
                    model.train()
                if len(loss_history) % 100 == 0:
                    print(
                        f"Avg loss from last 1000 batch: {(sum(loss_history[-10:-1])/10):.4f}"
                    )

print(f"Total training time: {time() - start_time} ms")
torch.save(
    model.state_dict(),
    f"{datetime.fromtimestamp(time())}_tfm_diffusion_{model_layer}_{pep_layer}_{batch_size}_{epoch}e.pth",
)

plot_loss_history(loss_history)
plot_loss_history(validate_pcc, "PCC_SCORE_tfm")
plot_loss_history(validate_sa, "SA_SCORE_tfm")
