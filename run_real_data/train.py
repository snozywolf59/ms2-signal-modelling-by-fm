import sys
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

from gen_path import get_xt
from models import HCDFlowResMLP, HCDFlow
from utils import plot_loss_history

train_path = r"E:\Dai hoc\2526I\dacn\flow-matching\data\traintest_hcd.hdf5"
with h5py.File(train_path, "r") as f:
    print("Keys:", list(f.keys()))

    seqs = f["sequence_integer"][:]
    charges_oh = f["precursor_charge_onehot"][:]
    intensities = f["intensities_raw"][:]  
    
charges = np.argmax(charges_oh, axis=1) + 1
del charges_oh

min_charge = 10
max_charge = 0

for charge in charges:
    min_charge = min(min_charge, charge)
    max_charge = max(charge, max_charge)
    
print(f"Min charge: {min_charge}")
print(f"Max charge: {max_charge}")

epoch = 8
batch_size = 256
# model_path = r"E:\Dai hoc\2526I\dacn\flow-matching\run_real_data\checkpoints\tfmemb_adaln6_8e.pth"
model = HCDFlowResMLP(noise_dim=174, pep_dim=256, time_dim=128, charge_dim=9, num_blocks=8, min_charge=min_charge, max_charge=max_charge)
optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, lr=2e-4,weight_decay=1e-3)
# model.load_state_dict(torch.load(model_path))

print(f"Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
loss_history = []
last_100_loss = []


pbar = tqdm(range(int(epoch)), desc="Training")
num_samples = len(seqs)
num_batches = math.ceil(num_samples / batch_size)

for ep in pbar:
    model.train()
    
    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, num_samples)

        batch_intensities = torch.tensor(
            intensities[start:end], dtype=torch.float32
        )
        batch_pep_seq = torch.tensor(
            seqs[start:end], dtype=torch.long
        )
        batch_charge = torch.tensor(
            charges[start:end], dtype=torch.long
        ).unsqueeze(1)

        cur_bs = batch_intensities.shape[0]

        noise = torch.randn_like(batch_intensities)
        t = torch.rand(cur_bs, 1)

        x_t = get_xt(batch_intensities, noise, t)
        u_pred = model(x_t, t=t, pep_seq=batch_pep_seq, charge=batch_charge)

        loss = nn.MSELoss()(u_pred, batch_intensities - noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_100_loss.append(loss.item())

        if len(last_100_loss) == 100:
            mean_last_100 = sum(last_100_loss) / 100
            last_100_loss.clear()
            loss_history.append(mean_last_100)

            pbar.set_postfix({
                "Last100": f"{mean_last_100:.4f}",
                "Avg": f"{(sum(loss_history)/len(loss_history)):.4f}",
            })
            if len(loss_history) % 100 == 0:
                print(f"Avg loss from last 1000 batch: {(sum(loss_history[-10:-1])/10):.4f}")

torch.save(model.state_dict(), "tfmemb_adalm8_512_8e.pth")

plot_loss_history(loss_history)