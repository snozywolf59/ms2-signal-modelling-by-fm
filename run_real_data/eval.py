"""
evaluate.py
───────────
Eval model trên tập holdout theo 3 phương pháp:
  • single-sample : 1 lần sample / peptide
  • best-of-K     : sample K lần, chọn PCC cao nhất
  • mean          : trung bình K sample rồi tính metric

Kết quả được báo cáo theo:
  • Tổng thể (overall)
  • Độ dài peptide  : [1-10] | [11-20] | [>20]
  • Điện tích       : [1]    | [2-4]   | [≥5]
  • Số mẫu/peptide  : [1-3]  | [4-10]  | [11-20] | [>20]

Chỉnh CONFIG bên dưới rồi chạy:
    python evaluate.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from time import time
from typing import NamedTuple

import h5py
import numpy as np
import torch
from rich import print
from rich.table import Table
from rich.console import Console
from tqdm.auto import tqdm

import config as C
from models import HCDFlow, HCDFlowResMLP
from preprocess import Preprocessor, PreprocessMode
from utils.metrics import pcc, sa
from utils.utils import (
    create_batch_fragment_mask_from_peptide,
    process_intensity_vector,
)

console = Console()

# ════════════════════════════════════════════════════════════
# CONFIG  ← chỉnh ở đây
# ════════════════════════════════════════════════════════════
HOLDOUT_PATH = r"E:\Dai hoc\2526I\dacn\flow-matching\data\holdout_hcd.hdf5"
MODEL_PATH   = ""            # đường dẫn tới file .pth

K            = 8             # số sample cho best-of-K và mean
EVAL_BATCH   = 64            # batch size khi chạy inference
ODE_STEPS    = C.ODE_STEPS
# ════════════════════════════════════════════════════════════

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

preprocessor = Preprocessor(
    mode=C.PREPROCESS_MODE,
    logit_eps=C.LOGIT_EPS,
    sphere_eps=C.SPHERE_EPS,
)
print(f"[bold cyan]Device:[/bold cyan]       {device}")
print(f"[bold cyan]Preprocessor:[/bold cyan] {preprocessor}")


# ────────────────────────────────────────────────────────────
# Load & group data
# ────────────────────────────────────────────────────────────
print(f"\n[bold]Loading holdout:[/bold] {HOLDOUT_PATH}")
with h5py.File(HOLDOUT_PATH, "r") as f:
    seqs        = f["sequence_integer"][:]
    intensities = np.array(f["intensities_raw"][:], dtype=np.float64)
    charge_oh   = f["precursor_charge_onehot"][:]

charges  = np.argmax(charge_oh, axis=1) + 1
seq_lens = np.count_nonzero(seqs, axis=1)
N, _     = intensities.shape

# group: key=(seq_tuple, charge) → list of replicate intensities
groups: dict[tuple, list[np.ndarray]] = defaultdict(list)
for i in range(N):
    ln = int(seq_lens[i])
    if ln == 0:
        continue
    key = (tuple(int(x) for x in seqs[i, :ln]), int(charges[i]))
    groups[key].append(intensities[i])
    if (i + 1) % 50_000 == 0:
        print(f"  grouped {i+1:,} / {N:,}")

# Flatten thành arrays
keys_list    = list(groups.keys())
n_unique     = len(keys_list)
n_replicates = np.array([len(v) for v in groups.values()], dtype=np.int64)

# Ground-truth = mean của các replicate
gt_arr = np.stack(
    [np.mean(v, axis=0).astype(np.float64) for v in groups.values()], axis=0
)  # (n_unique, 174)

seq_arr    = np.zeros((n_unique, seqs.shape[1]), dtype=seqs.dtype)
charge_arr = np.zeros(n_unique, dtype=np.int64)
len_arr    = np.zeros(n_unique, dtype=np.int64)

for idx, (seq_tup, ch) in enumerate(keys_list):
    ln = len(seq_tup)
    seq_arr[idx, :ln] = seq_tup
    charge_arr[idx]   = ch
    len_arr[idx]      = ln

print(f"[bold]Unique peptides:[/bold] {n_unique:,}  (raw rows: {N:,})")


# ────────────────────────────────────────────────────────────
# Grouping masks  (bool arrays, shape=(n_unique,))
# ────────────────────────────────────────────────────────────
GROUPS: dict[str, dict[str, np.ndarray]] = {
    "Peptide length": {
        "1–10" : (len_arr >= 1)  & (len_arr <= 10),
        "11–20": (len_arr >= 11) & (len_arr <= 20),
        ">20"  : (len_arr > 20),
    },
    "Charge": {
        "1"  : charge_arr == 1,
        "2–4": (charge_arr >= 2) & (charge_arr <= 4),
        "≥5" : charge_arr >= 5,
    },
    "# replicates": {
        "1–3"  : (n_replicates >= 1)  & (n_replicates <= 3),
        "4–10" : (n_replicates >= 4)  & (n_replicates <= 10),
        "11–20": (n_replicates >= 11) & (n_replicates <= 20),
        ">20"  : n_replicates > 20,
    },
}


# ────────────────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────────────────
# Đổi sang HCDFlowResMLP nếu cần
model = HCDFlow(noise_dim=174, pep_dim=C.D_MODEL, time_dim=128, charge_dim=8)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"[bold]Model params:[/bold] {sum(p.numel() for p in model.parameters()):,}")


# ────────────────────────────────────────────────────────────
# Core inference — 1 pass, thu scores per-peptide cho cả 3 method
# ────────────────────────────────────────────────────────────
class PerPeptideScores(NamedTuple):
    single_pcc: np.ndarray   # (n_unique,)
    single_sa:  np.ndarray
    bok_pcc:    np.ndarray
    bok_sa:     np.ndarray
    mean_pcc:   np.ndarray
    mean_sa:    np.ndarray


def _per_item(metric_fn, gen, gt, mask) -> list[float]:
    """
    Gọi metric_fn(gen, gt, mask) và luôn trả về list float per-peptide.
    Xử lý cả 2 trường hợp: [1] là tensor (batch>1) hoặc float (batch=1).
    """
    result = metric_fn(gen, gt, mask)[1]
    if isinstance(result, (float, int)):
        return [float(result)]
    if isinstance(result, torch.Tensor):
        return result.tolist()
    return list(result)


def _make_noise(bs: int) -> torch.Tensor:
    noise = torch.randn(bs, 174, dtype=torch.float64, device=device)
    if preprocessor.mode == PreprocessMode.SPHERE:
        noise = preprocessor.encode(torch.sigmoid(noise))
    return noise


def collect_scores() -> PerPeptideScores:
    """
    Chạy inference 1 lần duy nhất qua toàn bộ tập test.
    Với mỗi batch:
      - 1 noise draw  → single-sample score
      - K noise draws → shared cho best-of-K (argmax PCC) và mean
    """
    single_pcc_l, single_sa_l = [], []
    bok_pcc_l,    bok_sa_l    = [], []
    mean_pcc_l,   mean_sa_l   = [], []

    num_batches = math.ceil(n_unique / EVAL_BATCH)

    for b in tqdm(range(num_batches), desc="Inference"):
        s, e   = b * EVAL_BATCH, min((b + 1) * EVAL_BATCH, n_unique)
        bs_cur = e - s

        pep_np  = seq_arr[s:e]
        ch_np   = charge_arr[s:e]
        gt_np   = gt_arr[s:e]

        mask_np  = create_batch_fragment_mask_from_peptide(pep_np, ch_np, reshape=False)
        gt_01    = torch.tensor(
            process_intensity_vector(gt_np, reshape=False), dtype=torch.float64
        )
        mask_t   = torch.tensor(mask_np, dtype=torch.bool)
        pep_seq  = torch.tensor(pep_np,  dtype=torch.long)
        charge_t = torch.tensor(ch_np,   dtype=torch.long).unsqueeze(1)

        # ── Single-sample ────────────────────────────────────
        with torch.no_grad():
            gen_latent = model.sample(_make_noise(bs_cur), pep_seq, charge_t, step=ODE_STEPS)
        single_gen = preprocessor.decode(gen_latent)
        single_pcc_l.extend(_per_item(pcc, single_gen, gt_01, mask_t))
        single_sa_l.extend( _per_item(sa,  single_gen, gt_01, mask_t))

        # ── K samples (shared cho BoK + Mean) ────────────────
        samples: list[torch.Tensor] = []
        sum_gen = torch.zeros(bs_cur, 174, dtype=torch.float64, device=device)

        for _ in range(K):
            with torch.no_grad():
                gen_latent = model.sample(_make_noise(bs_cur), pep_seq, charge_t, step=ODE_STEPS)
            gen = preprocessor.decode(gen_latent)
            samples.append(gen)
            sum_gen += gen

        # Mean
        mean_gen = (sum_gen / K).clamp(0.0, 1.0)
        mean_pcc_l.extend(_per_item(pcc, mean_gen, gt_01, mask_t))
        mean_sa_l.extend( _per_item(sa,  mean_gen, gt_01, mask_t))

        # Best-of-K: chọn sample có PCC cao nhất per peptide
        pcc_matrix = torch.stack(
            [torch.tensor(_per_item(pcc, s_, gt_01, mask_t)) for s_ in samples], dim=0
        )  # (K, bs_cur)
        best_idx = pcc_matrix.argmax(dim=0)   # (bs_cur,)

        for i in range(bs_cur):
            best  = samples[best_idx[i]][i].unsqueeze(0)
            gt_i  = gt_01[i].unsqueeze(0)
            msk_i = mask_t[i].unsqueeze(0)
            bok_pcc_l.extend(_per_item(pcc, best, gt_i, msk_i))
            bok_sa_l.extend( _per_item(sa,  best, gt_i, msk_i))

    return PerPeptideScores(
        single_pcc = np.array(single_pcc_l),
        single_sa  = np.array(single_sa_l),
        bok_pcc    = np.array(bok_pcc_l),
        bok_sa     = np.array(bok_sa_l),
        mean_pcc   = np.array(mean_pcc_l),
        mean_sa    = np.array(mean_sa_l),
    )


# ────────────────────────────────────────────────────────────
# Report helpers
# ────────────────────────────────────────────────────────────
def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _compute_row(
    scores: PerPeptideScores,
    mask: np.ndarray | None = None,
) -> tuple[str, str, str, str, str, str, str]:
    """Trả về (N, single_pcc, single_sa, bok_pcc, bok_sa, mean_pcc, mean_sa) cho subset."""
    idx = mask if mask is not None else np.ones(n_unique, dtype=bool)
    n   = idx.sum()
    if n == 0:
        return ("0", "—", "—", "—", "—", "—", "—")
    return (
        str(n),
        _fmt(scores.single_pcc[idx].mean()),
        _fmt(scores.single_sa[idx].mean()),
        _fmt(scores.bok_pcc[idx].mean()),
        _fmt(scores.bok_sa[idx].mean()),
        _fmt(scores.mean_pcc[idx].mean()),
        _fmt(scores.mean_sa[idx].mean()),
    )


def _build_table(
    title: str,
    col_label: str,
    group_items: dict[str, np.ndarray],
    scores: PerPeptideScores,
) -> Table:
    t = Table(title=title, show_lines=True)
    t.add_column(col_label,        style="bold",   justify="left")
    t.add_column("N",              style="dim",    justify="right")
    t.add_column("Single PCC",                     justify="right")
    t.add_column("Single SA",                      justify="right")
    t.add_column(f"BoK-{K} PCC",                   justify="right")
    t.add_column(f"BoK-{K} SA",                    justify="right")
    t.add_column(f"Mean-{K} PCC",                  justify="right")
    t.add_column(f"Mean-{K} SA",                   justify="right")

    # Overall
    t.add_row("Overall", *_compute_row(scores), style="bold yellow")

    for label, mask in group_items.items():
        t.add_row(label, *_compute_row(scores, mask))

    return t


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time()

    print(f"\n[bold yellow]Running inference  (K={K}, ODE steps={ODE_STEPS}) …[/bold yellow]")
    scores = collect_scores()

    # ── Overall summary table ────────────────────────────────
    ov = Table(
        title=f"\nOverall Results  (K={K}, ODE steps={ODE_STEPS}, n={n_unique:,})",
        show_lines=True,
    )
    ov.add_column("Method",     style="bold cyan", justify="left")
    ov.add_column("PCC ↑",      justify="right")
    ov.add_column("SA  ↑",      justify="right")
    ov.add_row("Single-sample",
               _fmt(scores.single_pcc.mean()), _fmt(scores.single_sa.mean()))
    ov.add_row(f"Best-of-{K}",
               _fmt(scores.bok_pcc.mean()),    _fmt(scores.bok_sa.mean()))
    ov.add_row(f"Mean ({K})",
               _fmt(scores.mean_pcc.mean()),   _fmt(scores.mean_sa.mean()))
    console.print(ov)

    # ── Grouped tables ───────────────────────────────────────
    for group_name, group_items in GROUPS.items():
        tbl = _build_table(
            title       = f"\nGrouped by: {group_name}",
            col_label   = group_name,
            group_items = group_items,
            scores      = scores,
        )
        console.print(tbl)

    print(f"\n[bold green]Total eval time: {time() - t0:.1f}s[/bold green]")
