"""
=============================================================
  CONFIG — chỉnh tất cả hyperparameter và preprocessing ở đây
=============================================================

PREPROCESS_MODE choices:
  "raw"       — giữ nguyên intensity [0,1], không transform
  "logit"     — logit transform: log(x/(1-x)), đưa về R
  "sphere"    — L2-normalize lên unit sphere S^(d-1) trong R^d
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Data ───────────────────────────────────────────────────
TRAIN_PATH: str = os.getenv("TRAIN_PATH", "data/train.h5")

# ─── Preprocessing ──────────────────────────────────────────
# Chọn một trong: "raw" | "logit" | "sphere"
PREPROCESS_MODE: str = "logit"

# Dùng cho mode "logit"
LOGIT_EPS: float = 1e-4

# Dùng cho mode "sphere": clamp trước khi normalize
SPHERE_EPS: float = 1e-8

# ─── Model ──────────────────────────────────────────────────
D_NOISE: int = 6
D_MODEL: int = 256
MODEL_LAYERS: int = 4
PEP_LAYERS: int = 4

# ─── Training ───────────────────────────────────────────────
TRAIN_SAMPLE_SIZE: int = 256000

EPOCHS: int = 10
BATCH_SIZE: int = 256
LR: float = 2e-4
WEIGHT_DECAY: float = 2e-3
ADAM_EPS: float = 1e-8

# Flow matching sigma (noise schedule)
SIGMA: float = 1e-5

# Bước ODE khi sample
ODE_STEPS: int = 10

# ─── Logging ────────────────────────────────────────────────
# Tổng hợp loss mỗi bao nhiêu batch
LOG_EVERY_N_BATCHES: int = 100

# Validate mỗi bao nhiêu lần log
VALIDATE_EVERY_N_LOGS: int = 1

# In score ra console mỗi bao nhiêu lần log
PRINT_SCORE_EVERY_N_LOGS: int = 100

# Số sample dùng cho validation
VALIDATE_BATCH_SIZE: int = 32
