"""Configuration for training and evaluation on real data.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Data Path
TRAIN_PATH: str = os.getenv("TRAIN_PATH", "data/train.h5")
TEST_PATH: str = os.getenv("TEST_PATH", "data/test.h5")

# Preprocessing for spectra
# "raw" | "logit"
PREPROCESS_MODE: str = "raw"

# use for logit mode: clamp intensity in [LOGIT_EPS, 1 - LOGIT_EPS]
LOGIT_EPS: float = 1e-4

# use for sphere projection: work in progress
SPHERE_EPS: float = 1e-8

# Model architecture params
D_NOISE: int = 6
D_MODEL: int = 256
MODEL_LAYERS: int = 4
PEP_LAYERS: int = 4

# Training params
TRAIN_SAMPLE_SIZE: int = -1

# CFG Scaling - in progress
GUIDANCE_SCALE: float = 3.0
COND_DROP_PROB: float = 0.1

# training configuration

EPOCHS: int = 8
BATCH_SIZE: int = 512
LR: float = 3e-4
WEIGHT_DECAY: float = 1e-2
ADAM_EPS: float = 1e-8

# Flow matching sigma (noise schedule)
SIGMA: float = 1e-2

# ODE steps when sampling
ODE_STEPS: int = 6

# logging while training config
LOG_EVERY_N_BATCHES: int = 100

# Validation config, by pcc and sa on latent and true spectra
VALIDATE_EVERY_N_LOGS: int = 50

PRINT_SCORE_EVERY_N_LOGS: int = 1

VALIDATE_BATCH_SIZE: int = 16
