import os
import sys
import math
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning as L  # optional
except Exception:
    L = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

ROUTES = ["L", "N", "I", "LN", "LI", "NI", "LNI"]
BLOCKS = {"uni": ["L", "N", "I"], "bi": ["LN", "LI", "NI"], "tri": ["LNI"]}
TASKS  = ["mort", "pe", "ph"]  

@dataclass
class Config:
    # Model / training
    d: int = 256
    alpha: float = 4.0
    dropout: float = 0.1
    lr: float = 2e-4
    batch_size: int = 16
    max_epochs_uni: int = 5
    max_epochs_bi: int = 5
    max_epochs_tri: int = 5
    num_workers: int = 4
    ema_decay: float = 0.98
    steps_per_epoch_hint: int = 500

    # Text
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_text_len: int = 512

    # Structured seq
    structured_seq_len: int = 24
    structured_n_feats: int = 128

    # Image
    image_model_name: str = "resnet34"

    # Fairness
    sensitive_keys: List[str] = field(
        default_factory=lambda: ["age_group", "race", "ethnicity", "insurance"]
    )
    lambda_fair: float = 0.0  

    # Routing backend
    routing_backend: str = "learned_gate"

    # Paths (edit to your data locations)
    data_root: str = "./data"
    ckpt_root: str = "./checkpoints"

    task_name: str = "mort"

    use_cudnn_benchmark: bool = True

    precision_amp: str = "auto"


CFG = Config()
print(CFG)

TASK2IDX: Dict[str, int] = {name: i for i, name in enumerate(TASKS)}
SELECTED_TASK_IDX: int = TASK2IDX.get(CFG.task_name, 0)  

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = bool(CFG.use_cudnn_benchmark)

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Freeze/unfreeze a module in-place."""
    for p in module.parameters():
        p.requires_grad = flag

def is_cuda_device(dev) -> bool:
    return (
        torch.cuda.is_available()
        and (
            (isinstance(dev, torch.device) and dev.type == "cuda")
            or (isinstance(dev, str) and "cuda" in dev)
        )
    )

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
