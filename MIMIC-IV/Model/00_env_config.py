# This notebook sets up the environment and shared configuration for the **MultimodalRouting** project.
## **Modalities:** L (structured labs/vitals), N (notes), I (images)  
## **Tasks:** mortality, pulmonary embolism (PE), pulmonary hypertension (PH)

## It also defines the route/block taxonomy and a few helper utilities used across other notebooks.

## **Routing backends:**
### - `"ema_logits"` — 7 route **logit** heads + task-wise EMA loss→weight routing (two-stage: route→block).
### - `"capsule_features"` — 7 route **feature** heads + **capsule** dynamic routing from route features to task concepts (per-sample couplings).

import os, sys, math, json, random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning as L
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
BLOCKS = {"uni": ["L","N","I"], "bi": ["LN","LI","NI"], "tri": ["LNI"]}
TASKS = ["mort", "pe", "ph"]

@dataclass
class Config:
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
    image_model_name: str = "resnet50"         
    # Fairness
    sensitive_keys: List[str] = field(default_factory=lambda: ["age_group","race","ethnicity","insurance"])
    lambda_fair: float = 0.0                   
    # Routing backend
    routing_backend: str = "ema_logits"       
    # Paths (edit to your data locations)
    data_root: str = "./data"
    ckpt_root: str = "./checkpoints"

CFG = Config()
print(CFG)

# Helper: set requires_grad for freezing stages
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
