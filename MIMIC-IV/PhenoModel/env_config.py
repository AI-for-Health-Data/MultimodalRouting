from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
try:
    import lightning as L  
except Exception:
    L = None

try:
    import yaml  
except Exception:
    yaml = None


ROUTES: List[str] = ["L", "N", "I", "LN", "LI", "NI", "LNI"]
BLOCKS: Dict[str, List[str]] = {
    "uni": ["L", "N", "I"],
    "bi":  ["LN", "LI", "NI"],
    "tri": ["LNI"],
}

# Single task (phenotyping)
task = "ph"

@dataclass
class Config:
    d: int = 256
    dropout: float = 0.1

    lr: float = 2e-4
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    grad_clip_norm: float = 1.0

    task_name: str = "ph"

    batch_size: int = 16
    num_workers: int = 4

    max_epochs_uni: int = 5
    max_epochs_bi: int = 5
    max_epochs_tri: int = 5
    steps_per_epoch_hint: int = 500
    ema_decay: float = 0.98

    precision_amp: str = "auto"        
    deterministic: bool = False
    use_cudnn_benchmark: bool = True
    seed: int = 42
    verbose: bool = True
    use_compile: bool = False          

    # Text
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_text_len: int = 512
    text_note_agg: str = "mean"        
    max_notes_concat: int = 8         

    # Structured EHR (2h bins, dynamic length)
    structured_seq_len: int = 24       
    structured_n_feats: int = 128      
    structured_layers: int = 2
    structured_heads: int = 8
    structured_pool: str = "last"      

    # Image
    image_model_name: str = "resnet34" 
    image_agg: str = "last"            

    routing_backend: str = "embedding_concat"  
    use_gates: bool = True

    # Gate computation & mixing
    route_gate_mode: str = "loss_based"  
    gamma: float = 1.0                   
    loss_gate_alpha: float = 4.0         
    l2norm_each: bool = False            

    # Fusion choices
    bi_fusion_mode: str = "mlp"
    tri_fusion_mode: str = "mlp"
    feature_mode: str = "concat"         

    # Per-route head MLP hidden dims
    route_head_hidden: int = 256
    final_head_hidden: int = 256


    num_phenotypes: int = 25
    capsule_in_n_capsules: int = 7            # 7 routes
    capsule_in_d_capsules: int = 256          # = d (route embed size)
    capsule_out_n_capsules: int = 25          # = num_phenotypes
    capsule_out_d_capsules: int = 16          # phenotype pose dim
    capsule_iters: int = 3                    # routing iterations
    capsule_dp: float = 0.1                   # dropout inside capsule
    capsule_act_type: str = "EM"              
    capsule_uniform_routing: bool = False     # hard-uniform for ablation/debug
    capsule_small_std: bool = True            # identity nonlinearity for interpretability
    capsule_dim_pose_to_vote: int = 0         # API placeholder
    capsule_uniform_routing_coefficient: bool = False

    sensitive_keys: List[str] = field(
        default_factory=lambda: ["age_group", "race", "ethnicity", "insurance"]
    )
    lambda_fair: float = 0.0


    data_root: str = "./data"
    ckpt_root: str = "./checkpoints"
    cache_root: str = "./.cache"

    log_every_n_steps: int = 50
    save_every_n_epochs: int = 1
    monitor_metric: str = "val_auroc"
    monitor_mode: str = "max"

    # Legacy alias (kept for env compat)
    alpha: float = 4.0

CFG: Config = Config()
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TASK2IDX: Dict[str, int] = {"ph": 0}


def get_selected_task_idx() -> int:
    return TASK2IDX["ph"]


SELECTED_TASK_IDX: int = get_selected_task_idx()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled: bool) -> None:
    try:
        torch.use_deterministic_algorithms(bool(enabled))
    except Exception:
        pass
    torch.backends.cudnn.deterministic = bool(enabled)
    if enabled:
        torch.backends.cudnn.benchmark = False


def amp_autocast_dtype(precision_amp: str) -> Optional[torch.dtype]:
    p = (precision_amp or "auto").lower()
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    if p == "fp32":
        return None
    if torch.cuda.is_available():
        return torch.float16
    return torch.bfloat16 if hasattr(torch, "bfloat16") else None


def autocast_context() -> torch.autocast:
    dtype = amp_autocast_dtype(CFG.precision_amp)
    dev_type = "cuda" if DEVICE == "cuda" else "cpu"
    if dtype is None:
        return torch.autocast(device_type=dev_type, dtype=torch.float32, enabled=False)
    return torch.autocast(device_type=dev_type, dtype=dtype, enabled=True)


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


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(upd or {})
    return out


def _coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    def conv(x):
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl in {"true", "false"}:
                return xl == "true"
            try:
                if "." in xl:
                    return float(x)
                return int(x)
            except Exception:
                return x
        return x

    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        if isinstance(v, list):
            out[k] = [conv(e) for e in v]
        else:
            out[k] = conv(v)
    return out


def load_cfg(yaml_path: Optional[str] = None,
             overrides: Optional[Dict[str, Any]] = None) -> Config:
    global CFG, DEVICE, TASK2IDX, SELECTED_TASK_IDX

    cfg_dict = asdict(Config())

    if yaml_path and os.path.isfile(yaml_path):
        if yaml is None:
            raise RuntimeError("YAML path provided but PyYAML is not installed.")
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        if not isinstance(y, dict):
            raise ValueError(f"Config YAML at {yaml_path} must be a mapping.")
        cfg_dict = _merge(cfg_dict, _coerce_types(y))

    if overrides:
        cfg_dict = _merge(cfg_dict, _coerce_types(overrides))

    if "MIMICIV_CFG_JSON" in os.environ:
        try:
            blob = json.loads(os.environ["MIMICIV_CFG_JSON"])
            if isinstance(blob, dict):
                cfg_dict = _merge(cfg_dict, _coerce_types(blob))
        except Exception:
            pass

    env_map_simple = {
        "MIMICIV_DATA_ROOT": "data_root",
        "MIMICIV_CKPT_ROOT": "ckpt_root",
        "MIMICIV_TASK": "task_name",
        "MIMICIV_BACKEND": "routing_backend",
        "MIMICIV_BI_FUSION": "bi_fusion_mode",
        "MIMICIV_TRI_FUSION": "tri_fusion_mode",
        "MIMICIV_FEATURE_MODE": "feature_mode",
        "MIMICIV_TEXT_MODEL": "text_model_name",
        "MIMICIV_MAX_TEXT_LEN": "max_text_len",
        "MIMICIV_NOTES_CONCAT": "max_notes_concat",
        "MIMICIV_STRUCT_POOL": "structured_pool",
        "MIMICIV_ALPHA": "alpha",  # legacy alias
    }
    for env_key, cfg_key in env_map_simple.items():
        if env_key in os.environ:
            cfg_dict[cfg_key] = _coerce_types({cfg_key: os.environ[env_key]})[cfg_key]

    if "MIMICIV_BI_LAYERS" in os.environ:
        cfg_dict["bi_layers"] = int(os.environ["MIMICIV_BI_LAYERS"])
    if "MIMICIV_BI_HEADS" in os.environ:
        cfg_dict["bi_heads"] = int(os.environ["MIMICIV_BI_HEADS"])
    if "MIMICIV_TRI_LAYERS" in os.environ:
        cfg_dict["tri_layers"] = int(os.environ["MIMICIV_TRI_LAYERS"])
    if "MIMICIV_TRI_HEADS" in os.environ:
        cfg_dict["tri_heads"] = int(os.environ["MIMICIV_TRI_HEADS"])

    # Phenotype count override
    if "MIMICIV_PHENO_K" in os.environ:
        k = int(os.environ["MIMICIV_PHENO_K"])
        cfg_dict["num_phenotypes"] = k
        cfg_dict["capsule_out_n_capsules"] = k

    # Lock to phenotyping task
    if cfg_dict.get("task_name") != "ph":
        if cfg_dict.get("task_name") is not None:
            print(f"[env_config] Only 'ph' is supported; overriding task_name='{cfg_dict['task_name']}' -> 'ph'")
        cfg_dict["task_name"] = "ph"

    CFG = Config(**cfg_dict)

    set_global_seed(int(CFG.seed))
    set_deterministic(bool(CFG.deterministic))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = bool(CFG.use_cudnn_benchmark and not CFG.deterministic)

    TASK2IDX = {"ph": 0}
    SELECTED_TASK_IDX = get_selected_task_idx()

    ensure_dir(CFG.ckpt_root)
    ensure_dir(CFG.data_root)
    ensure_dir(CFG.cache_root)

    if CFG.verbose:
        print(f"[env_config] Device: {DEVICE}")
        try:
            cfg_json = json.dumps(asdict(CFG), indent=2)
        except Exception:
            cfg_json = str(CFG)
        print(f"[env_config] CFG: {cfg_json}")

    return CFG


load_cfg(yaml_path=None, overrides=None)


def get_device() -> torch.device:
    return torch.device(DEVICE)


def bfloat16_supported() -> bool:
    """Best-effort runtime check for bfloat16 tensor creation."""
    try:
        _ = torch.tensor([1.0], dtype=torch.bfloat16)
        return True
    except Exception:
        return False
