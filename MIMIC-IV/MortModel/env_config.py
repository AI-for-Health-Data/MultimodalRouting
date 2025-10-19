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
TASKS: List[str] = ["mort"]

@dataclass
class Config:
    d: int = 768
    dropout: float = 0.1
    lr: float = 2e-4
    batch_size: int = 16
    max_epochs_uni: int = 5
    max_epochs_bi: int = 5
    max_epochs_tri: int = 5
    num_workers: int = 4
    steps_per_epoch_hint: int = 500

    # Text encoder
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_text_len: int = 512

    # Structured 
    structured_seq_len: int = 24
    structured_n_feats: int = 128
    structured_layers: int = 2
    structured_heads: int = 8

    # Image encoder
    image_model_name: str = "resnet34"

    # Fusion / routing
    feature_mode: str = "concat"             
    routing_backend: str = "capsule"         
    use_gates: bool = True                    

    # Capsule head hyperparams 
    capsule_pc_dim: int = 32                 
    capsule_mc_caps_dim: int = 64             
    capsule_num_routing: int = 3
    capsule_act_type: str = "EM"             
    capsule_layer_norm: bool = False          
    capsule_dim_pose_to_vote: int = 0       

    # Paths & misc
    data_root: str = "./data"
    ckpt_root: str = "./checkpoints"
    task_name: str = "mort"

    # System
    use_cudnn_benchmark: bool = True
    precision_amp: str = "auto"              
    deterministic: bool = False
    seed: int = 42
    verbose: bool = True


# Global singletons
CFG: Config = Config()
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TASK2IDX: Dict[str, int] = {name: i for i, name in enumerate(TASKS)}
SELECTED_TASK_IDX: int = TASK2IDX.get(CFG.task_name, 0)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled: bool) -> None:
    try:
        torch.use_deterministic_algorithms(bool(enabled)) 
    except Exception:
        pass
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = bool(not enabled and CFG.use_cudnn_benchmark)


def amp_autocast_dtype(precision_amp: str) -> Optional[torch.dtype]:
    p = (precision_amp or "auto").lower()
    if p in {"fp32", "off"}:
        return None
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.bfloat16 if hasattr(torch, "bfloat16") else None


def autocast_context() -> torch.autocast:
    dtype = amp_autocast_dtype(CFG.precision_amp)
    if dtype is None:
        return torch.autocast(
            device_type="cuda" if DEVICE == "cuda" else "cpu",
            dtype=torch.float32,
            enabled=False,
        )
    return torch.autocast(
        device_type="cuda" if DEVICE == "cuda" else "cpu",
        dtype=dtype,
        enabled=True,
    )


def is_cuda_device(dev) -> bool:
    return (
        torch.cuda.is_available()
        and (
            (isinstance(dev, torch.device) and dev.type == "cuda")
            or (isinstance(dev, str) and "cuda" in dev)
        )
    )


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
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

    cfg_dict: Dict[str, Any] = asdict(Config())

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

    env_map = {
        "MIMICIV_DATA_ROOT": ("data_root", str),
        "MIMICIV_CKPT_ROOT": ("ckpt_root", str),
        "MIMICIV_TASK": ("task_name", str),
        "MIMICIV_TEXT_MODEL": ("text_model_name", str),
        "MIMICIV_MAX_TEXT_LEN": ("max_text_len", int),
        "MIMICIV_STRUCT_SEQ_LEN": ("structured_seq_len", int),
        "MIMICIV_STRUCT_N_FEATS": ("structured_n_feats", int),
        "MIMICIV_FEATURE_MODE": ("feature_mode", str),
        "MIMICIV_ROUTING_BACKEND": ("routing_backend", str),
        "MIMICIV_CAP_PC_DIM": ("capsule_pc_dim", int),
        "MIMICIV_CAP_MC_DIM": ("capsule_mc_caps_dim", int),
        "MIMICIV_CAP_ITERS": ("capsule_num_routing", int),
        "MIMICIV_CAP_ACT": ("capsule_act_type", str),
        "MIMICIV_CAP_LN": ("capsule_layer_norm", lambda s: str(s).lower() in {"1","true","yes"}),
        "MIMICIV_CAP_DPOSE2VOTE": ("capsule_dim_pose_to_vote", int),
        "MIMICIV_LR": ("lr", float),
        "MIMICIV_BS": ("batch_size", int),
        "MIMICIV_DROPOUT": ("dropout", float),
        "MIMICIV_NUM_WORKERS": ("num_workers", int),
        "MIMICIV_PRECISION": ("precision_amp", str),
        "MIMICIV_DETERMINISTIC": ("deterministic", lambda s: str(s).lower() in {"1","true","yes"}),
        "MIMICIV_SEED": ("seed", int),
        "MIMICIV_VERBOSE": ("verbose", lambda s: str(s).lower() in {"1","true","yes"}),
    }
    for env_key, (cfg_key, caster) in env_map.items():
        if env_key in os.environ:
            try:
                cfg_dict[cfg_key] = caster(os.environ[env_key])
            except Exception:
                cfg_dict[cfg_key] = os.environ[env_key]

    CFG = Config(**cfg_dict)

    set_global_seed(int(CFG.seed))
    set_deterministic(bool(CFG.deterministic))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = bool(CFG.use_cudnn_benchmark and not CFG.deterministic)

    TASK2IDX = {name: i for i, name in enumerate(TASKS)}
    SELECTED_TASK_IDX = TASK2IDX.get(CFG.task_name, 0)

    ensure_dir(CFG.ckpt_root)
    ensure_dir(CFG.data_root)

    if CFG.verbose:
        print(f"[env_config] Device: {DEVICE}")
        try:
            print(f"[env_config] CFG: {json.dumps(asdict(CFG), indent=2)}")
        except TypeError:
            print("[env_config] CFG loaded (json serialization skipped)")

    return CFG


load_cfg(yaml_path=None, overrides=None)


def get_device() -> torch.device:
    return torch.device(DEVICE)


def bfloat16_supported() -> bool:
    try:
        _ = torch.tensor([1.0], dtype=torch.bfloat16)
        return True
    except Exception:
        return False
