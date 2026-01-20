from __future__ import annotations
import os
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    import lightning as L  
except Exception:
    L = None  
try:
    import yaml  
except Exception:
    yaml = None  


def configure_reproducibility(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False if deterministic else bool(cudnn_benchmark)

ROUTES = ["L", "N", "I", "LN", "NL", "LI", "IL", "NI", "IN", "LNI"]
BLOCKS = {
    "uni": ["L", "N", "I"],
    "bi": ["LN", "NL", "LI", "IL", "NI", "IN"],
    "tri": ["LNI"],
}
ROUTE_NAMES = ["L","N","I","LN","NL","LI","IL","NI","IN","LNI"]
N_ROUTES = len(ROUTE_NAMES)
TASKS: List[str] = ["pheno"]
PHENO_NAMES: List[str] = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease",
    "Complications of surgical/medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and related",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia",
    "Respiratory failure; insufficiency; arrest",
    "Septicemia (except in labor)",
    "Shock",
]

@dataclass
class Config:
    # Core training
    d: int = 256
    dropout: float = 0.0
    lr: float = 2e-4
    batch_size: int = 16
    max_epochs_uni: int = 0
    max_epochs_bi: int = 0
    max_epochs_tri: int = 50
    num_workers: int = 4
    steps_per_epoch_hint: int = 500

    # Text encoder
    finetune_text: bool = False
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_text_len: int = 512          # per chunk token length
    notes_chunk_len: int = 512       
    notes_max_chunks: int = -1       
    bert_chunk_bs: int = 8

    # Structured
    feature_mode: str = "seq"        
    structured_seq_len: int = 256     
    structured_pos_max_len: int = 2048
    structured_n_feats: int = 61
    structured_layers: int = 2
    structured_heads: int = 8

    # Image encoder
    image_model_name: str = "resnet34"

    # Fusion / routing
    routing_backend: str = "cross_attn"
    use_gates: bool = True

    route_gate_temp: float = 3.0
    route_gate_min: float = 0.05
    route_gate_max: float = 0.95
    route_entropy_lambda: float = 0.01
    route_uniform_lambda: float = 0.10
    route_entropy_warmup_epochs: float = 0.0
    route_uniform_warmup_epochs: float = 0.0
    lambda_route_entropy: float = 0.01
    lambda_route_balance: float = 0.10
    grad_clip_norm: float = 1.0
    route_num: int = len(ROUTES)
    cross_attn_heads: int = 8
    cross_attn_dropout: float = 0.0
    cross_attn_pool: str = "mean"    # "mean" | "first"

    capsule_pc_dim: int = 32
    capsule_mc_caps_dim: int = 64
    capsule_num_routing: int = 3
    capsule_act_type: str = "EM"
    capsule_layer_norm: bool = False
    capsule_dim_pose_to_vote: int = 0

    loss_type: str = "bce"

    route_dropout_p: float = 0.0
    routing_warmup_epochs: int = 5

    route_prior_floor: float = 0.02
    route_prior_ceiling: float = 0.98
    label_smoothing: float = 0.0
    entropy_use_rc: bool = False

    routing_coef_mode: str = "gate_norm"  
    routing_coef_eps: float = 1e-6

    data_root: str = "./data"
    ckpt_root: str = "./checkpoints"
    image_root: str = "/users/nikkieh/mimic-iv"
    task_name: str = "pheno"

    # System
    use_cudnn_benchmark: bool = False
    precision_amp: str = "auto"     
    deterministic: bool = True
    seed: int = 42
    verbose: bool = False
    debug_samples: int = 3

CFG: Config = Config()
DEVICE: str = "cpu"
TASK2IDX: Dict[str, int] = {name: i for i, name in enumerate(TASKS)}
SELECTED_TASK_IDX: int = TASK2IDX.get(CFG.task_name, 0)


def _str2bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t", "on"}:
        return True
    if s in {"0", "false", "no", "n", "f", "off"}:
        return False
    return bool(s)


def _pick_device() -> str:
    want = str(os.environ.get("MIMICIV_DEVICE", "auto")).lower().strip()
    if want == "cpu":
        return "cpu"
    if want.startswith("cuda"):
        if torch.cuda.is_available():
            if want == "cuda":
                return "cuda"
            # Accept "cuda:0" etc.
            try:
                _ = torch.cuda.device_count()
                return want
            except Exception:
                return "cuda"
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def set_deterministic(enabled: bool) -> None:
    try:
        torch.use_deterministic_algorithms(bool(enabled))
    except Exception:
        pass
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = False if enabled else bool(CFG.use_cudnn_benchmark)


def _cuda_bf16_supported() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def amp_autocast_dtype(precision_amp: str) -> Optional[torch.dtype]:
    p = (precision_amp or "auto").lower().strip()
    if p in {"fp32", "off", "none"}:
        return None
    if p == "bf16":
        return torch.bfloat16
    if p == "fp16":
        return torch.float16
    if _cuda_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context():
    if not DEVICE.startswith("cuda"):
        return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    dt = amp_autocast_dtype(CFG.precision_amp)
    if dt is None:
        return torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False)
    return torch.autocast(device_type="cuda", dtype=dt, enabled=True)


def is_cuda_device(dev: Any) -> bool:
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
        p.requires_grad = bool(flag)

def _merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(upd or {})
    return out

def _coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    def conv(x: Any) -> Any:
        if isinstance(x, str):
            s = x.strip()
            sl = s.lower()
            if sl in {"true", "false"}:
                return sl == "true"
            if sl.startswith("cuda"):
                return s
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return x
        return x

    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        if isinstance(v, list):
            out[k] = [conv(e) for e in v]
        else:
            out[k] = conv(v)
    return out


def _sync_routing_aliases(cfg: Config) -> None:
    ent = float(getattr(cfg, "route_entropy_lambda", 0.0) or getattr(cfg, "lambda_route_entropy", 0.0))
    uni = float(getattr(cfg, "route_uniform_lambda", 0.0) or getattr(cfg, "lambda_route_balance", 0.0))

    cfg.route_entropy_lambda = ent
    cfg.route_uniform_lambda = uni
    cfg.lambda_route_entropy = ent
    cfg.lambda_route_balance = uni
    if not hasattr(cfg, "gate_temp"):
        setattr(cfg, "gate_temp", cfg.route_gate_temp)
    else:
        setattr(cfg, "gate_temp", float(getattr(cfg, "gate_temp")))

    if not hasattr(cfg, "gate_min"):
        setattr(cfg, "gate_min", cfg.route_gate_min)
    else:
        setattr(cfg, "gate_min", float(getattr(cfg, "gate_min")))

    if not hasattr(cfg, "gate_max"):
        setattr(cfg, "gate_max", cfg.route_gate_max)
    else:
        setattr(cfg, "gate_max", float(getattr(cfg, "gate_max")))

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
        "MIMICIV_IMAGE_ROOT": ("image_root", str),


        "MIMICIV_FINETUNE_TEXT": ("finetune_text", _str2bool),
        "MIMICIV_TEXT_MODEL": ("text_model_name", str),
        "MIMICIV_MAX_TEXT_LEN": ("max_text_len", int),
        "MIMICIV_NOTES_CHUNK_LEN": ("notes_chunk_len", int),
        "MIMICIV_NOTES_MAX_CHUNKS": ("notes_max_chunks", int),
        "MIMICIV_BERT_CHUNK_BS": ("bert_chunk_bs", int),

        "MIMICIV_STRUCT_SEQ_LEN": ("structured_seq_len", int),
        "MIMICIV_STRUCT_N_FEATS": ("structured_n_feats", int),

        "MIMICIV_CROSS_ATTN_HEADS": ("cross_attn_heads", int),
        "MIMICIV_CROSS_ATTN_DROPOUT": ("cross_attn_dropout", float),
        "MIMICIV_CROSS_ATTN_POOL": ("cross_attn_pool", str),

        "MIMICIV_USE_GATES": ("use_gates", _str2bool),
        "MIMICIV_ROUTE_GATE_TEMP": ("route_gate_temp", float),
        "MIMICIV_ROUTE_GATE_MIN": ("route_gate_min", float),
        "MIMICIV_ROUTE_GATE_MAX": ("route_gate_max", float),

        # Regularizers: accept legacy and canonical env names
        "MIMICIV_ROUTE_ENTROPY_LAMBDA": ("route_entropy_lambda", float),
        "MIMICIV_ROUTE_UNIFORM_LAMBDA": ("route_uniform_lambda", float),
        "MIMICIV_LAMBDA_ROUTE_ENTROPY": ("lambda_route_entropy", float),
        "MIMICIV_LAMBDA_ROUTE_BALANCE": ("lambda_route_balance", float),

        "MIMICIV_ROUTE_ENTROPY_WARM": ("route_entropy_warmup_epochs", float),
        "MIMICIV_ROUTE_UNIFORM_WARM": ("route_uniform_warmup_epochs", float),

        "MIMICIV_GRAD_CLIP_NORM": ("grad_clip_norm", float),

        "MIMICIV_CAP_PC_DIM": ("capsule_pc_dim", int),
        "MIMICIV_CAP_MC_DIM": ("capsule_mc_caps_dim", int),
        "MIMICIV_CAP_ITERS": ("capsule_num_routing", int),
        "MIMICIV_CAP_ACT": ("capsule_act_type", str),
        "MIMICIV_CAP_LN": ("capsule_layer_norm", _str2bool),
        "MIMICIV_CAP_DPOSE2VOTE": ("capsule_dim_pose_to_vote", int),

        "MIMICIV_LOSS": ("loss_type", str),
        "MIMICIV_LR": ("lr", float),
        "MIMICIV_BS": ("batch_size", int),
        "MIMICIV_DROPOUT": ("dropout", float),
        "MIMICIV_NUM_WORKERS": ("num_workers", int),
        "MIMICIV_PRECISION": ("precision_amp", str),
        "MIMICIV_DETERMINISTIC": ("deterministic", _str2bool),
        "MIMICIV_SEED": ("seed", int),
        "MIMICIV_VERBOSE": ("verbose", _str2bool),
        "MIMICIV_DEBUG_SAMPLES": ("debug_samples", int),
        "MIMICIV_ROUTING_PRINT_EVERY": ("routing_print_every", int),

        "MIMICIV_ROUTE_DROPOUT_P": ("route_dropout_p", float),
        "MIMICIV_ROUTING_WARMUP_EPOCHS": ("routing_warmup_epochs", int),

        "MIMICIV_ROUTE_PRIOR_FLOOR": ("route_prior_floor", float),
        "MIMICIV_ROUTE_PRIOR_CEILING": ("route_prior_ceiling", float),

        "MIMICIV_LABEL_SMOOTHING": ("label_smoothing", float),
        "MIMICIV_ENTROPY_USE_RC": ("entropy_use_rc", _str2bool),
    }

    for env_key, (cfg_key, caster) in env_map.items():
        if env_key in os.environ:
            try:
                cfg_dict[cfg_key] = caster(os.environ[env_key])
            except Exception:
                cfg_dict[cfg_key] = os.environ[env_key]

    CFG = Config(**cfg_dict)
    _sync_routing_aliases(CFG)
    if CFG.routing_backend != "cross_attn":
        if CFG.verbose:
            print(f"[env_config] Warning: routing_backend={CFG.routing_backend}; forcing to 'cross_attn'.")
        CFG.routing_backend = "cross_attn"

    if CFG.route_num != len(ROUTES) and CFG.verbose:
        print(f"[env_config] Warning: route_num={CFG.route_num} but len(ROUTES)={len(ROUTES)}; forcing to {len(ROUTES)}.")
    CFG.route_num = len(ROUTES)

    if CFG.cross_attn_heads <= 0:
        if CFG.verbose:
            print("[env_config] Warning: cross_attn_heads must be > 0; setting to 8.")
        CFG.cross_attn_heads = 8
    if not (0.0 <= CFG.cross_attn_dropout < 1.0):
        if CFG.verbose:
            print("[env_config] Warning: cross_attn_dropout must be in [0,1); setting to 0.0.")
        CFG.cross_attn_dropout = 0.0
    if CFG.cross_attn_pool not in {"mean", "first"}:
        if CFG.verbose:
            print(f"[env_config] Warning: invalid cross_attn_pool={CFG.cross_attn_pool}; defaulting to 'mean'.")
        CFG.cross_attn_pool = "mean"

    CFG.route_prior_floor = float(max(0.0, min(1.0, CFG.route_prior_floor)))
    CFG.route_prior_ceiling = float(max(0.0, min(1.0, CFG.route_prior_ceiling)))
    if CFG.route_prior_floor >= CFG.route_prior_ceiling:
        # ensure valid interval
        CFG.route_prior_floor = min(CFG.route_prior_floor, 0.49)
        CFG.route_prior_ceiling = max(CFG.route_prior_ceiling, 0.51)

    DEVICE = _pick_device()

    configure_reproducibility(
        seed=int(CFG.seed),
        deterministic=bool(CFG.deterministic),
        cudnn_benchmark=bool(CFG.use_cudnn_benchmark),
    )

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


def apply_cli_overrides(args: Any) -> None:
    global CFG
    if hasattr(args, "data_root") and args.data_root is not None:
        CFG.data_root = str(args.data_root)
    if hasattr(args, "ckpt_root") and args.ckpt_root is not None:
        CFG.ckpt_root = str(args.ckpt_root)
    if hasattr(args, "lr") and args.lr is not None:
        CFG.lr = float(args.lr)
    if hasattr(args, "batch_size") and args.batch_size is not None:
        CFG.batch_size = int(args.batch_size)
    if hasattr(args, "num_workers") and args.num_workers is not None:
        CFG.num_workers = int(args.num_workers)
    if hasattr(args, "precision") and args.precision is not None:
        CFG.precision_amp = str(args.precision)
    if hasattr(args, "structured_seq_len") and args.structured_seq_len is not None:
        CFG.structured_seq_len = int(args.structured_seq_len)
    if hasattr(args, "notes_max_chunks") and args.notes_max_chunks is not None:
        CFG.notes_max_chunks = int(args.notes_max_chunks)
    if hasattr(args, "notes_chunk_len") and args.notes_chunk_len is not None:
        CFG.notes_chunk_len = int(args.notes_chunk_len)
    if hasattr(args, "max_text_len") and args.max_text_len is not None:
        CFG.max_text_len = int(args.max_text_len)
    if hasattr(args, "bert_chunk_bs") and args.bert_chunk_bs is not None:
        CFG.bert_chunk_bs = int(args.bert_chunk_bs)
    if hasattr(args, "cross_attn_heads") and args.cross_attn_heads is not None:
        CFG.cross_attn_heads = int(args.cross_attn_heads)
    if hasattr(args, "cross_attn_dropout") and args.cross_attn_dropout is not None:
        CFG.cross_attn_dropout = float(args.cross_attn_dropout)
    if hasattr(args, "cross_attn_pool") and args.cross_attn_pool is not None:
        CFG.cross_attn_pool = str(args.cross_attn_pool)
    if hasattr(args, "use_gates") and args.use_gates is not None:
        CFG.use_gates = _str2bool(args.use_gates)
    if hasattr(args, "epochs") and args.epochs is not None:
        CFG.max_epochs_tri = int(args.epochs)
    if hasattr(args, "route_dropout_p") and args.route_dropout_p is not None:
        CFG.route_dropout_p = float(args.route_dropout_p)
    if hasattr(args, "route_entropy_lambda") and getattr(args, "route_entropy_lambda") is not None:
        CFG.route_entropy_lambda = float(getattr(args, "route_entropy_lambda"))
    if hasattr(args, "route_uniform_lambda") and getattr(args, "route_uniform_lambda") is not None:
        CFG.route_uniform_lambda = float(getattr(args, "route_uniform_lambda"))
    if hasattr(args, "route_entropy_warmup_epochs") and args.route_entropy_warmup_epochs is not None:
        CFG.route_entropy_warmup_epochs = float(args.route_entropy_warmup_epochs)
    if hasattr(args, "route_uniform_warmup_epochs") and args.route_uniform_warmup_epochs is not None:
        CFG.route_uniform_warmup_epochs = float(args.route_uniform_warmup_epochs)
    if hasattr(args, "finetune_text") and getattr(args, "finetune_text") is not None:
        CFG.finetune_text = _str2bool(getattr(args, "finetune_text"))
    CFG.route_num = len(ROUTES)
    _sync_routing_aliases(CFG)
    ensure_dir(CFG.ckpt_root)
    ensure_dir(CFG.data_root)


def get_pheno_name(idx: int) -> str:
    if 0 <= idx < len(PHENO_NAMES):
        return PHENO_NAMES[idx]
    return f"pheno_{idx:02d}"


def get_device() -> torch.device:
    return torch.device(DEVICE)


def bfloat16_supported() -> bool:
    if _cuda_bf16_supported():
        return True
    try:
        _ = torch.tensor([1.0], dtype=torch.bfloat16)
        return True
    except Exception:
        return False
