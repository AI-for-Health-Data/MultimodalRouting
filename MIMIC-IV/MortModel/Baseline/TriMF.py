from __future__ import annotations
import os
import json
import math
import time
import random
import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ast
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from transformers import AutoTokenizer
except Exception as e:
    AutoTokenizer = None

HAS_PROJECT = True
try:
    from env_config import CFG, DEVICE as PROJECT_DEVICE, load_cfg, ensure_dir, apply_cli_overrides
except Exception:
    HAS_PROJECT = False
    CFG = None
    PROJECT_DEVICE = None
    load_cfg = None
    ensure_dir = None
    apply_cli_overrides = None

try:
    from encoders import BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder, EncoderConfig
except Exception:
    BEHRTLabEncoder = None
    BioClinBERTEncoder = None
    ImageEncoder = None
    EncoderConfig = None

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_struct_in_dim(df: pd.DataFrame, struct_cols: List[str], max_rows: int = 2000) -> int:
    if not struct_cols:
        return 0
    if len(struct_cols) != 1:
        return int(len(struct_cols))

    col = struct_cols[0]
    dims: List[int] = []
    for v in df[col].head(max_rows).tolist():
        a = coerce_to_1d_float(v)
        if a is not None and a.size > 0:
            dims.append(int(a.size))

    if not dims:
        return 0

    vals, counts = np.unique(np.array(dims), return_counts=True)
    mode_dim = int(vals[np.argmax(counts)])
    top = sorted(zip(vals.tolist(), counts.tolist()), key=lambda x: -x[1])[:5]
    print(f"[STRUCT] dim candidates (top): {top} | picked mode={mode_dim}")
    return mode_dim

def infer_struct_cols(struct_df: pd.DataFrame, id_col: str, label_col: str) -> List[str]:
    preferred = [
        "x_ehr_48h_2h_76",
        "x_ehr",
        "ehr_vec",
        "ehr_features",
        "x",
        "features",
    ]
    for c in preferred:
        if c in struct_df.columns:
            return [c]
    cols = []
    for c in struct_df.columns:
        if c == id_col or c == label_col:
            continue
        if pd.api.types.is_numeric_dtype(struct_df[c]) or pd.api.types.is_bool_dtype(struct_df[c]):
            cols.append(c)
    return cols

def infer_labs_n_feats_from_df(df: pd.DataFrame, max_rows: int = 10000) -> Optional[int]:
    if df is None or len(df) == 0:
        return None

    df_small = df.head(max_rows)
    if "lab_codes" in df_small.columns:
        mx = -1
        for s in df_small["lab_codes"].dropna().astype(str).tolist():
            for tok in s.strip().split():
                try:
                    mx = max(mx, int(tok))
                except Exception:
                    pass
        return (mx + 1) if mx >= 0 else None

    if "labs_json" in df_small.columns:
        mx = -1
        for s in df_small["labs_json"].dropna().astype(str).tolist():
            try:
                d = safe_json_load(s)
                ids = d.get("input_ids", None)
                if ids is None:
                    continue
                if isinstance(ids, torch.Tensor):
                    mx = max(mx, int(ids.max().item()))
                else:
                    mx = max(mx, int(max(ids)))
            except Exception:
                pass
        return (mx + 1) if mx >= 0 else None

    if "labs_pt" in df_small.columns:
        mx = -1
        for p in df_small["labs_pt"].dropna().astype(str).tolist():
            if not (isinstance(p, str) and os.path.isfile(p)):
                continue
            try:
                d = torch.load(p, map_location="cpu")
                ids = d.get("input_ids", None)
                if ids is None:
                    continue
                if isinstance(ids, torch.Tensor):
                    mx = max(mx, int(ids.max().item()))
                else:
                    mx = max(mx, int(max(ids)))
            except Exception:
                pass
        return (mx + 1) if mx >= 0 else None

    return None

def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def as_tensor(x: Any, dtype=torch.long) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=dtype)


def safe_json_load(s: str) -> Dict[str, Any]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return {}
    s = str(s)
    if os.path.isfile(s) and s.lower().endswith(".json"):
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(s)

def coerce_to_1d_float(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None

    if isinstance(v, torch.Tensor):
        arr = v.detach().cpu().float().view(-1).numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr if arr.size > 0 else None

    if isinstance(v, np.ndarray):
        if v.dtype == object:
            parts = []
            for e in v.ravel():
                a = coerce_to_1d_float(e)
                if a is not None and a.size > 0:
                    parts.append(a)
            if not parts:
                return None
            arr = np.concatenate(parts, axis=0)
        else:
            arr = v.astype(np.float32).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr if arr.size > 0 else None

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        obj = None
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                try:
                    toks = s.replace(",", " ").split()
                    obj = [float(t) for t in toks]
                except Exception:
                    return None
        return coerce_to_1d_float(obj)

    if isinstance(v, (list, tuple)):
        parts = []
        for e in v:
            a = coerce_to_1d_float(e) if isinstance(e, (list, tuple, np.ndarray, torch.Tensor, str)) else None
            if a is None:
                try:
                    parts.append(np.asarray([e], dtype=np.float32))
                except Exception:
                    continue
            else:
                parts.append(a)
        if not parts:
            return None
        arr = np.concatenate(parts, axis=0).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr if arr.size > 0 else None

    try:
        arr = np.asarray([v], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr if arr.size > 0 else None
    except Exception:
        return None

def try_import_sklearn_metrics():
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        return roc_auc_score, f1_score
    except Exception:
        return None, None


def compute_auc_f1(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Tuple[float, float]:
    roc_auc_score, f1_score = try_import_sklearn_metrics()
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)
    auc = float("nan")
    if roc_auc_score is not None:
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_prob))
    else:
        if len(np.unique(y_true)) == 2:
            order = np.argsort(y_prob)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(y_prob))
            pos = y_true == 1
            n_pos = pos.sum()
            n_neg = (~pos).sum()
            auc = (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg + 1e-12)

    f1 = float("nan")
    if f1_score is not None:
        f1 = float(f1_score(y_true, y_pred))
    else:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    return auc, f1


def best_f1_over_thresholds(y_true: np.ndarray, y_prob: np.ndarray, n: int = 200) -> Tuple[float, float]:
    best_thr, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, n):
        _, f1 = compute_auc_f1(y_true, y_prob, thr=float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return best_thr, float(best_f1)


def pad_1d(seqs: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs)
    out = seqs[0].new_full((len(seqs), max_len), pad_value)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out


def init_with_supported_kwargs(cls, **kwargs):
    sig = inspect.signature(cls.__init__)
    supported = set(sig.parameters.keys())
    supported.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return cls(**filtered)


class MortalityTriModalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "mortality",
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        notes_max_len: int = 256,
        image_size: int = 224,
        labs_mode: str = "behrt",
        struct_cols: Optional[List[str]] = None,
        struct_in_dim: int = 0,                
    ):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.notes_max_len = notes_max_len
        self.struct_cols = struct_cols or []
        self.struct_in_dim = int(struct_in_dim) if struct_in_dim else 0
        self.labs_mode = labs_mode
        self.tokenizer = None
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            except Exception:
                self.tokenizer = None
        self.image_tf = torch.nn.Sequential()
        self.image_size = image_size


    def __len__(self) -> int:
        return len(self.df)

    def _load_labs(self, row: pd.Series) -> Tuple[Dict[str, Any], bool]:
        if self.labs_mode == "tabular":
            if not self.struct_cols:
                return {}, False

            if len(self.struct_cols) == 1:
                v = row[self.struct_cols[0]]
                a = coerce_to_1d_float(v)
                if a is None or a.size == 0:
                    return {}, False

                if self.struct_in_dim and a.size != self.struct_in_dim:
                    if a.size < self.struct_in_dim:
                        a = np.pad(a, (0, self.struct_in_dim - a.size), mode="constant")
                    else:
                        a = a[: self.struct_in_dim]

                x = torch.from_numpy(a.astype(np.float32))
                return {"x": x}, True


            vals = row[self.struct_cols].to_numpy()
            vals = np.nan_to_num(vals.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            return {"x": torch.from_numpy(vals)}, True

        if "labs_pt" in row and isinstance(row["labs_pt"], str) and os.path.isfile(row["labs_pt"]):
            d = torch.load(row["labs_pt"], map_location="cpu")
            return d, True

        if "labs_json" in row and pd.notna(row["labs_json"]):
            d = safe_json_load(str(row["labs_json"]))
            return d, (len(d) > 0)

        if "lab_codes" in row and pd.notna(row["lab_codes"]):
            codes = [int(x) for x in str(row["lab_codes"]).strip().split()]
            if len(codes) == 0:
                return {}, False
            attn = [1] * len(codes)
            d = {"input_ids": codes, "attention_mask": attn}
            return d, True

        return {}, False

    def _load_notes(self, row: pd.Series) -> Tuple[Dict[str, Any], bool]:
        if "notes_pt" in row and isinstance(row["notes_pt"], str) and os.path.isfile(row["notes_pt"]):
            d = torch.load(row["notes_pt"], map_location="cpu")
            return d, True

        text = None
        if "note_text" in row and pd.notna(row["note_text"]):
            text = str(row["note_text"])
        elif "note_path" in row and isinstance(row["note_path"], str) and os.path.isfile(row["note_path"]):
            with open(row["note_path"], "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        if text is None or len(text.strip()) == 0:
            return {}, False

        if "input_ids_000" in row and row["input_ids_000"] is not None:
            ids = row["input_ids_000"]
            attn = row["attention_mask_000"] if "attention_mask_000" in row else None

            ids = torch.tensor(np.asarray(ids, dtype=np.int64).reshape(-1), dtype=torch.long)
            if attn is None or (isinstance(attn, float) and math.isnan(attn)):
                attn = (ids != 0).long()
            else:
                attn = torch.tensor(np.asarray(attn, dtype=np.int64).reshape(-1), dtype=torch.long)

            return {"input_ids": ids, "attention_mask": attn}, True


        if self.tokenizer is None:
            return {"text": text}, True

        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.notes_max_len,
            padding=False,
            return_tensors=None,
        )
        return tok, True

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  
        arr = np.transpose(arr, (2, 0, 1)) 
        return torch.from_numpy(arr)

    def _load_image(self, row: pd.Series) -> Tuple[torch.Tensor, bool]:
        if "image_pt" in row and isinstance(row["image_pt"], str) and os.path.isfile(row["image_pt"]):
            t = torch.load(row["image_pt"], map_location="cpu")
            if isinstance(t, torch.Tensor):
                if t.dtype == torch.uint8:
                    t = t.float() / 255.0
                return t, True

        if "image_path" in row and isinstance(row["image_path"], str) and os.path.isfile(row["image_path"]):
            try:
                img = Image.open(row["image_path"])
                return self._pil_to_tensor(img), True
            except Exception:
                return torch.zeros(3, self.image_size, self.image_size), False

        return torch.zeros(3, self.image_size, self.image_size), False

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        y = float(row[self.label_col]) if self.label_col in row else float(row.get("label", 0.0))

        labs, has_labs = self._load_labs(row)
        notes, has_notes = self._load_notes(row)
        img, has_img = self._load_image(row)

        return {
            "labs": labs,
            "notes": notes,
            "image": img,
            "has_labs": has_labs,
            "has_notes": has_notes,
            "has_image": has_img,
            "label": y,
        }


def collate_trimodal(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

    out["has_labs"] = torch.tensor([1 if b["has_labs"] else 0 for b in batch], dtype=torch.bool)
    out["has_notes"] = torch.tensor([1 if b["has_notes"] else 0 for b in batch], dtype=torch.bool)
    out["has_image"] = torch.tensor([1 if b["has_image"] else 0 for b in batch], dtype=torch.bool)

    labs_dicts = [b["labs"] if isinstance(b["labs"], dict) else {} for b in batch]
    if any(("x" in d) for d in labs_dicts):
        xs = []
        feat_dim = None
        for d in labs_dicts:
            x = d.get("x", None)
            if x is None:
                if feat_dim is None:
                    # find first available feat_dim
                    for dd in labs_dicts:
                        if "x" in dd:
                            feat_dim = int(dd["x"].numel())
                            break
                    feat_dim = feat_dim or 1
                x = torch.zeros(feat_dim, dtype=torch.float32)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.float().view(-1)
            if feat_dim is None:
                feat_dim = int(x.numel())
            xs.append(x)
        out["labs"] = {"x": torch.stack(xs, dim=0)}
    else:
        labs_keys = sorted({k for d in labs_dicts for k in d.keys()})
        labs_batch: Dict[str, torch.Tensor] = {}
        for k in labs_keys:
            seqs = []
            for d in labs_dicts:
                v = d.get(k, [0])
                if isinstance(v, torch.Tensor):
                    t = v
                else:
                    t = torch.tensor(v, dtype=torch.long)
                if t.ndim == 0:
                    t = t.view(1)
                seqs.append(t)
            labs_batch[k] = pad_1d(seqs, pad_value=0)
        out["labs"] = labs_batch


    notes_dicts = [b["notes"] if isinstance(b["notes"], dict) else {} for b in batch]
    notes_keys = sorted({k for d in notes_dicts for k in d.keys() if k != "text"})
    notes_batch: Dict[str, Any] = {}
    if any(("text" in d) for d in notes_dicts):
        notes_batch["text"] = [d.get("text", "") for d in notes_dicts]

    for k in notes_keys:
        seqs = []
        for d in notes_dicts:
            v = d.get(k, [0])
            if isinstance(v, torch.Tensor):
                t = v
            else:
                t = torch.tensor(v, dtype=torch.long)
            if t.ndim == 0:
                t = t.view(1)
            seqs.append(t)
        notes_batch[k] = pad_1d(seqs, pad_value=0)
    out["notes"] = notes_batch

    imgs = [b["image"] for b in batch]
    out["image"] = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in imgs], dim=0)

    return out

class PooledEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, out_dim: int, name: str):
        super().__init__()
        self.encoder = encoder
        self.out_dim = out_dim
        self.name = name

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.encoder(inputs)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"{self.name} encoder returned non-tensor type: {type(x)}")

        if x.ndim == 2:
            return x
        if x.ndim == 3:
            # pool over sequence
            attn = inputs.get("attention_mask", None)
            if attn is None:
                return x.mean(dim=1)
            attn = attn.float()
            if attn.ndim == 2:
                w = attn.unsqueeze(-1)
            else:
                w = attn
            num = (x * w).sum(dim=1)
            den = w.sum(dim=1).clamp_min(1.0)
            return num / den
        raise RuntimeError(f"{self.name} encoder output has unsupported shape: {tuple(x.shape)}")

class TriMF(nn.Module):
    """
    A practical tri-modal fusion block:
      - project each modality to d_model
      - compute gated pairwise interactions (elementwise products)
      - compute gated triple interaction
      - concatenate everything and pass through an MLP
    """
    def __init__(self, d_in: Dict[str, int], d_model: int = 256, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model

        self.proj_l = nn.Linear(d_in["labs"], d_model)
        self.proj_t = nn.Linear(d_in["notes"], d_model)
        self.proj_i = nn.Linear(d_in["image"], d_model)

        self.ln_l = nn.LayerNorm(d_model)
        self.ln_t = nn.LayerNorm(d_model)
        self.ln_i = nn.LayerNorm(d_model)

        # gates
        self.g_lt = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.g_li = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.g_ti = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.g_lti = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Sigmoid())

        fused_dim = 3 * d_model + 3 * d_model + d_model  # unimodal + pairwise + triple
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, z_l: torch.Tensor, z_t: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        zl = self.ln_l(self.proj_l(z_l))
        zt = self.ln_t(self.proj_t(z_t))
        zi = self.ln_i(self.proj_i(z_i))

        lt_gate = self.g_lt(torch.cat([zl, zt], dim=-1))
        li_gate = self.g_li(torch.cat([zl, zi], dim=-1))
        ti_gate = self.g_ti(torch.cat([zt, zi], dim=-1))
        lti_gate = self.g_lti(torch.cat([zl, zt, zi], dim=-1))

        phi_lt = lt_gate * (zl * zt)
        phi_li = li_gate * (zl * zi)
        phi_ti = ti_gate * (zt * zi)
        phi_lti = lti_gate * (zl * zt * zi)

        fused = torch.cat([zl, zt, zi, phi_lt, phi_li, phi_ti, phi_lti], dim=-1)
        return self.fuse(fused)


class TriMFMortalityModel(nn.Module):
    def __init__(
        self,
        labs_encoder: nn.Module,
        notes_encoder: nn.Module,
        image_encoder: nn.Module,
        d_labs: int,
        d_notes: int,
        d_image: int,
        d_model: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.labs_encoder = labs_encoder
        self.notes_encoder = notes_encoder
        self.image_encoder = image_encoder
        self.missing_l = nn.Parameter(torch.zeros(d_labs))
        self.missing_t = nn.Parameter(torch.zeros(d_notes))
        self.missing_i = nn.Parameter(torch.zeros(d_image))
        self.trimf = TriMF({"labs": d_labs, "notes": d_notes, "image": d_image}, d_model=d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        has_l = batch["has_labs"].to(torch.bool)
        has_t = batch["has_notes"].to(torch.bool)
        has_i = batch["has_image"].to(torch.bool)

        B = int(batch["label"].shape[0])
        device = batch["label"].device

        z_l_default = self.missing_l.unsqueeze(0).repeat(B, 1)

        if has_l.any():
            labs_all = self.labs_encoder(batch["labs"])  # (B, d_labs) from pooled encoder
            # select labs_all where present, otherwise missing
            z_l = torch.where(has_l.unsqueeze(-1), labs_all, z_l_default)
        else:
            z_l = z_l_default

        z_t_default = self.missing_t.unsqueeze(0).repeat(B, 1)

        if has_t.any():
            # build notes input as full batch (no indexing tricks needed)
            notes_in = {}
            for k, v in batch["notes"].items():
                if k == "text":
                    notes_in["text"] = v  # list[str]
                else:
                    notes_in[k] = v       # tensor
            notes_all = self.notes_encoder(notes_in)
            z_t = torch.where(has_t.unsqueeze(-1), notes_all, z_t_default)
        else:
            z_t = z_t_default

        z_i_default = self.missing_i.unsqueeze(0).repeat(B, 1)

        if has_i.any():
            img = batch["image"]
            try:
                img_all = self.image_encoder({"pixel_values": img})
            except Exception:
                img_all = self.image_encoder(img)
            z_i = torch.where(has_i.unsqueeze(-1), img_all, z_i_default)
        else:
            z_i = z_i_default

        fused = self.trimf(z_l, z_t, z_i)
        logits = self.head(fused).squeeze(-1)
        return logits


class StructuredMLPEncoder(nn.Module):
    def __init__(self, in_dim: int, d_out: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = inputs["x"]
        return self.net(x)

def build_encoders(args, device: torch.device, struct_in_dim: int = 0):
    if BioClinBERTEncoder is None or ImageEncoder is None:
        raise RuntimeError("Missing encoders import (BioClinBERTEncoder/ImageEncoder).")

    enc_cfg = None
    if EncoderConfig is not None:
        try:
            enc_cfg = EncoderConfig()
        except Exception:
            enc_cfg = None

    if getattr(args, "labs_mode", "behrt") == "tabular":
        labs_enc = StructuredMLPEncoder(in_dim=int(struct_in_dim), d_out=int(args.labs_d), dropout=0.1)
    else:
        if BEHRTLabEncoder is None:
            raise RuntimeError("BEHRTLabEncoder not available but labs_mode=behrt.")
        labs_enc = init_with_supported_kwargs(
            BEHRTLabEncoder,
            n_feats=int(args.labs_n_feats),
            d=int(args.labs_d),
            cfg=enc_cfg,
            device=device,
            freeze=args.freeze_encoders,
        )

    notes_enc = init_with_supported_kwargs(
        BioClinBERTEncoder,
        d=int(args.labs_d),          
        cfg=enc_cfg,
        device=device,
        freeze=args.freeze_encoders
    )

    img_enc   = init_with_supported_kwargs(
        ImageEncoder,
        d=int(args.labs_d),         
        cfg=enc_cfg,
        device=device,
        freeze=args.freeze_encoders
    )

    labs_enc  = labs_enc.to(device)
    notes_enc = notes_enc.to(device)
    img_enc   = img_enc.to(device)

    with torch.no_grad():
        if getattr(args, "labs_mode", "behrt") == "tabular":
            labs_in = {"x": torch.zeros(2, int(struct_in_dim), device=device)}
        else:
            labs_in = {"input_ids": torch.ones(2, 4, dtype=torch.long, device=device),
                       "attention_mask": torch.ones(2, 4, dtype=torch.long, device=device)}
        zl = labs_enc(labs_in)
        if isinstance(zl, (tuple, list)): zl = zl[0]
        if zl.ndim == 3: zl = zl.mean(dim=1)
        d_labs = int(zl.shape[-1])

        notes_in = {"input_ids": torch.ones(2, 8, dtype=torch.long, device=device),
                    "attention_mask": torch.ones(2, 8, dtype=torch.long, device=device)}
        zt = notes_enc(notes_in)
        if isinstance(zt, (tuple, list)): zt = zt[0]
        if zt.ndim == 3: zt = zt.mean(dim=1)
        d_notes = int(zt.shape[-1])

        img = torch.zeros(2, 3, args.image_size, args.image_size, device=device)
        try:
            zi = img_enc({"pixel_values": img})
        except Exception:
            zi = img_enc(img)
        if isinstance(zi, (tuple, list)): zi = zi[0]
        if zi.ndim == 3: zi = zi.mean(dim=1)
        d_img = int(zi.shape[-1])
    labs_pool  = PooledEncoder(labs_enc,  out_dim=d_labs,  name="Labs")
    notes_pool = PooledEncoder(notes_enc, out_dim=d_notes, name="Notes")
    img_pool   = PooledEncoder(img_enc,   out_dim=d_img,   name="Image")
    return labs_pool, notes_pool, img_pool, d_labs, d_notes, d_img

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch["label"].detach().cpu().numpy()
        ys.append(y)
        ps.append(prob)

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)

    auc, f1_05 = compute_auc_f1(y_true, y_prob, thr=0.5)
    best_thr, best_f1 = best_f1_over_thresholds(y_true, y_prob, n=200)
    return {
        "auroc": float(auc),
        "f1@0.5": float(f1_05),
        "best_thr": float(best_thr),
        "best_f1": float(best_f1),
    }

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, opt, device, grad_clip: float = 1.0) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        y = batch["label"]
        loss = F.binary_cross_entropy_with_logits(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")

def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _abs(root: str, p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)

def load_splits_json(path: str) -> Dict[str, set]:
    with open(path, "r") as f:
        j = json.load(f)

    if "splits" in j and isinstance(j["splits"], dict):
        j = j["splits"]

    def pick(keys):
        for k in keys:
            if k in j:
                return j[k]
        return []

    train = pick(["train", "train_ids", "train_id", "train_stay_ids", "train_hadm_ids"])
    val   = pick(["val", "valid", "validation", "val_ids", "valid_ids", "validation_ids"])
    test  = pick(["test", "test_ids", "test_id"])

    return {
        "train": set(train),
        "val": set(val),
        "test": set(test),
    }

def infer_id_col(dfs: List[pd.DataFrame], override: str = "") -> str:
    if override:
        return override
    priority = ["stay_id", "hadm_id", "subject_id", "icustay_id"]
    cols = set(dfs[0].columns)
    for d in dfs[1:]:
        cols = cols.intersection(set(d.columns))
    for c in priority:
        if c in cols:
            return c
    for c in sorted(cols):
        if c.endswith("_id"):
            return c
    raise ValueError(f"Could not infer a shared id column across: {[list(d.columns)[:20] for d in dfs]}")

def reduce_one_row_per_id(df: pd.DataFrame, id_col: str, prefer_cols: List[str]) -> pd.DataFrame:
    if df[id_col].is_unique:
        return df
    agg = {c: "first" for c in df.columns if c != id_col}

    if prefer_cols:
        df2 = df.sort_values(prefer_cols, na_position="last")
    else:
        df2 = df

    out = df2.groupby(id_col, as_index=False).agg(agg)
    return out


def normalize_modal_columns(struct_df: pd.DataFrame, notes_df: pd.DataFrame, images_df: pd.DataFrame, labels_df: pd.DataFrame):
    rename_map = {}

    if "label" in labels_df.columns and "mortality" not in labels_df.columns:
        labels_df = labels_df.rename(columns={"label": "mortality"})

    if "text" in notes_df.columns and "note_text" not in notes_df.columns:
        notes_df = notes_df.rename(columns={"text": "note_text"})

    if "cxr_path" in images_df.columns and "image_path" not in images_df.columns:
        images_df = images_df.rename(columns={"cxr_path": "image_path"})


    return struct_df, notes_df, images_df, labels_df

def build_split_dfs_from_parquets(
    data_root: str,
    splits_path: str,
    structured_path: str,
    notes_path: str,
    images_path: str,
    labels_path: str,
    id_col_override: str = "",
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    splits = load_splits_json(splits_path)

    struct_df = pd.read_parquet(structured_path)
    notes_df  = pd.read_parquet(notes_path)
    images_df = pd.read_parquet(images_path)
    labels_df = pd.read_parquet(labels_path)

    struct_df, notes_df, images_df, labels_df = normalize_modal_columns(struct_df, notes_df, images_df, labels_df)

    id_col = infer_id_col([struct_df, notes_df, images_df, labels_df], override=id_col_override)
    print(f"[DATA] Using id_col={id_col}")
    struct_cols = infer_struct_cols(struct_df, id_col=id_col, label_col="mortality")
    print(f"[STRUCT] numeric feature cols: {len(struct_cols)}")

    images_df = reduce_one_row_per_id(images_df, id_col, prefer_cols=[c for c in ["image_pt", "image_path"] if c in images_df.columns])
    notes_df  = reduce_one_row_per_id(notes_df,  id_col, prefer_cols=[c for c in ["notes_pt", "note_text", "note_path"] if c in notes_df.columns])

    df = struct_df.merge(notes_df, on=id_col, how="left", suffixes=("", "_notes"))
    df = df.merge(images_df, on=id_col, how="left", suffixes=("", "_img"))
    df = df.merge(labels_df[[id_col, "mortality"]] if "mortality" in labels_df.columns else labels_df, on=id_col, how="left")

    def filt(split_name: str) -> pd.DataFrame:
        ids = splits.get(split_name, set())
        if not ids:
            return df.iloc[0:0].copy()
        return df[df[id_col].isin(ids)].reset_index(drop=True)

    df_train = filt("train")
    df_val   = filt("val")
    df_test  = filt("test")

    print("[DEBUG] df_train columns sample:", list(df_train.columns)[:80])
    lab_like = [c for c in df_train.columns if ("lab" in c.lower()) or ("code" in c.lower()) or ("token" in c.lower())]
    print("[DEBUG] lab-like columns:", lab_like)
    print(df_train[lab_like].head(2).to_string())


    print(f"[DATA] merged rows: {len(df)} | train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    if "mortality" in df.columns:
        print("[DATA] mortality missing rate:", float(df["mortality"].isna().mean()))

    return df_train, df_val, df_test, struct_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--splits_json", type=str, default="splits.json")
    ap.add_argument("--structured", type=str, default="structured_mortality.parquet")
    ap.add_argument("--notes", type=str, default="notes_wide_clean.parquet")
    ap.add_argument("--images", type=str, default="images.parquet")
    ap.add_argument("--labels", type=str, default="labels_mortality.parquet")
    ap.add_argument("--id_col", type=str, default="", help="Override merge key (e.g., stay_id/hadm_id/subject_id)")
    ap.add_argument("--labs_n_feats", type=int, default=0, help="Lab vocab size / n_feats for BEHRTLabEncoder (0=auto)")
    ap.add_argument("--labs_d", type=int, default=768, help="Embedding dim d for BEHRTLabEncoder")
    ap.add_argument("--weight_decay", type=float, default=1e-4) 
    ap.add_argument("--labs_mode", type=str, default="auto", choices=["auto", "behrt", "tabular"])

    ap.add_argument("--train", type=str, default="")
    ap.add_argument("--val", type=str, default="")
    ap.add_argument("--test", type=str, default="")
    ap.add_argument("--label_col", type=str, default="mortality")

    ap.add_argument("--tokenizer_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--notes_max_len", type=int, default=256)
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--freeze_encoders", action="store_true")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", type=str, default="trimf_best.pt")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--load_path", type=str, default="")

    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    if isinstance(device, str):
        device = torch.device(device)
    print(f"Device: {device} | cuda_available={torch.cuda.is_available()}")


    data_root = os.path.abspath(args.data_root)

    if args.train or args.val or args.test:
        if not args.eval_only and not args.train:
            raise ValueError("Provide --train (csv/parquet) or omit and use --data_root + splits/parquets.")
        df_train = read_table(args.train) if (args.train and not args.eval_only) else None
        df_val   = read_table(args.val) if args.val else None
        df_test  = read_table(args.test) if args.test else None
    else:
        splits_path    = _abs(data_root, args.splits_json)
        structured_path= _abs(data_root, args.structured)
        notes_path     = _abs(data_root, args.notes)
        images_path    = _abs(data_root, args.images)
        labels_path    = _abs(data_root, args.labels)

        for p in [splits_path, structured_path, notes_path, images_path, labels_path]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing required file: {p}")

        df_train, df_val, df_test, struct_cols = build_split_dfs_from_parquets(
            data_root=data_root,
            splits_path=splits_path,
            structured_path=structured_path,
            notes_path=notes_path,
            images_path=images_path,
            labels_path=labels_path,
            id_col_override=args.id_col,
        )

        struct_in_dim = infer_struct_in_dim(df_train if df_train is not None else df_val, struct_cols)
        print(f"[STRUCT] struct_cols={struct_cols} | struct_in_dim={struct_in_dim}")

        if args.eval_only:
            df_train = None


    def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = MortalityTriModalDataset(
            df,
            label_col=args.label_col,
            tokenizer_name=args.tokenizer_name,
            notes_max_len=args.notes_max_len,
            image_size=args.image_size,
            labs_mode=args.labs_mode,
            struct_cols=struct_cols if args.labs_mode == "tabular" else None,
            struct_in_dim=struct_in_dim if args.labs_mode == "tabular" else 0,   # NEW
        )

        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_trimodal,
        )


    train_loader = make_loader(df_train, shuffle=True) if df_train is not None else None
    val_loader = make_loader(df_val, shuffle=False) if df_val is not None else None
    test_loader = make_loader(df_test, shuffle=False) if df_test is not None else None

    if args.labs_mode == "behrt":
        if args.labs_n_feats <= 0:
            probe_df = df_train if df_train is not None else (df_val if df_val is not None else df_test)
            inferred = infer_labs_n_feats_from_df(probe_df)
            if inferred is None:
                raise RuntimeError(
                    "Could not infer labs_n_feats (no lab_codes/labs_json/labs_pt with input_ids found). "
                    "Pass --labs_n_feats explicitly or use --labs_mode tabular."
                )
            args.labs_n_feats = int(inferred)
        print(f"[LABS] labs_n_feats={args.labs_n_feats} | labs_d={args.labs_d}")
    else:
        print(f"[LABS] tabular in_dim={struct_in_dim} | labs_d={args.labs_d}")

    labs_enc, notes_enc, img_enc, d_labs, d_notes, d_img = build_encoders(args, device, struct_in_dim=struct_in_dim)

    model = TriMFMortalityModel(
        labs_encoder=labs_enc,
        notes_encoder=notes_enc,
        image_encoder=img_enc,
        d_labs=d_labs,
        d_notes=d_notes,
        d_image=d_img,
        d_model=args.d_model,
        dropout=args.dropout,
    ).to(device)

    if args.load_path and os.path.isfile(args.load_path):
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
        print(f"Loaded weights from: {args.load_path}")

    if args.eval_only:
        if val_loader is not None:
            m = evaluate(model, val_loader, device)
            print(f"[VAL]  AUROC={m['auroc']:.4f} | F1@0.5={m['f1@0.5']:.4f} | bestF1={m['best_f1']:.4f} @thr={m['best_thr']:.3f}")
        if test_loader is not None:
            m = evaluate(model, test_loader, device)
            print(f"[TEST] AUROC={m['auroc']:.4f} | F1@0.5={m['f1@0.5']:.4f} | bestF1={m['best_f1']:.4f} @thr={m['best_thr']:.3f}")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, device, grad_clip=1.0)

        msg = f"Epoch {epoch}/{args.epochs} | train_loss={loss:.4f}"
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            msg += f" | val_AUROC={metrics['auroc']:.4f} | val_F1@0.5={metrics['f1@0.5']:.4f} | val_bestF1={metrics['best_f1']:.4f}@{metrics['best_thr']:.3f}"
            val_auc = metrics["auroc"]
            if not math.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        dt = time.time() - t0
        print(msg + f" | {dt:.1f}s")

    if best_state is not None:
        payload = {"model": best_state, "best_val_auroc": best_val_auc, "args": vars(args)}
        torch.save(payload, args.save_path)
        print(f"Saved best checkpoint to: {args.save_path} (best_val_auroc={best_val_auc:.4f})")
        model.load_state_dict(best_state, strict=False)

    if test_loader is not None:
        m = evaluate(model, test_loader, device)
        print(f"[TEST] AUROC={m['auroc']:.4f} | F1@0.5={m['f1@0.5']:.4f} | bestF1={m['best_f1']:.4f} @thr={m['best_thr']:.3f}")


if __name__ == "__main__":
    main()
