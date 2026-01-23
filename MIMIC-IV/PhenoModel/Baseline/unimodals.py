from __future__ import annotations

import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass
try:
    import torchvision.transforms as T
except Exception:
    T = None
import encoders as enc_mod
from env_config import CFG

DEFAULT_PHENOS_25 = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
]

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)([to_device(v, device) for v in x])
    return x


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    if not np.isfinite(y_score).all():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if average_precision_score is None:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    if not np.isfinite(y_score).all():
        return float("nan")
    return float(average_precision_score(y_true, y_score))

def _prf1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    return prec, rec, f1


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray, n_grid: int = 101) -> np.ndarray:
    assert y_true.shape == y_prob.shape
    k = y_true.shape[1]
    grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    thr = np.full((k,), 0.5, dtype=np.float32)

    for j in range(k):
        yj = y_true[:, j]
        pj = y_prob[:, j]
        if np.unique(yj).size < 2:
            thr[j] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (pj >= t).astype(np.int32)
            _, _, f1 = _prf1(yj, pred)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[j] = best_t
    return thr


def compute_report(y_true: np.ndarray, y_prob: np.ndarray, names: List[str], thresholds: np.ndarray) -> Dict[str, Any]:
    assert y_true.shape == y_prob.shape
    k = y_true.shape[1]
    assert len(names) == k
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)
    per = []
    aucs, aprs, f1s, precs, recs = [], [], [], [], []
    for j in range(k):
        yj, pj, predj = y_true[:, j], y_prob[:, j], y_pred[:, j]
        auroc = _safe_auc(yj, pj)
        auprc = _safe_auprc(yj, pj)
        prec, rec, f1 = _prf1(yj, predj)

        per.append({
            "phenotype": names[j],
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "threshold": float(thresholds[j]),
            "pos_rate": float(np.mean(yj)),
        })

        if not math.isnan(auroc):
            aucs.append(auroc)
        if not math.isnan(auprc):
            aprs.append(auprc)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    macro = {
        "auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "auprc": float(np.mean(aprs)) if aprs else float("nan"),
        "f1": float(np.mean(f1s)) if f1s else float("nan"),
        "precision": float(np.mean(precs)) if precs else float("nan"),
        "recall": float(np.mean(recs)) if recs else float("nan"),
    }
    return {"macro": macro, "per_label": per}


def print_report(tag: str, rep: Dict[str, Any]) -> None:
    m = rep["macro"]
    print(f"\n[{tag}] MACRO  AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
          f"F1={m['f1']:.4f} Recall={m['recall']:.4f} Precision={m['precision']:.4f}")
    print(f"[{tag}] Per-phenotype metrics:")
    for row in rep["per_label"]:
        print(f"  {row['phenotype']:<60s} "
              f"AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} "
              f"F1={row['f1']:.4f} Recall={row['recall']:.4f} Precision={row['precision']:.4f} "
              f"thr={row['threshold']:.2f}")

def load_splits(path: Path) -> Dict[str, List[Any]]:
    obj = json.loads(path.read_text())
    out = {}
    for k in obj.keys():
        lk = k.lower()
        if "train" in lk:
            out["train"] = obj[k]
        elif lk in ("val", "valid", "validation") or "val" in lk:
            out["val"] = obj[k]
        elif "test" in lk:
            out["test"] = obj[k]
    if not all(s in out for s in ("train", "val", "test")):
        raise RuntimeError(f"splits.json must contain train/val/test lists. Found keys={list(obj.keys())}")
    return out

def detect_id_column(dfs: List[pd.DataFrame]) -> str:
    preferred = ["stay_id", "icustay_id", "hadm_id", "admission_id", "subject_id", "dicom_id", "study_id", "image_id"]
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)
    common_lower = {c.lower(): c for c in common}

    for p in preferred:
        if p in common_lower:
            return common_lower[p]

    id_like = [c for c in common if c.lower().endswith("_id")]
    if id_like:
        id_like.sort(key=lambda x: (len(x), x))
        return id_like[0]

    raise RuntimeError(f"Could not detect a common id column across dataframes. Common={sorted(list(common))[:50]}")


def detect_label_columns(labels_df: pd.DataFrame, id_col: str) -> List[str]:
    exclude = {id_col.lower(), "subject_id", "hadm_id", "stay_id", "icustay_id"}
    cols = []
    for c in labels_df.columns:
        if c.lower() in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(labels_df[c]):
            continue
        s = labels_df[c].dropna().unique()
        if len(s) == 0:
            continue
        if set(np.unique(s)).issubset({0, 1}):
            cols.append(c)
    if len(cols) != 25:
        raise RuntimeError(f"Expected exactly 25 binary label cols, found {len(cols)}: {cols}")
    return cols


def build_split_df(
    modality_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    split_ids: List[Any],
    id_col: str,
    label_cols: List[str],
) -> pd.DataFrame:
    sid = pd.Series(split_ids, name=id_col)
    keep = modality_df.merge(sid.to_frame(), on=id_col, how="inner")
    merged = keep.merge(labels_df[[id_col] + label_cols], on=id_col, how="inner")
    return merged

class StructuredDataset(torch.utils.data.Dataset):
    """
    Expects structured content in one of these forms:
      - a single column containing array/list (e.g., "x", "xehr", "features", "seq")
      - or many numeric columns (tabular vector)
    """
    def __init__(self, df: pd.DataFrame, id_col: str, label_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.label_cols = label_cols

        cand_cols = ["x", "xehr", "features", "feat", "seq", "sequence", "L", "structured"]
        self.arr_col = None
        for c in cand_cols:
            if c in self.df.columns:
                self.arr_col = c
                break

        if self.arr_col is None:
            excl = set([id_col] + label_cols)
            num_cols = [c for c in self.df.columns if c not in excl and pd.api.types.is_numeric_dtype(self.df[c])]
            if not num_cols:
                raise RuntimeError("StructuredDataset: could not find an array column or numeric feature columns.")
            self.num_cols = num_cols
        else:
            self.num_cols = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = torch.tensor(row[self.label_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        sid = row[self.id_col]

        if self.arr_col is not None:
            x = row[self.arr_col]
            x = np.asarray(x)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            if x.ndim == 1:
                xt = torch.tensor(x.astype(np.float32), dtype=torch.float32)
                return {"x": xt, "mask": None}, y, sid
            elif x.ndim == 2:
                xt = torch.tensor(x.astype(np.float32), dtype=torch.float32)
                return {"x": xt, "mask": None}, y, sid
            else:
                raise RuntimeError(f"Structured array has ndim={x.ndim}, expected 1 or 2.")
        else:
            vec = row[self.num_cols].to_numpy(dtype=np.float32)
            if not np.isfinite(vec).all():
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

            xt = torch.tensor(vec, dtype=torch.float32)
            return {"x": xt, "mask": None}, y, sid


def collate_structured(batch):
    xs, ys, ids = [], [], []
    for x, y, sid in batch:
        ids.append(sid)
        ys.append(y)
        xs.append(x["x"])

    y = torch.stack(ys, dim=0)

    def _norm_one(xx: torch.Tensor) -> torch.Tensor:
        if xx.ndim == 1:
            f = xx.numel()
            if f == 62:
                xx = xx[:61]
                f = 61
            return xx.unsqueeze(0)  # [1,f]

        if xx.ndim == 2:
            T, F = xx.shape

            if T in (61, 62) and F not in (61, 62):
                xx = xx.t().contiguous()
                T, F = xx.shape

            if F == 62:
                xx = xx[:, :61]

            return xx
        raise RuntimeError(f"Structured sample has shape {tuple(xx.shape)}; expected 1D or 2D.")


    xs = [ _norm_one(xx) for xx in xs ]
    is_seq = any(xx.shape[0] > 1 for xx in xs)

    if not is_seq:
        x = torch.stack(xs, dim=0)  # [B,1,F]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -10.0, 10.0)
        mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool)
        return {"x": x, "mask": mask}, y, ids

    maxT = max(xx.shape[0] for xx in xs)
    Fdim = xs[0].shape[1]
    xpad = torch.zeros(len(xs), maxT, Fdim, dtype=torch.float32)
    mask = torch.zeros(len(xs), maxT, dtype=torch.bool)
    for i, xx in enumerate(xs):
        Tlen = xx.shape[0]
        xpad[i, :Tlen] = xx
        mask[i, :Tlen] = True

    xpad = torch.nan_to_num(xpad, nan=0.0, posinf=0.0, neginf=0.0)
    xpad = torch.clamp(xpad, -10.0, 10.0)

    return {"x": xpad, "mask": mask}, y, ids

class NotesDataset(torch.utils.data.Dataset):
    """
    Returns chunked tensors:
      input_ids:      [S, L]
      attention_mask: [S, L]
      chunk_mask:     [S]  (1 if this chunk is valid)
    """
    def __init__(self, df: pd.DataFrame, id_col: str, label_cols: List[str],
                 max_len: int = 256, max_chunks: int = 16):
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.label_cols = label_cols
        self.max_len = int(max_len)
        self.max_chunks = int(max_chunks)
        self.input_cols = [c for c in self.df.columns if c.startswith("input_ids_")]
        self.attn_cols  = [c for c in self.df.columns if c.startswith("attn_mask_")]

        if not self.input_cols or not self.attn_cols:
            raise RuntimeError(
                f"NotesDataset: expected input_ids_* and attn_mask_* columns. "
                f"Found inputs={len(self.input_cols)} attn={len(self.attn_cols)}"
            )

        def _idx(c):
            try: return int(c.split("_")[-1])
            except: return 10**9

        self.input_cols.sort(key=_idx)
        self.attn_cols.sort(key=_idx)
        self.input_cols = self.input_cols[:self.max_chunks]
        self.attn_cols  = self.attn_cols[:self.max_chunks]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = torch.tensor(row[self.label_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        sid = row[self.id_col]
        S = len(self.input_cols)
        L = self.max_len
        ids_out  = torch.zeros(S, L, dtype=torch.long)
        attn_out = torch.zeros(S, L, dtype=torch.long)
        cmask    = torch.zeros(S, dtype=torch.float32)

        for s, (ic, ac) in enumerate(zip(self.input_cols, self.attn_cols)):
            ids = row[ic]
            attn = row[ac]

            if isinstance(ids, np.ndarray):  ids = ids.tolist()
            if isinstance(attn, np.ndarray): attn = attn.tolist()
            if ids is None or attn is None:
                continue

            # truncate/pad to L
            ids = [int(t) for t in ids[:L]]
            attn = [int(t) for t in attn[:L]]
            if len(ids) == 0:
                continue

            ids_out[s, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_out[s, :len(attn)] = torch.tensor(attn, dtype=torch.long)
            cmask[s] = 1.0 if sum(attn) > 0 else 0.0

        return {
            "input_ids": ids_out,                 # [S,L]
            "attention_mask": attn_out,           # [S,L]
            "chunk_mask": cmask,                  # [S]
        }, y, sid

def collate_notes(batch):
    xs = [b[0] for b in batch]
    ys = torch.stack([b[1] for b in batch], dim=0)
    ids = [b[2] for b in batch]

    input_ids = torch.stack([x["input_ids"] for x in xs], dim=0)          # [B,S,L]
    attn_mask = torch.stack([x["attention_mask"] for x in xs], dim=0)     # [B,S,L]
    chunk_mask = torch.stack([x["chunk_mask"] for x in xs], dim=0)        # [B,S]

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "chunk_mask": chunk_mask,
    }, ys, ids


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, id_col: str, label_cols: List[str], image_root: Optional[str], size: int = 224):
        if Image is None or T is None:
            raise RuntimeError("ImageDataset requires pillow + torchvision installed.")
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.label_cols = label_cols
        self.image_root = image_root
        self.size = size

        cand = ["cxr_path", "path", "image_path", "img_path", "png_path", "jpg_path", "filepath", "file_path"]
        self.path_col = None
        for c in cand:
            if c in self.df.columns:
                self.path_col = c
                break
        if self.path_col is None:
            raise RuntimeError(f"ImageDataset: could not find image path column in {list(self.df.columns)[:50]}")

        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),  
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = torch.tensor(row[self.label_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        sid = row[self.id_col]
        p = str(row[self.path_col])

        if self.image_root is not None and not os.path.isabs(p):
            p = str(Path(self.image_root) / p) 

        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        return {"image": x}, y, sid


def collate_images(batch):
    xs = torch.stack([b[0]["image"] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    ids = [b[2] for b in batch]
    return {"image": xs}, ys, ids

def patch_cfg_for_encoders(cfg) -> None:
    def _setdefault(name: str, val):
        if not hasattr(cfg, name):
            setattr(cfg, name, val)

    _setdefault("d", 256)
    _setdefault("dropout", 0.1)
    _setdefault("structured_pool", "mean")   
    _setdefault("structured_seq_len", 256)
    _setdefault("structured_in_dim", None)
    _setdefault("structured_heads", 4)
    _setdefault("structured_layers", 2)
    _setdefault("structured_hidden", cfg.d if hasattr(cfg, "d") else 256)
    _setdefault("ehr_pool", getattr(cfg, "structured_pool", "mean"))
    _setdefault("ehr_seq_len", getattr(cfg, "structured_seq_len", 256))
    _setdefault("ehr_heads", getattr(cfg, "structured_heads", 4))
    _setdefault("text_model", "emilyalsentzer/Bio_ClinicalBERT")
    _setdefault("text_max_len", 256)
    _setdefault("text_pool", "cls")         
    _setdefault("freeze_text", False)
    _setdefault("notes_model", getattr(cfg, "text_model"))
    _setdefault("notes_max_len", getattr(cfg, "text_max_len"))
    _setdefault("image_size", 224)
    _setdefault("image_arch", "resnet18")
    _setdefault("freeze_image", False)
    _setdefault("img_agg", "mean")         
    _setdefault("img_pool", getattr(cfg, "img_agg", "mean"))
    _setdefault("img_pretrained", True)
    _setdefault("img_channels", 3)

def freeze_bert_backbone_if_present(encoder: nn.Module) -> None:
    if hasattr(encoder, "bert"):
        for p in encoder.bert.parameters():
            p.requires_grad = False

def freeze_vision_backbone_if_present(encoder: nn.Module) -> None:
    for attr in ["backbone", "vision", "vision_model", "cnn", "resnet", "vit", "encoder"]:
        if hasattr(encoder, attr) and isinstance(getattr(encoder, attr), nn.Module):
            for p in getattr(encoder, attr).parameters():
                p.requires_grad = False
            break

def list_encoder_candidates(enc_mod) -> List[str]:
    out = []
    for name in dir(enc_mod):
        if name.startswith("_"):
            continue
        obj = getattr(enc_mod, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            out.append(name)
    return out


def build_encoder(modality: str, d: int, encoder_name: Optional[str] = None) -> nn.Module:
    import encoders as enc_mod  
    modality = modality.upper()
    cfg = None
    try:
        from env_config import CFG as _CFG 
        cfg = _CFG
    except Exception:
        cfg = None

    for fn_name in ["build_unimodal_encoder", "make_unimodal_encoder", "get_unimodal_encoder"]:
        if hasattr(enc_mod, fn_name):
            fn = getattr(enc_mod, fn_name)
            try:
                return fn(modality=modality, d=d, cfg=cfg)  
            except TypeError:
                pass
            try:
                return fn(modality=modality, d=d)            
            except TypeError:
                pass
            try:
                return fn(modality=modality)               
            except TypeError:
                return fn(modality)

    for fn_name in ["build_encoders"]:
        if hasattr(enc_mod, fn_name):
            fn = getattr(enc_mod, fn_name)

            enc_cfg = None
            if hasattr(enc_mod, "EncoderConfig"):
                EC = getattr(enc_mod, "EncoderConfig")

                enc_cfg = EC(
                    d=getattr(cfg, "d", d),
                    dropout=getattr(cfg, "dropout", 0.0),

                    structured_seq_len=getattr(cfg, "structured_seq_len", 256),
                    structured_n_feats=getattr(cfg, "structured_n_feats", 61),
                    structured_layers=getattr(cfg, "structured_layers", 2),
                    structured_heads=getattr(cfg, "structured_heads", 8),
                    structured_pool=getattr(cfg, "structured_pool", "cls"),

                    text_model_name=getattr(cfg, "text_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
                    text_max_len=getattr(cfg, "max_text_len", 512),
                    note_agg=getattr(cfg, "note_agg", "cls"),
                    bert_chunk_bs=getattr(cfg, "bert_chunk_bs", 8),

                    img_agg=getattr(cfg, "img_agg", "last"),
                    vision_backbone=getattr(cfg, "vision_backbone",
                                            getattr(cfg, "image_model_name", "resnet34")),
                    vision_num_classes=getattr(cfg, "vision_num_classes", 14),
                    vision_pretrained=getattr(cfg, "vision_pretrained", True),
                )

            if enc_cfg is not None:
                behrt, bbert, imgenc = fn(enc_cfg)
            else:
                behrt, bbert, imgenc = fn(cfg)

            if modality == "L":
                return behrt
            if modality == "N":
                return bbert
            if modality == "I":
                return imgenc

    if encoder_name is not None:
        if not hasattr(enc_mod, encoder_name):
            raise RuntimeError(f"--encoder_name {encoder_name} not found in encoders.py")
        cls = getattr(enc_mod, encoder_name)
        if not isinstance(cls, type):
            raise RuntimeError(f"{encoder_name} exists but is not a class.")
        try:
            return cls(d=d, cfg=cfg)
        except TypeError:
            pass
        try:
            return cls(d=d)
        except TypeError:
            return cls()

    candidates = list_encoder_candidates(enc_mod)
    pats = {
        "L": ["Structured", "EHR", "Lab", "BEHRT", "Tabular"],
        "N": ["Text", "Notes", "BERT", "Clinical"],
        "I": ["Image", "CXR", "Vision", "ResNet", "ViT"],
    }[modality]

    scored = []
    for name in candidates:
        score = sum(1 for p in pats if p.lower() in name.lower())
        if score > 0:
            scored.append((score, name))
    scored.sort(reverse=True)

    if scored:
        best = scored[0][1]
        cls = getattr(enc_mod, best)
        try:
            return cls(d=d, cfg=cfg)
        except TypeError:
            pass
        try:
            return cls(d=d)
        except TypeError:
            return cls()

    raise RuntimeError(
        f"Could not auto-build encoder for modality {modality} from encoders.py.\n"
        f"Available nn.Module classes in encoders.py:\n  {candidates}\n"
        f"Fix by passing --encoder_name <ClassName> or adding a factory function in encoders.py."
    )

class UniModel(nn.Module):
    """
    Adapts batch format to match encoders.py expectations:
      - L (structured): encoder expects Tensor x: [B,T,F] (optionally mask)
      - N (notes): encoder expects dict with input_ids/attention_mask/chunk_mask
      - I (images): encoder expects Tensor [B,C,H,W] or list, NOT dict
    """
    def __init__(self, encoder: nn.Module, d: int, num_labels: int = 25, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_labels))

    def _unwrap_embedding(self, z: Any) -> torch.Tensor:
        if torch.is_tensor(z):
            emb = z
        elif isinstance(z, (tuple, list)) and len(z) > 0 and torch.is_tensor(z[0]):
            emb = z[0]
        elif (z, dict):
            for k in ["emb", "z", "pooled", "cls", "feat", "features", "h"]:
                if k in z and torch.is_tensor(z[k]):
                    emb = z[k]
                    break
            else:
                tens = [v for v in z.values() if torch.is_tensor(v)]
                if len(tens) == 1:
                    emb = tens[0]
                else:
                    raise RuntimeError(f"Encoder returned dict without recognizable embedding: keys={list(z.keys())}")
        else:
            raise RuntimeError(f"Unsupported encoder output type: {type(z)}")

        # [B,T,d] -> [B,d]
        if emb.ndim == 3:
            emb = emb.mean(dim=1)

        if not torch.isfinite(emb).all():
            print("[NaN/Inf] encoder embedding non-finite!")
            print(" emb stats:", emb.nan_to_num().min().item(), emb.nan_to_num().max().item())
            raise RuntimeError("Non-finite encoder embedding")

        return emb

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(batch, dict):
            raise RuntimeError(f"Expected dict batch, got {type(batch)}")

        if "x" in batch:
            x = batch["x"]  # [B,T,F]
            mask = batch.get("mask", None)
            try:
                z = self.encoder(x, mask=mask)
            except TypeError:
                z = self.encoder(x)

        elif "image" in batch:
            z = self.encoder(batch["image"])

        elif "input_ids" in batch and "attention_mask" in batch:
            z = self.encoder(batch)

        else:
            raise RuntimeError(f"Unknown batch keys: {list(batch.keys())}")

        emb = self._unwrap_embedding(z)
        return self.head(emb)

@torch.no_grad()
def eval_loop(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    model.eval()
    all_logits, all_y, all_ids = [], [], []
    for x, y, ids in loader:
        x = to_device(x, device)
        y = to_device(y, device)
        logits = model(x)

        if not torch.isfinite(logits).all():
            bad = (~torch.isfinite(logits)).nonzero(as_tuple=False)
            print(f"[NaN/Inf] logits has non-finite values! batch_ids_sample={ids[:3]}")
            print(f"[NaN/Inf] logits stats: min={logits.nan_to_num().min().item():.4g} "
                  f"max={logits.nan_to_num().max().item():.4g}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

        all_logits.append(logits.detach().float().cpu().numpy())
        all_y.append(y.detach().float().cpu().numpy())
        all_ids.extend(ids)
    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    prob_np = sigmoid_np(logits_np)
    return logits_np, prob_np, y_np, all_ids


def train_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    names: List[str],
    device: torch.device,
    epochs: int,
    lr: float,
    wd: float,
    grad_clip: float,
) -> Tuple[nn.Module, np.ndarray, Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_score = -1e9
    best_state = None
    best_thr = np.full((25,), 0.5, dtype=np.float32)
    best_rep = None

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for x, y, _ in train_loader:
            x = to_device(x, device)
            y = to_device(y, device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                print("[NaN/Inf] train logits went non-finite")
                if isinstance(x, dict) and "x" in x and torch.is_tensor(x["x"]):
                    xx = x["x"]
                    print(f"[NaN/Inf] input x stats: min={xx.nan_to_num().min().item():.4g} "
                          f"max={xx.nan_to_num().max().item():.4g}")
                    print(f"[NaN/Inf] input x has_nonfinite={not torch.isfinite(xx).all()}")
                raise RuntimeError("Non-finite logits in train_loop")

            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.item()))

        # VAL
        _, val_prob, val_y, _ = eval_loop(model, val_loader, device)
        thr = find_best_thresholds(val_y, val_prob)
        rep = compute_report(val_y, val_prob, names=names, thresholds=thr)

        score = 0.5 * rep["macro"]["auprc"] + 0.5 * rep["macro"]["auroc"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr
            best_rep = rep

        dt = time.time() - t0
        print(f"[epoch {ep:03d}] loss={np.mean(losses):.4f} "
              f"VAL(AUROC={rep['macro']['auroc']:.4f} AUPRC={rep['macro']['auprc']:.4f} F1={rep['macro']['f1']:.4f}) "
              f"time={dt:.1f}s {'*' if score == best_score else ''}")

    assert best_state is not None and best_rep is not None
    model.load_state_dict(best_state)
    return model, best_thr, best_rep


def save_preds_csv(path: Path, ids: List[Any], y_true: np.ndarray, y_prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    K = y_true.shape[1]
    out = pd.DataFrame({ "id": ids })
    for j in range(K):
        out[f"y_{j:02d}"] = y_true[:, j].astype(int)
    for j in range(K):
        out[f"p_{j:02d}"] = y_prob[:, j].astype(float)
    out.to_csv(path, index=False)

def get_pheno_names() -> List[str]:
    return DEFAULT_PHENOS_25

def run_modality(args, modality: str) -> None:
    modality = modality.upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    ckpt_dir = ensure_dir(Path(args.ckpt_root) / f"uni_{modality}")
    splits = load_splits(data_root / args.splits_json)
    dfL = pd.read_parquet(data_root / args.xehr_parquet)
    dfN = pd.read_parquet(data_root / args.notes_parquet)
    dfI = pd.read_parquet(data_root / args.images_parquet)
    dfY = pd.read_parquet(data_root / args.labels_parquet)

    if modality == "L":
        id_col = detect_id_column([dfL, dfY])
    elif modality == "N":
        id_col = detect_id_column([dfN, dfY])
    elif modality == "I":
        id_col = detect_id_column([dfI, dfY])
    else:
        raise ValueError(modality)

    label_cols = detect_label_columns(dfY, id_col=id_col)
    pheno_names = label_cols if len(label_cols) == 25 else get_pheno_names()

    # Build split dataframes for this modality
    if modality == "L":
        train_df = build_split_df(dfL, dfY, splits["train"], id_col, label_cols)
        val_df   = build_split_df(dfL, dfY, splits["val"],   id_col, label_cols)
        test_df  = build_split_df(dfL, dfY, splits["test"],  id_col, label_cols)
        train_ds = StructuredDataset(train_df, id_col, label_cols)
        val_ds   = StructuredDataset(val_df,   id_col, label_cols)
        test_ds  = StructuredDataset(test_df,  id_col, label_cols)
        collate_fn = collate_structured

    elif modality == "N":
        train_df = build_split_df(dfN, dfY, splits["train"], id_col, label_cols)
        val_df   = build_split_df(dfN, dfY, splits["val"],   id_col, label_cols)
        test_df  = build_split_df(dfN, dfY, splits["test"],  id_col, label_cols)
        train_ds = NotesDataset(train_df, id_col, label_cols, max_len=args.max_len, max_chunks=args.max_chunks)
        val_ds   = NotesDataset(val_df,   id_col, label_cols, max_len=args.max_len, max_chunks=args.max_chunks)
        test_ds  = NotesDataset(test_df,  id_col, label_cols, max_len=args.max_len, max_chunks=args.max_chunks)
        collate_fn = collate_notes

    elif modality == "I":
        train_df = build_split_df(dfI, dfY, splits["train"], id_col, label_cols)
        val_df   = build_split_df(dfI, dfY, splits["val"],   id_col, label_cols)
        test_df  = build_split_df(dfI, dfY, splits["test"],  id_col, label_cols)
        train_ds = ImageDataset(train_df, id_col, label_cols, image_root=args.image_root, size=args.image_size)
        val_ds   = ImageDataset(val_df,   id_col, label_cols, image_root=args.image_root, size=args.image_size)
        test_ds  = ImageDataset(test_df,  id_col, label_cols, image_root=args.image_root, size=args.image_size)
        collate_fn = collate_images

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    bx, by, _ = next(iter(train_loader))
    if isinstance(bx, dict):
        print("[dbg] bx keys:", list(bx.keys()))
        for k, v in bx.items():
            if torch.is_tensor(v):
                print(f"[dbg] {k}: {tuple(v.shape)} dtype={v.dtype}")
    else:
        print("[dbg] bx:", type(bx))
    print("[dbg] y:", tuple(by.shape))

    try:
        from env_config import CFG
        CFG.d = args.d
        patch_cfg_for_encoders(CFG)
        CFG.finetune_text = True if modality == "N" else getattr(CFG, "finetune_text", False)
        if not hasattr(CFG, "note_agg"):
            CFG.note_agg = "cls"
        if not hasattr(CFG, "bert_chunk_bs"):
            CFG.bert_chunk_bs = 8
        if not hasattr(CFG, "structured_n_feats"):
            CFG.structured_n_feats = 61
        if not hasattr(CFG, "img_agg"):
            CFG.img_agg = "mean"
    except Exception as e:
        print(f"[warn] could not patch CFG: {e}")

    encoder = build_encoder(modality=modality, d=args.d, encoder_name=args.encoder_name)

    if modality == "N" and (not args.finetune_text):
        freeze_bert_backbone_if_present(encoder)

    if modality == "I" and (not args.finetune_image):
        freeze_vision_backbone_if_present(encoder)
   
    model = UniModel(encoder=encoder, d=args.d, num_labels=25, dropout=args.dropout).to(device)

    model, val_thr, val_rep = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.weight_decay,
        grad_clip=args.grad_clip,
        names=pheno_names,
    )

    _, val_prob, val_y, val_ids = eval_loop(model, val_loader, device)
    val_rep = compute_report(val_y, val_prob, names=pheno_names, thresholds=val_thr)
    print_report(f"VAL ({modality}, VAL thresholds)", val_rep)

    _, test_prob, test_y, test_ids = eval_loop(model, test_loader, device)
    test_rep = compute_report(test_y, test_prob, names=pheno_names, thresholds=val_thr)
    print_report(f"TEST ({modality}, VAL thresholds)", test_rep)

    torch.save({"model_state": model.state_dict(), "val_thresholds": val_thr}, ckpt_dir / "best.pt")
    (ckpt_dir / "reports").mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "reports" / "val_report.json").open("w") as f:
        json.dump(val_rep, f, indent=2)
    with (ckpt_dir / "reports" / "test_report.json").open("w") as f:
        json.dump(test_rep, f, indent=2)

    save_preds_csv(ckpt_dir / "preds_val.csv",  val_ids,  val_y,  val_prob)
    save_preds_csv(ckpt_dir / "preds_test.csv", test_ids, test_y, test_prob)

    print(f"\n[done] modality={modality} saved -> {ckpt_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt_root", type=str, required=True)
    p.add_argument("--splits_json", type=str, default="splits.json")
    p.add_argument("--xehr_parquet", type=str, default="xehr_haru17_2h_76.parquet")
    p.add_argument("--notes_parquet", type=str, default="notes_fullstay_radiology_TEXTCHUNKS_11230.parquet")
    p.add_argument("--images_parquet", type=str, default="images.parquet")
    p.add_argument("--labels_parquet", type=str, default="labels_pheno.parquet")
    p.add_argument("--modality", type=str, default="L", choices=["L", "N", "I", "all", "ALL"])
    p.add_argument("--d", type=int, default=256, help="embedding dim expected from encoder")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--encoder_name", type=str, default=None,
                   help="Optional explicit encoder class name from encoders.py")
    p.add_argument("--text_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--max_chunks", type=int, default=32)
    p.add_argument("--finetune_text", action="store_true",
                   help="Finetune BERT backbone (notes). Default: freeze backbone but train projection/head.")
    p.add_argument("--finetune_image", action="store_true",
                   help="Finetune vision backbone (images). Default: freeze backbone.")
    p.add_argument("--image_root", type=str, default=None,
                   help="If image paths in images.parquet are relative, they are resolved under this root.")
    p.add_argument("--image_size", type=int, default=224)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    mod = args.modality.upper()
    if mod == "ALL":
        for m in ["L", "N", "I"]:
            print(f"\n==================== UNIMODAL {m} ====================")
            run_modality(args, m)
    else:
        run_modality(args, mod)


if __name__ == "__main__":
    main()
