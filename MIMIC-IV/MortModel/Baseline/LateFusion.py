from __future__ import annotations
import os as _os
_os.environ.setdefault("HF_HOME", _os.path.expanduser("~/.cache/huggingface"))
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import os
import json
import argparse
import random
from dataclasses import asdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch import amp as torch_amp
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import matplotlib
matplotlib.use("Agg")
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from env_config import CFG, DEVICE, load_cfg, ensure_dir
from env_config import apply_cli_overrides

from encoders import (
    EncoderConfig,
    build_encoders,
    encode_unimodal_pooled,  
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TOKENIZER: Optional[Any] = None
MAXLEN: int = 512
CHUNK_STRIDE: int = 128

def _cfg(name: str, default):
    return getattr(CFG, name, default)

def grads_are_finite(param_list):
    for p in param_list:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True

def seed_worker(worker_id: int):
    import env_config as E
    if not hasattr(E, "CFG") or E.CFG is None:
        E.load_cfg()
    global CFG, DEVICE
    CFG = E.CFG
    DEVICE = E.DEVICE

    ws = (int(CFG.seed) + int(worker_id)) % (2**32)
    np.random.seed(ws); random.seed(ws); torch.manual_seed(ws)

    global TOKENIZER, MAXLEN
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(CFG.text_model_name)
        MAXLEN = int(getattr(CFG, "max_text_len", 512))

def _standardize_id_column(df: pd.DataFrame, name="stay_id") -> pd.DataFrame:
    candidates = [name, "icustay_id", "stay", "sample_id"]
    found = next((c for c in candidates if c in df.columns), None)
    if found is None:
        raise ValueError(f"Missing id column. Tried {candidates}. Found: {list(df.columns)[:50]}")
    if found != name:
        df = df.rename(columns={found: name})
    df[name] = df[name].astype(int)
    return df

def _standardize_image_path_column(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "cxr_path", "CXR_PATH",
        "image_path", "img_path", "path",
        "dicom_path", "png_path", "jpg_path",
    ]
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        raise ValueError(
            f"[ICUStayDataset] images.parquet missing an image path column. "
            f"Tried: {candidates}. Found columns: {list(df.columns)[:50]}"
        )
    if found != "cxr_path":
        df = df.rename(columns={found: "cxr_path"})
    return df

VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dcm"}

def is_probably_image_file(p: str) -> bool:
    ext = os.path.splitext(str(p).lower())[1]
    return ext in VALID_IMG_EXTS

def resolve_image_path(p: str, dataset_root: str) -> str:
    p = str(p).strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    p = p.replace("\\", "/").lstrip("/")
    image_root = str(getattr(CFG, "image_root", "") or "").strip()
    if image_root:
        return os.path.join(image_root, p)
    return os.path.join(dataset_root, p)

def build_image_transform(split: str) -> T.Compose:
    split = str(split).lower()
    if split == "train":
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(256),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def pad_or_trim_struct(x: torch.Tensor, Tlen: int, Fdim: int) -> torch.Tensor:
    t = x.shape[0]
    if t >= Tlen:
        return x[-Tlen:]
    pad = torch.zeros(Tlen - t, Fdim, dtype=x.dtype)
    return torch.cat([pad, x], dim=0)

def _chunk_long_ids(ids: List[int], attn: List[int], maxlen: int, stride: int):
    if len(ids) <= maxlen:
        return [ids], [attn]
    out_ids, out_attn = [], []
    step = max(1, maxlen - max(stride, 0))
    i = 0
    while i < len(ids):
        s = ids[i:i + maxlen]
        a = attn[i:i + maxlen]
        out_ids.append(s)
        out_attn.append(a)
        if i + maxlen >= len(ids):
            break
        i += step
    return out_ids, out_attn

def pretok_batch_notes(batch_notes: List[List[str]]):
    global TOKENIZER, MAXLEN
    if TOKENIZER is None:
        raise RuntimeError("TOKENIZER not initialized; call main() after load_cfg().")
    MAXLEN = int(_cfg("max_text_len", 512))

    cleaned = []
    for texts in batch_notes:
        cleaned.append([
            t.replace("[CLS]", "").replace("[SEP]", "").strip()
            for t in texts if t and str(t).strip()
        ])

    out = []
    pad_id = TOKENIZER.pad_token_id or 0
    for texts in cleaned:
        if not texts:
            out.append({
                "input_ids": torch.zeros(0, MAXLEN, dtype=torch.long),
                "attention_mask": torch.zeros(0, MAXLEN, dtype=torch.long),
            })
            continue

        all_ids, all_attn = [], []
        for t in texts:
            enc = TOKENIZER(
                t,
                truncation=True,
                max_length=MAXLEN,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=True,
            )
            ids, attn = enc["input_ids"], enc["attention_mask"]
            ids_chunks, attn_chunks = _chunk_long_ids(ids, attn, MAXLEN, CHUNK_STRIDE)
            all_ids.extend(ids_chunks)
            all_attn.extend(attn_chunks)

        def _pad(x, L=MAXLEN, v=pad_id):
            return x + [v] * (L - len(x))

        ids_mat  = torch.tensor([_pad(ch) for ch in all_ids],  dtype=torch.long)  
        attn_mat = torch.tensor([_pad(ch, MAXLEN, 0) for ch in all_attn], dtype=torch.long)  
        out.append({"input_ids": ids_mat, "attention_mask": attn_mat})
    return out

def _detect_notes_schema(notes_df: pd.DataFrame):
    input_id_cols = sorted([c for c in notes_df.columns if str(c).startswith("input_ids_")])
    attn_cols = sorted([c for c in notes_df.columns if str(c).startswith("attention_mask_")])
    if len(attn_cols) == 0:
        attn_cols = sorted([c for c in notes_df.columns if str(c).startswith("attn_mask_")])

    if len(input_id_cols) > 0 and len(attn_cols) > 0:
        id_sufs = {c.split("input_ids_")[-1] for c in input_id_cols}
        if any(str(c).startswith("attention_mask_") for c in attn_cols):
            mk_sufs = {c.split("attention_mask_")[-1] for c in attn_cols}
        else:
            mk_sufs = {c.split("attn_mask_")[-1] for c in attn_cols}

        common = sorted(list(id_sufs & mk_sufs), key=lambda s: int(s) if str(s).isdigit() else s)
        if len(common) == 0:
            raise ValueError("Found input_ids_* and masks but no matching suffixes.")

        aligned_ids = [f"input_ids_{s}" for s in common]
        aligned_attn = []
        for s in common:
            if f"attention_mask_{s}" in notes_df.columns:
                aligned_attn.append(f"attention_mask_{s}")
            elif f"attn_mask_{s}" in notes_df.columns:
                aligned_attn.append(f"attn_mask_{s}")

        if len(aligned_ids) != len(aligned_attn):
            raise ValueError("Pretokenized notes columns not aligned (ids vs masks).")

        return ("pretokenized", aligned_ids, aligned_attn)

    for pref in ("chunk_", "text_chunk_", "note_chunk_"):
        cols = sorted([c for c in notes_df.columns if str(c).startswith(pref)])
        if len(cols) > 0:
            return ("text", cols, None)

    raise ValueError("notes schema not recognized")

def _cell_to_list(x):
    import ast
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "as_py"):
        try:
            return _cell_to_list(x.as_py())
        except Exception:
            pass
    if isinstance(x, (str, bytes)):
        s = x.decode("utf-8") if isinstance(x, bytes) else x
        s = s.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        return []
    if hasattr(x, "__iter__"):
        try:
            return list(x)
        except Exception:
            return []
    return []

def coerce_packed_ehr(x, stay_id: int, T: int = 48, F: int = 76) -> np.ndarray:
    import ast, json

    def _as_py(v):
        if hasattr(v, "as_py"):
            try:
                return v.as_py()
            except Exception:
                pass
        return v

    def _parse_if_string(v):
        if isinstance(v, (str, bytes)):
            s = v.decode("utf-8") if isinstance(v, bytes) else v
            s = s.strip()
            if not s:
                return []
            try:
                return json.loads(s)
            except Exception:
                pass
            try:
                return ast.literal_eval(s)
            except Exception:
                return []
        return v

    x = _parse_if_string(_as_py(x))

    if isinstance(x, np.ndarray) and x.ndim == 2:
        arr = x.astype(np.float32, copy=False)
        return arr

    if isinstance(x, (list, tuple, np.ndarray)):
        arr0 = np.asarray(x, dtype=object)
        if arr0.ndim == 1:
            if arr0.dtype == object and arr0.size > 0 and isinstance(arr0[0], (list, tuple, np.ndarray)):
                rows = [np.asarray(r, dtype=np.float32) for r in arr0.tolist()]
                try:
                    arr = np.stack(rows, axis=0)
                except Exception as e:
                    raise ValueError(
                        f"[packed_ehr] stay_id={stay_id} cannot stack rows: "
                        f"len={len(rows)} first_row_shape={getattr(rows[0],'shape',None)} err={e}"
                    )
            else:
                try:
                    arr_num = np.asarray(x, dtype=np.float32)
                except Exception:
                    arr_num = None

                if arr_num is not None and arr_num.ndim == 1 and arr_num.size == T * F:
                    arr = arr_num.reshape(T, F)
                else:
                    raise ValueError(
                        f"[packed_ehr] stay_id={stay_id} unexpected 1D content: "
                        f"type={type(x)} len={len(x) if hasattr(x,'__len__') else 'NA'} "
                        f"sample_type={type(arr0[0]) if arr0.size>0 else None}"
                    )
        else:
            arr = np.asarray(x, dtype=np.float32).reshape(T, F)

    else:
        raise ValueError(f"[packed_ehr] stay_id={stay_id} unsupported type: {type(x)}")

    if arr.ndim != 2:
        raise ValueError(f"[packed_ehr] stay_id={stay_id} got ndim={arr.ndim} shape={arr.shape}")
    if arr.shape != (T, F):
        if arr.shape[1] != F:
            raise ValueError(f"[packed_ehr] stay_id={stay_id} bad feat dim: shape={arr.shape} expect F={F}")
        if arr.shape[0] < T:
            pad = np.zeros((T - arr.shape[0], F), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > T:
            arr = arr[:T]

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def _coerce_any_2d(x, stay_id: int) -> np.ndarray:
    import ast, json

    def _as_py(v):
        if hasattr(v, "as_py"):
            try: return v.as_py()
            except Exception: pass
        return v

    def _parse(v):
        v = _as_py(v)
        if isinstance(v, (str, bytes)):
            s = v.decode("utf-8") if isinstance(v, bytes) else v
            s = s.strip()
            if not s:
                return None
            try: return json.loads(s)
            except Exception: pass
            try: return ast.literal_eval(s)
            except Exception: return None
        return v

    v = _parse(x)
    if v is None:
        raise ValueError(f"[struct] stay_id={stay_id} empty/unparseable cell")

    if isinstance(v, np.ndarray):
        arr = v
    else:
        arr = np.asarray(v)

    if arr.ndim == 2:
        return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 1 and arr.dtype == object and arr.size > 0 and isinstance(arr[0], (list, tuple, np.ndarray)):
        rows = [np.asarray(r, dtype=np.float32) for r in arr.tolist()]
        out = np.stack(rows, axis=0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    raise ValueError(f"[struct] stay_id={stay_id} got non-2D content: ndim={arr.ndim} shape={arr.shape}")

def detect_struct_col(struct_df: pd.DataFrame, T_cfg: int, F_cfg: int) -> Tuple[str, Tuple[int,int]]:
    cand = [c for c in struct_df.columns if str(c).startswith("x_") or "ehr" in str(c).lower()]
    cand = [c for c in cand if c != "stay_id"]
    if not cand:
        raise ValueError(f"[struct] no candidate structured columns found. Columns={list(struct_df.columns)[:50]}")

    sample_df = struct_df.dropna(subset=cand, how="all")
    if sample_df.empty:
        raise ValueError("[struct] all candidate structured columns are empty")

    row = sample_df.iloc[0]
    stay_id = int(row["stay_id"]) if "stay_id" in row else -1

    shapes = {}
    for c in cand:
        try:
            arr = _coerce_any_2d(row[c], stay_id=stay_id)
            shapes[c] = (int(arr.shape[0]), int(arr.shape[1]))
        except Exception:
            continue

    if not shapes:
        raise ValueError("[struct] could not parse any candidate structured columns into 2D arrays")

    for c, (t,f) in shapes.items():
        if int(t) == int(T_cfg) and int(f) == int(F_cfg):
            return c, (t,f)

    for c, (t,f) in shapes.items():
        if int(f) == int(F_cfg):
            return c, (t,f)

    c0 = next(iter(shapes.keys()))
    return c0, shapes[c0]

def load_cxr_tensor(paths: List[str], tfms: T.Compose, return_path: bool = False):
    if not paths:
        tensor = torch.zeros(3, 224, 224)
        return (tensor, "<none>") if return_path else tensor
    p_full = str(paths[-1]).strip()
    if not p_full or not os.path.exists(p_full):
        print(f"[warn] image path missing/does not exist: {p_full} -> returning zero tensor")
        tensor = torch.zeros(3, 224, 224)
        return (tensor, p_full) if return_path else tensor

    ext = os.path.splitext(p_full.lower())[1]

    try:
        if ext == ".dcm":
            try:
                import pydicom
                ds = pydicom.dcmread(p_full)
                arr = ds.pixel_array.astype(np.float32)
                arr = arr - arr.min()
                if arr.max() > 0:
                    arr = arr / arr.max()
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            except Exception as e:
                print(f"[warn] failed to read DICOM: {p_full} ({e}) -> zero tensor")
                tensor = torch.zeros(3, 224, 224)
                return (tensor, p_full) if return_path else tensor
        else:
            img = Image.open(p_full)
        img = img.convert("RGB")
        tensor = tfms(img)
    except Exception as e:
        print(f"[warn] failed to open image: {p_full} ({e}) -> returning zero tensor")
        tensor = torch.zeros(3, 224, 224)
    return (tensor, p_full) if return_path else tensor

def prepare_notes_batch(notes_batch: List[Dict[str, Any]]):
    out = []
    pad_id = int(getattr(TOKENIZER, "pad_token_id", 0) or 0)
    L = int(_cfg("max_text_len", 512))

    def _pad_to_len(seq, pad_value: int, max_len: int):
        seq = list(seq)
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [pad_value] * (max_len - len(seq))

    def _flatten_if_nested(seq):
        if not isinstance(seq, (list, tuple)):
            return []
        if len(seq) == 0:
            return []
        if isinstance(seq[0], (list, tuple, np.ndarray)):
            flat = []
            for part in seq:
                if isinstance(part, np.ndarray):
                    part = part.tolist()
                if isinstance(part, (list, tuple)):
                    flat.extend(part)
                else:
                    flat.append(part)
            return flat
        return list(seq)

    def _to_int_list(seq):
        seq = _flatten_if_nested(seq)
        out_ints = []
        for v in seq:
            if isinstance(v, (np.integer,)):
                out_ints.append(int(v))
            elif isinstance(v, (int,)):
                out_ints.append(v)
            elif isinstance(v, float) and np.isnan(v):
                continue
            else:
                try:
                    out_ints.append(int(v))
                except Exception:
                    continue
        return out_ints

    for item in notes_batch:
        mode = item.get("mode", "text")

        if mode == "text":
            tmp = pretok_batch_notes([item["chunks"]])[0]
            out.append(tmp)
            continue

        ids_chunks  = item.get("input_ids", [])
        attn_chunks = item.get("attention_mask", [])

        if not isinstance(ids_chunks, (list, tuple)):
            ids_chunks = []
        if not isinstance(attn_chunks, (list, tuple)):
            attn_chunks = []

        ids_chunks  = [_to_int_list(x) for x in ids_chunks]
        attn_chunks = [_to_int_list(x) for x in attn_chunks]

        paired = [
            (a, b) for a, b in zip(ids_chunks, attn_chunks)
            if len(a) > 0 and len(b) > 0 and (np.sum(np.asarray(b, dtype=np.int64)) > 0)
        ]
        if len(paired) == 0:
            out.append({
                "input_ids": torch.zeros(0, L, dtype=torch.long),
                "attention_mask": torch.zeros(0, L, dtype=torch.long),
            })
            continue

        ids_chunks, attn_chunks = zip(*paired)
        ids_mat = torch.tensor([_pad_to_len(x, pad_id, L) for x in ids_chunks], dtype=torch.long)
        attn_mat = torch.tensor([_pad_to_len(x, 0, L) for x in attn_chunks], dtype=torch.long)
        attn_mat = (attn_mat > 0).long()
        out.append({"input_ids": ids_mat, "attention_mask": attn_mat})
    return out

class ICUStayDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"[ICUStayDataset] data root not found: {root}")
        self.root = root
        self.split = split
        self.img_tfms = build_image_transform(split)

        req_files = [
            "splits.json",
            "structured_mortality.parquet",
            "notes_wide_clean.parquet",
            "images.parquet",
            "labels_mortality.parquet",
        ]
        missing = [p for p in req_files if not os.path.exists(os.path.join(root, p))]
        if missing:
            raise FileNotFoundError(
                f"[ICUStayDataset] missing files under {root}: {missing}\n"
                f"Expected exactly: {', '.join(req_files)}"
            )

        with open(os.path.join(root, "splits.json")) as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(
                f"[ICUStayDataset] split '{split}' not in splits.json keys: {list(splits.keys())}"
            )
        split_ids: List[int] = [int(x) for x in splits[split]]
        ids_set = set(split_ids)

        struct_fp = os.path.join(root, "structured_mortality.parquet")
        notes_fp  = os.path.join(root, "notes_wide_clean.parquet")
        images_fp = os.path.join(root, "images.parquet")
        labels_fp = os.path.join(root, "labels_mortality.parquet")

        self.struct = _standardize_id_column(pd.read_parquet(struct_fp))

        wanted = str(getattr(CFG, "structured_x_col", "") or "").strip()
        if wanted and wanted in self.struct.columns:
            self.struct_col = wanted
            sample_df = self.struct.dropna(subset=[self.struct_col])
            if sample_df.empty:
                raise ValueError(f"[struct] column '{self.struct_col}' exists but is all-NaN")
            row0 = sample_df.iloc[0]
            sid0 = int(row0["stay_id"])
            arr0 = _coerce_any_2d(row0[self.struct_col], stay_id=sid0)
            t0, f0 = int(arr0.shape[0]), int(arr0.shape[1])
        else:
            T_cfg = int(getattr(CFG, "structured_seq_len", 48) or 48)
            F_cfg = int(getattr(CFG, "structured_n_feats", 76) or 76)
            self.struct_col, (t0, f0) = detect_struct_col(self.struct, T_cfg=T_cfg, F_cfg=F_cfg)

        if int(getattr(CFG, "structured_seq_len", t0)) != t0 or int(getattr(CFG, "structured_n_feats", f0)) != f0:
            print(f"[struct] CFG mismatch → syncing CFG.structured_seq_len={t0}, CFG.structured_n_feats={f0}")
            CFG.structured_seq_len = t0
            CFG.structured_n_feats = f0

        print(f"[dataset:{split}] structured_col='{self.struct_col}' sample_shape=({t0},{f0}) cfg_TF=({CFG.structured_seq_len},{CFG.structured_n_feats})")

        self.notes  = _standardize_id_column(pd.read_parquet(notes_fp))
        self.images = _standardize_id_column(pd.read_parquet(images_fp))
        self.images = _standardize_image_path_column(self.images)
        self.labels = _standardize_id_column(pd.read_parquet(labels_fp))

        self.struct = self.struct.drop_duplicates("stay_id").set_index("stay_id", drop=False)
        self.notes  = self.notes.drop_duplicates("stay_id").set_index("stay_id", drop=False)
        self.labels = self.labels.drop_duplicates("stay_id").set_index("stay_id", drop=False)

        # notes schema
        self.notes_mode, self.note_a_cols, self.note_b_cols = _detect_notes_schema(self.notes)
        if self.notes_mode == "text":
            self.chunk_cols = self.note_a_cols
            print(f"[dataset:{split}] notes_mode=text (chunks={len(self.chunk_cols)})")
        else:
            self.input_id_cols = self.note_a_cols
            self.attn_mask_cols = self.note_b_cols
            print(f"[dataset:{split}] notes_mode=pretokenized (chunks={len(self.input_id_cols)})")
            if "n_chunks" in self.notes.columns:
                print(f"[dataset:{split}] notes has n_chunks col; will use it to trim valid chunks")

        self.label_cols: List[str] = [c for c in self.labels.columns if c != "stay_id"]
        self.label_cols.sort()
        if len(self.label_cols) == 0:
            raise ValueError("[ICUStayDataset] labels file must contain at least one label column (besides stay_id).")
        print(f"[dataset:{split}] found {len(self.label_cols)} labels: {self.label_cols}")

        struct_ids = set(self.struct["stay_id"].astype(int).unique().tolist())
        label_ids  = set(self.labels["stay_id"].astype(int).unique().tolist())

        if self.notes_mode == "text":
            nonempty = np.zeros(len(self.notes), dtype=bool)
            for c in self.chunk_cols:
                if c in self.notes.columns:
                    nonempty |= self.notes[c].fillna("").astype(str).str.strip().ne("")
            note_ids = set(self.notes.loc[nonempty, "stay_id"].astype(int).unique().tolist())
        else:
            def _valid_chunk(ids_cell, msk_cell) -> bool:
                ids = _cell_to_list(ids_cell)
                msk = _cell_to_list(msk_cell)
                if len(ids) == 0 or len(msk) == 0:
                    return False
                return (np.sum(np.asarray(msk, dtype=np.int64)) > 0)

            nonempty = np.zeros(len(self.notes), dtype=bool)
            for c_id, c_m in zip(self.input_id_cols, self.attn_mask_cols):
                nonempty |= self.notes[[c_id, c_m]].apply(
                    lambda r: _valid_chunk(r[c_id], r[c_m]),
                    axis=1
                )
            note_ids = set(self.notes.loc[nonempty, "stay_id"].astype(int).unique().tolist())

        img_map: Dict[int, List[str]] = {}
        for sid, g in self.images.groupby("stay_id", sort=False):
            raw_paths = g["cxr_path"].dropna().astype(str).tolist()
            raw_paths = [p for p in raw_paths if p.strip()]
            if not raw_paths:
                continue
            cand = [resolve_image_path(p, self.root) for p in raw_paths]
            cand = [p for p in cand if is_probably_image_file(p) and os.path.exists(p)]
            if cand:
                img_map[int(sid)] = cand
        self.img_map = img_map
        img_ids = set(img_map.keys()) & ids_set
        print(f"[dataset:{split}] img_ids={len(img_ids)} (precomputed)")

        keep_ids = ids_set & struct_ids & label_ids & img_ids & note_ids
        self.ids = sorted(list(keep_ids))

        print(f"[dataset:{split}] kept {len(self.ids)} / {len(ids_set)}")
        if len(self.ids) == 0:
            raise RuntimeError(f"[ICUStayDataset] After filtering, split '{self.split}' is empty.")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = int(self.ids[idx])

        row_s = self.struct.loc[stay_id]
        if isinstance(row_s, pd.DataFrame):
            row_s = row_s.iloc[0]
        x = row_s[self.struct_col]
        Tlen = int(getattr(CFG, "structured_seq_len", 48) or 48)
        Fdim = int(getattr(CFG, "structured_n_feats", 76) or 76)
        xs_np = coerce_packed_ehr(x, stay_id=stay_id, T=Tlen, F=Fdim)
        xs = torch.from_numpy(xs_np)  # [T,F]

        # notes
        row_n = self.notes.loc[stay_id]
        if isinstance(row_n, pd.DataFrame):
            row_n = row_n.iloc[0]

        if self.notes_mode == "text":
            notes_list = []
            for c in self.chunk_cols:
                val = row_n.get(c, "")
                if pd.notna(val) and str(val).strip():
                    notes_list.append(str(val))
            if not notes_list:
                raise RuntimeError(f"[ICUStayDataset] stay_id={stay_id} has no non-empty chunk text")
            M = int(getattr(CFG, "notes_max_chunks", -1))
            if M > 0:
                notes_list = notes_list[:M]
            notes_payload = {"mode": "text", "chunks": notes_list}
        else:
            n_chunks = int(row_n.get("n_chunks", 0) or 0)
            ids_chunks, attn_chunks = [], []
            for j, (c_id, c_m) in enumerate(zip(self.input_id_cols, self.attn_mask_cols)):
                if n_chunks > 0 and j >= n_chunks:
                    break
                ids = _cell_to_list(row_n.get(c_id, None))
                msk = _cell_to_list(row_n.get(c_m, None))
                if len(ids) == 0 or len(msk) == 0 or len(ids) != len(msk):
                    continue
                if np.sum(np.asarray(msk, dtype=np.int64)) <= 0:
                    continue
                ids_chunks.append(ids)
                attn_chunks.append(msk)

            if len(ids_chunks) == 0:
                raise RuntimeError(f"[ICUStayDataset] stay_id={stay_id} has no valid chunks")

            M = int(getattr(CFG, "notes_max_chunks", -1))
            if M > 0:
                ids_chunks = ids_chunks[:M]
                attn_chunks = attn_chunks[:M]
            notes_payload = {"mode": "pretokenized", "input_ids": ids_chunks, "attention_mask": attn_chunks}

        cand = self.img_map.get(stay_id, [])
        if not cand:
            raise RuntimeError(f"[ICUStayDataset] stay_id={stay_id} has no valid image in img_map")
        img_paths = [cand[-1]]

        row_y = self.labels.loc[stay_id]
        if isinstance(row_y, pd.DataFrame):
            row_y = row_y.iloc[0]
        col = self.label_cols[0]
        y0 = row_y[col]
        y0 = 0 if (y0 is None or (isinstance(y0, float) and np.isnan(y0))) else int(float(y0) > 0.0)
        y = torch.tensor(y0, dtype=torch.long)  

        return {
            "stay_id": stay_id,
            "x_struct": xs,
            "notes": notes_payload,
            "image_paths": img_paths,
            "y": y,
        }

def collate_fn_factory(img_tfms: T.Compose):
    first_print = {"done": False}

    def _collate(batch: List[Dict[str, Any]]):
        F_dim = int(batch[0]["x_struct"].shape[1])
        T_len_cfg = int(getattr(CFG, "structured_seq_len", 48))
        T_len = T_len_cfg if (T_len_cfg is not None and T_len_cfg > 0) else max(int(b["x_struct"].shape[0]) for b in batch)
        T_len = max(T_len, 1)

        xL_batch = torch.stack([pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0)  # [B,T,F]
        lengths = torch.tensor([b["x_struct"].shape[0] for b in batch], dtype=torch.long)
        idx = torch.arange(T_len).unsqueeze(0)  # [1,T]
        start = (T_len - torch.clamp(lengths, max=T_len)).unsqueeze(1)  # [B,1]
        mL_batch = (idx >= start).float()  # [B,T]

        notes_list = [b["notes"] for b in batch]
        prep = prepare_notes_batch(notes_list) 

        L = int(getattr(CFG, "max_text_len", 512))
        S_cap = int(getattr(CFG, "notes_max_chunks", -1))
        S_max = max((p["input_ids"].shape[0] for p in prep), default=0)
        S = min(S_max, S_cap) if (S_cap and S_cap > 0) else S_max
        S = max(S, 1)

        B = len(prep)
        notes_ids  = torch.zeros((B, S, L), dtype=torch.long)
        notes_attn = torch.zeros((B, S, L), dtype=torch.long)
        notes_cmask = torch.zeros((B, S), dtype=torch.float32)

        for i, p in enumerate(prep):
            ids_i  = p["input_ids"]        # [S_i,L]
            att_i  = p["attention_mask"]   # [S_i,L]
            s_i = min(ids_i.shape[0], S)
            if s_i > 0:
                ids_cpu  = ids_i[:s_i].cpu()
                att_cpu  = att_i[:s_i].cpu()
                notes_ids[i, :s_i]  = ids_cpu
                notes_attn[i, :s_i] = att_cpu
                valid = (att_cpu.sum(dim=1) > 0).float()
                notes_cmask[i, :s_i] = valid

        notes_batch = {
            "input_ids": notes_ids,
            "attention_mask": notes_attn,
            "chunk_mask": notes_cmask, 
            "mode": "batched",
        }

        # images
        imgs_list, img_paths_list = [], []
        for b in batch:
            img_t, path = load_cxr_tensor(b["image_paths"], img_tfms, return_path=True)
            imgs_list.append(img_t)
            img_paths_list.append(path)
        imgs_batch = torch.stack(imgs_list, dim=0)  # [B,3,224,224]

        # labels
        ys = []
        for b in batch:
            yi = b["y"]
            ys.append(yi if yi.ndim > 0 else yi.view(()))
        y_batch = torch.stack(ys, dim=0)  # [B]

        dbg = {"stay_ids": [b["stay_id"] for b in batch], "img_paths": img_paths_list}

        if not first_print["done"]:
            zero_frac = float((imgs_batch.abs().sum(dim=(1,2,3)) == 0).float().mean().item())
            print(f"[collate] image_zero_fraction(first batch) = {zero_frac:.3f}")
            print(f"[collate] xL_batch={tuple(xL_batch.shape)} mL_batch={tuple(mL_batch.shape)} "
                  f"notes_ids={tuple(notes_ids.shape)} imgs={tuple(imgs_batch.shape)} y={tuple(y_batch.shape)}")
            first_print["done"] = True

        return xL_batch, mL_batch, notes_batch, imgs_batch, y_batch, dbg

    return _collate

def encode_pooled(
    behrt, bbert, imgenc,
    xL: torch.Tensor,
    mL: torch.Tensor,
    notes: Dict[str, torch.Tensor],
    imgs: torch.Tensor,
    amp_ctx_enc,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with amp_ctx_enc:
        xL_d = xL.to(DEVICE, non_blocking=True)
        mL_d = mL.to(DEVICE, non_blocking=True).float()

        notes_batch = {
            "input_ids": notes["input_ids"].to(DEVICE, non_blocking=True),
            "attention_mask": notes["attention_mask"].to(DEVICE, non_blocking=True),
        }
        if "chunk_mask" in notes and notes["chunk_mask"] is not None:
            notes_batch["chunk_mask"] = notes["chunk_mask"].to(DEVICE, non_blocking=True).float()

        imgs_d = imgs.to(DEVICE, non_blocking=True)

        pooled = encode_unimodal_pooled(
            behrt, bbert, imgenc,
            xL_d, notes_batch, imgs_d,
            mL_d
        )
        zL = pooled.get("L", None)
        zN = pooled.get("N", None)
        zI = pooled.get("I", None)

        if zL is None or zN is None or zI is None:
            raise RuntimeError(f"encode_unimodal_pooled returned keys={list(pooled.keys())}, expected L/N/I")

    return zL, zN, zI

class LateFusionMortality(nn.Module):
    def __init__(self, d_l: int, d_n: int, d_i: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        d_in = int(d_l + d_n + d_i)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),  
        )

    def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
        x = torch.cat([zL, zN, zI], dim=1)
        return self.mlp(x)

@torch.no_grad()
def collect_probs_and_logits(
    loader,
    behrt, bbert, imgenc,
    head: nn.Module,
    amp_ctx_enc,
    amp_ctx_head,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    behrt.eval(); bbert.eval(); imgenc.eval(); head.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()

    ys, logits_all, ids_all = [], [], []
    for xL, mL, notes, imgs, y, dbg in loader:
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE, non_blocking=True)

        zL, zN, zI = encode_pooled(behrt, bbert, imgenc, xL, mL, notes, imgs, amp_ctx_enc)

        with amp_ctx_head:
            logits = head(zL, zN, zI)

        ys.append(y.detach().cpu().numpy().reshape(-1))
        logits_all.append(logits.detach().float().cpu().numpy())
        ids_all += dbg.get("stay_ids", [])

    y_true = np.concatenate(ys, axis=0)
    logits_np = np.concatenate(logits_all, axis=0)  # [N,2]
    probs = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    p_death = probs[:, 1]
    return y_true, logits_np, ids_all

def compute_binary_metrics(y_true: np.ndarray, logits_np: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    logits_np = np.asarray(logits_np, dtype=np.float32)
    probs = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    p = probs[:, 1]
    y_pred = (p >= float(thr)).astype(int)

    out = {}
    out["AUROC"] = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan")
    out["AUPRC"] = float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan")
    out["F1"] = float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    out["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["CM"] = confusion_matrix(y_true, y_pred).tolist()
    out["prevalence"] = float(y_true.mean())
    return out

def find_best_threshold(y_true: np.ndarray, logits_np: np.ndarray, n_steps: int = 101) -> Tuple[float, float]:
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    logits_np = np.asarray(logits_np, dtype=np.float32)
    probs = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    p = probs[:, 1]

    best_thr = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.0, 1.0, n_steps):
        y_pred = (p >= float(t)).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(t)
    return best_thr, best_f1

def save_checkpoint(path: str, state: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_checkpoint(path: str, behrt, bbert, imgenc, head, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    behrt.load_state_dict(ckpt["behrt"])
    bbert.load_state_dict(ckpt["bbert"])
    imgenc.load_state_dict(ckpt["imgenc"])
    head.load_state_dict(ckpt["head"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] loaded epoch={ckpt.get('epoch', 0)} best_val_auroc={ckpt.get('best_val_auroc', -1):.4f}")
    return int(ckpt.get("epoch", 0))


def parse_args():
    ap = argparse.ArgumentParser("Late-fusion mortality prediction (your data + your encoders)")
    ap.add_argument("--data_root", type=str, default=_cfg("data_root", "./data"))
    ap.add_argument("--ckpt_root", type=str, default=_cfg("ckpt_root", "./ckpts"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=int(_cfg("batch_size", 16)))
    ap.add_argument("--lr", type=float, default=float(_cfg("lr", 2e-4)))
    ap.add_argument("--weight_decay", type=float, default=float(_cfg("weight_decay", 1e-4)))
    ap.add_argument("--num_workers", type=int, default=int(_cfg("num_workers", 4)))
    ap.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "off"])
    ap.add_argument("--finetune_text", action="store_true", help="Unfreeze Bio_ClinicalBERT if set.")
    ap.add_argument("--encoder_warmup_epochs", type=int, default=int(_cfg("encoder_warmup_epochs", 2)))
    ap.add_argument("--head_hidden", type=int, default=int(_cfg("latefuse_hidden", 256)))
    ap.add_argument("--dropout", type=float, default=float(_cfg("dropout", 0.1)))
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--log_every", type=int, default=300)
    ap.add_argument("--patience_epochs", type=int, default=int(_cfg("patience_epochs", 5)))
    ap.add_argument("--min_epochs", type=int, default=int(_cfg("min_epochs", 10)))
    ap.add_argument("--min_delta", type=float, default=float(_cfg("min_delta", 1e-4)))
    ap.add_argument("--pos_weight_max", type=float, default=float(_cfg("pos_weight_max", 20.0)))
    return ap.parse_args()

def main():
    import env_config as E
    E.load_cfg()
    args = parse_args()
    apply_cli_overrides(args)

    global CFG, DEVICE, TOKENIZER, MAXLEN
    CFG = E.CFG
    DEVICE = E.DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print("[forced] DEVICE =", DEVICE)

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if args.finetune_text:
        CFG.finetune_text = True

    print("[env_config] CFG:", json.dumps(asdict(CFG), indent=2))

    TOKENIZER = AutoTokenizer.from_pretrained(CFG.text_model_name)
    MAXLEN = int(_cfg("max_text_len", 512))

    use_cuda = (str(DEVICE).startswith("cuda") and torch.cuda.is_available())
    precision = str(args.precision).lower()
    use_amp = use_cuda and (precision != "off")

    if use_amp:
        if precision == "fp16":
            amp_ctx_enc = torch_amp.autocast(device_type="cuda", dtype=torch.float16)
            amp_ctx_head = torch_amp.autocast(device_type="cuda", dtype=torch.float16)
        elif precision == "bf16":
            amp_ctx_enc = torch_amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            amp_ctx_head = torch_amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            amp_ctx_enc = torch_amp.autocast(device_type="cuda")
            amp_ctx_head = torch_amp.autocast(device_type="cuda")
    else:
        amp_ctx_enc = nullcontext()
        amp_ctx_head = nullcontext()

    from torch.amp import GradScaler
    scaler = GradScaler("cuda", enabled=(use_amp and precision in {"auto", "fp16"}))

    print(f"[amp] use_amp={use_amp} precision={precision} scaler_enabled={scaler.is_enabled()}")

    train_ds = ICUStayDataset(args.data_root, split="train")
    val_ds   = ICUStayDataset(args.data_root, split="val")
    test_ds  = ICUStayDataset(args.data_root, split="test")

    collate_train = collate_fn_factory(img_tfms=build_image_transform("train"))
    collate_eval  = collate_fn_factory(img_tfms=build_image_transform("val"))

    pin = use_cuda
    g_train = torch.Generator().manual_seed(int(CFG.seed) + 123)
    g_eval  = torch.Generator().manual_seed(int(CFG.seed) + 456)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_train,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g_train,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_eval,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g_eval,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_eval,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g_eval,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    from dataclasses import fields
    allowed = {f.name for f in fields(EncoderConfig)}

    image_name = (
        getattr(CFG, "image_model_name", None)
        or getattr(CFG, "img_model_name", None)
        or getattr(CFG, "img_model", None)
        or getattr(CFG, "image_model", None)
        or "resnet50"
    )
    cfg_kwargs = dict(
        text_model_name=getattr(CFG, "text_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        structured_seq_len=int(getattr(CFG, "structured_seq_len", 48) or 48),
        structured_n_feats=int(getattr(CFG, "structured_n_feats", 76) or 76),
        max_text_len=int(getattr(CFG, "max_text_len", 512) or 512),
        notes_max_chunks=int(getattr(CFG, "notes_max_chunks", -1) or -1),
        finetune_text=bool(getattr(CFG, "finetune_text", False) or False),
        finetune_image=bool(getattr(CFG, "finetune_image", False) or False),
        finetune_struct=bool(getattr(CFG, "finetune_struct", False) or False),
    )

    for k in ("image_model_name", "img_model_name", "image_model", "img_model", "image_backbone", "img_backbone"):
        if k in allowed:
            cfg_kwargs[k] = image_name
            break

    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in allowed}

    print("[EncoderConfig] using keys:", sorted(cfg_kwargs.keys()))
    enc_cfg = EncoderConfig(**cfg_kwargs)

    behrt, bbert, imgenc = build_encoders(enc_cfg)
    behrt = behrt.to(DEVICE)
    bbert = bbert.to(DEVICE)
    imgenc = imgenc.to(DEVICE)

    def _get_pool_dim(m, fallback: int = 768):
        for k in ("d_pool", "pool_dim", "hidden_size", "d_model", "embed_dim", "out_dim"):
            if hasattr(m, k):
                try:
                    v = int(getattr(m, k))
                    if v > 0:
                        return v
                except Exception:
                    pass
        for sub in ("proj", "fc", "classifier", "head"):
            if hasattr(m, sub):
                sm = getattr(m, sub)
                if isinstance(sm, nn.Linear):
                    return int(sm.out_features)
        return int(fallback)

    d_l = _get_pool_dim(behrt, fallback=int(getattr(CFG, "structured_embed_dim", 256) or 256))
    d_n = _get_pool_dim(bbert, fallback=768)
    d_i = _get_pool_dim(imgenc, fallback=512)

    head = LateFusionMortality(d_l=d_l, d_n=d_n, d_i=d_i, hidden=args.head_hidden, dropout=args.dropout).to(DEVICE)
    print(f"[model] dL={d_l} dN={d_n} dI={d_i} head_hidden={args.head_hidden}")

    def _estimate_pos_weight(ds: Dataset, max_items: int = 20000) -> float:
        n = min(len(ds), int(max_items))
        if n <= 0:
            return 1.0
        pos = 0
        for i in range(n):
            y = ds[i]["y"]
            yv = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
            pos += (1 if yv == 1 else 0)
        neg = n - pos
        if pos <= 0:
            return float(args.pos_weight_max)
        return float(neg) / float(pos)

    raw_posw = _estimate_pos_weight(train_ds)
    posw = float(min(float(args.pos_weight_max), max(1.0, raw_posw)))
    ce_weight = torch.tensor([1.0, posw], dtype=torch.float32, device=DEVICE)
    print(f"[loss] estimated pos_weight={raw_posw:.3f} -> using w1={posw:.3f}")

    def _set_requires_grad(mod: nn.Module, flag: bool):
        for p in mod.parameters():
            p.requires_grad = bool(flag)

    def freeze_all_encoders():
        _set_requires_grad(behrt, False)
        _set_requires_grad(imgenc, False)
        _set_requires_grad(bbert, False)
        if getattr(bbert, "bert", None) is not None:
            _set_requires_grad(bbert.bert, False)

    def unfreeze_after_warmup():
        _set_requires_grad(behrt, True)
        _set_requires_grad(imgenc, True)

        if args.finetune_text or bool(getattr(CFG, "finetune_text", False)):
            _set_requires_grad(bbert, True)
            if getattr(bbert, "bert", None) is not None:
                _set_requires_grad(bbert.bert, True)
            print("[finetune] text encoder UNFROZEN")
        else:
            _set_requires_grad(bbert, False)
            if getattr(bbert, "bert", None) is not None:
                _set_requires_grad(bbert.bert, False)
            print("[finetune] text encoder remains FROZEN")

    warmup_epochs = int(max(0, args.encoder_warmup_epochs))
    if warmup_epochs > 0:
        print(f"[warmup] freezing encoders for first {warmup_epochs} epoch(s)")
        freeze_all_encoders()
    else:
        unfreeze_after_warmup()

    def _param_groups():
        params = []
        for m in (behrt, bbert, imgenc, head):
            ps = [p for p in m.parameters() if p.requires_grad]
            if ps:
                params.append({"params": ps})
        return params

    optimizer = torch.optim.AdamW(_param_groups(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    start_epoch = 0
    best_val_auroc = -1.0
    best_epoch = -1
    best_thr = 0.5

    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, behrt, bbert, imgenc, head, optimizer)
        start_epoch = int(start_epoch) + 1

    def _train_one_epoch(epoch: int) -> Dict[str, float]:
        behrt.train(); bbert.train(); imgenc.train(); head.train()
        if getattr(bbert, "bert", None) is not None and not (args.finetune_text or getattr(CFG, "finetune_text", False)):
            bbert.bert.eval()  

        total_loss = 0.0
        total = 0
        correct = 0

        for step, (xL, mL, notes, imgs, y, dbg) in enumerate(train_loader, start=1):
            xL = xL.to(DEVICE, non_blocking=True)
            mL = mL.to(DEVICE, non_blocking=True)
            imgs = imgs.to(DEVICE, non_blocking=True)
            y   = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            zL, zN, zI = encode_pooled(behrt, bbert, imgenc, xL, mL, notes, imgs, amp_ctx_enc)
            with amp_ctx_head:
                logits = head(zL, zN, zI)
                loss = F.cross_entropy(logits, y, weight=ce_weight)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in head.parameters() if p.requires_grad], max_norm=1.0
                )
                if grads_are_finite([p for p in head.parameters() if p.requires_grad]):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    print("[warn] non-finite grads; skipping step")
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in head.parameters() if p.requires_grad], max_norm=1.0
                )
                if grads_are_finite([p for p in head.parameters() if p.requires_grad]):
                    optimizer.step()
                else:
                    print("[warn] non-finite grads; skipping step")
                    optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().cpu().item()) * int(y.shape[0])
            total += int(y.shape[0])
            pred = logits.detach().argmax(dim=1)
            correct += int((pred == y).sum().item())

            if (step % int(max(1, args.log_every))) == 0:
                avg_loss = total_loss / max(1, total)
                acc = correct / max(1, total)
                lr_now = float(optimizer.param_groups[0]["lr"])
                print(f"[train] epoch={epoch} step={step}/{len(train_loader)} loss={avg_loss:.4f} acc={acc:.4f} lr={lr_now:g}")

        return {
            "loss": total_loss / max(1, total),
            "acc":  correct / max(1, total),
        }

    def _eval_split(name: str, loader) -> Dict[str, Any]:
        y_true, logits_np, ids_all = collect_probs_and_logits(
            loader, behrt, bbert, imgenc, head, amp_ctx_enc, amp_ctx_head
        )
        m = compute_binary_metrics(y_true, logits_np, thr=0.5)
        thr, thr_f1 = find_best_threshold(y_true, logits_np, n_steps=201)
        m["best_thr_f1"] = float(thr_f1)
        m["best_thr"] = float(thr)
        print(f"[eval:{name}] AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f} F1@0.5={m['F1']:.4f} best_thr={thr:.3f} bestF1={thr_f1:.4f}")
        return m

    ckpt_dir = os.path.join(args.ckpt_root, "latefusion_mortality")
    ensure_dir(ckpt_dir)

    bad_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, int(args.epochs)):
        if warmup_epochs > 0 and epoch == warmup_epochs:
            print(f"[warmup] finished -> unfreezing encoders at epoch={epoch}")
            unfreeze_after_warmup()
            optimizer = torch.optim.AdamW(_param_groups(), lr=float(args.lr), weight_decay=float(args.weight_decay))
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
            print("[warmup] rebuilt optimizer/scheduler to include unfrozen encoder params")

        train_stats = _train_one_epoch(epoch)
        val_stats = _eval_split("val", val_loader)

        if np.isfinite(val_stats.get("AUROC", float("nan"))):
            scheduler.step(float(val_stats["AUROC"]))

        val_auroc = float(val_stats.get("AUROC", float("nan")))
        improved = np.isfinite(val_auroc) and (val_auroc > best_val_auroc + float(args.min_delta))

        history_row = {
            "epoch": int(epoch),
            "train": train_stats,
            "val": val_stats,
            "best_val_auroc_so_far": float(best_val_auroc),
        }
        history.append(history_row)

        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(last_path, {
            "epoch": int(epoch),
            "behrt": behrt.state_dict(),
            "bbert": bbert.state_dict(),
            "imgenc": imgenc.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_auroc": float(best_val_auroc),
            "best_epoch": int(best_epoch),
            "best_thr": float(best_thr),
            "cfg": asdict(CFG),
            "args": vars(args),
            "history": history,
        })

        if improved:
            best_val_auroc = val_auroc
            best_epoch = int(epoch)
            best_thr = float(val_stats.get("best_thr", 0.5))
            bad_epochs = 0
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(best_path, {
                "epoch": int(epoch),
                "behrt": behrt.state_dict(),
                "bbert": bbert.state_dict(),
                "imgenc": imgenc.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_auroc": float(best_val_auroc),
                "best_epoch": int(best_epoch),
                "best_thr": float(best_thr),
                "cfg": asdict(CFG),
                "args": vars(args),
                "history": history,
            })
            print(f"[ckpt] new best: epoch={epoch} val_AUROC={best_val_auroc:.4f} saved -> {best_path}")
        else:
            bad_epochs += 1
            print(f"[earlystop] no improvement: bad_epochs={bad_epochs}/{args.patience_epochs}")

        if (epoch + 1) >= int(args.min_epochs) and bad_epochs >= int(args.patience_epochs):
            print(f"[earlystop] stopping at epoch={epoch} (best_epoch={best_epoch} best_val_AUROC={best_val_auroc:.4f})")
            break

    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        _ = load_checkpoint(best_path, behrt, bbert, imgenc, head, optimizer=None)
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        best_thr = float(ckpt.get("best_thr", best_thr))
        best_val_auroc = float(ckpt.get("best_val_auroc", best_val_auroc))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        print(f"[best] loaded best.pt (best_epoch={best_epoch} best_val_AUROC={best_val_auroc:.4f} best_thr={best_thr:.3f})")
    else:
        print("[best] best.pt not found; using current weights")


    test_stats = _eval_split("test", test_loader)
    y_true, logits_np, _ = collect_probs_and_logits(
        test_loader, behrt, bbert, imgenc, head, amp_ctx_enc, amp_ctx_head
    )
    test_at_best = compute_binary_metrics(y_true, logits_np, thr=float(best_thr))
    print(f"[test@best_thr] thr={best_thr:.3f} F1={test_at_best['F1']:.4f} Prec={test_at_best['Precision']:.4f} Rec={test_at_best['Recall']:.4f}")

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_auroc": float(best_val_auroc),
        "best_thr_from_val": float(best_thr),
        "test_default_thr_0.5": test_stats,
        "test_at_best_thr": test_at_best,
        "args": vars(args),
        "cfg": asdict(CFG),
    }
    out_json = os.path.join(ckpt_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {out_json}")

if __name__ == "__main__":
    main()
