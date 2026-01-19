from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HOME", _os.path.expanduser("~/.cache/huggingface"))
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import os

import json
import argparse
from typing import Any, Dict, List, Tuple, Optional
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch import amp as torch_amp
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from env_config import CFG, DEVICE, load_cfg, ensure_dir
from encoders import (
    BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder,
    EncoderConfig, build_encoders,
)
from routing_and_heads import (
    build_fusions,
    RoutePrimaryProjector,
    CapsuleMortalityHead,            # outputs [B,2] logits
    forward_capsule_from_routes,     # returns (logits, prim_acts, route_embs [, routing_coef])
)

def _cfg(name: str, default):
    return getattr(CFG, name, default)

TASK_MAP = {"mort": 0, "mortality": 0}
COL_MAP  = {"mort": "mort", "mortality": "mort"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TOKENIZER = None
MAXLEN = 512
CHUNK_STRIDE = 128

def _chunk_long_ids(ids: list[int], attn: list[int], maxlen: int, stride: int):
    if len(ids) <= maxlen:
        return [ids], [attn]
    out_ids, out_attn = [], []
    step = max(1, maxlen - max(stride, 0))
    i = 0
    while i < len(ids):
        s = ids[i:i+maxlen]; a = attn[i:i+maxlen]
        out_ids.append(s); out_attn.append(a)
        if i + maxlen >= len(ids): break
        i += step
    return out_ids, out_attn
def prepare_notes_batch(notes_payloads):
    global TOKENIZER, MAXLEN
    if TOKENIZER is None:
        raise RuntimeError("TOKENIZER not initialized; call after load_cfg().")

    MAXLEN = int(_cfg("max_text_len", 512))
    pad_id = TOKENIZER.pad_token_id or 0

    if len(notes_payloads) > 0 and isinstance(notes_payloads[0], list):
        notes_payloads = [{"mode": "text", "chunks": x} for x in notes_payloads]

    def _pad_list(x, L=MAXLEN, v=pad_id):
        x = list(x)
        return x[:L] if len(x) >= L else x + [v] * (L - len(x))

    out = []
    for p in notes_payloads:
        mode = str(p.get("mode", "")).lower().strip()
        all_ids, all_attn = [], []

        if mode == "text":
            texts = [t.replace("[CLS]", "").replace("[SEP]", "").strip()
                     for t in p.get("chunks", []) if t and str(t).strip()]
            for t in texts:
                enc = TOKENIZER(t, truncation=True, max_length=MAXLEN, padding=False,
                                return_attention_mask=True, add_special_tokens=True)
                ids_chunks, attn_chunks = _chunk_long_ids(enc["input_ids"], enc["attention_mask"], MAXLEN, CHUNK_STRIDE)
                all_ids.extend(ids_chunks)
                all_attn.extend(attn_chunks)

        elif mode == "pretokenized":
            for ids, attn in zip(p.get("input_ids", []), p.get("attention_mask", [])):
                if ids and attn:
                    all_ids.append(_pad_list(ids, MAXLEN, pad_id))
                    all_attn.append(_pad_list(attn, MAXLEN, 0))

        else:
            raise ValueError(f"[prepare_notes_batch] Unknown mode='{mode}' keys={list(p.keys())}")

        if len(all_ids) == 0:
            ids_mat  = torch.zeros((0, MAXLEN), dtype=torch.long, device=DEVICE)
            attn_mat = torch.zeros((0, MAXLEN), dtype=torch.long, device=DEVICE)
        else:
            ids_mat  = torch.tensor([_pad_list(x, MAXLEN, pad_id) for x in all_ids], dtype=torch.long, device=DEVICE)
            attn_mat = torch.tensor([_pad_list(x, MAXLEN, 0)      for x in all_attn], dtype=torch.long, device=DEVICE)

        out.append({"input_ids": ids_mat, "attention_mask": attn_mat})
    return out


def pretok_batch_notes(batch_notes: list[list[str]]):
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
            out.append({"input_ids": torch.zeros(0, MAXLEN, dtype=torch.long, device=DEVICE),
                        "attention_mask": torch.zeros(0, MAXLEN, dtype=torch.long, device=DEVICE)})
            continue
        all_ids, all_attn = [], []
        for t in texts:
            enc = TOKENIZER(t, truncation=True, max_length=MAXLEN, padding=False,
                            return_attention_mask=True, add_special_tokens=True)
            ids, attn = enc["input_ids"], enc["attention_mask"]
            ids_chunks, attn_chunks = _chunk_long_ids(ids, attn, MAXLEN, CHUNK_STRIDE)
            all_ids.extend(ids_chunks); all_attn.extend(attn_chunks)
        def _pad(x, L=MAXLEN, v=pad_id): return x + [v]*(L-len(x))
        ids_mat  = torch.tensor([_pad(ch) for ch in all_ids],  dtype=torch.long, device=DEVICE)
        attn_mat = torch.tensor([_pad(ch, MAXLEN, 0) for ch in all_attn], dtype=torch.long, device=DEVICE)
        out.append({"input_ids": ids_mat, "attention_mask": attn_mat})
    return out

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

def parse_args():
    ap = argparse.ArgumentParser(description="Mortality (binary) with 7-route capsule routing (2-class CE)")
    ap.add_argument("--task", type=str, default=_cfg("task_name", "mort"),
                    choices=["mort", "mortality"])

    ap.add_argument("--require_all_modalities", action="store_true", default=True,
                    help="Only keep stays that have structured + notes + image.")
    ap.add_argument("--data_root", type=str, default=_cfg("data_root", "./data"))
    ap.add_argument("--ckpt_root", type=str, default=_cfg("ckpt_root", "./ckpts"))
    ap.add_argument("--epochs", type=int, default=max(1, _cfg("max_epochs_tri", 5)))
    ap.add_argument("--batch_size", type=int, default=_cfg("batch_size", 16))
    ap.add_argument("--lr", type=float, default=_cfg("lr", 1e-4))
    ap.add_argument("--weight_decay", type=float, default=_cfg("weight_decay", 1e-4))
    ap.add_argument("--num_workers", type=int, default=_cfg("num_workers", 4))
    ap.add_argument("--finetune_text", action="store_true", help="Unfreeze Bio_ClinicalBERT if set.")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt).")
    ap.add_argument("--log_every", type=int, default=300, help="Print training stats every N steps.")
    ap.add_argument("--precision", type=str, default="auto",
                    choices=["auto", "fp16", "bf16", "off"])
    ap.add_argument("--peek_first_batch", action="store_true", default=True)
    ap.add_argument("--verbose_sanity", action="store_true", default=False)
    ap.add_argument("--route_debug", action="store_true")
    ap.add_argument("--calib_bins", type=int, default=10)
    return ap.parse_args()


def infer_structured_dim(struct_df: pd.DataFrame, feat_cols: List[str]) -> int:
    if len(feat_cols) >= 1:
        return int(len(feat_cols))

    for c in struct_df.columns:
        if c in ("stay_id", "hour", "split", "y_true"):
            continue
        v = struct_df[c].dropna()
        if len(v) == 0:
            continue
        v0 = v.iloc[0]
        arr2d = _to_2d_float32(v0)  
        if arr2d.size == 0:
            continue
        return int(arr2d.shape[1])

    raise RuntimeError("[infer_structured_dim] Could not infer structured feature dimension F from structured parquet.")


import ast

def _to_2d_float32(v) -> np.ndarray:
    if v is None:
        return np.zeros((0, 0), dtype=np.float32)

    if isinstance(v, str):
        s = v.strip()
        if len(s) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        try:
            if s.startswith("array(") and s.endswith(")"):
                s = s[len("array("):-1]
            v = ast.literal_eval(s)
        except Exception:
            return np.zeros((0, 0), dtype=np.float32)

    try:
        arr = np.asarray(v)
    except Exception:
        return np.zeros((0, 0), dtype=np.float32)

    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if arr.dtype == object:
        try:
            arr = np.array([np.asarray(x, dtype=np.float32) for x in arr], dtype=object)
        except Exception:
            return np.zeros((0, 0), dtype=np.float32)

    try:
        arr = arr.astype(np.float32)
    except Exception:
        try:
            arr = np.vectorize(lambda x: float(x))(arr).astype(np.float32)
        except Exception:
            return np.zeros((0, 0), dtype=np.float32)

    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim >= 2:
        f = arr.shape[-1]
        t = int(np.prod(arr.shape[:-1]))
        return arr.reshape(t, f)

    return np.zeros((0, 0), dtype=np.float32)

class ICUStayDataset(Dataset):
    """
    Strict tri-modal dataset:
      REQUIRED under data_root:
        - splits.json
        - labels_mort.parquet          (stay_id, mort)
        - structured_24h.parquet       (stay_id, hour, <17 feats>)
        - notes_48h_chunks.parquet     (stay_id, chunk_000..chunk_XXX)
        - images_24h.parquet           (stay_id, image_path)
    """
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
            "labels_mortality.parquet",
            "structured_mortality.parquet",
            "notes_wide_clean.parquet",
            "images.parquet",
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
            raise KeyError(f"[ICUStayDataset] split '{split}' not in splits.json keys: {list(splits.keys())}")
        split_ids: List[int] = list(splits[split])

        struct_fp = os.path.join(root, "structured_mortality.parquet")
        notes_fp  = os.path.join(root, "notes_wide_clean.parquet")
        images_fp = os.path.join(root, "images.parquet")
        labels_fp = os.path.join(root, "labels_mortality.parquet")

        self.struct = pd.read_parquet(struct_fp)
        self.notes  = pd.read_parquet(notes_fp)
        self.images = pd.read_parquet(images_fp)
        self.labels = pd.read_parquet(labels_fp)

        time_candidates = [
            "hour", "hours", "hr",
            "chart_hour", "chart_hr",
            "offset_hour", "offset_hours",
            "hours_since_admit", "hrs_since_admit",
            "time", "charttime", "t",
            "step", "time_step", "time_idx", "time_index", "seq"
        ]

        self.time_col = None
        for c in time_candidates:
            if c in self.struct.columns:
                self.time_col = c
                break

        if self.time_col is None:
            hour_like = [c for c in self.struct.columns if "hour" in str(c).lower()]
            time_like = [c for c in self.struct.columns if "time" in str(c).lower()]
            if hour_like:
                self.time_col = hour_like[0]
            elif time_like:
                self.time_col = time_like[0]

        if self.time_col is None:
            print(
                f"[warn] structured parquet has no explicit time column (expected 'hour' or similar). "
                f"Will keep row order as-is per stay. columns={list(self.struct.columns)}"
            )
        else:
            print(f"[dataset:{split}] using structured time column: {self.time_col}")

        label_candidates = [
            "mort", "mortality", "death", "label", "y",
            "hospital_expire_flag", "expire_flag", "died"
        ]
        self.label_col = None
        for c in label_candidates:
            if c in self.labels.columns:
                self.label_col = c
                break

        if self.label_col is None:
            cand = [c for c in self.labels.columns if c != "stay_id"]
            if len(cand) == 1:
                self.label_col = cand[0]

        if self.label_col is None:
            raise KeyError(
                f"[ICUStayDataset] labels parquet must contain a mortality label column. "
                f"Expected one of {label_candidates} (or a single non-stay_id column). "
                f"Found columns: {list(self.labels.columns)}"
            )

        print(f"[dataset:{split}] using label column: {self.label_col}")

        img_path_candidates = [
            "image_path", "img_path", "path", "filepath", "file_path",
            "png_path", "jpg_path", "jpeg_path", "dicom_path", "dcm_path"
        ]
        self.img_path_col = None
        for c in img_path_candidates:
            if c in self.images.columns:
                self.img_path_col = c
                break

        if self.img_path_col is None:
            path_like = [c for c in self.images.columns if "path" in str(c).lower()]
            if path_like:
                self.img_path_col = path_like[0]

        if self.img_path_col is None:
            raise KeyError(
                f"[ICUStayDataset] images.parquet must contain an image path column. "
                f"Expected one of {img_path_candidates} or something with 'path'. "
                f"Found columns: {list(self.images.columns)}"
            )

        print(f"[dataset:{split}] using image path column: {self.img_path_col}")

        base_cols = {"stay_id"}
        if self.time_col is not None:
            base_cols.add(self.time_col)
        base_cols |= {"hour", "split", "y_true", "label", "mort", "mortality"}

        cand = [c for c in self.struct.columns if c not in base_cols]
        num_cand = [c for c in cand if pd.api.types.is_numeric_dtype(self.struct[c])]

        self.feat_cols: List[str] = num_cand
        print(f"[dataset:{split}] feat_cols={self.feat_cols} (wide numeric features)")

        file_F = infer_structured_dim(self.struct, self.feat_cols)
        cfg_F  = int(getattr(CFG, "structured_n_feats", file_F))

        if file_F != cfg_F:
            print(f"[warn] structured_n_feats mismatch: CFG={cfg_F} vs file={file_F} in {struct_fp}. "
                  f"Overriding CFG.structured_n_feats -> {file_F}")
        CFG.structured_n_feats = int(file_F)

        self.note_col: Optional[str] = None
        self.chunk_cols: List[str] = [c for c in self.notes.columns if str(c).startswith("chunk_")]
        self.chunk_cols.sort()
        if len(self.chunk_cols) == 0:
            raise ValueError("[ICUStayDataset] notes_48h_chunks.parquet must contain at least one 'chunk_*' column.")

        ids_set = set(split_ids)

        struct_ids = set(self.struct["stay_id"].unique().tolist())

        note_rows = self.notes.copy()
        any_text = np.zeros(len(note_rows), dtype=bool)
        for c in self.chunk_cols:
            any_text |= note_rows[c].fillna("").astype(str).str.strip().ne("")
        note_ids = set(note_rows.loc[any_text, "stay_id"].unique().tolist())

        img_rows = self.images.copy()
        img_ids = set(
            img_rows.loc[
                img_rows[self.img_path_col].fillna("").astype(str).str.strip().ne(""),
                "stay_id"
            ].unique().tolist()
        )

        label_ids = set(self.labels["stay_id"].unique().tolist())

        keep_ids = ids_set & struct_ids & note_ids & img_ids & label_ids
        dropped = len(ids_set) - len(keep_ids)
        self.ids: List[int] = sorted(list(keep_ids))
        if len(self.ids) == 0:
            raise RuntimeError(f"[ICUStayDataset] After tri-modal filtering, split '{self.split}' is empty.")
        print(f"[dataset:{split}] strict tri-modal -> kept {len(self.ids)} / {len(ids_set)} (dropped {dropped})")

        n_chunks = len(self.chunk_cols)
        print(f"[dataset:{split}] notes columns -> note_col=None chunk_cols={n_chunks}")
        print(
            f"[dataset:{split}] root={root} ids={len(self.ids)} "
            f"| struct rows={len(self.struct)} (F={len(self.feat_cols)}) "
            f"| notes rows={len(self.notes)} (chunks={n_chunks}) "
            f"| images rows={len(self.images)} | labels rows={len(self.labels)}"
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = self.ids[idx]
        df_s = self.struct[self.struct.stay_id == stay_id]
        if self.time_col is not None and self.time_col in df_s.columns:
            df_s = df_s.sort_values(self.time_col)

        if len(self.feat_cols) > 0:
            xs_np = df_s[self.feat_cols].astype("float32").fillna(0.0).to_numpy()
        else:
            cand_cols = [c for c in ["x", "values", "features", "x_ehr_48h_2h_76"] if c in df_s.columns]
            if not cand_cols:
                bad = {"stay_id", "split", "y_true"}
                if self.time_col is not None:
                    bad.add(self.time_col)
                cand_cols = [c for c in df_s.columns if c not in bad]
            col = cand_cols[0]

            if idx == 0:
                vv = df_s[col].iloc[0]
                print(f"[debug] structured col='{col}' type(first)={type(vv)}")


            seq = []
            for v in df_s[col].tolist():
                arr2d = _to_2d_float32(v)   # [t,f] or [0,0]
                if arr2d.size == 0:
                    continue
                seq.append(arr2d)

            if len(seq):
                xs_np = np.concatenate(seq, axis=0).astype(np.float32)
            else:
                xs_np = np.zeros((0, int(_cfg("structured_n_feats", 0))), dtype=np.float32)

        xs = torch.from_numpy(xs_np)  # [<=T,F]
        T = int(_cfg("structured_seq_len", 24))
        F = int(_cfg("structured_n_feats", xs.shape[1] if xs.ndim == 2 else CFG.structured_n_feats))
        xs = pad_or_trim_struct(xs, T=T, F=F)

        notes_list: List[str] = []
        df_n = self.notes[self.notes.stay_id == stay_id]
        if not df_n.empty:
            row = df_n.iloc[0]
            for c in self.chunk_cols:
                if c in row.index:
                    val = row[c]
                    if pd.notna(val) and str(val).strip():
                        notes_list.append(str(val))

        img_paths: List[str] = []
        df_i = self.images[self.images.stay_id == stay_id]
        if not df_i.empty:
            img_paths = df_i[self.img_path_col].dropna().astype(str).tolist()[-1:]  
        lab_row = self.labels.loc[self.labels.stay_id == stay_id, [self.label_col]]
        y = 0 if lab_row.empty else int(lab_row.values[0][0])

        y = torch.tensor(y, dtype=torch.long)

        return {"stay_id": stay_id, "x_struct": xs, "notes_list": notes_list,
                "image_paths": img_paths, "y": y}

def pad_or_trim_struct(x: torch.Tensor, T: int, F: int) -> torch.Tensor:
    # x: [t, f]
    if x.ndim != 2:
        raise ValueError(f"[pad_or_trim_struct] expected 2D tensor [t,f], got {tuple(x.shape)}")
    if x.shape[1] != F:
        raise ValueError(f"[pad_or_trim_struct] feature dim mismatch: x has F={x.shape[1]} but CFG says F={F}")
    t = x.shape[0]
    if t >= T:
        return x[-T:]
    pad = torch.zeros(T - t, F, dtype=x.dtype)
    return torch.cat([pad, x], dim=0)


def load_cxr_tensor(paths: List[str], tfms: T.Compose, return_path: bool = False):
    if not paths:
        tensor = torch.zeros(3, 224, 224)
        return (tensor, "<none>") if return_path else tensor
    p = paths[-1]
    try:
        with Image.open(p) as img:
            tensor = tfms(img)
    except Exception as e:
        print(f"[warn] failed to open image: {p} ({e}) -> returning zero tensor")
        tensor = torch.zeros(3, 224, 224)
    return (tensor, p) if return_path else tensor

def collate_fn_factory(tidx: int, img_tfms: T.Compose):
    first_print = {"done": False}

    def _collate(batch: List[Dict[str, Any]]):
        xL_batch = torch.stack([b["x_struct"] for b in batch], dim=0)  # [B,T,F]
        mL_batch = (xL_batch.abs().sum(dim=2) > 0).float()             # [B,T] 

        notes_payloads = [b["notes_list"] for b in batch]

        imgs_list, img_paths_list = [], []
        for b in batch:
            assert len(b["image_paths"]) > 0 and str(b["image_paths"][-1]).strip(), \
                "[collate] tri-modal strict: missing image path for a sample"
            img_t, path = load_cxr_tensor(b["image_paths"], img_tfms, return_path=True)
            imgs_list.append(img_t)
            img_paths_list.append(path)
        imgs_batch = torch.stack(imgs_list, dim=0)                     # [B,3,224,224]

        y_batch = torch.stack([b["y"].view(()) for b in batch], dim=0).long()  # [B]

        dbg = {"stay_ids": [b["stay_id"] for b in batch], "img_paths": img_paths_list}

        if not first_print["done"]:
            first_print["done"] = True
            print(f"[collate] xL_batch:{tuple(xL_batch.shape)} mL_batch:{tuple(mL_batch.shape)} "
                  f"imgs_batch:{tuple(imgs_batch.shape)} y_batch:{tuple(y_batch.shape)} "
                  f"notes_payloads:{len(notes_payloads)}")
        return xL_batch, mL_batch, notes_payloads, imgs_batch, y_batch, dbg

    return _collate

@torch.no_grad()
def pretty_print_small_batch(xL, mL, notes, dbg, k: int = 3) -> None:
    B, T, F = xL.shape
    k = min(k, B)
    print("\n[sample-inspect] ---- Top few samples ----")
    for i in range(k):
        sid = dbg["stay_ids"][i] if "stay_ids" in dbg else "<id?>"
        nz_rows = (mL[i] > 0.5).nonzero(as_tuple=False).flatten().tolist()
        show_rows = nz_rows[:2] if nz_rows else []
        ehr_rows = []
        for r in show_rows:
            vec = xL[i, r].detach().cpu().numpy()
            ehr_rows.append(np.round(vec[:min(5, F)], 3).tolist())
        note_text = notes[i][0] if len(notes[i]) > 0 else ""
        note_text = (note_text[:120] + "…") if len(note_text) > 120 else note_text
        imgp = dbg.get("img_paths", ["<path?>"] * B)[i]
        print(f"  • stay_id={sid} | ehr_rows(first2->first5feats)={ehr_rows} | "
              f"notes[0][:120]=\"{note_text}\" | cxr='{imgp}'")
    print("[sample-inspect] ---------------------------\n")

def _capsule_forward_safe(z, fusion, projector, cap_head,
                          route_mask=None, act_temperature=1.0,
                          detach_priors=False, return_routing=True):
    """Robust wrapper that supports both new and old signatures."""
    try:
        return forward_capsule_from_routes(
            z_unimodal=z, fusion=fusion, projector=projector, capsule_head=cap_head,
            route_mask=route_mask, act_temperature=act_temperature,
            detach_priors=detach_priors, return_routing=return_routing
        )
    except TypeError:
        return forward_capsule_from_routes(
            z_unimodal=z, fusion=fusion, projector=projector, capsule_head=cap_head,
            return_routing=return_routing
        )

@torch.no_grad()
def evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, loader, amp_ctx, loss_fn):
    behrt.eval(); imgenc.eval()
    if getattr(bbert, "bert", None) is not None: bbert.bert.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    act_sum = torch.zeros(7, dtype=torch.float32)
    printed_unimodal = False
    printed_caps_once = False
    rpt_every = int(_cfg("routing_print_every", 0) or 0)
    for bidx, (xL, mL, notes, imgs, y, dbg) in enumerate(loader):
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)
        with amp_ctx:
            zL = behrt(xL, mask=mL)
            zN = bbert(pretok_batch_notes(notes))

            zI = imgenc(imgs)
            z = {"L": zL, "N": zN, "I": zI}
            gates = torch.ones(7, device=DEVICE, dtype=torch.float32)
            route_mask = gates.unsqueeze(0).expand(zL.size(0), -1)
            out = _capsule_forward_safe(
                z, fusion, projector, cap_head,
                route_mask=route_mask, act_temperature=1.0,
                detach_priors=False, return_routing=True
            )
            logits, prim_acts, route_embs = out[0], out[1], out[2]
            routing_coef = out[3] if len(out) > 3 else None

            if not printed_unimodal:
                printed_unimodal = True
                print(f"[eval:unimodal] zL:{tuple(zL.shape)} zN:{tuple(zN.shape)} zI:{tuple(zI.shape)}")
                pretty_print_small_batch(xL, mL, notes, dbg, k=3)
            if (not printed_caps_once) or (rpt_every > 0 and ((bidx + 1) % rpt_every == 0)):
                printed_caps_once = True
                keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
                print(f"[eval:caps] logits:{tuple(logits.shape)} prim_acts:{tuple(prim_acts.shape)} routes -> {keys}")
            loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)
        act_sum += prim_acts.detach().float().cpu().sum(dim=0)
    avg_loss = total_loss / max(1, total)
    avg_acc  = total_correct / max(1, total)
    avg_act  = (act_sum / max(1, total)).tolist()
    route_names = ["L","N","I","LN","LI","NI","LNI"]
    avg_act_dict = {r: avg_act[i] for i, r in enumerate(route_names)}
    return avg_loss, avg_acc, avg_act_dict

def save_checkpoint(path: str, state: Dict):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_checkpoint(path: str, behrt, bbert, imgenc, fusion, projector, cap_head, optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    behrt.load_state_dict(ckpt["behrt"])
    bbert.load_state_dict(ckpt["bbert"])
    imgenc.load_state_dict(ckpt["imgenc"])
    for k in fusion.keys():
        fusion[k].load_state_dict(ckpt["fusion"][k])
    projector.load_state_dict(ckpt["projector"])
    cap_head.load_state_dict(ckpt["cap_head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] loaded epoch={ckpt.get('epoch', 0)} val_acc={ckpt.get('val_acc', -1):.4f}")
    return int(ckpt.get("epoch", 0))

@torch.no_grad()
def collect_epoch_outputs(loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx):
    behrt.eval(); imgenc.eval()
    if getattr(bbert, "bert", None) is not None: bbert.bert.eval()
    y_true, p1, y_pred, ids = [], [], [], []
    for xL, mL, notes, imgs, y, dbg in loader:
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)
        with amp_ctx:
            zL = behrt(xL, mask=mL)
            zN = bbert(pretok_batch_notes(notes))

            zI = imgenc(imgs)
            out = _capsule_forward_safe(
                {"L": zL, "N": zN, "I": zI}, fusion, projector, cap_head,
                route_mask=torch.ones(zL.size(0), 7, device=DEVICE, dtype=torch.float32),
                act_temperature=1.0, detach_priors=False, return_routing=True
            )
            logits = out[0]
        probs = torch.softmax(logits, dim=-1)[:, 1]
        probs = probs.float()
        y_true.append(y.detach().cpu())
        p1.append(probs.detach().cpu())
        y_pred.append(logits.argmax(dim=1).detach().cpu())
        ids += dbg.get("stay_ids", [])
    y_true = torch.cat(y_true).numpy()
    p1     = torch.cat(p1).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return y_true, p1, y_pred, ids

def epoch_metrics(y_true, p1, y_pred):
    out = {}
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p1     = np.asarray(p1).astype(float)

    try:
        out["AUROC"] = float(roc_auc_score(y_true, p1))
    except Exception:
        out["AUROC"] = float("nan")

    try:
        out["AUPRC"] = float(average_precision_score(y_true, p1))
    except Exception:
        out["AUPRC"] = float("nan")

    out["F1"]        = float(f1_score(y_true, y_pred, pos_label=1))
    out["Recall"]    = float(recall_score(y_true, y_pred, pos_label=1))
    out["Precision"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["CM"]        = confusion_matrix(y_true, y_pred)
    return out


def expected_calibration_error(p, y, n_bins=10):
    p = np.asarray(p); y = np.asarray(y).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids = 0.5 * (bins[1:] + bins[:-1])
    ece = 0.0
    bconf, bacc, bcnt = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            bconf.append(0.0); bacc.append(0.0); bcnt.append(0)
            continue
        conf = float(p[m].mean()); acc = float((y[m] == 1).mean())
        w = m.mean()
        ece += w * abs(acc - conf)
        bconf.append(conf); bacc.append(acc); bcnt.append(int(m.sum()))
    return float(ece), mids, np.array(bconf), np.array(bacc), np.array(bcnt)

def reliability_plot(bin_centers, bin_conf, bin_acc, out_path):
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(bin_conf, bin_acc, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _print_routing_mort(routing_coef, prim_acts, where=""):
    with torch.no_grad():
        beta = routing_coef * prim_acts.unsqueeze(-1)  # [B,7,2]
        bmean = beta.mean(dim=0).detach().cpu().numpy()  # [7,2]
        routes = ["L","N","I","LN","LI","NI","LNI"]
        msg = " | ".join(f"{r}:{bmean[i,0]:.3f}/{bmean[i,1]:.3f}" for i,r in enumerate(routes))
        print(f"[routing β] {where} {msg}")

def _set_all_seeds():
    seed = int(_cfg("seed", 1337))
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

def main():
    args = parse_args()
    load_cfg()
    _set_all_seeds()

    global TOKENIZER, MAXLEN
    TOKENIZER = AutoTokenizer.from_pretrained(CFG.text_model_name)
    MAXLEN = int(_cfg("max_text_len", 512))

    print(f"[setup] DEVICE={DEVICE} | batch_size={args.batch_size} | epochs={args.epochs}")

    use_cuda = (str(DEVICE).startswith("cuda") and torch.cuda.is_available())
    if use_cuda:
        if args.precision == "fp16":
            amp_ctx = torch_amp.autocast(device_type="cuda", dtype=torch.float16)
        elif args.precision == "bf16":
            amp_ctx = torch_amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            amp_ctx = torch_amp.autocast(device_type="cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        amp_ctx = nullcontext()
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    train_ds = ICUStayDataset(args.data_root, split="train")
    val_ds   = ICUStayDataset(args.data_root, split="val")
    test_ds  = ICUStayDataset(args.data_root, split="test")

    from torch.utils.data import WeightedRandomSampler

    label_col = train_ds.label_col

    lab = train_ds.labels[["stay_id", label_col]].copy()
    lab["stay_id"] = lab["stay_id"].astype(int)

    y_series = (
        lab.set_index("stay_id")[label_col]
           .reindex(pd.Index(train_ds.ids, dtype=int))
    )

    if y_series.isna().any():
        missing = y_series[y_series.isna()].index[:10].tolist()
        raise RuntimeError(
            f"[sampler] Missing labels for {int(y_series.isna().sum())} stay_ids. "
            f"Examples: {missing}. Check labels_mortality.parquet stay_id coverage."
        )

    yvals = y_series.astype(int).values

    pos = int(yvals.sum())
    neg = int(len(yvals) - pos)
    pos_ratio = (neg / max(1, pos))

    weights = [pos_ratio if int(y_series.loc[int(sid)]) == 1 else 1.0 for sid in train_ds.ids]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    print(f"[sampler] label_col='{label_col}' pos={pos} neg={neg} pos_weight={pos_ratio:.3f}")

    ce = nn.CrossEntropyLoss(
        reduction="mean",
        label_smoothing=float(_cfg("label_smoothing", 0.05))
    )
    print(f"[loss] CE with label_smoothing={_cfg('label_smoothing', 0.05)} and class-balanced sampler")

    collate_train = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("train"))
    collate_eval  = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("val"))
    pin = use_cuda

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_train, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_eval
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_eval
    )

    enc_cfg = EncoderConfig(
        d=_cfg("d", 256), dropout=_cfg("dropout", 0.5),
        structured_seq_len=_cfg("structured_seq_len", 24),
        structured_n_feats=_cfg("structured_n_feats", 17),
        structured_layers=_cfg("structured_layers", 2),
        structured_heads=_cfg("structured_heads", 4),
        structured_pool="mean",
        text_model_name=_cfg("text_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        text_max_len=_cfg("max_text_len", 512),
        note_agg="mean",
        max_notes_concat=8,
        img_agg="last",
    )
    behrt, bbert, imgenc = build_encoders(enc_cfg, device=DEVICE)
    print(f"[encoders] d={CFG.d} | BEHRT out_dim={behrt.out_dim} | "
          f"BERT hidden={getattr(bbert, 'hidden', 'NA')}→out_dim={bbert.out_dim} | "
          f"IMG out_dim={getattr(imgenc.proj, 'out_features', 'NA')}")

    if not args.finetune_text and getattr(bbert, "bert", None) is not None:
        for p in bbert.bert.parameters():
            p.requires_grad = False
        bbert.bert.eval()
        print("[encoders] Bio_ClinicalBERT frozen (feature extractor mode)")

    fusion = build_fusions(d=CFG.d, feature_mode=CFG.feature_mode, p_drop=CFG.dropout)
    for k in fusion.keys():
        fusion[k].to(DEVICE)
    projector = RoutePrimaryProjector(d_in=CFG.d, pc_dim=CFG.capsule_pc_dim).to(DEVICE)
    cap_head = CapsuleMortalityHead(
        pc_dim=CFG.capsule_pc_dim,
        mc_caps_dim=CFG.capsule_mc_caps_dim,
        num_routing=CFG.capsule_num_routing,
        dp=CFG.dropout,
        act_type=CFG.capsule_act_type,
        layer_norm=CFG.capsule_layer_norm,
        dim_pose_to_vote=CFG.capsule_dim_pose_to_vote,
    ).to(DEVICE)
    print(f"[capsule] pc_dim={CFG.capsule_pc_dim} mc_caps_dim={CFG.capsule_mc_caps_dim} "
          f"iters={CFG.capsule_num_routing} act_type={CFG.capsule_act_type}")

    params = list(behrt.parameters()) + list(bbert.parameters()) + list(imgenc.parameters())
    for k in fusion.keys(): params += list(fusion[k].parameters())
    params += list(projector.parameters()) + list(cap_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_acc = -1.0
    ckpt_dir = os.path.join(args.ckpt_root, "mort_capsule")
    ensure_dir(ckpt_dir)
    if args.resume and os.path.isfile(args.resume):
        print(f"[main] Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, behrt, bbert, imgenc, fusion, projector, cap_head, optimizer)

    printed_once = False

    route_dropout_p = float(_cfg("route_dropout_p", 0.10))
    routing_warmup_epochs = int(_cfg("routing_warmup_epochs", 1))
    route_entropy_lambda = float(_cfg("route_entropy_lambda", 1e-4))
    route_entropy_warm = int(_cfg("route_entropy_warmup_epochs", 2))
    entropy_use_rc = bool(_cfg("entropy_use_rc", True))

    for epoch in range(start_epoch, args.epochs):
        behrt.train(); imgenc.train()
        if args.finetune_text and getattr(bbert, "bert", None) is not None:
            bbert.bert.train()

        total_loss, total_correct, total = 0.0, 0, 0
        act_sum = torch.zeros(7, dtype=torch.float32)

        for step, (xL, mL, notes, imgs, y, dbg) in enumerate(train_loader):
            xL, mL = xL.to(DEVICE), mL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y = y.to(DEVICE)

            if (epoch == start_epoch) and (step == 0):
                pretty_print_small_batch(xL, mL, notes, dbg, k=3)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                zL = behrt(xL, mask=mL)
                notes_tok = prepare_notes_batch([{"mode":"text", "chunks": x} for x in notes])
                zN = bbert(notes_tok)

                zI = imgenc(imgs)
                z = {"L": zL, "N": zN, "I": zI}

                if not printed_once:
                    printed_once = True
                    print(f"[sanity] xL: {tuple(xL.shape)} | mL: {tuple(mL.shape)} "
                          f"| imgs: {tuple(imgs.shape)} | y: {tuple(y.shape)}")
                    print(f"[sanity] zL: {tuple(zL.shape)} | zN: {tuple(zN.shape)} | zI: {tuple(zI.shape)}")
                    with torch.no_grad():
                        for i in range(min(3, zL.size(0))):
                            print(f"[emb-norms] i={i} ||zL||={zL[i].norm().item():.3f} "
                                  f"||zN||={zN[i].norm().item():.3f} ||zI||={zI[i].norm().item():.3f}")

                B = zL.size(0)
                route_mask = torch.ones(B, 7, device=DEVICE, dtype=torch.float32)
                if route_dropout_p > 0.0 and (torch.rand(()) < route_dropout_p):
                    drop_idx = int(torch.randint(low=3, high=7, size=(1,)))
                    route_mask[:, drop_idx] = 0.0

                detach_priors_flag = (epoch - start_epoch) < routing_warmup_epochs
                temp = (2.0 if epoch < 2 else 1.0)

                out = _capsule_forward_safe(
                    z, fusion, projector, cap_head,
                    route_mask=route_mask, act_temperature=temp,
                    detach_priors=detach_priors_flag, return_routing=True
                )

                logits, prim_acts, route_embs = out[0], out[1], out[2]
                routing_coef = out[3] if len(out) > 3 else None

                if getattr(args, "route_debug", False) and routing_coef is not None and (step % 100 == 0):
                    _print_routing_mort(routing_coef, prim_acts, where=f"TRAIN@step{step}")

                if printed_once and step == 0:
                    keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
                    print(f"[sanity] routes -> {keys} | logits: {tuple(logits.shape)} "
                          f"| prim_acts: {tuple(prim_acts.shape)}")

                loss = ce(logits, y)

                if route_entropy_lambda > 0.0 and ((epoch - start_epoch) < route_entropy_warm):
                    if (routing_coef is not None) and entropy_use_rc:
                        p = torch.clamp(routing_coef, 1e-6, 1.0)       # [B,7,2]
                        H = -(p * p.log()).sum(dim=1).mean()           
                    else:
                        pa = prim_acts                                   # [B,7]
                        pa = pa / (pa.sum(dim=1, keepdim=True) + 1e-6)
                        pa = torch.clamp(pa, 1e-6, 1.0)
                        H = -(pa * pa.log()).sum(dim=1).mean()
                    loss = loss + (-route_entropy_lambda) * H

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total += y.size(0)
            act_sum += prim_acts.detach().cpu().sum(dim=0)

            # periodic logs
            if args.log_every > 0 and ((step + 1) % args.log_every == 0):
                avg_act = (act_sum / max(1, total)).tolist()
                msg = (f"[epoch {epoch+1} step {step+1}] "
                       f"loss={total_loss/max(1,total):.4f} "
                       f"acc={total_correct/max(1,total):.4f} "
                       f"avg_prim_act(L,N,I,LN,LI,NI,LNI)="
                       f"{', '.join(f'{a:.3f}' for a in avg_act)}")
                if routing_coef is not None:
                    rc = routing_coef.detach().float().cpu()
                    rc_mean = rc.mean(dim=0)  # [7,2]
                    routes = ["L","N","I","LN","LI","NI","LNI"]
                    rc_str = " | ".join(
                        f"{r}:({rc_mean[i,0]:.3f},{rc_mean[i,1]:.3f})" for i, r in enumerate(routes)
                    )
                    msg += f" | [routing mean] {rc_str}"
                print(msg)

                # collapse alarm
                if max(avg_act) > 0.95:
                    print(f"[alert] potential collapse → route={int(np.argmax(avg_act))} mean={max(avg_act):.3f}")

        train_loss = total_loss / max(1, total)
        train_acc = total_correct / max(1, total)
        train_avg_act = (act_sum / max(1, total)).tolist()
        print(f"[epoch {epoch+1}] TRAIN loss={train_loss:.4f} acc={train_acc:.4f} "
              f"avg_prim_act={', '.join(f'{a:.3f}' for a in train_avg_act)}")

        # Validation
        val_loss, val_acc, val_act = evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, val_loader, amp_ctx, ce)
        print(f"[epoch {epoch+1}] VAL loss={val_loss:.4f} acc={val_acc:.4f} "
              f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k,v in val_act.items())}")

        y_true, p1, y_pred, _ = collect_epoch_outputs(val_loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx)
        m = epoch_metrics(y_true, p1, y_pred)
        print(
            f"[epoch {epoch+1}] VAL "
            f"AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f} "
            f"F1={m['F1']:.4f} Recall={m['Recall']:.4f} Precision={m['Precision']:.4f}"
        )
        print(f"[epoch {epoch+1}] VAL Confusion Matrix:\n{m['CM']}")
        ece, centers, bconf, bacc, bcnt = expected_calibration_error(p1, y_true, n_bins=args.calib_bins)
        print(f"[epoch {epoch+1}] VAL ECE({args.calib_bins} bins) = {ece:.4f}")
        rel_path = os.path.join(ckpt_dir, f"reliability_val_epoch{epoch+1:03d}.png")
        reliability_plot(centers, bconf, bacc, out_path=rel_path)
        print(f"[epoch {epoch+1}] Saved reliability diagram → {rel_path}")

        # Save checkpoints
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        ckpt = {
            "epoch": epoch + 1,
            "behrt": behrt.state_dict(),
            "bbert": bbert.state_dict(),
            "imgenc": imgenc.state_dict(),
            "fusion": {k: v.state_dict() for k, v in fusion.items()},
            "projector": projector.state_dict(),
            "cap_head": cap_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc,
        }
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), ckpt)
        if is_best:
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), ckpt)
            print(f"[epoch {epoch+1}] Saved BEST checkpoint (acc={val_acc:.4f})")

    # Final test
    print("[main] Evaluating BEST checkpoint on TEST...")
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        behrt.load_state_dict(ckpt["behrt"]); bbert.load_state_dict(ckpt["bbert"]); imgenc.load_state_dict(ckpt["imgenc"])
        for k in fusion.keys(): fusion[k].load_state_dict(ckpt["fusion"][k])
        projector.load_state_dict(ckpt["projector"]); cap_head.load_state_dict(ckpt["cap_head"])

    test_loss, test_acc, test_act = evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, test_loader, amp_ctx, ce)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} "
          f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k,v in test_act.items())}")

    y_true_t, p1_t, y_pred_t, _ = collect_epoch_outputs(test_loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx)
    mt = epoch_metrics(y_true_t, p1_t, y_pred_t)
    print(
        f"[TEST] "
        f"AUROC={mt['AUROC']:.4f} AUPRC={mt['AUPRC']:.4f} "
        f"F1={mt['F1']:.4f} Recall={mt['Recall']:.4f} Precision={mt['Precision']:.4f}"
    )
    print(f"[TEST] Confusion Matrix:\n{mt['CM']}")
    ece_t, centers_t, bconf_t, bacc_t, bcnt_t = expected_calibration_error(p1_t, y_true_t, n_bins=args.calib_bins)
    print(f"[TEST] ECE({args.calib_bins} bins) = {ece_t:.4f}")
    rel_test_path = os.path.join(ckpt_dir, "reliability_test.png")
    reliability_plot(centers_t, bconf_t, bacc_t, out_path=rel_test_path)
    print(f"[TEST] Saved reliability diagram → {rel_test_path}")

if __name__ == "__main__":
    main()
