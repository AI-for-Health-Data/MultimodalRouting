from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HOME", _os.path.expanduser("~/.cache/huggingface"))
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch import amp as torch_amp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from env_config import CFG, DEVICE, load_cfg, ensure_dir
from env_config import get_pheno_name, apply_cli_overrides

from encoders import (
    BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder,
    EncoderConfig, build_encoders,
)
from routing_and_heads import (
    build_fusions,
    RoutePrimaryProjector,
    CapsuleMortalityHead,
    forward_capsule_from_routes,
)

def print_route_matrix_detailed(
    routing_coef: torch.Tensor,
    prim_acts: torch.Tensor,
    label_names: List[str],
    where: str = "",
):
    """
    routing_coef: [B, 7, K] – routing weights (softmax over routes or caps)
    prim_acts:    [B, 7]     – primary activations (sigmoid over 7 routes)
    label_names:  list of K phenotype names (for display)
    """
    with torch.no_grad():
        rc = routing_coef.detach().float().cpu()  # [B,7,K]
        pa = prim_acts.detach().float().cpu()     # [B,7]

        B, R, K = rc.shape

        # Average over batch
        rc_mean = rc.mean(dim=0).numpy()          # [7,K]
        pa_mean = pa.mean(dim=0).numpy()          # [7]

        # Effective weights (per route × phenotype)
        effective = rc_mean * pa_mean[:, np.newaxis]  # [7,K]

        routes = ["L", "N", "I", "LN", "LI", "NI", "LNI"]

        print(f"\n{'=' * 120}")
        print(f"[ROUTING ANALYSIS] {where}")
        print(f"{'=' * 120}")

        # Primary activations
        print(f"\n1. PRIMARY ACTIVATIONS (sigmoid, same across all phenotypes):")
        print("   " + " | ".join(f"{r:4s}={pa_mean[i]:.3f}" for i, r in enumerate(routes)))

        # Per-phenotype routing + effective weights
        print(f"\n2. PER-PHENOTYPE ROUTING WEIGHTS:")
        print("   Format: phenotype_name | L | N | I | LN | LI | NI | LNI")
        print("   Each cell shows: routing_coef(effective_weight)")
        print(f"   {'-' * 116}")

        for k in range(K):
            cells = []
            for i in range(R):
                cells.append(f"{rc_mean[i, k]:.4f}({effective[i, k]:.4f})")

            name = label_names[k] if k < len(label_names) else f"label_{k}"
            row = f"   {name:15s} | " + " | ".join(f"{cell:>10s}" for cell in cells)
            print(row)

        # Route importance aggregated across phenotypes
        print(f"\n3. ROUTE IMPORTANCE (averaged across all phenotypes):")
        rc_avg = rc_mean.mean(axis=1)      # [7]
        eff_avg = effective.mean(axis=1)   # [7]

        for i, r in enumerate(routes):
            print(
                f"   {r:4s}: routing={rc_avg[i]:.3f} "
                f"| effective={eff_avg[i]:.3f} "
                f"| primary_act={pa_mean[i]:.3f}"
            )

        print(f"\n4. ROUTE HEALTH CHECK:")
        for i, r in enumerate(routes):
            if pa_mean[i] < 0.01:
                print(f"    {r:4s} COLLAPSED (primary activation = {pa_mean[i]:.4f})")
            elif pa_mean[i] < 0.1:
                print(f"    {r:4s} WEAK (primary activation = {pa_mean[i]:.4f})")
            elif eff_avg[i] < 0.01:
                print(f"    {r:4s} LOW EFFECTIVE WEIGHT (avg = {eff_avg[i]:.4f})")
            else:
                print(f"    {r:4s} HEALTHY (effective weight = {eff_avg[i]:.3f})")

        print(f"{'=' * 120}\n")


def print_phenotype_routing_heatmap(
    routing_coef: torch.Tensor,
    prim_acts: torch.Tensor,
    label_names: Optional[List[str]] = None,
    where: str = "",
    top_k: Optional[int] = None,
):
    with torch.no_grad():
        rc = routing_coef.detach().float().cpu()
        pa = prim_acts.detach().float().cpu()

        B, R, K = rc.shape
        rc_mean = rc.mean(dim=0).numpy()  # [7,K]
        pa_mean = pa.mean(dim=0).numpy()  # [7]

        effective = rc_mean * pa_mean[:, np.newaxis]  # [7,K]

        # decide which phenotypes to show
        if top_k is None or top_k >= K:
            top_indices = np.arange(K)
        else:
            variance = effective.var(axis=0)  # [K]
            top_indices = variance.argsort()[-top_k:][::-1]

        routes = ["L", "N", "I", "LN", "LI", "NI", "LNI"]

        print(f"\n{'=' * 120}")
        print(f"[PHENOTYPE ROUTING HEATMAP] {where}")
        print("Showing effective weights (primary_act × routing_coef):")
        print(f"{'-' * 120}")

        for idx in top_indices:
            if label_names is not None and idx < len(label_names):
                name = label_names[idx]
            else:
                name = get_pheno_name(idx)

            weights = effective[:, idx]  # [7]

            dominant_idx = weights.argmax()
            dominant_route = routes[dominant_idx]
            dominant_weight = float(weights[dominant_idx])

            weight_str = " | ".join(
                f"{r}:{float(weights[i]):.3f}" for i, r in enumerate(routes)
            )
            print(
                f"  {name:60s} → DOMINANT: {dominant_route:4s} ({dominant_weight:.3f}) "
                f"| ALL: {weight_str}"
            )
        print(f"{'=' * 120}\n")


def save_routing_heatmap(
    routing_coef: torch.Tensor,
    prim_acts: torch.Tensor,
    label_names: List[str],
    where: str,
    out_dir: str,
):
    with torch.no_grad():
        rc = routing_coef.detach().float().cpu()
        pa = prim_acts.detach().float().cpu()

        B, R, K = rc.shape          # [B, 7, K]
        rc_mean = rc.mean(dim=0).numpy()     # [7, K]
        pa_mean = pa.mean(dim=0).numpy()     # [7]
        effective = rc_mean * pa_mean[:, np.newaxis]  # [7, K]

        routes = ["L", "N", "I", "LN", "LI", "NI", "LNI"]

        mat = effective.T  # [K, 7]

        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        im = plt.imshow(mat, aspect="auto")
        plt.colorbar(
            im,
            label="Effective weight (primary_act × routing_coef)"
        )

        plt.xticks(ticks=np.arange(len(routes)), labels=routes)
        plt.yticks(ticks=np.arange(K), labels=label_names, fontsize=6)

        plt.xlabel("Route")
        plt.ylabel("Phenotype")
        plt.title(f"Phenotype Routing Heatmap ({where}, effective weights)")
        plt.tight_layout()

        fname = os.path.join(out_dir, f"phenotype_routing_{where}_heatmap.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"[routing] saved phenotype routing heatmap → {fname}")

def _cfg(name: str, default):
    return getattr(CFG, name, default)


TASK_MAP = {"pheno": 0}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TOKENIZER = None
MAXLEN = 512
CHUNK_STRIDE = 128

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

def prepare_notes_batch(notes_batch: List[Dict[str, Any]]):
    out = []
    pad_id = int(TOKENIZER.pad_token_id or 0)
    L = int(_cfg("max_text_len", 512))

    def _pad_to_len(seq, pad_value: int, max_len: int):
        seq = list(seq)
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [pad_value] * (max_len - len(seq))

    def _flatten_if_nested(seq):
        """
        Accepts:
          - [int, int, ...]
          - [[int,...], [int,...], ...]   (extra nesting)
        Returns a flat [int, int, ...]
        """
        if not isinstance(seq, (list, tuple)):
            return []
        if len(seq) == 0:
            return []
        # if first element is itself list-like => flatten one level
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
            # handle numpy scalars etc.
            if isinstance(v, (np.integer,)):
                out_ints.append(int(v))
            elif isinstance(v, (int,)):
                out_ints.append(v)
            elif isinstance(v, float) and np.isnan(v):
                continue
            else:
                # last resort: try cast
                try:
                    out_ints.append(int(v))
                except Exception:
                    # if it's still a list/object, skip it
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

        # Make sure both are list-of-chunks
        if not isinstance(ids_chunks, (list, tuple)):
            ids_chunks = []
        if not isinstance(attn_chunks, (list, tuple)):
            attn_chunks = []

        # Normalize each chunk to flat List[int]
        ids_chunks  = [_to_int_list(x) for x in ids_chunks]
        attn_chunks = [_to_int_list(x) for x in attn_chunks]

        # Keep only non-empty aligned pairs
        paired = [(a, b) for a, b in zip(ids_chunks, attn_chunks) if len(a) > 0 and len(b) > 0]
        if len(paired) == 0:
            out.append({
                "input_ids": torch.zeros(0, L, dtype=torch.long, device=DEVICE),
                "attention_mask": torch.zeros(0, L, dtype=torch.long, device=DEVICE),
            })
            continue

        ids_chunks, attn_chunks = zip(*paired)

        ids_mat = torch.tensor(
            [_pad_to_len(x, pad_id, L) for x in ids_chunks],
            dtype=torch.long, device=DEVICE
        )
        attn_mat = torch.tensor(
            [_pad_to_len(x, 0, L) for x in attn_chunks],
            dtype=torch.long, device=DEVICE
        )

        out.append({"input_ids": ids_mat, "attention_mask": attn_mat})

    return out



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
                "input_ids": torch.zeros(0, MAXLEN, dtype=torch.long, device=DEVICE),
                "attention_mask": torch.zeros(0, MAXLEN, dtype=torch.long, device=DEVICE),
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
    ap = argparse.ArgumentParser(
        description="Phenotype prediction with 7-route capsule routing (multi-label)"
    )
    ap.add_argument("--task", type=str, default=_cfg("task_name", "pheno"),
                    choices=list(TASK_MAP.keys()))
    ap.add_argument("--require_all_modalities", action="store_true", default=True,
                    help="Only keep stays that have structured + notes + image.")
    ap.add_argument("--data_root", type=str, default=_cfg("data_root", "./data"))
    ap.add_argument("--ckpt_root", type=str, default=_cfg("ckpt_root", "./ckpts"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=_cfg("batch_size", 16))
    ap.add_argument("--lr", type=float, default=_cfg("lr", 2e-4))
    ap.add_argument("--weight_decay", type=float, default=_cfg("weight_decay", 1e-4))
    ap.add_argument("--num_workers", type=int, default=_cfg("num_workers", 4))
    ap.add_argument("--finetune_text", action="store_true",
                    help="Unfreeze Bio_ClinicalBERT if set.")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt).")
    ap.add_argument("--log_every", type=int, default=300,
                    help="Print training stats every N steps.")
    ap.add_argument("--precision", type=str, default="auto",
                    choices=["auto", "fp16", "bf16", "off"])
    ap.add_argument("--peek_first_batch", action="store_true", default=True)
    ap.add_argument("--verbose_sanity", action="store_true", default=False)
    ap.add_argument("--route_debug", action="store_true")
    ap.add_argument("--calib_bins", type=int, default=10)
    return ap.parse_args()

def _standardize_image_path_column(df: pd.DataFrame) -> pd.DataFrame:
    # Accept common variants and normalize to "cxr_path"
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


def _detect_notes_schema(notes_df: pd.DataFrame):
    chunk_cols = sorted([c for c in notes_df.columns if str(c).startswith("chunk_")])

    input_id_cols = sorted([c for c in notes_df.columns if str(c).startswith("input_ids_")])
    attn_cols = sorted([c for c in notes_df.columns if str(c).startswith("attention_mask_")])
    if len(attn_cols) == 0:
        attn_cols = sorted([c for c in notes_df.columns if str(c).startswith("attn_mask_")])

    if len(input_id_cols) > 0 and len(attn_cols) > 0:
        # match by suffix (e.g., _000)
        def suf(x): return str(x).split("input_ids_")[-1] if "input_ids_" in str(x) else str(x).split("mask_")[-1]
        id_sufs = {c.split("input_ids_")[-1] for c in input_id_cols}
        mk_sufs = {c.split("_")[-1] for c in attn_cols}  # works for attention_mask_XXX or attn_mask_XXX
        common = sorted(list(id_sufs & mk_sufs))
        if len(common) == 0:
            raise ValueError("[ICUStayDataset] Found input_ids_* and *_mask_* but no matching suffixes.")
        # rebuild aligned lists in common order
        aligned_ids = [f"input_ids_{s}" for s in common if f"input_ids_{s}" in notes_df.columns]
        # prefer attention_mask_*
        aligned_attn = []
        for s in common:
            if f"attention_mask_{s}" in notes_df.columns:
                aligned_attn.append(f"attention_mask_{s}")
            elif f"attn_mask_{s}" in notes_df.columns:
                aligned_attn.append(f"attn_mask_{s}")
        return ("pretokenized", aligned_ids, aligned_attn)

    if len(chunk_cols) > 0:
        return ("text", chunk_cols, None)

    raise ValueError(
        "[ICUStayDataset] notes.parquet must contain either chunk_* text columns "
        "OR (input_ids_* + attention_mask_* / attn_mask_*) columns."
    )


def _cell_to_list(x):
    """Normalize a parquet cell to a python list of ints (or empty list)."""
    import ast

    if x is None:
        return []

    # pandas may give NaN float
    if isinstance(x, float) and np.isnan(x):
        return []

    # already a list
    if isinstance(x, list):
        return x

    # numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    # pyarrow scalar / other scalar wrappers
    # (some objects support .as_py())
    if hasattr(x, "as_py"):
        try:
            return _cell_to_list(x.as_py())
        except Exception:
            pass

    # strings: try parse
    if isinstance(x, (str, bytes)):
        s = x.decode("utf-8") if isinstance(x, bytes) else x
        s = s.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return []
        # try JSON first
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        # try python literal list: "[1,2,3]"
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        return []

    # generic iterable (but not string)
    if hasattr(x, "__iter__"):
        try:
            return list(x)
        except Exception:
            return []

    return []
def _pick_existing(root: str, candidates: List[str]) -> str:
    """Return the first candidate path that exists under root."""
    for fn in candidates:
        p = os.path.join(root, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these files exist under {root}: {candidates}")

def _standardize_id_column(df: pd.DataFrame, *, want: str = "stay_id") -> pd.DataFrame:
    """
    Unify ID column naming to `stay_id`.
    Common in partial preprocessing: `sample_id`.
    """
    if want in df.columns:
        df[want] = df[want].astype(int)
        return df

    # common alternatives
    candidates = ["sample_id", "icustay_id", "hadm_id"]
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        raise ValueError(
            f"[dataset] No id column found. Expected '{want}' or one of {candidates}. "
            f"Found columns: {list(df.columns)[:50]}"
        )

    df = df.rename(columns={found: want})
    df[want] = df[want].astype(int)
    return df

class ICUStayDataset(Dataset):
    def __init__(self, root: str, split: str = "train", use_partial: Optional[bool] = None):
        super().__init__()
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"[ICUStayDataset] data root not found: {root}")
        self.root = root
        self.split = split
        self.img_tfms = build_image_transform(split)

        # ----------------------------
        # Decide FULL vs PARTIAL
        # ----------------------------
        # If user didn't specify, auto-detect by presence of partial files
        if use_partial is None:
            use_partial = os.path.exists(os.path.join(root, "structured_partial.parquet"))

        # splits file
        splits_path = _pick_existing(root, [
            "splits_partial.json" if use_partial else "splits.json",
            "splits_partial.json",
            "splits.json",
        ])

        # parquet files
        structured_path = _pick_existing(root, [
            "structured_partial.parquet" if use_partial else "structured.parquet",
            "structured_partial.parquet",
            "structured.parquet",
        ])

        notes_path = _pick_existing(root, [
            "notes_partial.parquet" if use_partial else "notes.parquet",
            "notes_partial.parquet",
            "notes.parquet",
        ])

        # images: for TRAIN you usually want augmented; for VAL/TEST usually non-augmented
        if split.lower() == "train":
            images_path = _pick_existing(root, [
                "images_partial_augmented.parquet" if use_partial else "images.parquet",
                "images_partial_augmented.parquet",
                "images_partial.parquet",
                "images.parquet",
            ])
        else:
            images_path = _pick_existing(root, [
                "images_partial.parquet" if use_partial else "images.parquet",
                "images_partial.parquet",
                "images.parquet",
            ])

        labels_path = _pick_existing(root, [
            "labels_pheno_partial.parquet" if use_partial else "labels_pheno.parquet",
            "labels_pheno_partial.parquet",
            "labels_pheno.parquet",
        ])

        print(f"[dataset:{split}] use_partial={use_partial}")
        print(f"[dataset:{split}] splits   = {os.path.basename(splits_path)}")
        print(f"[dataset:{split}] struct   = {os.path.basename(structured_path)}")
        print(f"[dataset:{split}] notes    = {os.path.basename(notes_path)}")
        print(f"[dataset:{split}] images   = {os.path.basename(images_path)}")
        print(f"[dataset:{split}] labels   = {os.path.basename(labels_path)}")

        # ----------------------------
        # Read splits (robust to bad ids like "nan_...")
        # ----------------------------
        with open(splits_path) as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(
                f"[ICUStayDataset] split '{split}' not in {os.path.basename(splits_path)} keys: {list(splits.keys())}"
            )

        split_ids = []
        bad_split_ids = []
        for x in splits[split]:
            try:
                # handles int, "123", 123.0, "123.0"
                xi = int(float(x))
                split_ids.append(xi)
            except Exception:
                bad_split_ids.append(str(x))

        if bad_split_ids:
            print(f"[dataset:{split}] WARNING: dropped {len(bad_split_ids)} non-numeric split ids (e.g. {bad_split_ids[:3]})")

        ids_set = set(split_ids)


        # ----------------------------
        # Load tables
        # ----------------------------
        self.struct = pd.read_parquet(structured_path)
        self.notes  = pd.read_parquet(notes_path)
        self.images = pd.read_parquet(images_path)
        self.labels = pd.read_parquet(labels_path)

        # ----------------------------
        # Standardize ID column name to `stay_id` everywhere
        # (partial data often uses sample_id)
        # ----------------------------
        self.struct = _standardize_id_column(self.struct, want="stay_id")
        self.notes  = _standardize_id_column(self.notes,  want="stay_id")
        self.images = _standardize_id_column(self.images, want="stay_id")
        self.labels = _standardize_id_column(self.labels, want="stay_id")

        # normalize image path column to cxr_path
        self.images = _standardize_image_path_column(self.images)

        # ---- everything below here can remain exactly as you already have it ----
        # structured feature columns
        base_cols = {"stay_id", "hour"}
        self.feat_cols: List[str] = [c for c in self.struct.columns if c not in base_cols]
        self.feat_cols.sort()
        if hasattr(CFG, "structured_n_feats"):
            assert len(self.feat_cols) == CFG.structured_n_feats, \
                f"CFG.structured_n_feats={CFG.structured_n_feats}, " \
                f"found {len(self.feat_cols)} in {structured_path}"

        # notes schema
        self.notes_mode, self.note_a_cols, self.note_b_cols = _detect_notes_schema(self.notes)
        if self.notes_mode == "text":
            self.chunk_cols = self.note_a_cols
            print(f"[dataset:{split}] notes_mode=text (chunks={len(self.chunk_cols)})")
        else:
            self.input_id_cols = self.note_a_cols
            self.attn_mask_cols = self.note_b_cols
            print(f"[dataset:{split}] notes_mode=pretokenized (chunks={len(self.input_id_cols)})")

        # phenotype label columns
        self.label_cols: List[str] = [c for c in self.labels.columns if c != "stay_id"]
        self.label_cols.sort()
        if len(self.label_cols) == 0:
            raise ValueError("[ICUStayDataset] labels parquet must contain at least one phenotype column.")
        self.num_labels = len(self.label_cols)
        print(f"[dataset:{split}] found {len(self.label_cols)} phenotype labels: {self.label_cols[:5]}{' ...' if len(self.label_cols) > 5 else ''}")

        # tri-modal filtering (same as your existing logic)
        struct_ids = set(self.struct["stay_id"].astype(int).unique().tolist())
        label_ids  = set(self.labels["stay_id"].astype(int).unique().tolist())

        if self.notes_mode == "text":
            nonempty = np.zeros(len(self.notes), dtype=bool)
            for c in self.chunk_cols:
                if c in self.notes.columns:
                    nonempty |= self.notes[c].fillna("").astype(str).str.strip().ne("")
            note_ids = set(self.notes.loc[nonempty, "stay_id"].astype(int).unique().tolist())
        else:
            nonempty = np.zeros(len(self.notes), dtype=bool)
            for c_id, c_m in zip(self.input_id_cols, self.attn_mask_cols):
                ids_len = self.notes[c_id].apply(lambda x: len(_cell_to_list(x)))
                msk_len = self.notes[c_m].apply(lambda x: len(_cell_to_list(x)))
                nonempty |= (ids_len > 0) & (msk_len > 0)
            note_ids = set(self.notes.loc[nonempty, "stay_id"].astype(int).unique().tolist())

        img_ids = set(
            self.images.loc[
                self.images["cxr_path"].fillna("").astype(str).str.strip().ne(""),
                "stay_id"
            ].astype(int).unique().tolist()
        )

        base = ids_set & label_ids

        has_L = base & struct_ids
        has_N = base & note_ids
        has_I = base & img_ids

        # counts per id (set-based)
        keep_2of3 = (has_L & has_N) | (has_L & has_I) | (has_N & has_I)

        self.ids = sorted(list(keep_2of3))
        print(f"[dataset:{split}] >=2 modalities -> kept {len(self.ids)} / {len(ids_set)}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = self.ids[idx]

        # structured EHR
        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = (
            df_s[self.feat_cols]
            .astype("float32")
            .fillna(0.0)
            .to_numpy()
        )
        xs = torch.from_numpy(xs_np)  # [<=T,F]

        # notes: use the first row for this stay_id (you can change policy later if needed)
        df_n = self.notes[self.notes.stay_id == stay_id]
        notes_payload = None

        if df_n.empty:
            notes_payload = {"mode": "text", "chunks": []}  # missing notes
        else:
            row = df_n.iloc[0]
            if self.notes_mode == "text":
                notes_list = []
                for c in self.chunk_cols:
                    val = row.get(c, "")
                    if pd.notna(val) and str(val).strip():
                        notes_list.append(str(val))
                notes_payload = {"mode": "text", "chunks": notes_list}
            else:
                ids_chunks, attn_chunks = [], []
                for c_id, c_m in zip(self.input_id_cols, self.attn_mask_cols):
                    ids = _cell_to_list(row.get(c_id, None))
                    msk = _cell_to_list(row.get(c_m, None))
                    if len(ids) > 0 and len(msk) > 0:
                        ids_chunks.append(ids)
                        attn_chunks.append(msk)
                notes_payload = {"mode": "pretokenized", "input_ids": ids_chunks, "attention_mask": attn_chunks}


        # images: use last path
        df_i = self.images[self.images.stay_id == stay_id]
        img_paths = []
        if not df_i.empty:
            img_paths = df_i.cxr_path.dropna().astype(str).tolist()
            img_paths = [p for p in img_paths if str(p).strip()]
        img_paths = img_paths[-1:]  # last only (or empty)


        # multi-label phenotype target [K]
        lab_row = self.labels[self.labels.stay_id == stay_id]
        if lab_row.empty:
            raise RuntimeError(f"[ICUStayDataset] Missing labels for stay_id={stay_id}")
        y_vec = lab_row[self.label_cols].iloc[0].to_numpy()
        y = torch.tensor(y_vec, dtype=torch.float32)  # [K]

        return {
            "stay_id": stay_id,
            "x_struct": xs,
            "notes": notes_payload,
            "image_paths": img_paths,
            "y": y,
        }


def pad_or_trim_struct(x: torch.Tensor, T: int, F: int) -> torch.Tensor:
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

    if not os.path.isabs(p):
        p_full = os.path.join(CFG.data_root, p)
    else:
        p_full = p

    try:
        with Image.open(p_full) as img:
            tensor = tfms(img)
    except Exception as e:
        print(f"[warn] failed to open image: {p_full} ({e}) -> returning zero tensor")
        tensor = torch.zeros(3, 224, 224)

    return (tensor, p_full) if return_path else tensor


def build_route_mask(has_L, has_N, has_I):
    # each is [B] float 0/1
    L  = has_L
    N  = has_N
    I  = has_I
    LN = has_L * has_N
    LI = has_L * has_I
    NI = has_N * has_I
    LNI= has_L * has_N * has_I
    return torch.stack([L, N, I, LN, LI, NI, LNI], dim=1)  # [B,7]


def collate_fn_factory(tidx: int, img_tfms: T.Compose):
    first_print = {"done": False}

    def _collate(batch: List[Dict[str, Any]]):
        T_len, F_dim = CFG.structured_seq_len, CFG.structured_n_feats
        xL_batch = torch.stack(
            [pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0
        )
        mL_batch = (xL_batch.abs().sum(dim=2) > 0).float()

        notes_batch = [b["notes"] for b in batch]
        has_N = torch.tensor(
            [ (nb.get("mode")=="text" and len(nb.get("chunks",[]))>0) or
              (nb.get("mode")!="text" and len(nb.get("input_ids",[]))>0)
              for nb in notes_batch ],
            dtype=torch.float32, device=DEVICE
        )

        imgs_list, img_paths_list = [], []
        has_I_list = []
        for b in batch:
            paths = b["image_paths"]
            has_I_list.append(1.0 if (paths and str(paths[-1]).strip()) else 0.0)
            img_t, path = load_cxr_tensor(paths, img_tfms, return_path=True)
            imgs_list.append(img_t)
            img_paths_list.append(path)

        imgs_batch = torch.stack(imgs_list, dim=0)
        has_I = torch.tensor(has_I_list, dtype=torch.float32, device=DEVICE)

        # structured always present in your current pipeline
        has_L = torch.ones(len(batch), dtype=torch.float32, device=DEVICE)

        y_batch = torch.stack([b["y"].float().view(-1) for b in batch], dim=0)

        dbg = {
            "stay_ids": [b["stay_id"] for b in batch],
            "img_paths": img_paths_list,
            "has_L": has_L.detach().cpu().tolist(),
            "has_N": has_N.detach().cpu().tolist(),
            "has_I": has_I.detach().cpu().tolist(),
        }

        if not first_print["done"]:
            first_print["done"] = True
            print(
                f"[collate] xL_batch: {tuple(xL_batch.shape)} "
                f"| mL_batch: {tuple(mL_batch.shape)} "
                f"| notes_batch: len={len(notes_batch)} "
                f"| imgs_batch: {tuple(imgs_batch.shape)} "
                f"| y_batch: {tuple(y_batch.shape)}"
            )
            print(
                f"[collate] modality availability (first batch): "
                f"L={has_L.mean().item():.3f}, N={has_N.mean().item():.3f}, I={has_I.mean().item():.3f}"
            )

        # ✅ IMPORTANT: return availability so training/eval can build route_mask
        return xL_batch, mL_batch, notes_batch, imgs_batch, y_batch, dbg, has_L, has_N, has_I

    return _collate



@torch.no_grad()
def pretty_print_small_batch(xL, mL, notes, dbg, k: int = 3) -> None:
    """
    notes is expected to be the collated notes_batch:
      - List[{"mode":"text","chunks":[...]}]  OR
      - List[{"mode":"pretokenized","input_ids":[[...],...],"attention_mask":[[...],...]}]
    """
    B, T, F = xL.shape
    k = min(k, B)

    print("\n[sample-inspect] ---- Top few samples ----")
    for i in range(k):
        sid = dbg.get("stay_ids", ["<id?>"] * B)[i]
        imgp = dbg.get("img_paths", ["<path?>"] * B)[i]

        # show a couple non-zero EHR rows (first 5 feats)
        nz_rows = (mL[i] > 0.5).nonzero(as_tuple=False).flatten().tolist()
        show_rows = nz_rows[:2] if nz_rows else []
        ehr_rows = []
        for r in show_rows:
            vec = xL[i, r].detach().cpu().numpy()
            ehr_rows.append(np.round(vec[:min(5, F)], 3).tolist())

        # notes summary (robust to new schema)
        note_obj = notes[i]

        note_preview = "<no-notes>"
        try:
            mode = note_obj.get("mode", "text") if isinstance(note_obj, dict) else "unknown"

            if mode == "text":
                chunks = note_obj.get("chunks", [])
                if chunks:
                    s = str(chunks[0])
                    note_preview = (s[:120] + "…") if len(s) > 120 else s
                else:
                    note_preview = "<empty-text-chunks>"

            elif mode == "pretokenized":
                ids_chunks = note_obj.get("input_ids", [])
                n_chunks = len(ids_chunks)
                if n_chunks > 0:
                    first = ids_chunks[0]
                    # first might already be list[int] here (CPU python list), but be defensive
                    first_list = list(first) if hasattr(first, "__iter__") else []
                    note_preview = (
                        f"<pretokenized: chunks={n_chunks}, len0={len(first_list)}, "
                        f"ids0[:10]={first_list[:10]}>"
                    )
                else:
                    note_preview = "<empty-pretokenized-chunks>"

            else:
                note_preview = f"<unknown notes mode: {mode}>"

        except Exception as e:
            note_preview = f"<notes preview failed: {type(e).__name__}: {e}>"

        print(
            f"  • stay_id={sid} | ehr_rows(first2->first5feats)={ehr_rows} | "
            f'notes_preview="{note_preview}" | cxr="{imgp}"'
        )

    print("[sample-inspect] ---------------------------\n")



def _capsule_forward_safe(z, fusion, projector, cap_head,
                          route_mask=None, act_temperature=1.0,
                          detach_priors=False, return_routing=True):
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


def _clamp_norm(x: torch.Tensor, max_norm: float = 20.0) -> torch.Tensor:
    if x.ndim < 2:
        return x
    n = x.norm(dim=1, keepdim=True) + 1e-6
    scale = torch.clamp(max_norm / n, max=1.0)
    return x * scale


def _safe_tensor(x: torch.Tensor, name: str = "") -> torch.Tensor:
    if not torch.isfinite(x).all():
        n_nan = (~torch.isfinite(x)).sum().item()
        print(f"[NaN/Inf GUARD] {name}: found {n_nan} non-finite entries, clamping.")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    return x

@torch.no_grad()
def evaluate_epoch(
    behrt, bbert, imgenc, fusion, projector, cap_head,
    loader, amp_ctx, loss_fn,
    route_debug: bool = False,
    label_names: Optional[List[str]] = None,
    epoch_idx: Optional[int] = None,
    split_name: str = "VAL",
    routing_out_dir: Optional[str] = None,
):
    behrt.eval()
    imgenc.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()

    total_loss, total_correct, total = 0.0, 0, 0
    act_sum = torch.zeros(7, dtype=torch.float32)
    num_samples = 0
    printed_unimodal = False
    printed_caps_once = False
    rpt_every = int(_cfg("routing_print_every", 0) or 0)

    # per-route, per-phenotype routing importance
    rc_sum_mat = None      # will become [7, K]
    has_routing = False


    for bidx, (xL, mL, notes, imgs, y, dbg, has_L, has_N, has_I) in enumerate(loader):
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)

        with amp_ctx:
            zL = behrt(xL, mask=mL)
            zN = bbert(prepare_notes_batch(notes))
            zI = imgenc(imgs)

            zL = _safe_tensor(zL, "eval.zL")
            zN = _safe_tensor(zN, "eval.zN")
            zI = _safe_tensor(zI, "eval.zI")

            z = {"L": zL, "N": zN, "I": zI}
            route_mask = build_route_mask(has_L, has_N, has_I)


            out = _capsule_forward_safe(
                z, fusion, projector, cap_head,
                route_mask=route_mask, act_temperature=1.0,
                detach_priors=False, return_routing=True
            )
            logits, prim_acts, route_embs = out[0], out[1], out[2]
            routing_coef = out[3] if len(out) > 3 else None
            if routing_coef is not None:
                has_routing = True
                rc = routing_coef.detach().float().cpu()   # [B, 7, K]
                rc_mean_batch = rc.mean(dim=0)             # [7, K]

                if rc_sum_mat is None:
                    rc_sum_mat = torch.zeros_like(rc_mean_batch)  # [7, K]

                rc_sum_mat += rc_mean_batch * y.size(0)           # weight by batch size

            # routing debugging / heatmaps for first batch
            if route_debug and routing_coef is not None and bidx == 0:
                names = label_names if label_names is not None else \
                    [get_pheno_name(i) for i in range(routing_coef.size(2))]
                print_route_matrix_detailed(
                    routing_coef, prim_acts, names,
                    where=f"{split_name} Batch {bidx}"
                )
                print_phenotype_routing_heatmap(
                    routing_coef, prim_acts, names,
                    where=f"{split_name} Epoch {epoch_idx if epoch_idx is not None else '?'}",
                    top_k=None
                )
                if routing_out_dir is not None and epoch_idx is not None:
                    where_tag = f"{split_name.lower()}_epoch{epoch_idx:03d}"
                    save_routing_heatmap(
                        routing_coef, prim_acts, names,
                        where=where_tag,
                        out_dir=routing_out_dir
                    )

            if not printed_unimodal:
                printed_unimodal = True
                print(
                    f"[eval:unimodal] zL:{tuple(zL.shape)} "
                    f"zN:{tuple(zN.shape)} zI:{tuple(zI.shape)}"
                )
                pretty_print_small_batch(xL, mL, notes, dbg, k=3)

            if (not printed_caps_once) or (rpt_every > 0 and ((bidx + 1) % rpt_every == 0)):
                printed_caps_once = True
                keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
                print(
                    f"[eval:caps] logits:{tuple(logits.shape)} "
                    f"prim_acts:{tuple(prim_acts.shape)} routes -> {keys}"
                )

            loss = loss_fn(logits, y)

        total_loss += loss.item() * y.size(0)
        probs = torch.sigmoid(logits)
        pred  = (probs >= 0.5).float()
        total_correct += (pred == y.float()).sum().item()
        total += y.numel()
        num_samples += y.size(0)
        act_sum += prim_acts.detach().float().cpu().sum(dim=0)

    avg_loss = total_loss / max(1, num_samples)
    avg_acc  = total_correct / max(1, total)
    avg_act  = (act_sum / max(1, num_samples)).tolist()
    route_names = ["L", "N", "I", "LN", "LI", "NI", "LNI"]
    avg_act_dict = {r: avg_act[i] for i, r in enumerate(route_names)}

    if num_samples > 0 and has_routing and rc_sum_mat is not None:
        avg_rc_mat = (rc_sum_mat / num_samples).numpy()   # [7, K]
    else:
        avg_rc_mat = None

    return avg_loss, avg_acc, avg_act_dict, avg_rc_mat

def save_checkpoint(path: str, state: Dict):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_checkpoint(path: str, behrt, bbert, imgenc, fusion, projector, cap_head, optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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
    behrt.eval()
    imgenc.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()

    y_true, p1, ids = [], [], []
    for xL, mL, notes, imgs, y, dbg, has_L, has_N, has_I in loader:
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)

        with amp_ctx:
            zL = behrt(xL, mask=mL)
            zN = bbert(prepare_notes_batch(notes))
            zI = imgenc(imgs)

            zL = _safe_tensor(zL, "collect.zL")
            zN = _safe_tensor(zN, "collect.zN")
            zI = _safe_tensor(zI, "collect.zI")

            route_mask = build_route_mask(has_L, has_N, has_I)

            out = _capsule_forward_safe(
                {"L": zL, "N": zN, "I": zI}, fusion, projector, cap_head,
                route_mask=route_mask,
                act_temperature=1.0, detach_priors=False, return_routing=True
            )
            logits = out[0]

        probs = torch.sigmoid(logits)
        y_true.append(y.detach().cpu())
        p1.append(probs.detach().cpu())
        ids += dbg.get("stay_ids", [])

    y_true = torch.cat(y_true, dim=0).numpy()
    p1     = torch.cat(p1, dim=0).numpy()
    return y_true, p1, ids



def epoch_metrics(y_true, p, y_pred):
    """
    y_true, p, y_pred: numpy arrays of shape [N, K]
      - y_true: {0,1}
      - p: probabilities in [0,1]
      - y_pred: {0,1} after thresholding (e.g., 0.5)
    """
    import numpy as np
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        recall_score,
        confusion_matrix,
    )

    y_true = np.asarray(y_true)
    p      = np.asarray(p)
    y_pred = np.asarray(y_pred)

    N, K = y_true.shape

    aurocs, auprcs, f1s, recs = [], [], [], []
    auroc_per_label = np.full(K, np.nan, dtype=float)
    auprc_per_label = np.full(K, np.nan, dtype=float)
    f1_per_label    = np.full(K, np.nan, dtype=float)
    rec_per_label   = np.full(K, np.nan, dtype=float)

    for k in range(K):
        yk  = y_true[:, k]
        pk  = p[:, k]
        ypk = y_pred[:, k]

        if len(np.unique(yk)) < 2:
            continue

        try:
            au = roc_auc_score(yk, pk)
            aurocs.append(au)
            auroc_per_label[k] = au
        except Exception:
            pass

        try:
            ap = average_precision_score(yk, pk)
            auprcs.append(ap)
            auprc_per_label[k] = ap
        except Exception:
            pass

        try:
            f1k = f1_score(yk, ypk)
            f1s.append(f1k)
            f1_per_label[k] = f1k
        except Exception:
            pass

        try:
            rk = recall_score(yk, ypk)
            recs.append(rk)
            rec_per_label[k] = rk
        except Exception:
            pass

    out = {}

    # Macro metrics
    out["AUROC_macro"]  = float(np.nanmean(aurocs)) if len(aurocs) > 0 else float("nan")
    out["AUPRC_macro"]  = float(np.nanmean(auprcs)) if len(auprcs) > 0 else float("nan")
    out["F1_macro"]     = float(np.nanmean(f1s))    if len(f1s) > 0 else float("nan")
    out["Recall_macro"] = float(np.nanmean(recs))   if len(recs) > 0 else float("nan")

    out["AUROC"]  = out["AUROC_macro"]
    out["AUPRC"]  = out["AUPRC_macro"]
    out["F1"]     = out["F1_macro"]
    out["Recall"] = out["Recall_macro"]

    # Micro metrics
    y_flat  = y_true.reshape(-1)
    p_flat  = p.reshape(-1)
    yp_flat = y_pred.reshape(-1)

    try:
        out["AUROC_micro"] = float(roc_auc_score(y_flat, p_flat))
    except Exception:
        out["AUROC_micro"] = float("nan")
    try:
        out["AUPRC_micro"] = float(average_precision_score(y_flat, p_flat))
    except Exception:
        out["AUPRC_micro"] = float("nan")

    tp = np.logical_and(y_flat == 1, yp_flat == 1).sum()
    fp = np.logical_and(y_flat == 0, yp_flat == 1).sum()
    fn = np.logical_and(y_flat == 1, yp_flat == 0).sum()

    micro_prec = float(tp) / float(tp + fp + 1e-8)
    micro_rec  = float(tp) / float(tp + fn + 1e-8)
    micro_f1   = (
        2.0 * micro_prec * micro_rec / (micro_prec + micro_rec + 1e-8)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    out["Precision_micro"] = micro_prec
    out["Recall_micro"]    = micro_rec
    out["F1_micro"]        = micro_f1

    # Example-based F1
    example_f1s = []
    for i in range(N):
        true_i = (y_true[i] == 1)
        pred_i = (y_pred[i] == 1)

        if true_i.sum() == 0 and pred_i.sum() == 0:
            example_f1s.append(1.0)
            continue

        inter = np.logical_and(true_i, pred_i).sum()
        denom = true_i.sum() + pred_i.sum()
        if denom == 0:
            example_f1s.append(0.0)
        else:
            example_f1s.append(2.0 * inter / float(denom))

    out["F1_example"] = float(np.mean(example_f1s)) if len(example_f1s) > 0 else float("nan")

    out["Hamming"] = float(np.mean(y_flat != yp_flat))
    out["CM"] = confusion_matrix(y_flat, yp_flat)

    out["AUROC_per_label"]  = auroc_per_label
    out["AUPRC_per_label"]  = auprc_per_label
    out["F1_per_label"]     = f1_per_label
    out["Recall_per_label"] = rec_per_label
    return out


def expected_calibration_error(p, y, n_bins=10):
    p = np.asarray(p)
    y = np.asarray(y).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids = 0.5 * (bins[1:] + bins[:-1])
    ece = 0.0
    bconf, bacc, bcnt = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            bconf.append(0.0)
            bacc.append(0.0)
            bcnt.append(0)
            continue
        conf = float(p[m].mean())
        acc = float((y[m] == 1).mean())
        w = m.mean()
        ece += w * abs(acc - conf)
        bconf.append(conf)
        bacc.append(acc)
        bcnt.append(int(m.sum()))
    return float(ece), mids, np.array(bconf), np.array(bacc), np.array(bcnt)


def reliability_plot(bin_centers, bin_conf, bin_acc, out_path):
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(bin_conf, bin_acc, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _set_all_seeds():
    seed = int(_cfg("seed", 42))
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(CFG.deterministic)
    torch.backends.cudnn.benchmark = bool(CFG.use_cudnn_benchmark and not CFG.deterministic)


def find_best_thresholds(y_true, p, n_steps: int = 50):
    import numpy as np
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    p      = np.asarray(p)

    N, K = p.shape
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t = np.full(K, 0.5, dtype=float)

    for k in range(K):
        yk = y_true[:, k]
        pk = p[:, k]

        if len(np.unique(yk)) < 2:
            continue

        best_f1 = 0.0
        best_thr = 0.5
        for t in thresholds:
            yhat = (pk >= t).astype(int)
            f1 = f1_score(yk, yhat)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = t
        best_t[k] = best_thr

    return best_t


def compute_split_prevalence(
    y_true: np.ndarray,
    split_name: str = "VAL",
    label_names: Optional[List[str]] = None,
):
    y_true = np.asarray(y_true)
    prev = y_true.mean(axis=0)  # [K]

    print(f"\n[{split_name}] label prevalence:")
    for i, p in enumerate(prev):
        name = label_names[i] if label_names is not None else get_pheno_name(i)
        print(f"  {name}: {p:.4f}")
    return prev


def grid_search_thresholds(
    y_true: np.ndarray,
    p: np.ndarray,
    n_steps: int = 101,
):
    """
    Per-label threshold search that maximizes per-label F1.
    Returns:
      thresholds: [K]
      best_f1:    [K]
    """
    y_true = np.asarray(y_true)
    p      = np.asarray(p)
    N, K = p.shape

    thresholds = np.linspace(0.0, 1.0, n_steps)
    best_thr = np.full(K, 0.5, dtype=float)
    best_f1 = np.zeros(K, dtype=float)

    from sklearn.metrics import f1_score

    for k in range(K):
        yk = y_true[:, k]
        pk = p[:, k]

        if len(np.unique(yk)) < 2:
            continue

        best = 0.0
        best_t = 0.5
        for t in thresholds:
            yhat = (pk >= t).astype(int)
            f1 = f1_score(yk, yhat)
            if f1 > best:
                best = f1
                best_t = t

        best_thr[k] = best_t
        best_f1[k] = best

    return best_thr, best_f1


def save_split_thresholds(
    thr: np.ndarray,
    ckpt_dir: str,
    split_name: str = "VAL",
):
    thr = np.asarray(thr, dtype=float).reshape(-1)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"thresholds_{split_name.lower()}.npy")
    np.save(path, thr)
    print(f"[{split_name}] Saved thresholds → {path}")
    return path

def main():
    args = parse_args()

    apply_cli_overrides(args)

    if hasattr(args, "finetune_text") and args.finetune_text:
        CFG.finetune_text = True

    print("[env_config] Device:", DEVICE)
    print("[env_config] CFG:", json.dumps(asdict(CFG), indent=2))

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

    # Datasets
    train_ds = ICUStayDataset(args.data_root, split="train", use_partial=True)
    tri_ids = set(train_ds.ids)

    # Class-balancing pos_weight from TRAIN
    train_label_df = train_ds.labels[
        train_ds.labels["stay_id"].isin(tri_ids)
    ][train_ds.label_cols].astype(float)
    N_train = train_label_df.shape[0]

    compute_split_prevalence(
        train_label_df.values,
        split_name="TRAIN",
        label_names=[get_pheno_name(i) for i in range(train_ds.num_labels)]
    )
    pos_counts = train_label_df.sum(axis=0).values
    neg_counts = N_train - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

    val_ds   = ICUStayDataset(args.data_root, split="val",   use_partial=True)
    test_ds  = ICUStayDataset(args.data_root, split="test",  use_partial=True)

    num_phenos = train_ds.num_labels

    raw_label_cols = train_ds.label_cols
    label_names = [get_pheno_name(i) for i in range(num_phenos)]

    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_tensor)
    print("[loss] BCEWithLogitsLoss with per-label pos_weight")

    collate_train = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("train"))
    collate_eval  = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("val"))
    pin = use_cuda

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_train, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_eval
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_eval
    )

    # Encoders
    enc_cfg = EncoderConfig(
        d=_cfg("d", 256), dropout=_cfg("dropout", 0.0),
        structured_seq_len=_cfg("structured_seq_len", 24),
        structured_n_feats=_cfg("structured_n_feats", 17),
        structured_layers=_cfg("structured_layers", 2),
        structured_heads=_cfg("structured_heads", 8),
        structured_pool="cls",
        text_model_name=_cfg("text_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        text_max_len=_cfg("max_text_len", 512),
        note_agg="attention",
        img_agg="last",
        vision_backbone=_cfg("image_model_name", "resnet34"),
        vision_num_classes=14,
        vision_pretrained=True,
    )
    behrt, bbert, imgenc = build_encoders(enc_cfg, device=DEVICE)
    print(
        f"[encoders] d={CFG.d} | BEHRT out_dim={behrt.out_dim} | "
        f"BERT→out_dim={bbert.out_dim} | IMG out_dim={getattr(imgenc.proj, 'out_features', 'NA')}"
    )

    if not CFG.finetune_text and getattr(bbert, "bert", None) is not None:
        for p in bbert.bert.parameters():
            p.requires_grad = False
        bbert.bert.eval()
        print("[encoders] Bio_ClinicalBERT frozen (feature extractor mode)")

    # Fusions + capsule head
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
        num_classes=num_phenos,  # <<< K=25 phenotypes
    ).to(DEVICE)
    print(
        f"[capsule] pc_dim={CFG.capsule_pc_dim} mc_caps_dim={CFG.capsule_mc_caps_dim} "
        f"iters={CFG.capsule_num_routing} act_type={CFG.capsule_act_type} "
        f"out_caps={num_phenos}"
    )

    # Optimizer
    params = list(behrt.parameters()) + list(bbert.parameters()) + list(imgenc.parameters())
    for k in fusion.keys():
        params += list(fusion[k].parameters())
    params += list(projector.parameters()) + list(cap_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Checkpoint setup
    start_epoch = 0
    best_val_acc = -1.0
    ckpt_dir = os.path.join(args.ckpt_root, "pheno_capsule")
    ensure_dir(ckpt_dir)
    if args.resume and os.path.isfile(args.resume):
        print(f"[main] Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, behrt, bbert, imgenc, fusion, projector, cap_head, optimizer)

    printed_once = False

    # Routing regularization hyperparams
    route_dropout_p = float(_cfg("route_dropout_p", 0.0))
    routing_warmup_epochs = int(_cfg("routing_warmup_epochs", 1))

    route_entropy_lambda = float(_cfg("route_entropy_lambda", 1e-3))
    route_entropy_warm   = int(_cfg("route_entropy_warmup_epochs", 10))
    entropy_use_rc       = bool(_cfg("entropy_use_rc", False))  # kept for compatibility

    route_uniform_lambda = float(_cfg("route_uniform_lambda", 1e-3))
    route_uniform_warm   = int(_cfg("route_uniform_warmup_epochs", 5))

    max_train_patients = int(os.environ.get("MIMICIV_MAX_TRAIN_PATIENTS", "-1"))
    seen_patients = 0
    stop_training = False

    # Epoch-level early stopping based on VAL macro AUROC
    best_val_auroc = -float("inf")  
    epochs_no_improve = 0
    patience_epochs = 5    
    min_delta = 1e-4     
    min_epochs = 20  # <-- do not early-stop before this epoch
  

    # Training loop
    for epoch in range(start_epoch, args.epochs):

        if stop_training:
            print(f"[debug] Early stop flag set → breaking before epoch {epoch + 1}.")
            break

        behrt.train()
        imgenc.train()
        if CFG.finetune_text and getattr(bbert, "bert", None) is not None:
            bbert.bert.train()

        total_loss, total_correct, total = 0.0, 0, 0
        act_sum = torch.zeros(7, dtype=torch.float32)
        num_samples = 0

        for step, (xL, mL, notes, imgs, y, dbg, has_L, has_N, has_I) in enumerate(train_loader):

            if max_train_patients > 0 and seen_patients >= max_train_patients:
                print(
                    f"[debug] Reached MAX_TRAIN_PATIENTS={max_train_patients}, "
                    f"stopping at epoch {epoch + 1}, step {step}."
                )
                stop_training = True
                break
            batch_size = y.size(0)
            seen_patients += batch_size

            xL, mL = xL.to(DEVICE), mL.to(DEVICE)
            imgs   = imgs.to(DEVICE)
            y      = y.to(DEVICE)

            if (epoch == start_epoch) and (step == 0):
                pretty_print_small_batch(xL, mL, notes, dbg, k=3)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                zL = behrt(xL, mask=mL)
                notes_tok = prepare_notes_batch(notes)
                zN = bbert(notes_tok)
                zI = imgenc(imgs)

                zL = _clamp_norm(zL, max_norm=20.0)
                zN = _clamp_norm(zN, max_norm=20.0)
                zI = _clamp_norm(zI, max_norm=20.0)

                zL = _safe_tensor(zL, "zL(behrt)")
                zN = _safe_tensor(zN, "zN(bbert)")
                zI = _safe_tensor(zI, "zI(imgenc)")

                z = {"L": zL, "N": zN, "I": zI}

                if not printed_once:
                    printed_once = True
                    print(
                        f"[sanity] xL: {tuple(xL.shape)} | mL: {tuple(mL.shape)} "
                        f"| imgs: {tuple(imgs.shape)} | y: {tuple(y.shape)}"
                    )
                    print(
                        f"[sanity] zL: {tuple(zL.shape)} | zN: {tuple(zN.shape)} | zI: {tuple(zI.shape)}"
                    )
                    with torch.no_grad():
                        for i in range(min(3, zL.size(0))):
                            print(
                                f"[emb-norms] i={i} "
                                f"||zL||={zL[i].norm().item():.3f} "
                                f"||zN||={zN[i].norm().item():.3f} "
                                f"||zI||={zI[i].norm().item():.3f}"
                            )

                # has_L/has_N/has_I already on DEVICE from collate
                route_mask = build_route_mask(has_L, has_N, has_I)


                # Route dropout for regularization
                if route_dropout_p > 0.0:
                    if torch.rand(()) < route_dropout_p:
                        drop_idx = int(torch.randint(low=0, high=7, size=(1,)))
                        route_mask[:, drop_idx] = 0.0

                    # second drop with smaller prob in early epochs
                    if (epoch - start_epoch) < 2 and (torch.rand(()) < route_dropout_p * 0.5):
                        drop_idx2 = int(torch.randint(low=0, high=7, size=(1,)))
                        route_mask[:, drop_idx2] = 0.0

                detach_priors_flag = (epoch - start_epoch) < routing_warmup_epochs
                temp = 2.0 if epoch < 2 else 1.0

                out = _capsule_forward_safe(
                    {"L": zL, "N": zN, "I": zI},
                    fusion, projector, cap_head,
                    route_mask=route_mask, act_temperature=temp,
                    detach_priors=detach_priors_flag, return_routing=True
                )

                logits, prim_acts, route_embs = out[0], out[1], out[2]
                routing_coef = out[3] if len(out) > 3 else None

                # Routing debug on first step of each epoch
                if getattr(args, "route_debug", False) and routing_coef is not None and step == 0:
                    where_train = f"train_epoch{epoch + 1:03d}"
                    print_route_matrix_detailed(
                        routing_coef, prim_acts, label_names,
                        where=f"TRAIN Epoch {epoch + 1} Step {step}"
                    )
                    print_phenotype_routing_heatmap(
                        routing_coef, prim_acts, label_names,
                        where=f"TRAIN Epoch {epoch + 1}",
                        top_k=None
                    )
                    save_routing_heatmap(
                        routing_coef, prim_acts, label_names,
                        where=where_train,
                        out_dir=os.path.join(ckpt_dir, "routing")
                    )

                if printed_once and step == 0:
                    keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
                    print(
                        f"[sanity] routes -> {keys} | logits: {tuple(logits.shape)} "
                        f"| prim_acts: {tuple(prim_acts.shape)}"
                    )

                # Multi-label BCE loss
                loss = bce(logits, y.float())

                # Entropy bonus 
                if route_entropy_lambda > 0.0 and ((epoch - start_epoch) < route_entropy_warm):
                    pa = prim_acts
                    pa = pa / (pa.sum(dim=1, keepdim=True) + 1e-6)
                    pa = torch.clamp(pa, 1e-6, 1.0)
                    H = -(pa * pa.log()).sum(dim=1).mean()
                    loss = loss - route_entropy_lambda * H

                # Uniform prior bonus across routes
                if route_uniform_lambda > 0.0 and ((epoch - start_epoch) < route_uniform_warm):
                    pa_dist = prim_acts / (prim_acts.sum(dim=1, keepdim=True) + 1e-6)
                    p_mean = pa_dist.mean(dim=0)
                    target = torch.full_like(p_mean, 1.0 / p_mean.numel())
                    uniform_loss = ((p_mean - target) ** 2).sum()
                    loss = loss + route_uniform_lambda * uniform_loss

            # Backprop
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

            # Log running stats
            total_loss += loss.item() * y.size(0)

            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            total_correct += (pred == y.float()).sum().item()
            total += y.numel()

            act_sum += prim_acts.detach().cpu().sum(dim=0)
            num_samples += y.size(0)


            if args.log_every > 0 and ((step + 1) % args.log_every == 0):
                avg_loss_step = total_loss / max(1, total // num_phenos)
                avg_acc_step  = total_correct / max(1, total)
                avg_act = (act_sum / max(1, num_samples)).tolist()

                routes = ["L", "N", "I", "LN", "LI", "NI", "LNI"]
                msg = (
                    f"[epoch {epoch + 1} step {step + 1}] "
                    f"loss={avg_loss_step:.4f} acc={avg_acc_step:.4f} "
                    f"avg_prim_act(L,N,I,LN,LI,NI,LNI)="
                    f"{', '.join(f'{a:.3f}' for a in avg_act)}"
                )

                if routing_coef is not None:
                    rc = routing_coef.detach().float().cpu()
                    rc_route_mean = rc.mean(dim=(0, 2))
                    rc_str = " | ".join(
                        f"{r}:{rc_route_mean[i]:.3f}" for i, r in enumerate(routes)
                    )
                    msg += f" | [routing mean coeff] {rc_str}"

                print(msg)

                if max(avg_act) > 0.95:
                    dom_route = int(np.argmax(avg_act))
                    print(
                        f"[alert] potential collapse → route={routes[dom_route]} "
                        f"mean={max(avg_act):.3f}"
                    )

        # Epoch summary (TRAIN)
        train_loss = total_loss / max(1, total // num_phenos)
        train_acc  = total_correct / max(1, total)

        train_avg_act = (act_sum / max(1, num_samples)).tolist()
        print(
            f"[epoch {epoch + 1}] TRAIN loss={train_loss:.4f} acc={train_acc:.4f} "
            f"avg_prim_act={', '.join(f'{a:.3f}' for a in train_avg_act)}"
        )

        # VAL metrics (BCE + 0.5 threshold / F1-based thresholds)
        val_loss, val_acc, val_act, val_rc_mat = evaluate_epoch(
            behrt, bbert, imgenc, fusion, projector, cap_head,
            val_loader, amp_ctx, bce,
            route_debug=bool(getattr(args, "route_debug", False)),
            label_names=label_names,
            epoch_idx=epoch + 1,
            split_name="VAL",
            routing_out_dir=os.path.join(ckpt_dir, "routing"),
        )

        # Routing importance: global and per-phenotype (VAL) 
        routes = ["L", "N", "I", "LN", "LI", "NI", "LNI"]

        if val_rc_mat is not None:
            # Data-derived mean routing coefficients per route (averaged over phenotypes)
            rc_mean_val = val_rc_mat.mean(axis=1)   # [7]
            print("\n[epoch %d] [VAL] mean routing coefficient per route (data-derived):" % (epoch + 1))
            for i, r in enumerate(routes):
                print(f"  rc_mean_{r} = {rc_mean_val[i]:.4f}")

            # Per-route, per-phenotype routing importance (7 x K) from routing coefficients
            K = val_rc_mat.shape[1]
            pheno_names = [get_pheno_name(i) for i in range(K)]

            print("\n[epoch %d] [VAL] per-route, per-phenotype routing importance (mean routing coeff):" % (epoch + 1))
            for i, r in enumerate(routes):
                row = " | ".join(
                    f"{pheno_names[k]}:{val_rc_mat[i, k]:.3f}"
                    for k in range(K)
                )
                print(f"  {r}: {row}")

            # Heatmap of routing importance on VAL for this epoch
            mat_val = val_rc_mat.T  # [K, 7]

            plt.figure(figsize=(10, 8))
            im = plt.imshow(mat_val, aspect="auto")
            plt.colorbar(im, label="Mean routing coefficient")

            plt.xticks(range(len(routes)), routes)
            plt.yticks(range(K), pheno_names, fontsize=6)

            plt.xlabel("Route")
            plt.ylabel("Phenotype")
            plt.title(f"Per-route, per-phenotype routing importance (VAL, epoch {epoch + 1})")
            plt.tight_layout()

            fname = os.path.join(ckpt_dir, f"routing_val_mean_rc_epoch{epoch + 1:03d}.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"[routing] saved VAL per-route-per-phenotype map → {fname}")

        # Collect outputs for full VAL metrics
        y_true, p1, _ = collect_epoch_outputs(
            val_loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx
        )

        best_thr = find_best_thresholds(y_true, p1, n_steps=50)
        y_pred = (p1 >= best_thr[np.newaxis, :]).astype(float)

        m = epoch_metrics(y_true, p1, y_pred)

        print(
            f"[epoch {epoch + 1}] VAL MACRO  "
            f"AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f} "
            f"F1={m['F1']:.4f} Recall={m['Recall']:.4f}"
        )
        print(
            f"[epoch {epoch + 1}] VAL MICRO  "
            f"AUROC={m['AUROC_micro']:.4f} "
            f"AUPRC={m['AUPRC_micro']:.4f} "
            f"Precision={m['Precision_micro']:.4f} "
            f"Recall={m['Recall_micro']:.4f} "
            f"F1={m['F1_micro']:.4f}"
        )
        print(
            f"[epoch {epoch + 1}] VAL example-F1={m['F1_example']:.4f} "
            f"Hamming={m['Hamming']:.4f}"
        )
        print(f"[epoch {epoch + 1}] VAL Confusion Matrix:\n{m['CM']}")

        au_per  = m["AUROC_per_label"]
        ap_per  = m["AUPRC_per_label"]
        f1_per  = m["F1_per_label"]
        rec_per = m["Recall_per_label"]

        for k, name in enumerate(label_names):
            print(
                f"[epoch {epoch + 1}] VAL {name} "
                f"AUROC={au_per[k]:.4f} "
                f"AUPRC={ap_per[k]:.4f} "
                f"F1={f1_per[k]:.4f} "
                f"Recall={rec_per[k]:.4f}"
            )
        
        val_macro_auroc = float(m["AUROC"])   # macro AUROC

        # Epoch-level early stopping on VAL macro AUROC 
        if val_macro_auroc > best_val_auroc + min_delta:
            best_val_auroc = val_macro_auroc
            epochs_no_improve = 0
            print(f"[early-stop] AUROC improved to {best_val_auroc:.4f} (epoch {epoch + 1})")
        else:
            # Only start counting patience after min_epochs
            if (epoch + 1) >= min_epochs:
                epochs_no_improve += 1
                print(
                    f"[early-stop] No improvement for {epochs_no_improve}/{patience_epochs} "
                    f"epochs (best={best_val_auroc:.4f}, current={val_macro_auroc:.4f})"
                )
                if epochs_no_improve >= patience_epochs:
                    print(f"[early-stop] Stop: no improvement for {patience_epochs} epochs after min_epochs={min_epochs}.")
                    break
            else:
                print(
                    f"[early-stop] No improvement but epoch {epoch + 1} < min_epochs={min_epochs} "
                    f"→ not counting patience yet."
                )


        # Calibration
        ece, centers, bconf, bacc, bcnt = expected_calibration_error(
            p1.reshape(-1), y_true.reshape(-1), n_bins=args.calib_bins
        )
        print(f"[epoch {epoch + 1}] VAL ECE({args.calib_bins} bins) = {ece:.4f}")
        rel_path = os.path.join(ckpt_dir, f"reliability_val_epoch{epoch + 1:03d}.png")
        reliability_plot(centers, bconf, bacc, out_path=rel_path)
        print(f"[epoch {epoch + 1}] Saved reliability diagram → {rel_path}")

        # Save checkpoints based on VAL accuracy
        val_score = m["AUROC"]  # macro AUROC
        is_best = val_score > best_val_acc
        best_val_acc = max(best_val_acc, val_score)

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
            "best_thr": torch.from_numpy(best_thr.astype(np.float32)),
        }
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), ckpt)
        if is_best:
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), ckpt)
            print(f"[epoch {epoch + 1}] Saved BEST checkpoint (acc={val_acc:.4f})")

    # TEST evaluation using BEST checkpoint
    print("[main] Evaluating BEST checkpoint on TEST...")
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        behrt.load_state_dict(ckpt["behrt"])
        bbert.load_state_dict(ckpt["bbert"])
        imgenc.load_state_dict(ckpt["imgenc"])
        for k in fusion.keys():
            fusion[k].load_state_dict(ckpt["fusion"][k])
        projector.load_state_dict(ckpt["projector"])
        cap_head.load_state_dict(ckpt["cap_head"])

    test_loss, test_acc, test_act, test_rc_mat = evaluate_epoch(
        behrt, bbert, imgenc, fusion, projector, cap_head,
        test_loader, amp_ctx, bce,
        route_debug=False,
        label_names=label_names,
        split_name="TEST",
    )
    print(
        f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} "
        f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k, v in test_act.items())}"
    )


    # Routing importance: global and per-phenotype (TEST) 
    routes = ["L", "N", "I", "LN", "LI", "NI", "LNI"]

    if test_rc_mat is not None:
        # Data-derived mean routing coefficients per route (averaged over phenotypes)
        rc_mean_test = test_rc_mat.mean(axis=1)
        print("\n[TEST] mean routing coefficient per route (data-derived):")
        for i, r in enumerate(routes):
            print(f"  rc_mean_{r} = {rc_mean_test[i]:.4f}")

        # Per-route, per-phenotype routing importance on TEST (7 x K)
        K = test_rc_mat.shape[1]
        pheno_names = [get_pheno_name(i) for i in range(K)]

        print("\n[TEST] per-route, per-phenotype routing importance (mean routing coeff):")
        for i, r in enumerate(routes):
            row = " | ".join(
                f"{pheno_names[k]}:{test_rc_mat[i, k]:.3f}"
                for k in range(K)
            )
            print(f"  {r}: {row}")

        # Heatmap for TEST (phenotypes on y-axis, routes on x-axis)
        mat_test = test_rc_mat.T  # [K, 7]

        plt.figure(figsize=(10, 8))
        im = plt.imshow(mat_test, aspect="auto")
        plt.colorbar(im, label="Mean routing coefficient")

        plt.xticks(range(len(routes)), routes)
        plt.yticks(range(K), pheno_names, fontsize=6)

        plt.xlabel("Route")
        plt.ylabel("Phenotype")
        plt.title("Per-route, per-phenotype routing importance (TEST)")
        plt.tight_layout()

        fname = os.path.join(ckpt_dir, "routing_test_mean_rc.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"[routing] saved TEST per-route-per-phenotype map → {fname}")

    # Compute prevalence & thresholds on VAL
    y_true_v, p_v, _ = collect_epoch_outputs(
        val_loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx
    )
    prev_val = compute_split_prevalence(
        y_true_v, split_name="VAL", label_names=label_names
    )
    thr_val, f1_val_thr = grid_search_thresholds(y_true_v, p_v, n_steps=101)
    save_split_thresholds(thr_val, ckpt_dir, split_name="VAL")

    # Prevalence, thresholds and metrics on TEST
    y_true_t, p1_t, _ = collect_epoch_outputs(
        test_loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx
    )

    prev_test = compute_split_prevalence(
        y_true_t, split_name="TEST", label_names=label_names
    )

    print("\n[VAL vs TEST] Prevalence difference per label:")
    for i, name in enumerate(label_names):
        dv = prev_test[i] - prev_val[i]
        print(f"  {name}: VAL={prev_val[i]:.4f} TEST={prev_test[i]:.4f} Δ={dv:+.4f}")

    thr_test, f1_test_thr = grid_search_thresholds(
        y_true_t, p1_t, n_steps=101
    )
    save_split_thresholds(thr_test, ckpt_dir, split_name="TEST")

    # Evaluate TEST using VAL-optimized thresholds
    y_pred_t = (p1_t >= thr_val[np.newaxis, :]).astype(float)
    mt = epoch_metrics(y_true_t, p1_t, y_pred_t)

    print(
        f"[TEST] MACRO  AUROC={mt['AUROC']:.4f} "
        f"AUPRC={mt['AUPRC']:.4f} F1={mt['F1']:.4f} Recall={mt['Recall']:.4f}"
    )
    print(
        f"[TEST] MICRO  "
        f"AUROC={mt['AUROC_micro']:.4f} "
        f"AUPRC={mt['AUPRC_micro']:.4f} "
        f"Precision={mt['Precision_micro']:.4f} "
        f"Recall={mt['Recall_micro']:.4f} "
        f"F1={mt['F1_micro']:.4f}"
    )
    print(
        f"[TEST] example-F1={mt['F1_example']:.4f} "
        f"Hamming={mt['Hamming']:.4f}"
    )
    print(f"[TEST] Confusion Matrix:\n{mt['CM']}")

    au_per  = mt["AUROC_per_label"]
    ap_per  = mt["AUPRC_per_label"]
    f1_per  = mt["F1_per_label"]
    rec_per = mt["Recall_per_label"]

    for k, name in enumerate(label_names):
        print(
            f"[TEST] {name} "
            f"AUROC={au_per[k]:.4f} "
            f"AUPRC={ap_per[k]:.4f} "
            f"F1={f1_per[k]:.4f} "
            f"Recall={rec_per[k]:.4f}"
        )

    ece_t, centers_t, bconf_t, bacc_t, bcnt_t = expected_calibration_error(
        p1_t.reshape(-1), y_true_t.reshape(-1), n_bins=args.calib_bins
    )
    print(f"[TEST] ECE({args.calib_bins} bins) = {ece_t:.4f}")
    rel_test_path = os.path.join(ckpt_dir, "reliability_test.png")
    reliability_plot(centers_t, bconf_t, bacc_t, out_path=rel_test_path)
    print(f"[TEST] Saved reliability diagram → {rel_test_path}")


if __name__ == "__main__":
    main()
