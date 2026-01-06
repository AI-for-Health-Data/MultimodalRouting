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
import random

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
from mult_model import MULTModel
from routing_and_heads import (
    RoutePrimaryProjector,
    RouteDimAdapter,
    CapsuleMortalityHead,
    forward_capsule_from_multmodel,
    make_route_inputs_mult,
    forward_capsule_from_route_dict,
)



ROUTE_NAMES = ["L", "N", "I", "LN", "NL", "LI", "IL", "NI", "IN", "LNI"]
N_ROUTES = len(ROUTE_NAMES) 

def grads_are_finite(param_list):
    for p in param_list:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True

def safe_zero_grad(optimizer):
    optimizer.zero_grad(set_to_none=True)

def seed_worker(worker_id: int):
    ws = (int(CFG.seed) + int(worker_id)) % (2**32)
    np.random.seed(ws)
    random.seed(ws)
    torch.manual_seed(ws)

def _standardize_id_column(df: pd.DataFrame, name="stay_id") -> pd.DataFrame:
    candidates = [name, "icustay_id", "stay", "sample_id"]
    found = next((c for c in candidates if c in df.columns), None)
    if found is None:
        raise ValueError(f"Missing id column. Tried {candidates}. Found: {list(df.columns)[:50]}")
    if found != name:
        df = df.rename(columns={found: name})
    df[name] = df[name].astype(int)
    return df

def print_route_matrix_detailed(
    routing_coef: torch.Tensor,
    prim_acts: torch.Tensor,
    label_names: List[str],
    where: str = "",
):
    with torch.no_grad():
        rc = routing_coef.detach().float().cpu()  # [B,N_ROUTES,K]
        pa = prim_acts.detach().float().cpu()     # [B,N_ROUTES]

        B, R, K = rc.shape

        # Average over batch
        rc_mean = rc.mean(dim=0).numpy()          # [N_ROUTES,K]
        pa_mean = pa.mean(dim=0).numpy()          # [N_ROUTES]

        # Effective weights (per route × phenotype)
        effective = (rc * pa.unsqueeze(-1)).mean(dim=0).numpy()  # [N_ROUTES,K]

        routes = ROUTE_NAMES

        print(f"\n{'=' * 120}")
        print(f"[ROUTING ANALYSIS] {where}")
        print(f"{'=' * 120}")

        print(f"\n1. PRIMARY ACTIVATIONS (sigmoid, same across all phenotypes):")
        print("   " + " | ".join(f"{r:4s}={pa_mean[i]:.3f}" for i, r in enumerate(routes)))

        print(f"\n2. PER-PHENOTYPE ROUTING WEIGHTS:")
        print("   Format: phenotype_name | " + " | ".join(routes))
        print("   Each cell shows: routing_coef(effective_weight)")
        print(f"   {'-' * 116}")

        for k in range(K):
            cells = []
            for i in range(R):
                cells.append(f"{rc_mean[i, k]:.4f}({effective[i, k]:.4f})")

            name = label_names[k] if k < len(label_names) else f"label_{k}"
            row = f"   {name:15s} | " + " | ".join(f"{cell:>10s}" for cell in cells)
            print(row)

        print(f"\n3. ROUTE IMPORTANCE (averaged across all phenotypes):")
        rc_avg = rc_mean.mean(axis=1)      # [N_ROUTES]
        eff_avg = effective.mean(axis=1)   # [N_ROUTES]

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

def encode_all_modalities(behrt, bbert, imgenc, xL, mL, notes, imgs, amp_ctx_enc):
    with amp_ctx_enc:
        L_seq, L_mask, L_pool = behrt.encode_seq_and_pool(xL, mask=mL)

        notes_batch = prepare_notes_batch(notes)
        N_seq, N_mask = bbert.encode_seq(notes_batch)
        N_pool = bbert.pool_from_seq(N_seq, N_mask)

        I_seq, I_mask, I_pool = imgenc.encode_seq_and_pool(imgs)

    outL = {"seq": L_seq, "mask": L_mask.float(), "pool": L_pool}
    outN = {"seq": N_seq, "mask": N_mask.float(), "pool": N_pool}
    outI = {"seq": I_seq, "mask": I_mask.float(), "pool": I_pool}

    outL = _sanitize_encoder_out(outL, "L")
    outN = _sanitize_encoder_out(outN, "N")
    outI = _sanitize_encoder_out(outI, "I")
    return outL, outN, outI

def debug_routing_tensor(
    rc: torch.Tensor,
    name: str = "routing_coef",
    expect_routes: int = 10,
    expect_k: Optional[int] = None,
):
    with torch.no_grad():
        print(f"\n[debug] {name}.shape={tuple(rc.shape)} dtype={rc.dtype} device={rc.device}")
        if rc.ndim != 3:
            print(f"[debug] {name}: expected 3D [B,R,K], got {rc.ndim}D")
            return

        B, D1, D2 = rc.shape

        # figure out which dim is routes
        if D1 == expect_routes:
            routes_dim = 1
            k_dim = 2
        elif D2 == expect_routes:
            routes_dim = 2
            k_dim = 1
        else:
            routes_dim = 1
            k_dim = 2
            print(f"[debug] {name}: WARNING could not find routes axis by size={expect_routes}, using dim=1 as routes.")
        if expect_k is not None:
            K_found = rc.shape[k_dim]
            if int(K_found) != int(expect_k):
                print(f"[debug] {name}: WARNING K mismatch: got K={K_found} but expect_k={expect_k} (k_dim={k_dim})")
        s_routes = rc.sum(dim=routes_dim)
        print(f"[debug] sum_over_routes(dim={routes_dim}): mean={s_routes.float().mean().item():.6f} "
              f"min={s_routes.float().min().item():.6f} max={s_routes.float().max().item():.6f}")

        if routes_dim == 1:
            print("[debug] rc[0, :, 0] (routes for phenotype0):", rc[0, :, 0].detach().float().cpu().tolist())
            print("[debug] rc[0, :, 0].sum() =", float(rc[0, :, 0].sum().detach().cpu()))
        else:
            print("[debug] rc[0, 0, :] (routes for phenotype0):", rc[0, 0, :].detach().float().cpu().tolist())
            print("[debug] rc[0, 0, :].sum() =", float(rc[0, 0, :].sum().detach().cpu()))

def normalize_routing_coef_auto(
    rc: torch.Tensor,
    expect_routes: int = N_ROUTES,
    eps: float = 1e-12,
    mode: str = "gate_norm",  # {"gate_norm", "softmax", "none"}
    atol_already: float = 1e-3,
):
    if rc is None or rc.ndim != 3:
        return rc, {"ok": False, "reason": "rc None or not 3D"}

    rc = rc.float()
    B, D1, D2 = rc.shape
    info = {"orig_shape": (B, D1, D2), "expect_routes": int(expect_routes), "mode": str(mode)}

    transposed = False
    if D1 == expect_routes:
        pass
    elif D2 == expect_routes:
        rc = rc.transpose(1, 2)  # [B,K,R] -> [B,R,K]
        transposed = True
    else:
        s1 = rc.sum(dim=1)  # [B, D2]
        s2 = rc.sum(dim=2)  # [B, D1]
        info["err_sum_dim1"] = float((s1 - 1.0).abs().mean().item())
        info["err_sum_dim2"] = float((s2 - 1.0).abs().mean().item())
        info["warn"] = "could_not_identify_routes_dim_by_size"

    info["transposed_to_BRK"] = transposed
    info["shape_after_orient"] = tuple(rc.shape)

    mode_l = str(mode).lower()

    if mode_l == "none":
        info["ok"] = True
        return rc, info

    if mode_l == "softmax":
        rc = torch.softmax(rc, dim=1)
        s = rc.sum(dim=1)
        info["post_min_sum_routes"] = float(s.min().item())
        info["post_max_sum_routes"] = float(s.max().item())
        info["ok"] = True
        return rc, info

    if mode_l == "gate_norm":
        rc = rc.clamp_min(0.0)
        denom = rc.sum(dim=1, keepdim=True).clamp_min(eps)
        rc = rc / denom

        s = rc.sum(dim=1)
        info["post_min_sum_routes"] = float(s.min().item())
        info["post_max_sum_routes"] = float(s.max().item())
        info["ok"] = True
        return rc, info

    return rc, {"ok": False, "reason": f"unknown mode={mode}"}



def renorm_over_routes(rc: torch.Tensor, routes_dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    denom = rc.sum(dim=routes_dim, keepdim=True).clamp_min(eps)
    return rc / denom


def assert_routing_over_routes(rc: torch.Tensor, routes_dim: int = 1, atol: float = 1e-3, name: str = "routing_coef"):
    if rc.ndim != 3:
        raise AssertionError(f"{name}: expected [B,R,K], got {tuple(rc.shape)}")

    s = rc.sum(dim=routes_dim)  # [B,K]
    ones = torch.ones_like(s)

    if not torch.allclose(s, ones, atol=atol, rtol=0.0):
        with torch.no_grad():
            max_err = (s - 1.0).abs().max().item()
            mean_err = (s - 1.0).abs().mean().item()
            raise AssertionError(
                f"{name}: NOT normalized over routes dim={routes_dim}. "
                f"sum(dim={routes_dim}) -> min={s.min().item():.6f} max={s.max().item():.6f} "
                f"abs_err mean={mean_err:.3e} max={max_err:.3e}"
            )


def quantization_check(x: torch.Tensor, name: str = "x", k: int = 15):
    with torch.no_grad():
        xf = x.detach().float().flatten()
        xf = xf[torch.isfinite(xf)]
        if xf.numel() == 0:
            print(f"[debug] {name}: no finite values")
            return
        vals, counts = torch.unique(xf, return_counts=True)
        topk = torch.topk(counts, k=min(k, counts.numel()))
        print(f"\n[debug] {name}: numel={xf.numel()} unique={vals.numel()} min={xf.min().item():.6f} max={xf.max().item():.6f} mean={xf.mean().item():.6f}")
        print(f"[debug] {name}: top values (value,count):")
        for c, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            print(f"  {vals[idx].item():.6f}  {c}")

        for denom in [8, 16, 32, 64]:
            frac = torch.abs(xf * denom - torch.round(xf * denom)).mean().item()
            print(f"[debug] {name}: mean(|x*{denom}-round|)={frac:.6f}  (near 0 => step 1/{denom})")


def mask_stats(m: torch.Tensor, name: str = "m"):
    with torch.no_grad():
        m2 = m[..., 0] if m.ndim == 3 else m
        valid = m2.detach().float().sum(dim=1)
        print(f"\n[debug] {name}.shape={tuple(m.shape)}")
        print(f"[debug] valid timesteps: mean={valid.mean().item():.2f} min={valid.min().item():.0f} max={valid.max().item():.0f}")
        for t in [1,2,3,4,6,12,24]:
            print(f"  count(valid<= {t}): {(valid<=t).sum().item()}/{valid.numel()}")


def route_cosine_report(route_embs: Dict[str, torch.Tensor]):
    import torch.nn.functional as F
    def _cos(a, b):
        a = F.normalize(a.detach().float(), dim=-1)
        b = F.normalize(b.detach().float(), dim=-1)
        return (a * b).sum(dim=-1)

    pairs = [("L","LN"), ("N","NL"), ("I","LI"), ("NI","LNI"), ("L","LNI"), ("N","NI"), ("I","NI")]
    print("\n[debug] route cosine similarities (mean over batch):")
    for a,b in pairs:
        if a in route_embs and b in route_embs:
            c = _cos(route_embs[a], route_embs[b])
            print(f"  cos({a},{b}) mean={c.mean().item():.4f} min={c.min().item():.4f} max={c.max().item():.4f}")
        else:
            print(f"  missing {a} or {b}")

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
        rc_mean = rc.mean(dim=0).numpy()  # [N_ROUTES,K]
        pa_mean = pa.mean(dim=0).numpy()  # [N_ROUTES]

        effective = rc_mean * pa_mean[:, np.newaxis]  # [N_ROUTES,K]

        # decide which phenotypes to show
        if top_k is None or top_k >= K:
            top_indices = np.arange(K)
        else:
            variance = effective.var(axis=0)  # [K]
            top_indices = variance.argsort()[-top_k:][::-1]

        routes = ROUTE_NAMES

        print(f"\n{'=' * 120}")
        print(f"[PHENOTYPE ROUTING HEATMAP] {where}")
        print("Showing effective weights (primary_act × routing_coef):")
        print(f"{'-' * 120}")

        for idx in top_indices:
            if label_names is not None and idx < len(label_names):
                name = label_names[idx]
            else:
                name = get_pheno_name(idx)

            weights = effective[:, idx]  # [N_ROUTES]

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

        B, R, K = rc.shape          # [B, N_ROUTES, K]
        rc_mean = rc.mean(dim=0).numpy()     # [N_ROUTES, K]
        pa_mean = pa.mean(dim=0).numpy()     # [N_ROUTES]
        effective = rc_mean * pa_mean[:, np.newaxis]  # [N_ROUTES, K]

        routes = ROUTE_NAMES
        mat = effective.T  # [K, 10]

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
from matplotlib.colors import LinearSegmentedColormap

def _light_yellow_to_light_blue_cmap():
    return LinearSegmentedColormap.from_list(
        "yl_to_lb",
        ["#fff7cc", "#d6ecff"],  
        N=256
    )

def normalize_minmax(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))

    if not np.isfinite(a_min) or not np.isfinite(a_max) or (a_max - a_min) < eps:
        out = np.zeros_like(arr, dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    out = (arr - a_min) / (a_max - a_min)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def normalize_routes_per_phenotype(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, eps)
    return mat / row_sum

def save_array_with_versions(
    arr_raw: np.ndarray,
    out_dir: str,
    base_name: str,
    row_names: Optional[List[str]] = None,
    col_names: Optional[List[str]] = None,
    print_title: str = "",
    norm_fn = None,  
):
    os.makedirs(out_dir, exist_ok=True)
    arr_raw = np.asarray(arr_raw, dtype=np.float32)
    if norm_fn is None:
        arr_norm = normalize_minmax(arr_raw)
    else:
        arr_norm = norm_fn(arr_raw)


    raw_npy  = os.path.join(out_dir, f"{base_name}_raw.npy")
    norm_npy = os.path.join(out_dir, f"{base_name}_norm.npy")
    np.save(raw_npy, arr_raw)
    np.save(norm_npy, arr_norm)

    def _to_df(a):
        if a.ndim == 1:
            df = pd.DataFrame(a.reshape(1, -1))
        else:
            df = pd.DataFrame(a)
        if row_names is not None and df.shape[0] == len(row_names):
            df.index = row_names
        if col_names is not None and df.shape[1] == len(col_names):
            df.columns = col_names
        return df

    df_raw = _to_df(arr_raw)
    df_nrm = _to_df(arr_norm)

    raw_csv  = os.path.join(out_dir, f"{base_name}_raw.csv")
    norm_csv = os.path.join(out_dir, f"{base_name}_norm.csv")
    df_raw.to_csv(raw_csv, float_format="%.4f")
    df_nrm.to_csv(norm_csv, float_format="%.4f")

    # Print tables (4 decimals)
    if print_title:
        print(f"\n{'='*120}\n{print_title}\n{'='*120}")
    print(f"[saved] {raw_npy}\n[saved] {raw_csv}")
    print(df_raw.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n[saved] {norm_npy}\n[saved] {norm_csv}")
    print(df_nrm.to_string(float_format=lambda x: f"{x:.4f}"))

    return arr_norm, df_raw, df_nrm

def save_heatmap_with_numbers(
    mat_norm: np.ndarray,                
    mat_raw: np.ndarray,                 
    row_names: List[str],
    col_names: List[str],
    title: str,
    out_path: str,
    fontsize_cell: int = 7,
    fontsize_ticks: int = 9,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmap = _light_yellow_to_light_blue_cmap()

    mat_norm = np.asarray(mat_norm, dtype=np.float32)
    mat_raw  = np.asarray(mat_raw, dtype=np.float32)

    n_rows, n_cols = mat_norm.shape

    # Bigger figure for readability (scales with size)
    w = max(10, 0.9 * n_cols + 6)
    h = max(6,  0.35 * n_rows + 4)

    plt.figure(figsize=(w, h))
    im = plt.imshow(mat_norm, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Normalized (min–max)")

    plt.xticks(np.arange(n_cols), col_names, fontsize=fontsize_ticks, rotation=0)
    plt.yticks(np.arange(n_rows), row_names, fontsize=fontsize_ticks)

    # annotate raw values
    for i in range(n_rows):
        for j in range(n_cols):
            plt.text(
                j, i, f"{mat_raw[i, j]:.4f}",
                ha="center", va="center",
                fontsize=fontsize_cell
            )

    plt.xlabel("Route" if (col_names is not None and len(col_names) > 0) else "")
    plt.ylabel("Phenotype" if len(row_names) > 1 else "")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[heatmap] saved → {out_path}")

def save_vector_heatmap_with_numbers(
    vec_raw: np.ndarray,     # shape [M]
    labels: List[str],
    title: str,
    out_path: str,
    fontsize_cell: int = 10,
    fontsize_ticks: int = 10,
):
    vec_raw = np.asarray(vec_raw, dtype=np.float32).reshape(1, -1)
    vec_norm = normalize_minmax(vec_raw)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmap = _light_yellow_to_light_blue_cmap()

    plt.figure(figsize=(max(10, 0.9 * len(labels) + 4), 2.6))
    im = plt.imshow(vec_norm, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Normalized (min–max)")

    plt.xticks(np.arange(len(labels)), labels, fontsize=fontsize_ticks)
    plt.yticks([0], [" "], fontsize=fontsize_ticks)

    for j in range(len(labels)):
        plt.text(j, 0, f"{vec_raw[0, j]:.4f}", ha="center", va="center", fontsize=fontsize_cell)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[heatmap] saved → {out_path}")

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

        # Make sure both are list-of-chunks
        if not isinstance(ids_chunks, (list, tuple)):
            ids_chunks = []
        if not isinstance(attn_chunks, (list, tuple)):
            attn_chunks = []

        # Normalize each chunk to flat List[int]
        ids_chunks  = [_to_int_list(x) for x in ids_chunks]
        attn_chunks = [_to_int_list(x) for x in attn_chunks]

        paired = [
            (a, b) for a, b in zip(ids_chunks, attn_chunks)
            if len(a) > 0 and len(b) > 0 and (np.sum(np.asarray(b, dtype=np.int64)) > 0)
        ]
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
        attn_mat = (attn_mat > 0).long()
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

def parse_args():
    ap = argparse.ArgumentParser(
        description="Phenotype prediction with N_ROUTES-route capsule routing (multi-label)"
    )
    ap.add_argument("--task", type=str, default=_cfg("task_name", "pheno"),
                    choices=list(TASK_MAP.keys()))
    ap.add_argument("--data_root", type=str, default=_cfg("data_root", "./data"))
    ap.add_argument("--ckpt_root", type=str, default=_cfg("ckpt_root", "./ckpts"))
    ap.add_argument("--encoder_warmup_epochs", type=int, default=int(_cfg("encoder_warmup_epochs", 2)))

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
    ap.add_argument("--peek_first_batch", action="store_true", default=False)
    ap.add_argument("--verbose_sanity", action="store_true", default=False)
    ap.add_argument("--route_debug", action="store_true")
    ap.add_argument("--calib_bins", type=int, default=10)
    return ap.parse_args()

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

VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dcm"}

def resolve_image_path(p: str, dataset_root: str) -> str:
    p = str(p).strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    image_root = str(getattr(CFG, "image_root", "") or "").strip()
    if p.startswith("mimic-cxr-jpg/"):
        base = image_root if image_root else dataset_root
        return os.path.join(base, p)

    if image_root:
        return os.path.join(image_root, p)
    return os.path.join(dataset_root, p)


def is_probably_image_file(p: str) -> bool:
    ext = os.path.splitext(str(p).lower())[1]
    return ext in VALID_IMG_EXTS

def _detect_notes_schema(notes_df: pd.DataFrame):
    # pretokenized
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

        # sort suffixes numerically if possible
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

    # text mode 
    chunk_cols = sorted([c for c in notes_df.columns if str(c).startswith("chunk_")])
    if len(chunk_cols) > 0:
        return ("text", chunk_cols, None)

    text_chunk_cols = sorted([c for c in notes_df.columns if str(c).startswith("text_chunk_")])
    if len(text_chunk_cols) > 0:
        return ("text", text_chunk_cols, None)

    raise ValueError("notes schema not recognized")


def _cell_to_list(x):
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
            "structured_fullstay_medfuse_11230.parquet",
            "notes_fullstay_radiology_TEXTCHUNKS_11230.parquet",
            "images.parquet",
            "labels_pheno.parquet",
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

        struct_fp = os.path.join(root, "structured_fullstay_medfuse_11230.parquet")
        notes_fp  = os.path.join(root, "notes_fullstay_radiology_TEXTCHUNKS_11230.parquet")

        images_fp = os.path.join(root, "images.parquet")
        labels_fp = os.path.join(root, "labels_pheno.parquet")

        self.struct = _standardize_id_column(pd.read_parquet(struct_fp))
        self.notes  = _standardize_id_column(pd.read_parquet(notes_fp))
        self.images = _standardize_id_column(pd.read_parquet(images_fp))
        self.images = _standardize_image_path_column(self.images)
        self.labels = _standardize_id_column(pd.read_parquet(labels_fp))


        sample = self.images["cxr_path"].dropna().astype(str)
        sample = sample[sample.str.strip().ne("")]

        if len(sample) > 0:
            testp = resolve_image_path(sample.iloc[0], self.root)  
            if not os.path.exists(testp):
                raise RuntimeError(
                    f"[ICUStayDataset] Image root mismatch.\n"
                    f"  sample cxr_path = {sample.iloc[0]}\n"
                    f"  resolved        = {testp}\n"
                    f"  dataset_root    = {self.root}\n"
                    f"Set CFG.image_root to the directory that contains mimic-cxr-jpg/."
                )


        # unify stay_id dtype
        for attr in ["struct", "notes", "images", "labels"]:
            df = getattr(self, attr)
            if "stay_id" in df.columns:
                df["stay_id"] = df["stay_id"].astype(int)

        # structured feature columns
        base_cols = {"stay_id", "hour"}
        self.feat_cols: List[str] = [c for c in self.struct.columns if c not in base_cols]
        self.feat_cols.sort()
        if hasattr(CFG, "structured_feat_cols") and CFG.structured_feat_cols:
            self.feat_cols = list(CFG.structured_feat_cols)
        else:
            exclude = {"stay_id", "hour", "subject_id", "hadm_id", "icustay_id"}
            self.feat_cols = [
                c for c in self.struct.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(self.struct[c])
            ]
        self.feat_cols.sort()

        # notes schema: text chunks OR pretokenized ids/masks
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

        # phenotype label columns
        self.label_cols: List[str] = [c for c in self.labels.columns if c != "stay_id"]
        self.label_cols.sort()
        if len(self.label_cols) == 0:
            raise ValueError("[ICUStayDataset] labels_pheno.parquet must contain at least one phenotype column.")
        self.num_labels = len(self.label_cols)
        print(f"[dataset:{split}] found {len(self.label_cols)} phenotype labels: {self.label_cols[:5]}{' ...' if len(self.label_cols) > 5 else ''}")

        struct_ids = set(self.struct["stay_id"].astype(int).unique().tolist())
        label_ids  = set(self.labels["stay_id"].astype(int).unique().tolist())

        # notes ids with actual usable content
        if self.notes_mode == "text":
            nonempty = np.zeros(len(self.notes), dtype=bool)
            for c in self.chunk_cols:
                if c in self.notes.columns:
                    nonempty |= self.notes[c].fillna("").astype(str).str.strip().ne("")
            note_ids = set(self.notes.loc[nonempty, "stay_id"].astype(int).unique().tolist())

        else:
            # pretokenized: require at least one chunk with mask_sum > 0 (not just len==512)
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

        # image ids with at least one VALID existing image file
        def _has_existing_image(stay_id: int) -> bool:
            df_i = self.images[self.images.stay_id == int(stay_id)]
            if df_i.empty:
                return False
            raw_paths = df_i.cxr_path.dropna().astype(str).tolist()
            raw_paths = [p for p in raw_paths if str(p).strip()]
            if not raw_paths:
                return False

            # resolve relative -> absolute
            cand = [resolve_image_path(p, self.root) for p in raw_paths]

            # must look like an image and exist
            cand = [p for p in cand if is_probably_image_file(p) and os.path.exists(p)]
            return len(cand) > 0

        # only check candidates in the split to keep it fast
        split_img_candidates = set(
            self.images.loc[
                self.images["cxr_path"].fillna("").astype(str).str.strip().ne(""),
                "stay_id"
            ].astype(int).tolist()
        )

        split_img_candidates = split_img_candidates & ids_set  # limit to split ids
        img_ids = set([sid for sid in split_img_candidates if _has_existing_image(sid)])

        print(f"[dataset:{split}] ids_set={len(ids_set)}")
        print(f"[dataset:{split}] struct_ids={len(struct_ids)}")
        print(f"[dataset:{split}] label_ids={len(label_ids)}")
        print(f"[dataset:{split}] note_ids={len(note_ids)}")
        print(f"[dataset:{split}] img_ids={len(img_ids)}")


        # Always require structured + labels
        keep_ids = ids_set & struct_ids & label_ids & img_ids & note_ids


        dropped_total = len(ids_set) - len(keep_ids)
        dropped_no_notes = len(ids_set & struct_ids & label_ids & img_ids) - len(keep_ids)
        self.ids = sorted(list(keep_ids))
        print(f"[dataset:{split}] kept {len(self.ids)} / {len(ids_set)}")
        print(f"[dataset:{split}] dropped total={dropped_total} | dropped_missing_notes={dropped_no_notes}")

        if len(self.ids) == 0:
            raise RuntimeError(f"[ICUStayDataset] After filtering, split '{self.split}' is empty.")

        # small summary
        print(
            f"[dataset:{split}] root={root} ids={len(self.ids)} "
            f"| struct rows={len(self.struct)} (F={len(self.feat_cols)}) "
            f"| notes rows={len(self.notes)} "
            f"| images rows={len(self.images)} "
            f"| labels rows={len(self.labels)} (K={len(self.label_cols)})"
        )

    def __len__(self) -> int:
        return len(self.ids)

    def audit_images(self, n: int = 200, seed: int = 0) -> None:
        """
        Checks raw cxr_path values, resolved paths, extensions, and existence.
        """
        rng = np.random.default_rng(seed)
        ids = self.ids if len(self.ids) > 0 else self.images["stay_id"].astype(int).unique().tolist()
        if len(ids) == 0:
            print("[audit_images] no ids to audit")
            return

        pick = ids if len(ids) <= n else rng.choice(ids, size=n, replace=False).tolist()

        total = ok = bad_ext = missing = empty = 0
        exts = {}

        for stay_id in pick:
            df_i = self.images[self.images.stay_id == int(stay_id)]
            raw_paths = df_i.cxr_path.dropna().astype(str).tolist()
            raw_paths = [p for p in raw_paths if str(p).strip()]

            if not raw_paths:
                empty += 1
                total += 1
                continue

            cand = [resolve_image_path(p, self.root) for p in raw_paths]

            for p in cand:
                ext = os.path.splitext(p.lower())[1]
                exts[ext] = exts.get(ext, 0) + 1

            cand2 = [p for p in cand if is_probably_image_file(p)]
            if not cand2:
                bad_ext += 1
                total += 1
                continue

            cand3 = [p for p in cand2 if os.path.exists(p)]
            if not cand3:
                missing += 1
                total += 1
                continue

            ok += 1
            total += 1

        print(f"[audit_images] audited={total} | ok(existing+validext)={ok} "
              f"| empty_paths={empty} | bad_ext={bad_ext} | missing_files={missing}")
        top = sorted(exts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("[audit_images] ext_counts(top):", top)

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
        if xs_np.shape[1] != CFG.structured_n_feats:
            raise ValueError(f"Structured feat dim mismatch: got {xs_np.shape[1]} but CFG.structured_n_feats={CFG.structured_n_feats}")

        xs = torch.from_numpy(xs_np)  # [<=T,F]

        # notes: use the first row for this stay_id (you can change policy later if needed)
        df_n = self.notes[self.notes.stay_id == stay_id]
        if df_n.empty:
            raise RuntimeError(f"[ICUStayDataset] stay_id={stay_id} missing notes row")
        row = df_n.iloc[0]

        if self.notes_mode == "text":
            notes_list: List[str] = []
            for c in self.chunk_cols:
                val = row.get(c, "")
                if pd.notna(val) and str(val).strip():
                    notes_list.append(str(val))

            if not notes_list:
                raise RuntimeError(f"[ICUStayDataset] stay_id={stay_id} has no non-empty chunk_* text")

            M = int(getattr(CFG, "notes_max_chunks", -1))
            if M > 0:
                notes_list = notes_list[:M]

            notes_payload = {"mode": "text", "chunks": notes_list}

        else:
            try:
                n_chunks = int(row.get("n_chunks", 0) or 0)
            except Exception:
                n_chunks = 0

            ids_chunks: List[List[int]] = []
            attn_chunks: List[List[int]] = []

            for j, (c_id, c_m) in enumerate(zip(self.input_id_cols, self.attn_mask_cols)):
                if n_chunks > 0 and j >= n_chunks:
                    break

                ids = _cell_to_list(row.get(c_id, None))
                msk = _cell_to_list(row.get(c_m, None))

                # Must have content in both
                if len(ids) == 0 or len(msk) == 0:
                    continue

                # Safety: ids/mask should align
                if len(ids) != len(msk):
                    continue

                # Key fix: ignore all-padding chunks (mask sum == 0)
                if np.sum(np.asarray(msk, dtype=np.int64)) <= 0:
                    continue

                ids_chunks.append(ids)
                attn_chunks.append(msk)

            if len(ids_chunks) == 0:
                raise RuntimeError(
                    f"[ICUStayDataset] stay_id={stay_id} has no valid non-pad chunks "
                    f"(all attention masks are zero or missing)"
                )

            M = int(getattr(CFG, "notes_max_chunks", -1))
            if M > 0:
                ids_chunks = ids_chunks[:M]
                attn_chunks = attn_chunks[:M]

            notes_payload = {
                "mode": "pretokenized",
                "input_ids": ids_chunks,
                "attention_mask": attn_chunks,
            }
        # images: choose the last *valid existing* path for this stay
        df_i = self.images[self.images.stay_id == stay_id]
        if df_i.empty:
            raise RuntimeError(f"[ICUStayDataset] stay_id={stay_id} missing images row")

        raw_paths = df_i.cxr_path.dropna().astype(str).tolist()
        raw_paths = [p for p in raw_paths if str(p).strip()]

        # resolve relative -> absolute using *dataset* root
        cand = [resolve_image_path(p, self.root) for p in raw_paths]

        # filter: looks like image + exists
        cand = [p for p in cand if is_probably_image_file(p) and os.path.exists(p)]

        if not cand:
            # print a useful debug message before failing
            sample_show = raw_paths[:3]
            raise RuntimeError(
                f"[ICUStayDataset] stay_id={stay_id} has no valid existing image files. "
                f"Example raw paths: {sample_show} | dataset_root={self.root}"
            )

        img_paths = [cand[-1]]  # keep last valid only (stable policy)


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
                # normalize to 0..255
                arr = arr - arr.min()
                if arr.max() > 0:
                    arr = arr / arr.max()
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr)  # grayscale PIL
            except Exception as e:
                print(f"[warn] failed to read DICOM: {p_full} ({e}) -> zero tensor")
                tensor = torch.zeros(3, 224, 224)
                return (tensor, p_full) if return_path else tensor
        else:
            img = Image.open(p_full)

        # Make sure PIL image is in a consistent mode
        # (your transforms start with Grayscale->3ch; still good to ensure load works)
        img = img.convert("RGB")

        tensor = tfms(img)

    except Exception as e:
        print(f"[warn] failed to open image: {p_full} ({e}) -> returning zero tensor")
        tensor = torch.zeros(3, 224, 224)

    return (tensor, p_full) if return_path else tensor



def collate_fn_factory(tidx: int, img_tfms: T.Compose):
    first_print = {"done": False}

    def _collate(batch: List[Dict[str, Any]]):
        F_dim = int(CFG.structured_n_feats)

        # choose T safely
        T_len_cfg = int(getattr(CFG, "structured_seq_len", 256))
        if T_len_cfg is not None and T_len_cfg > 0:
            T_len = T_len_cfg
        else:
            # dynamic pad-to-batch-max when seq_len is -1 / unset
            T_len = max(int(b["x_struct"].shape[0]) for b in batch)
            T_len = max(T_len, 1)  # safety

        xL_batch = torch.stack(
            [pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0
        )

        # mask: valid timesteps = any non-zero feature at that t
        mL_batch = (xL_batch.abs().sum(dim=2) > 0).float()

        # safety: ensure mask not all-zero
        mask_sum = mL_batch.sum(dim=1)
        bad = (mask_sum == 0)
        if bad.any():
            mL_batch[bad, -1] = 1.0
            xL_batch[bad, -1, 0] = 1e-6
            print(f"[collate] fixed {int(bad.sum())} samples with empty structured mask")


        # notes payloads are already in the exact format prepare_notes_batch expects
        notes_batch = [b["notes"] for b in batch]
        for b in notes_batch:
            if b.get("mode") == "text":
                assert len(b.get("chunks", [])) > 0, "[collate] empty text chunks"
            else:
                assert len(b.get("input_ids", [])) > 0, "[collate] empty pretokenized chunks"

        imgs_list, img_paths_list = [], []
        for b in batch:
            assert len(b["image_paths"]) > 0 and str(b["image_paths"][-1]).strip(), \
                "[collate] tri-modal strict: missing image path for a sample"
            img_t, path = load_cxr_tensor(b["image_paths"], img_tfms, return_path=True)
            imgs_list.append(img_t)
            img_paths_list.append(path)
        imgs_batch = torch.stack(imgs_list, dim=0)
        # Quick health check: detect if images are all zeros 
        if not first_print["done"]:
            zero_frac = float((imgs_batch.abs().sum(dim=(1,2,3)) == 0).float().mean().item())
            print(f"[collate] image_zero_fraction(first batch) = {zero_frac:.3f}")
            print(f"[collate] xL_batch: {tuple(xL_batch.shape)} | ...")
            first_print["done"] = True

        y_batch = torch.stack([b["y"].float().view(-1) for b in batch], dim=0)
        dbg = {"stay_ids": [b["stay_id"] for b in batch], "img_paths": img_paths_list}
        return xL_batch, mL_batch, notes_batch, imgs_batch, y_batch, dbg

    return _collate


@torch.no_grad()
def pretty_print_small_batch(xL, mL, notes, dbg, k: int = 3) -> None:
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

def _enc_forward(enc, *args, **kwargs):
    try:
        return enc(*args, **kwargs)
    except TypeError as e:
        for k in ["return_seq", "return_attn", "return_mask"]:
            if k in kwargs:
                kwargs.pop(k, None)
        return enc(*args, **kwargs)


def _capsule_forward_mult(
    *,
    mult,
    route_adapter,
    x_l: torch.Tensor,
    x_n: torch.Tensor,
    x_i: torch.Tensor,
    projector,
    cap_head,
    route_mask=None,
    act_temperature: float = 1.0,
    detach_priors: bool = False,
    return_routing: bool = True,
):
    """
    MulT -> route dict -> (adapter) -> projector -> capsule head
    returns (logits, prim_acts, route_embs, routing_coef)
    """
    return forward_capsule_from_multmodel(
        multmodel=mult,
        x_l=x_l, x_n=x_n, x_i=x_i,
        projector=projector,
        capsule_head=cap_head,
        route_adapter=route_adapter,
        route_mask=route_mask,
        act_temperature=act_temperature,
        detach_priors=detach_priors,
        return_routing=return_routing,
    )

def capsule_forward_from_encoded(
    *,
    mult,
    route_adapter,
    outL: Dict[str, torch.Tensor],
    outN: Dict[str, torch.Tensor],
    outI: Dict[str, torch.Tensor],
    projector,
    cap_head,
    route_mask=None,
    act_temperature: float = 1.0,
    detach_priors: bool = False,
    return_routing: bool = True,
):
    """
    Build ALL 10 routes using your MulT cross-attention stack:
      L,N,I,LN,NL,LI,IL,NI,IN,LNI
    Then adapt dims -> projector -> capsule head.
    Returns (logits, prim_acts, route_embs, routing_coef)
    """
    z = {"L": outL, "N": outN, "I": outI}

    # 10 routes from encoders (MulT does cross-attn internally with masks/pools)
    route_embs_in = make_route_inputs_mult(z, mult)

    # ensure common d_in for projector/capsule
    if route_adapter is not None:
        route_embs_in = route_adapter(route_embs_in)

    return forward_capsule_from_route_dict(
        route_embs_in=route_embs_in,
        projector=projector,
        capsule_head=cap_head,
        route_mask=route_mask,
        act_temperature=act_temperature,
        detach_priors=detach_priors,
        return_routing=return_routing,
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


def _ensure_encoder_dict(out, name: str, mask_fallback: Optional[torch.Tensor] = None):
    if isinstance(out, dict):
        return out

    if torch.is_tensor(out):
        pool = out
        seq = pool[:, None, :]  # [B,1,D]
        if mask_fallback is None:
            mask = torch.ones(pool.size(0), 1, device=pool.device, dtype=torch.float32)
        else:
            # use first token mask if provided
            if mask_fallback.ndim == 2:
                mask = mask_fallback[:, :1].float()
            else:
                mask = torch.ones(pool.size(0), 1, device=pool.device, dtype=torch.float32)
        return {"seq": seq, "pool": pool, "mask": mask}

    raise TypeError(f"[{name}] unsupported encoder output type: {type(out)}")

def _sanitize_encoder_out(out: Dict[str, torch.Tensor], name: str) -> Dict[str, torch.Tensor]:
    out2 = dict(out)

    if "seq" in out2 and out2["seq"] is not None:
        out2["seq"] = _safe_tensor(_clamp_norm(out2["seq"].float(), 20.0), f"{name}.seq").float()

    if "pool" in out2 and out2["pool"] is not None:
        out2["pool"] = _safe_tensor(_clamp_norm(out2["pool"].float(), 20.0), f"{name}.pool").float()

    if "mask" in out2 and out2["mask"] is not None:
        out2["mask"] = out2["mask"].float()

    return out2


def _has_nonfinite(*tensors: torch.Tensor) -> bool:
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            return True
    return False

@torch.no_grad()
def evaluate_epoch(
    behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
    loader,
    amp_ctx_enc,
    amp_ctx_caps,
    loss_fn,
    act_temperature: float = 1.0,
    detach_priors: bool = False,
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
    act_sum = torch.zeros(N_ROUTES, dtype=torch.float32)  # N_ROUTES=10
    route_names = ROUTE_NAMES

    num_samples = 0
    printed_unimodal = False
    printed_caps_once = False
    rpt_every = int(_cfg("routing_print_every", 0) or 0)

    # per-route, per-phenotype routing importance
    rc_sum_mat = None      # [N_ROUTES, K]
    has_routing = False

    for bidx, (xL, mL, notes, imgs, y, dbg) in enumerate(loader):
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)

        # (A) Encoders under autocast (REAL seq + mask + pool)
        outL, outN, outI = encode_all_modalities(
            behrt, bbert, imgenc,
            xL=xL, mL=mL, notes=notes, imgs=imgs,
            amp_ctx_enc=amp_ctx_enc,
        )

        # (B) Capsule forward + loss in FP32
        with amp_ctx_caps:
            z = {"L": outL, "N": outN, "I": outI}
            route_mask = torch.ones(xL.size(0), N_ROUTES, device=DEVICE, dtype=torch.float32)

            out = capsule_forward_from_encoded(
                mult=mult,
                route_adapter=route_adapter,
                outL=outL, outN=outN, outI=outI,
                projector=projector,
                cap_head=cap_head,
                route_mask=route_mask,
                act_temperature=float(act_temperature),
                detach_priors=bool(detach_priors),
                return_routing=True,
            )


            logits, prim_acts, route_embs = out[0], out[1], out[2]
            routing_coef = out[3] if len(out) > 3 else None

            if routing_coef is not None:
                # DO NOT renormalize here — the capsule head should be the single source of truth.
                assert routing_coef.ndim == 3, \
                    f"routing_coef must be 3D [B,R,K], got {tuple(routing_coef.shape)}"

                assert int(routing_coef.shape[1]) == int(N_ROUTES), \
                    f"routing_coef must be [B,{N_ROUTES},K], got {tuple(routing_coef.shape)}"

                # Optional: just verify it sums to ~1 over routes WITHOUT changing it
                if bidx == 0:
                    debug_routing_tensor(
                        routing_coef,
                        name=f"{split_name}.routing_coef",
                        expect_routes=N_ROUTES,
                        expect_k=int(y.size(1))
                    )

            if routing_coef is not None and bidx == 0:
                quantization_check(routing_coef, name=f"{split_name}.routing_coef")
            if bidx == 0:
                mask_stats(mL, name=f"{split_name}.mL_batch")


            logits    = _safe_tensor(logits.float(),    "eval.logits(fp32)")
            prim_acts = _safe_tensor(prim_acts.float(), "eval.prim_acts(fp32)")

            if routing_coef is not None:
                has_routing = True
                rc = routing_coef.detach().float().cpu()   # [B, N_ROUTES, K]
                rc_mean_batch = rc.mean(dim=0)             # [N_ROUTES, K]
                if rc_sum_mat is None:
                    rc_sum_mat = torch.zeros_like(rc_mean_batch)
                rc_sum_mat += rc_mean_batch * y.size(0)

            # debug prints/heatmaps on first batch
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
                    f"[eval:unimodal] "
                    f"L.pool:{tuple(outL['pool'].shape)} L.seq:{tuple(outL['seq'].shape)} | "
                    f"N.pool:{tuple(outN['pool'].shape)} N.seq:{tuple(outN['seq'].shape)} | "
                    f"I.pool:{tuple(outI['pool'].shape)} I.seq:{tuple(outI['seq'].shape)}"
                )
                pretty_print_small_batch(xL, mL, notes, dbg, k=3)

            if (not printed_caps_once) or (rpt_every > 0 and ((bidx + 1) % rpt_every == 0)):
                printed_caps_once = True
                keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
                print(
                    f"[eval:caps] logits:{tuple(logits.shape)} "
                    f"prim_acts:{tuple(prim_acts.shape)} routes -> {keys}"
                )

                if bidx == 0:
                    route_cosine_report(route_embs)

            loss = loss_fn(logits, y.float())

        total_loss += float(loss.item()) * y.size(0)
        probs = torch.sigmoid(logits)
        pred  = (probs >= 0.5).float()
        total_correct += (pred == y.float()).sum().item()
        total += y.numel()
        num_samples += y.size(0)
        act_sum += prim_acts.detach().float().cpu().sum(dim=0)

    avg_loss = total_loss / max(1, num_samples)
    avg_acc  = total_correct / max(1, total)

    avg_pa = (act_sum / max(1, num_samples)).numpy()  # [10]
    avg_act_dict = {r: float(avg_pa[i]) for i, r in enumerate(route_names)}

    if num_samples > 0 and has_routing and rc_sum_mat is not None:
        avg_rc_mat = (rc_sum_mat / num_samples).numpy()   # [N_ROUTES, K]
        avg_effective_mat = avg_rc_mat * avg_pa[:, None]  # [N_ROUTES, K]
    else:
        avg_rc_mat = None
        avg_effective_mat = None

    return avg_loss, avg_acc, avg_act_dict, avg_rc_mat, avg_effective_mat, avg_pa


def save_checkpoint(path: str, state: Dict):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_checkpoint(path: str, behrt, bbert, imgenc, mult, route_adapter, projector, cap_head, optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    behrt.load_state_dict(ckpt["behrt"])
    bbert.load_state_dict(ckpt["bbert"])
    imgenc.load_state_dict(ckpt["imgenc"])
    mult.load_state_dict(ckpt["mult"])
    route_adapter.load_state_dict(ckpt["route_adapter"])


    projector.load_state_dict(ckpt["projector"])
    cap_head.load_state_dict(ckpt["cap_head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] loaded epoch={ckpt.get('epoch', 0)} val_acc={ckpt.get('val_acc', -1):.4f}")
    return int(ckpt.get("epoch", 0))


@torch.no_grad()
def collect_epoch_logits(
    loader,
    behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
    amp_ctx_enc,   
    amp_ctx_caps,  
    act_temperature: float = 1.0,
    detach_priors: bool = False,
):
    """
    Collect raw logits (NOT sigmoid) and y_true for temperature scaling & calibration.
    Returns:
      y_true: [N,K] float numpy
      logits: [N,K] float numpy
      ids: list
    """
    behrt.eval()
    imgenc.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()

    ys, ls, ids = [], [], []
    for xL, mL, notes, imgs, y, dbg in loader:
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)

        # (A) Encoders under autocast
        outL, outN, outI = encode_all_modalities(
            behrt, bbert, imgenc,
            xL=xL, mL=mL, notes=notes, imgs=imgs,
            amp_ctx_enc=amp_ctx_enc,
        )


        # (B) Capsule in fp32
        with amp_ctx_caps:
            z = {"L": outL, "N": outN, "I": outI}
            route_mask = torch.ones(xL.size(0), N_ROUTES, device=DEVICE, dtype=torch.float32)

            out = capsule_forward_from_encoded(
                mult=mult,
                route_adapter=route_adapter,
                outL=outL, outN=outN, outI=outI,
                projector=projector,
                cap_head=cap_head,
                route_mask=route_mask,
                act_temperature=float(act_temperature),
                detach_priors=bool(detach_priors),
                return_routing=True,
            )
            logits = _safe_tensor(out[0].float(), "collect_logits.logits(fp32)")


        ys.append(y.detach().cpu())
        ls.append(logits.detach().cpu())
        ids += dbg.get("stay_ids", [])

    y_true = torch.cat(ys, dim=0).numpy()
    logits = torch.cat(ls, dim=0).numpy()
    return y_true, logits, ids


def fit_temperature_scalar_from_val(
    val_logits: np.ndarray,
    val_y_true: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.01,
):
    """
    Fit a single scalar temperature T > 0 on VAL by minimizing BCEWithLogitsLoss(logits/T, y).
    Returns float T.
    """
    val_logits_t = torch.tensor(val_logits, dtype=torch.float32, device=DEVICE)
    val_y_t      = torch.tensor(val_y_true, dtype=torch.float32, device=DEVICE)

    # optimize logT so T is always positive
    logT = torch.zeros((), device=DEVICE, requires_grad=True)

    opt = torch.optim.Adam([logT], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    best_T = 1.0
    best_loss = float("inf")

    for _ in range(max_iter):
        opt.zero_grad(set_to_none=True)
        T = torch.exp(logT) + 1e-6
        loss = loss_fn(val_logits_t / T, val_y_t)
        loss.backward()
        opt.step()

        l = float(loss.detach().cpu().item())
        if l < best_loss:
            best_loss = l
            best_T = float((torch.exp(logT).detach().cpu().item()))

    # clamp for sanity
    best_T = float(np.clip(best_T, 0.05, 50.0))
    return best_T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return logits / float(T)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))

@torch.no_grad()
def collect_epoch_outputs(
    loader,
    behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
    amp_ctx_enc,  
    amp_ctx_caps, 
    act_temperature: float = 1.0, 
    detach_priors: bool = False,
):
    behrt.eval()
    imgenc.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()

    y_true, p1, ids = [], [], []
    for xL, mL, notes, imgs, y, dbg in loader:
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)

        outL, outN, outI = encode_all_modalities(
            behrt, bbert, imgenc,
            xL=xL, mL=mL, notes=notes, imgs=imgs,
            amp_ctx_enc=amp_ctx_enc,
        )   



        with amp_ctx_caps:
            z = {"L": outL, "N": outN, "I": outI}
            route_mask = torch.ones(xL.size(0), N_ROUTES, device=DEVICE, dtype=torch.float32)

            out = capsule_forward_from_encoded(
                mult=mult,
                route_adapter=route_adapter,
                outL=outL, outN=outN, outI=outI,
                projector=projector,
                cap_head=cap_head,
                route_mask=route_mask,
                act_temperature=float(act_temperature),
                detach_priors=bool(detach_priors),
                return_routing=True,
            )
            logits = _safe_tensor(out[0].float(), "collect.logits(fp32)")



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

def generate_split_heatmaps_and_tables(
    split_name: str,
    loader,
    behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
    amp_ctx_enc,
    amp_ctx_caps,
    label_names: List[str],
    ckpt_dir: str,
    T_cal: Optional[float] = None,          
    thr_val: Optional[np.ndarray] = None,
):

    """
    Creates & saves (TRAIN/TEST only):
      1) Primary activations heatmap (1xN_ROUTES)  [raw + norm saved/printed]
      2) Effective weights heatmap (KxN_ROUTES)    [raw + norm saved/printed]
      3) Mean routing coeff heatmap (KxN_ROUTES)   [raw + norm saved/printed]
      4) AUROC per label heatmap (1xK)      [raw + norm saved/printed]
      5) Prevalence per label heatmap (1xK) [raw + norm saved/printed]
      6) AUROC+Prevalence combined heatmap (2xK) [raw + norm saved/printed]
    """
    routes = ["L","N","I","LN","NL","LI","IL","NI","IN","LNI"]
    out_dir = os.path.join(ckpt_dir, "heatmaps", split_name.lower())
    os.makedirs(out_dir, exist_ok=True)

    dummy_bce = nn.BCEWithLogitsLoss(reduction="mean")
    loss, acc, act_dict, rc_mat, eff_mat, pa_vec = evaluate_epoch(
        behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
        loader, amp_ctx_enc, amp_ctx_caps, dummy_bce,
        route_debug=False,
        label_names=label_names,
        epoch_idx=None,
        split_name=split_name,
        routing_out_dir=None,
    )

    # pa_vec: [10], eff_mat: [N_ROUTES,K], rc_mat: [10,K]
    if rc_mat is None or eff_mat is None:
        raise RuntimeError(f"[{split_name}] routing matrices are None; routing_coef not returned in forward.")

    K = rc_mat.shape[1]
    rc_k10  = rc_mat.T       # [K,N_ROUTES]
    eff_k10 = eff_mat.T      # [K,N_ROUTES]

    y_true_log, logits, _ = collect_epoch_logits(
        loader, behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
        amp_ctx_enc, amp_ctx_caps
    )

    prev = y_true_log.mean(axis=0)  # [K]

    if T_cal is not None:
        p = sigmoid_np(apply_temperature(logits, float(T_cal)))
    else:
        p = sigmoid_np(logits)

    if thr_val is not None:
        thr_val = np.asarray(thr_val, dtype=np.float32).reshape(-1)
        y_pred = (p >= thr_val[np.newaxis, :]).astype(float)
    else:
        y_pred = (p >= 0.5).astype(float)

    m = epoch_metrics(y_true_log, p, y_pred)

    auroc_per = m["AUROC_per_label"]  # [K] may include NaN

    auroc_vis = np.nan_to_num(auroc_per, nan=0.0).astype(np.float32)
    prev_vis  = np.nan_to_num(prev,      nan=0.0).astype(np.float32)

    pa_norm, _, _ = save_array_with_versions(
        pa_vec, out_dir, f"{split_name.lower()}_primary_activations",
        row_names=None, col_names=routes,
        print_title=f"[{split_name}] PRIMARY ACTIVATIONS (raw + normalized)"
    )
    save_vector_heatmap_with_numbers(
        pa_vec, routes,
        title=f"{split_name} Primary Activations (normalized color, raw numbers)",
        out_path=os.path.join(out_dir, f"{split_name.lower()}_primary_activations.png"),
        fontsize_cell=12, fontsize_ticks=11
    )

    eff_norm, _, _ = save_array_with_versions(
        eff_k10, out_dir, f"{split_name.lower()}_effective_weights_kxN_ROUTES",
        row_names=label_names, col_names=routes,
        print_title=f"[{split_name}] EFFECTIVE WEIGHTS (primary_act × routing_coef) [KxN_ROUTES] (raw + normalized)",
        norm_fn=normalize_routes_per_phenotype, 
    )
    save_heatmap_with_numbers(
        mat_norm=eff_norm,
        mat_raw=eff_k10,
        row_names=label_names,
        col_names=routes,
        title=f"{split_name} Effective Weights (KxN_ROUTES) | normalized color, raw numbers",
        out_path=os.path.join(out_dir, f"{split_name.lower()}_effective_weights_kxN_ROUTES.png"),
        fontsize_cell=6,
        fontsize_ticks=9
    )

    rc_norm, _, _ = save_array_with_versions(
        rc_k10, out_dir, f"{split_name.lower()}_mean_routing_coeff_kx10",
        row_names=label_names, col_names=routes,
        print_title=f"[{split_name}] MEAN ROUTING COEFFICIENT (per-route, per-phenotype) [KxN_ROUTES] (raw + normalized)",
        norm_fn=normalize_routes_per_phenotype,   
    )
    save_heatmap_with_numbers(
        mat_norm=rc_norm,
        mat_raw=rc_k10,
        row_names=label_names,
        col_names=routes,
        title=f"{split_name} Mean Routing Coefficient (KxN_ROUTES) | normalized color, raw numbers",
        out_path=os.path.join(out_dir, f"{split_name.lower()}_mean_routing_coeff_kx10.png"),
        fontsize_cell=6,
        fontsize_ticks=9
    )

    auroc_norm, _, _ = save_array_with_versions(
        auroc_per, out_dir, f"{split_name.lower()}_auroc_per_label",
        row_names=None, col_names=label_names,
        print_title=f"[{split_name}] AUROC per label [1xK] (raw + normalized)"
    )
    save_vector_heatmap_with_numbers(
        auroc_vis, label_names,
        title=f"{split_name} AUROC per Label (normalized color, raw numbers)",
        out_path=os.path.join(out_dir, f"{split_name.lower()}_auroc_per_label.png"),
        fontsize_cell=7, fontsize_ticks=7
    )

    prev_norm, _, _ = save_array_with_versions(
        prev, out_dir, f"{split_name.lower()}_prevalence_per_label",
        row_names=None, col_names=label_names,
        print_title=f"[{split_name}] Prevalence per label [1xK] (raw + normalized)"
    )
    save_vector_heatmap_with_numbers(
        prev_vis, label_names,
        title=f"{split_name} Prevalence per Label (normalized color, raw numbers)",
        out_path=os.path.join(out_dir, f"{split_name.lower()}_prevalence_per_label.png"),
        fontsize_cell=7, fontsize_ticks=7
    )

    combo_raw = np.vstack([auroc_vis.reshape(1, -1), prev_vis.reshape(1, -1)])  # [2,K]
    combo_norm, _, _ = save_array_with_versions(
        combo_raw, out_dir, f"{split_name.lower()}_auroc_and_prevalence_2xk",
        row_names=["AUROC", "Prevalence"], col_names=label_names,
        print_title=f"[{split_name}] AUROC + Prevalence [2xK] (raw + normalized)"
    )
    save_heatmap_with_numbers(
        mat_norm=combo_norm,
        mat_raw=combo_raw,
        row_names=["AUROC", "Prevalence"],
        col_names=label_names,
        title=f"{split_name} AUROC + Prevalence (2xK) | normalized color, raw numbers",
        out_path=os.path.join(out_dir, f"{split_name.lower()}_auroc_prevalence_2xk.png"),
        fontsize_cell=7,
        fontsize_ticks=7
    )

    print(f"\n[{split_name}] Summary:")
    print(f"  loss={loss:.4f} acc@0.5={acc:.4f} AUROC_macro={m['AUROC']:.4f} AUPRC_macro={m['AUPRC']:.4f}")
    return {
        "loss": float(loss),
        "acc": float(acc),
        "AUROC_macro": float(m["AUROC"]),
        "AUPRC_macro": float(m["AUPRC"]),
        "primary_activations": pa_vec.astype(np.float32),
        "effective_kx10": eff_k10.astype(np.float32),
        "routing_coeff_kx10": rc_k10.astype(np.float32),
        "auroc_per_label": auroc_per.astype(np.float32),
        "prevalence_per_label": prev.astype(np.float32),
    }

def main():
    import env_config as E

    E.load_cfg()                    
    args = parse_args()             
    apply_cli_overrides(args)

    global CFG, DEVICE
    CFG = E.CFG
    DEVICE = E.DEVICE

    from env_config import ROUTES

    _expected = ["L", "N", "I", "LN", "NL", "LI", "IL", "NI", "IN", "LNI"]
    assert list(ROUTES) == _expected, f"ROUTES mismatch. expected={_expected} got={list(ROUTES)}"


    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False   

    if hasattr(args, "finetune_text") and args.finetune_text:
        CFG.finetune_text = True

    print("[env_config] Device:", DEVICE)
    print("[env_config] CFG:", json.dumps(asdict(CFG), indent=2))
    global TOKENIZER, MAXLEN
    TOKENIZER = AutoTokenizer.from_pretrained(CFG.text_model_name)
    MAXLEN = int(_cfg("max_text_len", 512))
    print(f"[setup] DEVICE={DEVICE} | batch_size={args.batch_size} | epochs={args.epochs}")


    use_cuda = (str(DEVICE).startswith("cuda") and torch.cuda.is_available())

    precision = str(args.precision).lower()
    use_amp = use_cuda and (precision != "off")

    # Encoder autocast only 
    if use_amp:
        if precision == "fp16":
            amp_ctx_enc = torch_amp.autocast(device_type="cuda", dtype=torch.float16)
        elif precision == "bf16":
            amp_ctx_enc = torch_amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            # "auto" -> let PyTorch decide (often bf16 on A100/H100, fp16 otherwise)
            amp_ctx_enc = torch_amp.autocast(device_type="cuda")
    else:
        amp_ctx_enc = nullcontext()

    amp_ctx_caps = nullcontext()

    from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=(use_amp and precision in {"auto", "fp16"}))
    print(f"[amp] use_amp={use_amp} precision={precision} scaler_enabled={scaler.is_enabled()}")

    # Datasets
    train_ds = ICUStayDataset(args.data_root, split="train")
    tri_ids = set(train_ds.ids)

    # Class-balancing pos_weight from TRAIN (tri-modal TRAIN only)
    train_label_df = (
        train_ds.labels[train_ds.labels["stay_id"].isin(tri_ids)]
        .loc[:, ["stay_id"] + train_ds.label_cols]
        .drop_duplicates(subset=["stay_id"], keep="first")
    )
    N_train = len(train_label_df)

    compute_split_prevalence(
        train_label_df[train_ds.label_cols].values,
        split_name="TRAIN",
        label_names=[get_pheno_name(i) for i in range(train_ds.num_labels)]
    )
    pos_counts = train_label_df[train_ds.label_cols].sum(axis=0).values
    neg_counts = N_train - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)

    # Prevent extreme imbalance from exploding gradients
    max_pw = float(_cfg("pos_weight_max", 20.0))
    pos_weight = np.clip(pos_weight, 1.0, max_pw)
    print(f"[loss] pos_weight: min={pos_weight.min():.3f} max={pos_weight.max():.3f} (clamped to {max_pw})")

    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

    val_ds   = ICUStayDataset(args.data_root, split="val")
    test_ds  = ICUStayDataset(args.data_root, split="test")
    num_phenos = train_ds.num_labels

    raw_label_cols = train_ds.label_cols
    label_names = [get_pheno_name(i) for i in range(num_phenos)]

    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_tensor)
    print("[loss] BCEWithLogitsLoss with per-label pos_weight")

    collate_train = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("train"))
    collate_eval  = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("val"))
    pin = use_cuda

    # Deterministic DataLoader RNGs (shuffle order + worker randomness)
    g_train = torch.Generator()
    g_train.manual_seed(int(CFG.seed) + 123)

    g_eval = torch.Generator()
    g_eval.manual_seed(int(CFG.seed) + 456)


    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_train, drop_last=False,
        worker_init_fn=seed_worker,
        generator=g_train,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_eval,
        worker_init_fn=seed_worker,
        generator=g_eval,
        persistent_workers=(args.num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_eval,
        worker_init_fn=seed_worker,
        generator=g_eval,
        persistent_workers=(args.num_workers > 0),
    )

    train_eval_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_eval,
        worker_init_fn=seed_worker,
        generator=g_eval,
        persistent_workers=(args.num_workers > 0),
    )


    # Encoders
    enc_cfg = EncoderConfig(
        d=_cfg("d", 256), dropout=_cfg("dropout", 0.0),
        structured_seq_len=_cfg("structured_seq_len", 256),
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



    # =========================
    # MULTModel + capsule head
    # =========================

    # Choose MulT output dims (these are MulT internal dims, NOT encoder dims)
    d_l = int(getattr(CFG, "mult_d_l", CFG.d))
    d_n = int(getattr(CFG, "mult_d_n", CFG.d))
    d_i = int(getattr(CFG, "mult_d_i", CFG.d))

    # Original encoder feature dims are CFG.d for all three in your current setup
    orig_d_l = int(CFG.d)
    orig_d_n = int(CFG.d)
    orig_d_i = int(CFG.d)

    mult = MULTModel(
        orig_d_l=orig_d_l, orig_d_n=orig_d_n, orig_d_i=orig_d_i,
        d_l=d_l, d_n=d_n, d_i=d_i,
        ionly=True, nonly=True, lonly=True,
        num_heads=int(getattr(CFG, "mult_num_heads", 8)),
        layers=int(getattr(CFG, "mult_layers", 4)),
        self_layers=int(getattr(CFG, "mult_self_layers", 0)),
        attn_dropout=float(getattr(CFG, "mult_attn_dropout", CFG.dropout)),
        attn_dropout_n=float(getattr(CFG, "mult_attn_dropout_n", CFG.dropout)),
        attn_dropout_i=float(getattr(CFG, "mult_attn_dropout_i", CFG.dropout)),
        relu_dropout=float(getattr(CFG, "mult_relu_dropout", CFG.dropout)),
        res_dropout=float(getattr(CFG, "mult_res_dropout", CFG.dropout)),
        out_dropout=float(getattr(CFG, "mult_out_dropout", CFG.dropout)),
        embed_dropout=float(getattr(CFG, "mult_embed_dropout", 0.0)),
        attn_mask=bool(getattr(CFG, "mult_attn_mask", False)),
    ).to(DEVICE)

    # Projector input dim (shared across routes)
    d_in = int(getattr(CFG, "mult_d_in", d_l))  # common choice: d_in=d_l if you set d_l=d_n=d_i

    route_adapter = RouteDimAdapter(d_in=d_in, d_l=d_l, d_n=d_n, d_i=d_i).to(DEVICE)

    projector = RoutePrimaryProjector(d_in=d_in, pc_dim=CFG.capsule_pc_dim).to(DEVICE)

    cap_head = CapsuleMortalityHead(
        pc_dim=CFG.capsule_pc_dim,
        mc_caps_dim=CFG.capsule_mc_caps_dim,
        num_routing=CFG.capsule_num_routing,
        dp=CFG.dropout,
        act_type=CFG.capsule_act_type,
        layer_norm=CFG.capsule_layer_norm,
        dim_pose_to_vote=CFG.capsule_dim_pose_to_vote,
        num_classes=num_phenos,
    ).to(DEVICE)

    print(
        f"[mult] d_l={d_l} d_n={d_n} d_i={d_i} d_in={d_in} "
        f"| heads={getattr(CFG,'mult_num_heads',8)} layers={getattr(CFG,'mult_layers',4)}"
    )
    print(
        f"[capsule] pc_dim={CFG.capsule_pc_dim} mc_caps_dim={CFG.capsule_mc_caps_dim} "
        f"iters={CFG.capsule_num_routing} act_type={CFG.capsule_act_type} out_caps={num_phenos}"
    )


    encoder_warmup_epochs = int(getattr(args, "encoder_warmup_epochs", _cfg("encoder_warmup_epochs", 2)))

    enc_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []

    # encoders 
    enc_params += [p for p in behrt.parameters() if p.requires_grad]
    enc_params += [p for p in imgenc.parameters() if p.requires_grad]

    if CFG.finetune_text:
        enc_params += [p for p in bbert.parameters() if p.requires_grad]
    else:
        head_params += [p for p in bbert.parameters() if p.requires_grad]


    head_params += [p for p in mult.parameters() if p.requires_grad]
    head_params += [p for p in route_adapter.parameters() if p.requires_grad]
    head_params += [p for p in projector.parameters() if p.requires_grad]
    head_params += [p for p in cap_head.parameters() if p.requires_grad]


    params = enc_params + head_params

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params,  "lr": args.lr, "weight_decay": args.weight_decay, "name": "enc"},
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay, "name": "head"},
        ]
    )

    print(f"[optim] enc_tensors={len(enc_params)} head_tensors={len(head_params)} total={len(params)}")
    print(f"[warmup] encoder_warmup_epochs={encoder_warmup_epochs}")


    # Checkpoint setup
    start_epoch = 0
    best_val_acc = -1.0
    ckpt_dir = os.path.join(args.ckpt_root, "pheno_capsule")
    ensure_dir(ckpt_dir)
    if args.resume and os.path.isfile(args.resume):
        print(f"[main] Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, behrt, bbert, imgenc, mult, route_adapter, projector, cap_head, optimizer)

    printed_once = False

    route_dropout_p       = float(_cfg("route_dropout_p", 0.0))
    routing_warmup_epochs = int(_cfg("routing_warmup_epochs", 5))

    route_entropy_lambda        = float(_cfg("route_entropy_lambda", 0.0))
    route_entropy_warmup_epochs = int(_cfg("route_entropy_warmup_epochs", 0.0))
    entropy_use_rc              = bool(_cfg("entropy_use_rc", False))  # compatibility only

    route_uniform_lambda        = float(_cfg("route_uniform_lambda", 0.0))
    route_uniform_warmup_epochs = int(_cfg("route_uniform_warmup_epochs", 0))

    routing_warmup_epochs        = max(0, int(routing_warmup_epochs))
    route_entropy_warmup_epochs  = max(0.0, float(route_entropy_warmup_epochs))
    route_uniform_warmup_epochs = max(0, route_uniform_warmup_epochs)


    max_train_patients = int(os.environ.get("MIMICIV_MAX_TRAIN_PATIENTS", "-1"))
    seen_patients = 0
    stop_training = False

    # Epoch-level early stopping based on VAL macro AUROC
    best_val_auroc    = -float("inf")
    best_ckpt_auroc   = -float("inf")
    epochs_no_improve = 0

    patience_epochs = int(_cfg("patience_epochs", 5))     
    min_delta       = float(_cfg("min_delta", 1e-4))      
    min_epochs      = int(_cfg("min_epochs", 20))         

    patience_epochs = max(1, patience_epochs)
    min_epochs      = max(0, min_epochs)
    min_delta       = max(0.0, min_delta)
  

    # Training loop
    for epoch in range(start_epoch, args.epochs):

        if stop_training:
            print(f"[debug] Early stop flag set → breaking before epoch {epoch + 1}.")
            break

        enc_lr = 0.0 if (epoch - start_epoch) < encoder_warmup_epochs else args.lr
        optimizer.param_groups[0]["lr"] = enc_lr
        optimizer.param_groups[1]["lr"] = args.lr

        enc_train = (enc_lr > 0.0)
        for p in enc_params:
            p.requires_grad = enc_train

        if epoch in {start_epoch, start_epoch + encoder_warmup_epochs}:
            print(f"[warmup] epoch={epoch+1} enc_lr={enc_lr} head_lr={args.lr} enc_train={enc_train}")

        behrt.train()
        imgenc.train()
        bbert.train()
        if getattr(bbert, "bert", None) is not None:
            if CFG.finetune_text:
                bbert.bert.train()
            else:
                bbert.bert.eval()


        total_loss, total_correct, total = 0.0, 0, 0
        act_sum = torch.zeros(N_ROUTES, dtype=torch.float32)

        num_samples = 0

        for step, (xL, mL, notes, imgs, y, dbg) in enumerate(train_loader):
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

            outL, outN, outI = encode_all_modalities(
                behrt, bbert, imgenc,
                xL=xL, mL=mL, notes=notes, imgs=imgs,
                amp_ctx_enc=amp_ctx_enc,
            )


            if _has_nonfinite(outL.get("pool"), outN.get("pool"), outI.get("pool"),
                              outL.get("seq"),  outN.get("seq"),  outI.get("seq")):
                print(f"[skip] non-finite encoder outputs at epoch={epoch+1} step={step+1} -> skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            with amp_ctx_caps:
                B = xL.size(0)
                route_mask = torch.ones(B, N_ROUTES, device=DEVICE, dtype=torch.float32)

                if route_dropout_p > 0.0:
                    if torch.rand((), device=DEVICE) < route_dropout_p:
                        drop_idx = int(torch.randint(low=0, high=N_ROUTES, size=(1,), device=DEVICE))
                        route_mask[:, drop_idx] = 0.0
                    if (epoch - start_epoch) < 2 and (torch.rand((), device=DEVICE) < route_dropout_p * 0.5):
                        drop_idx2 = int(torch.randint(low=0, high=N_ROUTES, size=(1,), device=DEVICE))
                        route_mask[:, drop_idx2] = 0.0

                
                detach_priors_flag = (epoch - start_epoch) < routing_warmup_epochs
                temp = 2.0 if (epoch - start_epoch) < 2 else 1.0   # or whatever schedule you want

                out = capsule_forward_from_encoded(
                    mult=mult,
                    route_adapter=route_adapter,
                    outL=outL, outN=outN, outI=outI,
                    projector=projector,
                    cap_head=cap_head,
                    route_mask=route_mask,
                    act_temperature=temp,
                    detach_priors=detach_priors_flag,
                    return_routing=True,
                )


                logits, prim_acts, route_embs = out[0], out[1], out[2]
                routing_coef = out[3] if len(out) > 3 else None

                if routing_coef is not None:
                    # no renorm here (CapsuleMortalityHead is the source of truth)
                    assert routing_coef.ndim == 3, \
                        f"routing_coef must be 3D [B,R,K], got {tuple(routing_coef.shape)}"

                    assert int(routing_coef.shape[1]) == int(N_ROUTES), \
                        f"routing_coef must be [B,{N_ROUTES},K], got {tuple(routing_coef.shape)}"

                    if bool(getattr(args, "route_debug", False)) and (epoch == start_epoch) and (step == 0):
                        debug_routing_tensor(
                            routing_coef,
                            name="TRAIN.routing_coef_precheck",
                            expect_routes=N_ROUTES,
                            expect_k=int(y.size(1)),
                        )

                logits    = _safe_tensor(logits.float(),     "logits(fp32)")
                prim_acts = _safe_tensor(prim_acts.float(), "prim_acts(fp32)")


                if _has_nonfinite(logits, prim_acts):
                    print(f"[skip] non-finite capsule outputs at epoch={epoch+1} step={step+1} -> skip")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss = bce(logits, y.float())

                cur_epoch = float(epoch + 1) 

                # Entropy bonus (if warmup_epochs <= 0 => apply ALWAYS)
                if route_entropy_lambda > 0.0 and (route_entropy_warmup_epochs <= 0 or cur_epoch <= route_entropy_warmup_epochs):
                    pa = prim_acts / (prim_acts.sum(dim=1, keepdim=True) + 1e-6)
                    pa = torch.clamp(pa, 1e-6, 1.0)
                    H = -(pa * pa.log()).sum(dim=1).mean()
                    loss = loss - route_entropy_lambda * H

                # Uniform bonus (if warmup_epochs <= 0 => apply ALWAYS)
                if route_uniform_lambda > 0.0 and (route_uniform_warmup_epochs <= 0 or cur_epoch <= route_uniform_warmup_epochs):
                    pa_dist = prim_acts / (prim_acts.sum(dim=1, keepdim=True) + 1e-6)
                    p_mean = pa_dist.mean(dim=0)
                    target = torch.full_like(p_mean, 1.0 / p_mean.numel())
                    uniform_loss = ((p_mean - target) ** 2).sum()
                    loss = loss + route_uniform_lambda * uniform_loss



            grad_clip = float(_cfg("grad_clip", 0.3))
            if not torch.isfinite(loss):
                print(f"[skip] non-finite loss at epoch={epoch+1} step={step+1} -> skip")
                safe_zero_grad(optimizer)
                continue

            trainable_params = [p for p in params if p.requires_grad]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                if not grads_are_finite(trainable_params):
                    print(f"[skip] non-finite grads (AMP) at epoch={epoch+1} step={step+1} -> skip")
                    safe_zero_grad(optimizer)
                    continue

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                if not grads_are_finite(trainable_params):
                    print(f"[skip] non-finite grads at epoch={epoch+1} step={step+1} -> skip")
                    safe_zero_grad(optimizer)
                    continue
                optimizer.step()
            safe_zero_grad(optimizer)
            total_loss += float(loss.item()) * y.size(0)

            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            total_correct += (pred == y.float()).sum().item()
            total += y.numel()

            act_sum += prim_acts.detach().cpu().sum(dim=0)
            num_samples += y.size(0)

            if args.log_every > 0 and ((step + 1) % args.log_every == 0):
                avg_loss_step = total_loss / max(1, num_samples)
                avg_acc_step  = total_correct / max(1, total)
                avg_act = (act_sum / max(1, num_samples)).tolist()

                routes = ROUTE_NAMES[:N_ROUTES]

                msg = (
                    f"[epoch {epoch + 1} step {step + 1}] "
                    f"loss={avg_loss_step:.4f} acc={avg_acc_step:.4f} "
                    f"avg_prim_act=" + " | ".join(
                        f"{r}:{avg_act[i]:.3f}" for i, r in enumerate(routes)
                    )
                )
                if routing_coef is not None:
                    # routing_coef is guaranteed [B,R,K] from the asserts above
                    rc_route_mean = routing_coef.detach().float().mean(dim=(0, 2))  # [R]
                    rc_route_mean = rc_route_mean.cpu().tolist()

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
        train_loss = total_loss / max(1, num_samples)
        train_acc  = total_correct / max(1, total)

        train_avg_act = (act_sum / max(1, num_samples)).tolist()
        print(
            f"[epoch {epoch + 1}] TRAIN loss={train_loss:.4f} acc={train_acc:.4f} "
            f"avg_prim_act={', '.join(f'{a:.3f}' for a in train_avg_act)}"
        )

        # VAL metrics (BCE + 0.5 threshold / F1-based thresholds)
        val_loss, val_acc, val_act, val_rc_mat, val_eff_mat, val_pa = evaluate_epoch(
            behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
            val_loader, amp_ctx_enc, amp_ctx_caps, bce,
            act_temperature=1.0,
            detach_priors=False,
            route_debug=bool(getattr(args, "route_debug", False)),
            label_names=label_names,
            epoch_idx=epoch + 1,
            split_name="VAL",
            routing_out_dir=os.path.join(ckpt_dir, "routing"),
        )

        # Routing importance: global and per-phenotype (VAL) 
        routes = ["L","N","I","LN","NL","LI","IL","NI","IN","LNI"]

        if val_rc_mat is not None:
            # Data-derived mean routing coefficients per route (averaged over phenotypes)
            rc_mean_val = val_rc_mat.mean(axis=1)   # [N_ROUTES]
            print("\n[epoch %d] [VAL] mean routing coefficient per route (data-derived):" % (epoch + 1))
            for i, r in enumerate(routes):
                print(f"  rc_mean_{r} = {rc_mean_val[i]:.4f}")

            # Per-route, per-phenotype routing importance (N_ROUTES x K) from routing coefficients
            K = val_rc_mat.shape[1]
            pheno_names = [get_pheno_name(i) for i in range(K)]

            print("\n[epoch %d] [VAL] per-route, per-phenotype routing importance (mean routing coeff):" % (epoch + 1))
            for i, r in enumerate(routes):
                row = " | ".join(
                    f"{pheno_names[k]}:{val_rc_mat[i, k]:.3f}"
                    for k in range(K)
                )
                print(f"  {r}: {row}")
            if val_eff_mat is not None and val_eff_mat.shape == val_rc_mat.shape:
                print("\n[epoch %d] [VAL] per-route, per-phenotype routing importance (EFFECTIVE = rc × primary_act):" % (epoch + 1))
                for i, r in enumerate(routes):
                    row = " | ".join(
                        f"{pheno_names[k]}:{val_eff_mat[i, k]:.3f}"
                        for k in range(K)
                    )
                    print(f"  {r}: {row}")
            # Heatmap of routing importance on VAL for this epoch
            #mat_val = val_rc_mat.T  # [K, N_ROUTES]

            #plt.figure(figsize=(10, 8))
            #im = plt.imshow(mat_val, aspect="auto")
            #plt.colorbar(im, label="Mean routing coefficient")

            #plt.xticks(range(len(routes)), routes)
            #plt.yticks(range(K), pheno_names, fontsize=6)

            #plt.xlabel("Route")
            #plt.ylabel("Phenotype")
            #plt.title(f"Per-route, per-phenotype routing importance (VAL, epoch {epoch + 1})")
            #plt.tight_layout()

            #fname = os.path.join(ckpt_dir, f"routing_val_mean_rc_epoch{epoch + 1:03d}.png")
            #plt.savefig(fname, dpi=300)
            #plt.close()
            #print(f"[routing] saved VAL per-route-per-phenotype map → {fname}")

        # Collect outputs for full VAL metrics
        y_true, p1, _ = collect_epoch_outputs(
            val_loader, behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
            amp_ctx_enc, amp_ctx_caps,
            act_temperature=1.0,
            detach_priors=False,

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

        # Save checkpoints based on VAL macro AUROC
        val_score = float(m["AUROC"])  # macro AUROC
        is_best = val_score > best_ckpt_auroc
        if is_best:
            best_ckpt_auroc = val_score

        ckpt = {
            "epoch": epoch + 1,
            "behrt": behrt.state_dict(),
            "bbert": bbert.state_dict(),
            "imgenc": imgenc.state_dict(),
            "mult": mult.state_dict(),
            "route_adapter": route_adapter.state_dict(),
            "projector": projector.state_dict(),
            "cap_head": cap_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": float(val_acc),  
            "val_auroc": float(val_score),    
            "best_thr": torch.from_numpy(best_thr.astype(np.float32)),
        }
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), ckpt)
        
        if is_best:
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), ckpt)
            print(f"[epoch {epoch + 1}] Saved BEST checkpoint (VAL AUROC={val_score:.4f} | val_acc@0.5={float(val_acc):.4f})")

    # TEST evaluation using BEST checkpoint
    print("[main] Evaluating BEST checkpoint on TEST...")
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        behrt.load_state_dict(ckpt["behrt"])
        bbert.load_state_dict(ckpt["bbert"])
        imgenc.load_state_dict(ckpt["imgenc"])
        mult.load_state_dict(ckpt["mult"])
        route_adapter.load_state_dict(ckpt["route_adapter"])


        projector.load_state_dict(ckpt["projector"])
        cap_head.load_state_dict(ckpt["cap_head"])

    test_loss, test_acc, test_act, test_rc_mat, test_eff_mat, test_pa = evaluate_epoch(
        behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
        test_loader, amp_ctx_enc, amp_ctx_caps, bce,
        act_temperature=1.0,
        detach_priors=False,
        route_debug=False,
        label_names=label_names,
        split_name="TEST",
    )

    print(
        f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} "
        f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k, v in test_act.items())}"
    )

    # Routing importance: global and per-phenotype (TEST) 
    routes = ["L","N","I","LN","NL","LI","IL","NI","IN","LNI"]

    if test_rc_mat is not None:
        rc_mean_test = test_rc_mat.mean(axis=1)
        print("\n[TEST] mean routing coefficient per route (data-derived):")
        for i, r in enumerate(routes):
            print(f"  rc_mean_{r} = {rc_mean_test[i]:.4f}")

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
        mat_test = test_rc_mat.T  # [K, N_ROUTES]

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
    y_true_v_log, logits_v, _ = collect_epoch_logits(
        val_loader, behrt, bbert, imgenc, mult, route_adapter, projector, cap_head,
        amp_ctx_enc, amp_ctx_caps
    )

    # Fit temperature on VAL only (post-hoc, no training change)
    T_star = fit_temperature_scalar_from_val(logits_v, y_true_v_log, max_iter=300, lr=0.05)
    print(f"[calibration] Fitted temperature on VAL only: T={T_star:.4f}")

    p_v = sigmoid_np(apply_temperature(logits_v, T_star))  # use calibrated probs for threshold search
    prev_val = compute_split_prevalence(y_true_v_log, split_name="VAL", label_names=label_names)
    thr_val, f1_val_thr = grid_search_thresholds(y_true_v_log, p_v, n_steps=101)
    save_split_thresholds(thr_val, ckpt_dir, split_name="VAL")  
    y_true_t_log, logits_t, _ = collect_epoch_logits(
        test_loader, behrt, bbert, imgenc, mult, route_adapter, projector, cap_head, amp_ctx_enc, amp_ctx_caps
    )

    # Uncalibrated + calibrated probabilities on TEST
    p_test_uncal = sigmoid_np(logits_t)
    p_test_cal   = sigmoid_np(apply_temperature(logits_t, T_star))

    # Evaluate TEST using VAL thresholds (frozen) — calibrated probabilities
    y_pred_t = (p_test_cal >= thr_val[np.newaxis, :]).astype(float)
    mt = epoch_metrics(y_true_t_log, p_test_cal, y_pred_t)

    print(
        f"[TEST] (VAL-thresholds, CALIBRATED probs) MACRO  AUROC={mt['AUROC']:.4f} "
        f"AUPRC={mt['AUPRC']:.4f} F1={mt['F1']:.4f} Recall={mt['Recall']:.4f}"
    )

    ece_unc, centers_unc, bconf_unc, bacc_unc, _ = expected_calibration_error(
        p_test_uncal.reshape(-1), y_true_t_log.reshape(-1), n_bins=args.calib_bins
    )
    print(f"[TEST] ECE({args.calib_bins} bins) UNCALIBRATED = {ece_unc:.4f}")
    reliability_plot(centers_unc, bconf_unc, bacc_unc, out_path=os.path.join(ckpt_dir, "reliability_test_uncal.png"))

    ece_cal, centers_cal, bconf_cal, bacc_cal, _ = expected_calibration_error(
        p_test_cal.reshape(-1), y_true_t_log.reshape(-1), n_bins=args.calib_bins
    )   
    print(f"[TEST] ECE({args.calib_bins} bins) CALIBRATED(T from VAL) = {ece_cal:.4f}")
    reliability_plot(centers_cal, bconf_cal, bacc_cal, out_path=os.path.join(ckpt_dir, "reliability_test_cal.png"))

    print("\n[main] Generating FINAL normalized heatmaps (TRAIN + TEST only) from BEST checkpoint...")

    generate_split_heatmaps_and_tables(
        split_name="TRAIN",
        loader=train_eval_loader,
        behrt=behrt, bbert=bbert, imgenc=imgenc,
        mult=mult, route_adapter=route_adapter, projector=projector, cap_head=cap_head,
        amp_ctx_enc=amp_ctx_enc,
        amp_ctx_caps=amp_ctx_caps,
        label_names=label_names,
        ckpt_dir=ckpt_dir,
        T_cal=T_star,         
        thr_val=thr_val,      
    )

    generate_split_heatmaps_and_tables(
        split_name="TEST",
        loader=test_loader,
        behrt=behrt, bbert=bbert, imgenc=imgenc,
        mult=mult, route_adapter=route_adapter, projector=projector, cap_head=cap_head,
        amp_ctx_enc=amp_ctx_enc,
        amp_ctx_caps=amp_ctx_caps,
        label_names=label_names,
        ckpt_dir=ckpt_dir,
        T_cal=T_star,        
        thr_val=thr_val,      
    )

if __name__ == "__main__":
    main()
