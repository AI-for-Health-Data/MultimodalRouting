import os, json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from env_config import CFG, DEVICE, ROUTES, TASKS
from encoders import build_fusions_enc
from train_step1_unimodal import ICUStayDataset, collate_fn

# Ensure fusion modules exist 
def _ensure_fusions_interpret():
    global fusion
    need = {"LN", "LI", "NI", "LNI"}
    if "fusion" in globals() and isinstance(fusion, dict) and need.issubset(fusion.keys()):
        for k in need:
            fusion[k] = fusion[k].to(DEVICE).eval()
        return

    # Build canonical fusions
    fusion = build_fusions_enc(d=CFG.d, p_drop=CFG.dropout, hidden=4 * 256)
    for k in need:
        fusion[k] = fusion[k].to(DEVICE).eval()

    step3_path = os.path.join(CFG.ckpt_root, "step3_trimodal_router.pt")
    if os.path.exists(step3_path):
        try:
            ckpt3 = torch.load(step3_path, map_location=DEVICE)
            if "fusion_LNI" in ckpt3:
                fusion["LNI"].load_state_dict(ckpt3["fusion_LNI"], strict=False)
                print("Loaded fusion['LNI'] weights from step3_trimodal_router.pt")
        except Exception as e:
            print(f"[warn] Could not load fusion_LNI from step3: {e}")

    class _PairwiseFusion(nn.Module):
        def __init__(self, d: int, hidden: int = None, p_drop: float = 0.1):
            super().__init__()
            hidden = hidden or (2 * d)
            self.net = nn.Sequential(
                nn.LayerNorm(2 * d),
                nn.Linear(2 * d, hidden),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden, d),
            )
            self.gate = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.Sigmoid(),
            )
        def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
            h = torch.cat([za, zb], dim=-1)  
            f = self.net(h)                  
            base = (za + zb) / 2.0
            g = self.gate(base)
            return g * f + (1.0 - g) * base

    class _TriFusion(nn.Module):
        def __init__(self, d: int, hidden: int = None, p_drop: float = 0.1):
            super().__init__()
            hidden = hidden or (3 * d)
            self.net = nn.Sequential(
                nn.LayerNorm(3 * d),
                nn.Linear(3 * d, hidden),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden, d),
            )
            self.gate = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.Sigmoid(),
            )
        def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
            h = torch.cat([zL, zN, zI], dim=-1)  
            f = self.net(h)                      
            base = (zL + zN + zI) / 3.0
            g = self.gate(base)
            return g * f + (1.0 - g) * base

    fusion = {
        "LN":  _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE).eval(),
        "LI":  _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE).eval(),
        "NI":  _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE).eval(),
        "LNI": _TriFusion     (d=CFG.d, p_drop=CFG.dropout).to(DEVICE).eval(),
    }

    step3_path = os.path.join(CFG.ckpt_root, "step3_trimodal_router.pt")
    if os.path.exists(step3_path):
        try:
            ckpt3 = torch.load(step3_path, map_location=DEVICE)
            if "fusion_LNI" in ckpt3:
                fusion["LNI"].load_state_dict(ckpt3["fusion_LNI"])
                print("Loaded fusion['LNI'] weights from step3_trimodal_router.pt")
        except Exception:
            pass

_ensure_fusions_interpret()

def build_masks(xL: torch.Tensor, notes_list: List[List[str]], imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)

    mN_list = []
    for notes in notes_list:
        present = 1.0 if (isinstance(notes, list) and any((isinstance(t, str) and len(t.strip()) > 0) for t in notes)) else 0.0
        mN_list.append(present)
    mN = torch.tensor(mN_list, device=xL.device, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        mI_vals = (imgs.abs().flatten(1).sum(dim=1) > 0).float()
    mI = mI_vals.to(xL.device).unsqueeze(1)

    return {"L": mL, "N": mN, "I": mI}

route_index = {r: i for i, r in enumerate(ROUTES)}

# Forward utilities
def embeddings_from_batch(xL: torch.Tensor, notes_list: List[List[str]], imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run encoders and return unimodal embeddings."""
    zL = behrt(xL)
    zN = bbert(notes_list)   
    zI = imgenc(imgs)
    return {"L": zL, "N": zN, "I": zI}

def route_logits_from_embeddings(z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Build logits for all 7 routes using route heads.
    Multi-modal routes use *fusion* embeddings (not raw concat).
    Returns dict r -> [B, C].
    """
    out = {
        "L": route_heads["L"](z["L"]),
        "N": route_heads["N"](z["N"]),
        "I": route_heads["I"](z["I"]),
    }
    # Pairwise fused embeddings
    zLN = fusion["LN"](z["L"], z["N"])
    zLI = fusion["LI"](z["L"], z["I"])
    zNI = fusion["NI"](z["N"], z["I"])
    out["LN"] = route_heads["LN"](zLN)
    out["LI"] = route_heads["LI"](zLI)
    out["NI"] = route_heads["NI"](zNI)

    # Trimodal fused embedding
    zLNI = fusion["LNI"](z["L"], z["N"], z["I"])
    out["LNI"] = route_heads["LNI"](zLNI)
    return out

def eval_f(
    z: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Evaluate the full model:
      returns (final logits, route_w, block_w, block_logits, route_logits).
    """
    route_logits = route_logits_from_embeddings(z)                                 
    ylogits, route_w, block_w, block_logits = router(z, route_logits, masks=masks)  
    return ylogits, route_w, block_w, block_logits, route_logits

# 1. Collect per-sample contributions 
def collect_contributions(loader: DataLoader, limit_batches: int | None = None) -> pd.DataFrame:
    behrt.eval(); bbert.eval(); imgenc.eval()
    for r in ROUTES: route_heads[r].eval()
    for k in ["LN","LI","NI","LNI"]: fusion[k].eval()
    router.eval()

    rows = []
    global_idx = 0

    with torch.no_grad():
        for bidx, (xL, notes_list, imgs, y, sens) in enumerate(tqdm(loader, desc="Collecting contributions", dynamic_ncols=True)):
            if (limit_batches is not None) and (bidx >= limit_batches):
                break

            xL   = xL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y    = y.to(DEVICE)

            z = embeddings_from_batch(xL, notes_list, imgs)
            masks = build_masks(xL, notes_list, imgs)

            # Full forward
            ylogits, route_w, block_w, block_logits, route_logits = eval_f(z, masks)
            probs = torch.sigmoid(ylogits)  # [B, C]

            B = y.size(0)

            # Record per-sample
            for i in range(B):
                rec = {"row_id": int(global_idx)}
                # labels & preds per task
                for t_idx, t_name in enumerate(TASKS):
                    rec[f"y_true__{t_name}"] = float(y[i, t_idx])
                    rec[f"y_prob__{t_name}"] = float(probs[i, t_idx])
                    rec[f"y_logit__{t_name}"] = float(ylogits[i, t_idx])

                    # block-level
                    rec[f"block_w__uni__{t_name}"] = float(block_w[i, t_idx, 0])
                    rec[f"block_w__bi__{t_name}"]  = float(block_w[i, t_idx, 1])
                    rec[f"block_w__tri__{t_name}"] = float(block_w[i, t_idx, 2])

                    rec[f"block_logit__uni__{t_name}"] = float(block_logits[i, 0, t_idx])
                    rec[f"block_logit__bi__{t_name}"]  = float(block_logits[i, 1, t_idx])
                    rec[f"block_logit__tri__{t_name}"] = float(block_logits[i, 2, t_idx])

                    # route-level
                    for r in ROUTES:
                        sr = float(route_logits[r][i, t_idx])                                  
                        wr = float(route_w[i, t_idx, route_index[r]])                          
                        rec[f"route_logit__{r}__{t_name}"] = sr
                        rec[f"route_w__{r}__{t_name}"]     = wr
                        rec[f"route_contrib__{r}__{t_name}"] = sr * wr                        
                rows.append(rec)
                global_idx += 1

    df = pd.DataFrame.from_records(rows)
    return df

# 2. Global summaries 
def global_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Mean absolute route *raw* contribution across all tasks (|s_r^{(c)}|)."""
    out: Dict[str, float] = {}
    for r in ROUTES:
        cols = [f"route_logit__{r}__{t}" for t in TASKS]
        if all(c in df.columns for c in cols):
            vals = df[cols].to_numpy()
            out[r] = float(np.nanmean(np.abs(vals)))
        else:
            out[r] = float("nan")
    return out

def global_summary_weighted(df: pd.DataFrame) -> Dict[str, float]:
    """Mean absolute *weighted* route contribution |w_r^{(c)} * s_r^{(c)}|."""
    out: Dict[str, float] = {}
    for r in ROUTES:
        cols = [f"route_contrib__{r}__{t}" for t in TASKS]
        if all(c in df.columns for c in cols):
            vals = df[cols].to_numpy()
            out[r] = float(np.nanmean(np.abs(vals)))
        else:
            out[r] = float("nan")
    return out

# 3. UC / BI / TI decomposition 
@torch.no_grad()
def compute_dataset_means(loader: DataLoader, max_batches: int | None = 8) -> Dict[str, torch.Tensor]:
    """Compute dataset-mean embeddings μL, μN, μI to approximate expectations."""
    behrt.eval(); bbert.eval(); imgenc.eval()
    sum_L = None; sum_N = None; sum_I = None
    total = 0

    for bidx, (xL, notes_list, imgs, y, sens) in enumerate(tqdm(loader, desc="Computing μL, μN, μI", dynamic_ncols=True)):
        if (max_batches is not None) and (bidx >= max_batches):
            break
        xL   = xL.to(DEVICE)
        imgs = imgs.to(DEVICE)

        z = embeddings_from_batch(xL, notes_list, imgs)
        B = z["L"].size(0)
        total += B

        if sum_L is None:
            sum_L = z["L"].sum(dim=0)
            sum_N = z["N"].sum(dim=0)
            sum_I = z["I"].sum(dim=0)
        else:
            sum_L += z["L"].sum(dim=0)
            sum_N += z["N"].sum(dim=0)
            sum_I += z["I"].sum(dim=0)

    if total == 0:
        d = CFG.d
        device = torch.device(DEVICE)
        return {"L": torch.zeros(d, device=device), "N": torch.zeros(d, device=device), "I": torch.zeros(d, device=device)}

    mu_L = (sum_L / total).unsqueeze(0)  
    mu_N = (sum_N / total).unsqueeze(0)
    mu_I = (sum_I / total).unsqueeze(0)
    return {"L": mu_L, "N": mu_N, "I": mu_I}

def _repeat_like(mu: torch.Tensor, B: int) -> torch.Tensor:
    return mu.expand(B, -1)  # [B, d]

@torch.no_grad()
def uc_bi_ti_for_batch(xL: torch.Tensor, notes_list: List[List[str]], imgs: torch.Tensor,
                       mus: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns per-sample, per-task (UC, BI, TI, F_full) as tensors [B, C].
    Uses μL, μN, μI for expectations. Masks are set to 1 for present, 0 for μ if you prefer;
    here we just pass masks=ones to focus on interaction in logit space (consistent gating).
    """
    # Prepare
    xL   = xL.to(DEVICE)
    imgs = imgs.to(DEVICE)
    z    = embeddings_from_batch(xL, notes_list, imgs)
    B    = z["L"].size(0)

    # mean embeddings tiled to batch
    muL = _repeat_like(mus["L"].to(DEVICE), B)
    muN = _repeat_like(mus["N"].to(DEVICE), B)
    muI = _repeat_like(mus["I"].to(DEVICE), B)

    ones_mask = {"L": torch.ones(B, 1, device=DEVICE),
                 "N": torch.ones(B, 1, device=DEVICE),
                 "I": torch.ones(B, 1, device=DEVICE)}

    # Define a small helper to evaluate f for given (L,N,I)
    def F(zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
        zz = {"L": zL, "N": zN, "I": zI}
        ylogits, _, _, _, _ = eval_f(zz, ones_mask)  
        return ylogits

    # Full and baselines
    F_full = F(z["L"], z["N"], z["I"])         
    F_mmm  = F(muL,   muN,   muI)              

    # Unimodal terms with others at μ
    F_Lmm = F(z["L"], muN,    muI)             
    F_mNm = F(muL,    z["N"], muI)             
    F_mmI = F(muL,    muN,    z["I"])          

    # Pairwise terms (others at μ)
    F_LNm = F(z["L"], z["N"], muI)             
    F_LmI = F(z["L"], muN,    z["I"])          
    F_mNI = F(muL,    z["N"], z["I"])          

    # UC, BI, TI (per-task, per-sample)
    UC = F_Lmm + F_mNm + F_mmI - 2.0 * F_mmm

    BI_TV = F_LNm - F_Lmm - F_mNm + F_mmm
    BI_TA = F_LmI - F_Lmm - F_mmI + F_mmm
    BI_VA = F_mNI - F_mNm - F_mmI + F_mmm
    BI = BI_TV + BI_TA + BI_VA

    TI = F_full - UC - BI

    return UC, BI, TI, F_full

@torch.no_grad()
def collect_uc_bi_ti(loader: DataLoader, mus: Dict[str, torch.Tensor], limit_batches: int | None = None) -> pd.DataFrame:
    """Compute UC/BI/TI per sample and task; return a tidy DataFrame."""
    for k in ["LN","LI","NI","LNI"]: fusion[k].eval()
    rows = []
    gid = 0
    for bidx, (xL, notes_list, imgs, y, sens) in enumerate(tqdm(loader, desc="UC/BI/TI", dynamic_ncols=True)):
        if (limit_batches is not None) and (bidx >= limit_batches):
            break
        xL   = xL.to(DEVICE)
        imgs = imgs.to(DEVICE)
        y    = y.to(DEVICE)

        UC, BI, TI, F_full = uc_bi_ti_for_batch(xL, notes_list, imgs, mus)
        B, _ = F_full.shape
        for i in range(B):
            rec = {"row_id": int(gid)}
            for t_idx, t_name in enumerate(TASKS):
                rec[f"UC__{t_name}"] = float(UC[i, t_idx])
                rec[f"BI__{t_name}"] = float(BI[i, t_idx])
                rec[f"TI__{t_name}"] = float(TI[i, t_idx])
                rec[f"F__{t_name}"]  = float(F_full[i, t_idx])
                rec[f"y_true__{t_name}"] = float(y[i, t_idx])
            rows.append(rec)
            gid += 1
    return pd.DataFrame.from_records(rows)

ROOT = os.path.join(CFG.data_root, "MIMIC-IV")  # or INSPECT
test_ds = ICUStayDataset(ROOT, split="test")
test_loader = DataLoader(
    test_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    collate_fn=collate_fn,
    pin_memory=True
)

# 1. Collect route-level contributions (logits, weights, contributions)
df_contrib = collect_contributions(test_loader)
print("Contribution columns:", len(df_contrib.columns), "rows:", len(df_contrib))
display_cols = [c for c in df_contrib.columns if c.startswith("route_logit__")][:9]  
print(df_contrib[["row_id"] + display_cols].head(3).to_string(index=False))

# 2. Global summaries
gs_raw = global_summary(df_contrib)
gs_w   = global_summary_weighted(df_contrib)
print("\nGlobal mean |route logits| per route:")
print(json.dumps(gs_raw, indent=2))
print("\nGlobal mean |weighted route contributions| per route:")
print(json.dumps(gs_w, indent=2))

# 3) UC / BI / TI (using dataset means for expectations)
mus = compute_dataset_means(test_loader, max_batches=8)
df_inter = collect_uc_bi_ti(test_loader, mus, limit_batches=None)
print("\nUC/BI/TI (first 5 rows):")
print(df_inter.head().to_string(index=False))

# Aggregate UC/BI/TI per task
agg = {}
for t in TASKS:
    agg[f"UC_mean__{t}"] = float(df_inter[f"UC__{t}"].mean())
    agg[f"BI_mean__{t}"] = float(df_inter[f"BI__{t}"].mean())
    agg[f"TI_mean__{t}"] = float(df_inter[f"TI__{t}"].mean())
print("\nMean UC/BI/TI by task:")
print(json.dumps(agg, indent=2))
