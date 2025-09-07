from __future__ import annotations

import os, json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from env_config import CFG, DEVICE, ROUTES, BLOCKS, TASKS
from encoders import EncoderConfig, build_encoders
from routing_and_heads import (
    build_fusions,            
    RouteGateNet,             
    FinalConcatHead,          
    make_route_inputs,        
    concat_routes,            
)

from train_step1_unimodal import ICUStayDataset, collate_fn


def _build_stack():
    # Encoders
    behrt, bbert, imgenc = build_encoders(
        EncoderConfig(
            d=CFG.d,
            dropout=CFG.dropout,
            structured_seq_len=CFG.structured_seq_len,
            structured_n_feats=CFG.structured_n_feats,
            text_model_name=CFG.text_model_name,
            text_max_len=CFG.max_text_len,
            note_agg="mean",
            max_notes_concat=8,
            img_agg="last",
        ),
        device=DEVICE,
    )

    # Fusion blocks
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)

    # Embedding-level router (7 per-route gates) + Final head (MLP over concat of 7*d)
    gate = RouteGateNet(d=CFG.d, hidden=4 * 256, p_drop=CFG.dropout, use_masks=True).to(DEVICE)
    final_head = FinalConcatHead(d=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE) 

    return behrt, bbert, imgenc, fusion, gate, final_head


def _load_checkpoints(
    behrt, bbert, imgenc, fusion, gate: RouteGateNet, final_head: FinalConcatHead, task: Optional[str] = None
) -> None:
    tname = task or getattr(CFG, "task_name", "mort")

    ckpt1 = os.path.join(CFG.ckpt_root, f"{tname}_step1_unimodal.pt")
    ckpt2 = os.path.join(CFG.ckpt_root, f"{tname}_step2_bimodal.pt")
    ckpt3 = os.path.join(CFG.ckpt_root, f"{tname}_step3_concat_gate.pt")

    if os.path.exists(ckpt1):
        s1 = torch.load(ckpt1, map_location=DEVICE)
        behrt.load_state_dict(s1["behrt"], strict=False)
        bbert.load_state_dict(s1["bbert"], strict=False)
        imgenc.load_state_dict(s1["imgenc"], strict=False)
        print(f"Loaded step1 encoders from {ckpt1}")
    else:
        print(f"[warn] step1 not found: {ckpt1}")

    if os.path.exists(ckpt2):
        s2 = torch.load(ckpt2, map_location=DEVICE)
        for k in ["fusion_LN", "fusion_LI", "fusion_NI"]:
            name = k.split("_", 1)[1]
            if k in s2 and name in fusion:
                try:
                    fusion[name].load_state_dict(s2[k], strict=False)
                    print(f"Loaded fusion['{name}'] from {ckpt2}")
                except Exception as e:
                    print(f"[warn] couldn't load {k} → fusion['{name}']: {e}")
    else:
        print(f"[warn] step2 not found: {ckpt2}")

    if os.path.exists(ckpt3):
        s3 = torch.load(ckpt3, map_location=DEVICE)
        key_gate = "gate" if "gate" in s3 else ("gate_net" if "gate_net" in s3 else None)
        if key_gate:
            gate.load_state_dict(s3[key_gate], strict=False)
            print(f"Loaded gate ({key_gate}) from {ckpt3}")
        else:
            print(f"[warn] no gate weights found in {ckpt3} (expected 'gate' or 'gate_net')")

        if "final_head" in s3:
            final_head.load_state_dict(s3["final_head"], strict=False)
            print(f"Loaded final_head from {ckpt3}")
        else:
            print(f"[warn] no final_head weights found in {ckpt3}")

        if "fusion_LNI" in s3 and "LNI" in fusion:
            try:
                fusion["LNI"].load_state_dict(s3["fusion_LNI"], strict=False)
                print(f"Loaded fusion['LNI'] from {ckpt3}")
            except Exception as e:
                print(f"[warn] couldn't load fusion_LNI: {e}")
    else:
        print(f"[warn] step3 not found: {ckpt3}")


def build_masks(xL: torch.Tensor, notes_list, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)

    # Notes present if any non-empty note
    present_N = [
        1.0 if (isinstance(notes, list) and any((isinstance(t, str) and len(t.strip()) > 0) for t in notes)) else 0.0
        for notes in notes_list
    ]
    mN = torch.tensor(present_N, device=xL.device, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        mI_vals = (imgs.abs().flatten(1).sum(dim=1) > 0).float()
    mI = mI_vals.to(xL.device).unsqueeze(1)

    return {"L": mL, "N": mN, "I": mI}


@torch.no_grad()
def embeddings_from_batch(behrt, bbert, imgenc, xL, notes_list, imgs) -> Dict[str, torch.Tensor]:
    zL = behrt(xL)
    zN = bbert(notes_list)
    zI = imgenc(imgs)
    return {"L": zL, "N": zN, "I": zI}


@torch.no_grad()
def forward_full(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
    gate: RouteGateNet,
    final_head: FinalConcatHead,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False,
):

    routes_raw = make_route_inputs(z, fusion)                      
    gates = gate(z, masks=masks)                                   
    x_cat, routes_weighted = concat_routes(routes_raw, gates=gates, l2norm=l2norm_each)
    ylogits = final_head(x_cat)                                    
    return ylogits, gates, routes_raw, routes_weighted


@torch.no_grad()
def route_contributions_occlusion(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
    gate: RouteGateNet,
    final_head: FinalConcatHead,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    routes_raw = make_route_inputs(z, fusion)          
    gates_full = gate(z, masks=masks)                  

    x_full, _ = concat_routes(routes_raw, gates=gates_full, l2norm=l2norm_each)
    y_full = final_head(x_full)                        

    contribs: Dict[str, torch.Tensor] = {}
    for ri, r in enumerate(ROUTES):
        gates_wo = gates_full.clone()
        gates_wo[:, ri] = 0.0                         
        x_wo, _ = concat_routes(routes_raw, gates=gates_wo, l2norm=l2norm_each)
        y_wo = final_head(x_wo)
        contribs[r] = y_full - y_wo                   
    return y_full, contribs


def block_weights_from_gates(gates: torch.Tensor) -> torch.Tensor:
    device = gates.device
    idx = {r: i for i, r in enumerate(ROUTES)}
    uni = gates[:, [idx["L"], idx["N"], idx["I"]]].sum(dim=1, keepdim=True)
    bi  = gates[:, [idx["LN"], idx["LI"], idx["NI"]]].sum(dim=1, keepdim=True)
    tri = gates[:, [idx["LNI"]]].sum(dim=1, keepdim=True)
    W = torch.cat([uni, bi, tri], dim=1)               
    denom = (W.sum(dim=1, keepdim=True) + 1e-12)
    return W / denom



def global_mean_abs_contrib(contrib_frame: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in ROUTES:
        col = f"route_contrib__{r}"
        if col in contrib_frame.columns:
            out[r] = float(np.nanmean(np.abs(contrib_frame[col].values)))
        else:
            out[r] = float("nan")
    return out


@torch.no_grad()
def compute_dataset_means(
    loader: DataLoader,
    behrt, bbert, imgenc,
    max_batches: Optional[int] = 8
) -> Dict[str, torch.Tensor]:
    behrt.eval(); bbert.eval(); imgenc.eval()
    sum_L = None; sum_N = None; sum_I = None; total = 0
    for bidx, (xL, notes_list, imgs, y, sens) in enumerate(tqdm(loader, desc="Estimating μL,μN,μI", dynamic_ncols=True)):
        if (max_batches is not None) and (bidx >= max_batches):
            break
        xL = xL.to(DEVICE); imgs = imgs.to(DEVICE)
        z = embeddings_from_batch(behrt, bbert, imgenc, xL, notes_list, imgs)
        B = z["L"].size(0); total += B
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
        zero = torch.zeros(d, device=DEVICE)
        return {"L": zero[None, :], "N": zero[None, :], "I": zero[None, :]}
    return {
        "L": (sum_L / total)[None, :],
        "N": (sum_N / total)[None, :],
        "I": (sum_I / total)[None, :],
    }


@torch.no_grad()
def F_embed(
    zL, zN, zI,
    fusion, gate: RouteGateNet, final_head: FinalConcatHead,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False
) -> torch.Tensor:
    z = {"L": zL, "N": zN, "I": zI}
    ylogits, _, _, _ = forward_full(z, fusion, gate, final_head, masks=masks, l2norm_each=l2norm_each)
    return ylogits  # [B,1]


@torch.no_grad()
def uc_bi_ti_for_batch(
    xL: torch.Tensor, notes_list, imgs: torch.Tensor,
    behrt, bbert, imgenc,
    fusion, gate: RouteGateNet, final_head: FinalConcatHead,
    mus: Dict[str, torch.Tensor],
    l2norm_each: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xL = xL.to(DEVICE); imgs = imgs.to(DEVICE)
    z  = embeddings_from_batch(behrt, bbert, imgenc, xL, notes_list, imgs)
    B  = z["L"].size(0)

    ones = {k: torch.ones(B, 1, device=DEVICE) for k in ["L", "N", "I"]}

    muL = mus["L"].expand(B, -1)
    muN = mus["N"].expand(B, -1)
    muI = mus["I"].expand(B, -1)

    F_full = F_embed(z["L"], z["N"], z["I"], fusion, gate, final_head, masks=ones, l2norm_each=l2norm_each)
    F_mmm  = F_embed(muL,   muN,   muI,   fusion, gate, final_head, masks=ones, l2norm_each=l2norm_each)

    # Unimodal contrasts
    F_Lmm = F_embed(z["L"], muN,   muI,   fusion, gate, final_head, masks=ones, l2norm_each=l2norm_each)
    F_mNm = F_embed(muL,   z["N"], muI,   fusion, gate, final_head, masks=ones, l2norm_each=l2norm_each)
    F_mmI = F_embed(muL,   muN,   z["I"], fusion, gate, final_head, masks=ones, l2norm_each=l2norm_each)

    # Pairwise contrasts
    F_LNm = F_embed(z["L"], z["N"], muI,  fusion, gate, final_head, masks=ones, l2norm_each=l2norm_each)
    F_LmI = F_embed(z["L"], muN,   z["I"], fusion, gate, final_head, masks=ones, l2norm_each=l2norm_each)
    F_mNI = F_embed(muL,   z["N"], z["I"], fusion, gate, final_head, masks=ones, l2norm_each=l2norm_each)

    # Decomposition 
    UC = F_Lmm + F_mNm + F_mmI - 2.0 * F_mmm

    BI_TV = F_LNm - F_Lmm - F_mNm + F_mmm
    BI_TA = F_LmI - F_Lmm - F_mmI + F_mmm
    BI_VA = F_mNI - F_mNm - F_mmI + F_mmm
    BI = BI_TV + BI_TA + BI_VA

    TI = F_full - UC - BI
    return UC, BI, TI, F_full

def collect_contributions(
    loader: DataLoader,
    behrt, bbert, imgenc, fusion, gate: RouteGateNet, final_head: FinalConcatHead,
    limit_batches: Optional[int] = None,
    l2norm_each: bool = False,
) -> pd.DataFrame:
    behrt.eval(); bbert.eval(); imgenc.eval()
    for k in ["LN", "LI", "NI", "LNI"]:
        fusion[k].eval()
    gate.eval(); final_head.eval()

    rows = []
    gid = 0
    with torch.no_grad():
        for bidx, (xL, notes_list, imgs, y, sens) in enumerate(tqdm(loader, desc="Collecting local contributions", dynamic_ncols=True)):
            if (limit_batches is not None) and (bidx >= limit_batches):
                break
            xL = xL.to(DEVICE); imgs = imgs.to(DEVICE); y = y.to(DEVICE)
            z = embeddings_from_batch(behrt, bbert, imgenc, xL, notes_list, imgs)
            masks = build_masks(xL, notes_list, imgs)

            # Full forward + gates
            y_full, gates, routes_raw, routes_weighted = forward_full(z, fusion, gate, final_head, masks=masks, l2norm_each=l2norm_each)
            probs = torch.sigmoid(y_full)                     
            B = y.size(0)

            _, contribs = route_contributions_occlusion(z, fusion, gate, final_head, masks=masks, l2norm_each=l2norm_each)

            bw = block_weights_from_gates(gates)            

            for i in range(B):
                rec = {"row_id": int(gid)}
                rec["y_true"]  = float(y[i, 0]) if y.dim() == 2 else float(y[i])
                rec["y_prob"]  = float(probs[i, 0])
                rec["y_logit"] = float(y_full[i, 0])

                # Block weights
                rec["block_w__uni"] = float(bw[i, 0])
                rec["block_w__bi"]  = float(bw[i, 1])
                rec["block_w__tri"] = float(bw[i, 2])

                # Per-route gate and contribution (logit-space Δ)
                idx = {r: k for k, r in enumerate(ROUTES)}
                for r in ROUTES:
                    rec[f"gate__{r}"] = float(gates[i, idx[r]])
                    rec[f"route_contrib__{r}"] = float(contribs[r][i, 0])
                    rec[f"route_emb_norm__{r}"] = float(routes_raw[r][i].norm(p=2).item())

                rows.append(rec); gid += 1

    return pd.DataFrame.from_records(rows)


def collect_uc_bi_ti(
    loader: DataLoader,
    behrt, bbert, imgenc, fusion, gate: RouteGateNet, final_head: FinalConcatHead,
    mus: Dict[str, torch.Tensor],
    limit_batches: Optional[int] = None,
    l2norm_each: bool = False,
) -> pd.DataFrame:
    for k in ["LN", "LI", "NI", "LNI"]:
        fusion[k].eval()
    gate.eval(); final_head.eval()

    rows = []
    gid = 0
    with torch.no_grad():
        for bidx, (xL, notes_list, imgs, y, sens) in enumerate(tqdm(loader, desc="UC/BI/TI", dynamic_ncols=True)):
            if (limit_batches is not None) and (bidx >= limit_batches):
                break
            xL = xL.to(DEVICE); imgs = imgs.to(DEVICE); y = y.to(DEVICE)

            UC, BI, TI, F_full = uc_bi_ti_for_batch(
                xL, notes_list, imgs, behrt, bbert, imgenc,
                fusion, gate, final_head, mus, l2norm_each=l2norm_each,
            )
            B = F_full.size(0)
            for i in range(B):
                rec = {
                    "row_id": int(gid),
                    "UC": float(UC[i, 0]),
                    "BI": float(BI[i, 0]),
                    "TI": float(TI[i, 0]),
                    "F":  float(F_full[i, 0]),
                    "y_true": float(y[i, 0]) if y.dim() == 2 else float(y[i]),
                }
                rows.append(rec); gid += 1
    return pd.DataFrame.from_records(rows)



if __name__ == "__main__":
    behrt, bbert, imgenc, fusion, gate, final_head = _build_stack()
    _load_checkpoints(behrt, bbert, imgenc, fusion, gate, final_head, task=getattr(CFG, "task_name", "mort"))

    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    test_ds = ICUStayDataset(ROOT, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    # Local contributions (per sample)
    df_contrib = collect_contributions(test_loader, behrt, bbert, imgenc, fusion, gate, final_head)
    print(f"[contrib] cols={len(df_contrib.columns)}, rows={len(df_contrib)}")
    print(df_contrib.head(3).to_string(index=False))

    # Global summary of |Δ| per route
    gs = global_mean_abs_contrib(df_contrib)
    print("\nGlobal mean |Δ(logit)| per route:")
    print(json.dumps(gs, indent=2))

    # UC/BI/TI with dataset-mean embeddings (expectation baselines)
    mus = compute_dataset_means(test_loader, behrt, bbert, imgenc, max_batches=8)
    df_inter = collect_uc_bi_ti(test_loader, behrt, bbert, imgenc, fusion, gate, final_head, mus)
    print("\nUC/BI/TI (first 5 rows):")
    print(df_inter.head().to_string(index=False))

    print("\nMean UC/BI/TI:")
    print(json.dumps({
        "UC_mean": float(df_inter["UC"].mean()),
        "BI_mean": float(df_inter["BI"].mean()),
        "TI_mean": float(df_inter["TI"].mean()),
    }, indent=2))
