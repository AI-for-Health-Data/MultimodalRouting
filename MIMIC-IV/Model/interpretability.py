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
    build_route_heads,
    RouteGateNet,
    FinalConcatHead,
    make_route_inputs,
    concat_routes,
    route_availability_mask,
)
from train_step1_unimodal import ICUStayDataset, collate_fn_factory

TASK_MAP = {"mort": 0, "pe": 1, "ph": 2}

def _build_stack():
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
    fusion       = build_fusions(d=CFG.d, p_drop=CFG.dropout)
    route_heads  = build_route_heads(d=CFG.d, n_tasks=1, p_drop=CFG.dropout)
    gate         = RouteGateNet(d=CFG.d, hidden=4 * 256, p_drop=CFG.dropout, use_masks=True).to(DEVICE)
    final_head   = FinalConcatHead(d=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE)
    return behrt, bbert, imgenc, fusion, route_heads, gate, final_head


def _load_checkpoints(
    behrt, bbert, imgenc, fusion, route_heads, gate: RouteGateNet, final_head: FinalConcatHead,
    task: Optional[str] = None
) -> Tuple[str, float, bool]:
    tname = task or getattr(CFG, "task_name", "mort")

    ckpt1 = os.path.join(CFG.ckpt_root, f"{tname}_step1_unimodal.pt")
    ckpt2 = os.path.join(CFG.ckpt_root, f"{tname}_step2_bimodal.pt")
    ckpt3 = os.path.join(CFG.ckpt_root, f"{tname}_step3_concat_gate.pt")

    # Step 1 (encoders + uni heads)
    if os.path.exists(ckpt1):
        s1 = torch.load(ckpt1, map_location=DEVICE)
        behrt.load_state_dict(s1["behrt"], strict=False)
        bbert.load_state_dict(s1["bbert"], strict=False)
        imgenc.load_state_dict(s1["imgenc"], strict=False)
        for r in ["L", "N", "I"]:
            if r in s1 and r in route_heads:
                route_heads[r].load_state_dict(s1[r], strict=False)
        print(f"Loaded step1 encoders/uni heads from {ckpt1}")
    else:
        print(f"[warn] step1 not found: {ckpt1}")

    # Step 2 (pairwise fusion + bi heads)
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
        for r in ["LN", "LI", "NI"]:
            if r in s2 and r in route_heads:
                route_heads[r].load_state_dict(s2[r], strict=False)
        print(f"Loaded step2 bi heads from {ckpt2}")
    else:
        print(f"[warn] step2 not found: {ckpt2}")

    # Step 3 (gate mode, alpha, optional LNI, final head)
    gate_mode = "learned"
    loss_gate_alpha = 4.0
    l2norm_each = False

    if os.path.exists(ckpt3):
        s3 = torch.load(ckpt3, map_location=DEVICE)
        gate_mode       = s3.get("gate_mode", gate_mode)
        loss_gate_alpha = float(s3.get("loss_gate_alpha", loss_gate_alpha))
        l2norm_each     = bool(s3.get("l2norm_each", l2norm_each))

        key_gate = "gate" if "gate" in s3 else ("gate_net" if "gate_net" in s3 else None)
        if key_gate:
            try:
                gate.load_state_dict(s3[key_gate], strict=False)
                print(f"Loaded gate ({key_gate}) from {ckpt3}")
            except Exception as e:
                print(f"[warn] couldn't load {key_gate}: {e}")
        else:
            print(f"[info] no learned gate in {ckpt3} (expected for gate_mode='{gate_mode}')")

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

        if "LNI_head" in s3 and "LNI" in route_heads:
            try:
                route_heads["LNI"].load_state_dict(s3["LNI_head"], strict=False)
                print(f"Loaded route_heads['LNI'] from {ckpt3}")
            except Exception as e:
                print(f"[warn] couldn't load LNI_head: {e}")
    else:
        print(f"[warn] step3 not found: {ckpt3}")

    return gate_mode, loss_gate_alpha, l2norm_each


def build_masks(xL: torch.Tensor, notes_list, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)
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
def _compute_gates(
    gate_mode: str,
    routes_emb: Dict[str, torch.Tensor],
    route_heads: Dict[str, nn.Module],
    gate: RouteGateNet,
    z_unimodal: Dict[str, torch.Tensor],
    y: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    loss_gate_alpha: float,
) -> torch.Tensor:
    B = y.size(0)
    avail = route_availability_mask(masks, batch_size=B, device=y.device)
    if gate_mode == "uniform":
        return avail / (avail.sum(dim=1, keepdim=True).clamp_min(1.0))
    if gate_mode == "learned":
        g = gate(z_unimodal, masks=masks)
        return g / (g.sum(dim=1, keepdim=True).clamp_min(1e-6))
    # loss_based
    per_route_losses = []
    for r in ROUTES:
        logits_r = route_heads[r](routes_emb[r])
        l_i = F.binary_cross_entropy_with_logits(logits_r, y, reduction="none").squeeze(1)
        per_route_losses.append(l_i)
    Lmat = torch.stack(per_route_losses, dim=1)  
    masked_logits = (-float(loss_gate_alpha) * Lmat) + torch.log(avail + 1e-12)
    return torch.softmax(masked_logits, dim=1)
    
@torch.no_grad()
def forward_full(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
    route_heads: Dict[str, nn.Module],
    gate: RouteGateNet,
    final_head: FinalConcatHead,
    y: torch.Tensor,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False,
    gate_mode: str = "learned",
    loss_gate_alpha: float = 4.0,
):
    routes_raw = make_route_inputs(z, fusion)
    if masks is None:
        B = next(iter(z.values())).size(0)
        masks = {"L": torch.ones(B, 1, device=DEVICE), "N": torch.ones(B, 1, device=DEVICE), "I": torch.ones(B, 1, device=DEVICE)}
    gates = _compute_gates(gate_mode, routes_raw, route_heads, gate, z, y, masks, loss_gate_alpha)
    x_cat, routes_weighted = concat_routes(routes_raw, gates=gates, l2norm=l2norm_each)
    ylogits = final_head(x_cat)
    return ylogits, gates, routes_raw, routes_weighted


@torch.no_grad()
def route_contributions_occlusion(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
    route_heads: Dict[str, nn.Module],
    gate: RouteGateNet,
    final_head: FinalConcatHead,
    y: torch.Tensor,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False,
    gate_mode: str = "learned",
    loss_gate_alpha: float = 4.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    routes_raw = make_route_inputs(z, fusion)
    gates_full = _compute_gates(gate_mode, routes_raw, route_heads, gate, z, y, (masks or {}), loss_gate_alpha)

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
        out[r] = float(np.nanmean(np.abs(contrib_frame[col].values))) if col in contrib_frame.columns else float("nan")
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
            sum_L = z["L"].sum(dim=0); sum_N = z["N"].sum(dim=0); sum_I = z["I"].sum(dim=0)
        else:
            sum_L += z["L"].sum(dim=0); sum_N += z["N"].sum(dim=0); sum_I += z["I"].sum(dim=0)
    if total == 0:
        d = CFG.d; zero = torch.zeros(d, device=DEVICE)
        return {"L": zero[None, :], "N": zero[None, :], "I": zero[None, :]}
    return {"L": (sum_L / total)[None, :], "N": (sum_N / total)[None, :], "I": (sum_I / total)[None, :]}


@torch.no_grad()
def F_embed(
    zL, zN, zI,
    fusion, route_heads, gate: RouteGateNet, final_head: FinalConcatHead,
    y: torch.Tensor,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False,
    gate_mode: str = "learned",
    loss_gate_alpha: float = 4.0,
) -> torch.Tensor:
    z = {"L": zL, "N": zN, "I": zI}
    ylogits, _, _, _ = forward_full(
        z, fusion, route_heads, gate, final_head, y=y, masks=masks,
        l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha
    )
    return ylogits 


@torch.no_grad()
def uc_bi_ti_for_batch(
    xL: torch.Tensor, notes_list, imgs: torch.Tensor,
    behrt, bbert, imgenc,
    fusion, route_heads, gate: RouteGateNet, final_head: FinalConcatHead,
    mus: Dict[str, torch.Tensor],
    y: torch.Tensor,
    l2norm_each: bool = False,
    gate_mode: str = "learned",
    loss_gate_alpha: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xL = xL.to(DEVICE); imgs = imgs.to(DEVICE)
    z  = embeddings_from_batch(behrt, bbert, imgenc, xL, notes_list, imgs)
    B  = z["L"].size(0)

    ones = {k: torch.ones(B, 1, device=DEVICE) for k in ["L", "N", "I"]}

    muL = mus["L"].expand(B, -1)
    muN = mus["N"].expand(B, -1)
    muI = mus["I"].expand(B, -1)

    F_full = F_embed(z["L"], z["N"], z["I"], fusion, route_heads, gate, final_head, y, masks=ones,
                     l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha)
    F_mmm  = F_embed(muL,   muN,   muI,   fusion, route_heads, gate, final_head, y, masks=ones,
                     l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha)

    # Unimodal contrasts
    F_Lmm = F_embed(z["L"], muN,   muI,   fusion, route_heads, gate, final_head, y, masks=ones,
                    l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha)
    F_mNm = F_embed(muL,   z["N"], muI,   fusion, route_heads, gate, final_head, y, masks=ones,
                    l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha)
    F_mmI = F_embed(muL,   muN,   z["I"], fusion, route_heads, gate, final_head, y, masks=ones,
                    l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha)

    # Pairwise contrasts
    F_LNm = F_embed(z["L"], z["N"], muI,  fusion, route_heads, gate, final_head, y, masks=ones,
                    l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha)
    F_LmI = F_embed(z["L"], muN,   z["I"], fusion, route_heads, gate, final_head, y, masks=ones,
                    l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha)
    F_mNI = F_embed(muL,   z["N"], z["I"], fusion, route_heads, gate, final_head, y, masks=ones,
                    l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha)

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
    behrt, bbert, imgenc, fusion, route_heads, gate: RouteGateNet, final_head: FinalConcatHead,
    gate_mode: str, loss_gate_alpha: float,
    limit_batches: Optional[int] = None,
    l2norm_each: bool = False,
) -> pd.DataFrame:
    behrt.eval(); bbert.eval(); imgenc.eval()
    for k in ["LN", "LI", "NI", "LNI"]:
        fusion[k].eval()
    for k in route_heads: route_heads[k].eval()
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

            y_full, gates, routes_raw, routes_weighted = forward_full(
                z, fusion, route_heads, gate, final_head, y=y, masks=masks,
                l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha
            )
            probs = torch.sigmoid(y_full)
            B = y.size(0)

            _, contribs = route_contributions_occlusion(
                z, fusion, route_heads, gate, final_head, y=y, masks=masks,
                l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha
            )

            bw = block_weights_from_gates(gates)

            for i in range(B):
                rec = {"row_id": int(gid)}
                rec["y_true"]  = float(y[i, 0]) if y.dim() == 2 else float(y[i])
                rec["y_prob"]  = float(probs[i, 0])
                rec["y_logit"] = float(y_full[i, 0])

                rec["block_w__uni"] = float(bw[i, 0])
                rec["block_w__bi"]  = float(bw[i, 1])
                rec["block_w__tri"] = float(bw[i, 2])

                idx = {r: k for k, r in enumerate(ROUTES)}
                for r in ROUTES:
                    rec[f"gate__{r}"] = float(gates[i, idx[r]])
                    rec[f"route_contrib__{r}"] = float(contribs[r][i, 0])
                    rec[f"route_emb_norm__{r}"] = float(routes_raw[r][i].norm(p=2).item())

                rows.append(rec); gid += 1

    return pd.DataFrame.from_records(rows)
    
def collect_uc_bi_ti(
    loader: DataLoader,
    behrt, bbert, imgenc, fusion, route_heads, gate: RouteGateNet, final_head: FinalConcatHead,
    mus: Dict[str, torch.Tensor],
    gate_mode: str, loss_gate_alpha: float,
    limit_batches: Optional[int] = None,
    l2norm_each: bool = False,
) -> pd.DataFrame:
    for k in ["LN", "LI", "NI", "LNI"]:
        fusion[k].eval()
    for k in route_heads: route_heads[k].eval()
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
                fusion, route_heads, gate, final_head, mus, y,
                l2norm_each=l2norm_each, gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha,
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
    behrt, bbert, imgenc, fusion, route_heads, gate, final_head = _build_stack()
    gate_mode, loss_gate_alpha, l2norm_each = _load_checkpoints(
        behrt, bbert, imgenc, fusion, route_heads, gate, final_head, task=getattr(CFG, "task_name", "mort")
    )

    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    test_ds = ICUStayDataset(ROOT, split="test")

    tidx = TASK_MAP.get(getattr(CFG, "task_name", "mort"), 0)
    base_collate = collate_fn_factory(tidx)
    def collate_fn(batch):
        xL, notes, imgs, y = base_collate(batch)
        sens = [{} for _ in batch]  
        return xL, notes, imgs, y, sens

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    # Local contributions (per sample)
    df_contrib = collect_contributions(
        test_loader, behrt, bbert, imgenc, fusion, route_heads, gate, final_head,
        gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha,
        l2norm_each=l2norm_each,
    )
    print(f"[contrib] cols={len(df_contrib.columns)}, rows={len(df_contrib)}")
    print(df_contrib.head(3).to_string(index=False))

    # Global summary of |Δ| per route
    gs = global_mean_abs_contrib(df_contrib)
    print("\nGlobal mean |Δ(logit)| per route:")
    print(json.dumps(gs, indent=2))

    # UC/BI/TI with dataset-mean embeddings (expectation baselines)
    mus = compute_dataset_means(test_loader, behrt, bbert, imgenc, max_batches=8)
    df_inter = collect_uc_bi_ti(
        test_loader, behrt, bbert, imgenc, fusion, route_heads, gate, final_head, mus,
        gate_mode=gate_mode, loss_gate_alpha=loss_gate_alpha, l2norm_each=l2norm_each
    )
    print("\nUC/BI/TI (first 5 rows):")
    print(df_inter.head().to_string(index=False))

    print("\nMean UC/BI/TI:")
    print(json.dumps({
        "UC_mean": float(df_inter["UC"].mean()),
        "BI_mean": float(df_inter["BI"].mean()),
        "TI_mean": float(df_inter["TI"].mean()),
    }, indent=2))
