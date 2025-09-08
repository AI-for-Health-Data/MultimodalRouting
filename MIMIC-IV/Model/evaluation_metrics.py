from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from env_config import CFG, DEVICE, ROUTES, BLOCKS, TASKS
from encoders import EncoderConfig, build_encoders
from routing_and_heads import (
    build_fusions,
    build_route_heads,
    RouteGateNet,
    FinalConcatHead,
    make_route_inputs,
    route_availability_mask,
    concat_routes,
)
from train_step1_unimodal import ICUStayDataset, collate_fn_factory


TASK_MAP = {"mort": 0, "pe": 1, "ph": 2}
COL_MAP  = {"mort": "mort", "pe": "pe", "ph": "ph"}


def _group_index(batch_groups: List[dict], key: str, device) -> Dict[str, torch.Tensor]:
    buckets: Dict[str, List[int]] = {}
    for i, meta in enumerate(batch_groups):
        g = str(meta.get(key, "UNK"))
        buckets.setdefault(g, []).append(i)
    return {g: torch.tensor(ix, device=device, dtype=torch.long) for g, ix in buckets.items()}

def eddi_sign_agnostic(errors: torch.Tensor, batch_groups: List[dict], keys: List[str]) -> torch.Tensor:
    if errors.numel() == 0:
        return torch.tensor(0.0, device=errors.device)
    overall = errors.mean()
    accum = 0.0
    nkeys = 0
    for k in keys:
        g2ix = _group_index(batch_groups, k, errors.device)
        if not g2ix:
            continue
        total = 0
        disp = 0.0
        for _, ix in g2ix.items():
            if ix.numel() == 0:
                continue
            gmean = errors.index_select(0, ix).mean()
            disp = disp + (gmean - overall).abs() * ix.numel()
            total += ix.numel()
        if total > 0:
            accum = accum + (disp / total)
            nkeys += 1
    if nkeys == 0:
        return torch.tensor(0.0, device=errors.device)
    return accum / nkeys

def compute_eddi_final(sens: List[dict], ylogits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if getattr(CFG, "lambda_fair", 0.0) <= 0.0:
        return torch.tensor(0.0, device=y.device)
    probs = torch.sigmoid(ylogits).squeeze(1)
    err = (probs - y.squeeze(1)).abs()
    return eddi_sign_agnostic(err, sens, getattr(CFG, "sensitive_keys", []))


def build_masks(xL: torch.Tensor, notes_list, imgs: torch.Tensor) -> dict:
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)

    # N present if any non-empty note
    mN_list = []
    for notes in notes_list:
        present = 1.0 if (isinstance(notes, list) and any((isinstance(t, str) and len(t.strip()) > 0) for t in notes)) else 0.0
        mN_list.append(present)
    mN = torch.tensor(mN_list, device=xL.device, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        mI_vals = (imgs.abs().flatten(1).sum(dim=1) > 0).float()
    mI = mI_vals.to(xL.device).unsqueeze(1)

    return {"L": mL, "N": mN, "I": mI}


def _build_stack(n_tasks: int = 1):
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
        )
    )

    # Fusion blocks
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)

    # Route heads (7) used for loss_based gating at eval
    route_heads = build_route_heads(d=CFG.d, n_tasks=1, p_drop=CFG.dropout)

    # Gate net 
    gate_net   = RouteGateNet(d=CFG.d, hidden=4 * 256, p_drop=CFG.dropout, use_masks=True).to(DEVICE)

    # Final head over concat(7*d)
    final_head = FinalConcatHead(d=CFG.d, n_tasks=n_tasks, p_drop=CFG.dropout).to(DEVICE)

    return behrt, bbert, imgenc, fusion, route_heads, gate_net, final_head


def _load_checkpoints(task: str,
                      behrt: nn.Module,
                      bbert: nn.Module,
                      imgenc: nn.Module,
                      fusion: Dict[str, nn.Module],
                      route_heads: Dict[str, nn.Module],
                      gate_net: RouteGateNet,
                      final_head: FinalConcatHead) -> Tuple[str, float, bool]:

    ckpt1_path = os.path.join(CFG.ckpt_root, f"{task}_step1_unimodal.pt")
    ckpt2_path = os.path.join(CFG.ckpt_root, f"{task}_step2_bimodal.pt")
    ckpt3_path = os.path.join(CFG.ckpt_root, f"{task}_step3_concat_gate.pt")

    if not os.path.exists(ckpt1_path):
        raise FileNotFoundError(f"Missing Step-1 checkpoint: {ckpt1_path}")
    if not os.path.exists(ckpt2_path):
        raise FileNotFoundError(f"Missing Step-2 checkpoint: {ckpt2_path}")
    if not os.path.exists(ckpt3_path):
        raise FileNotFoundError(f"Missing Step-3 (embedding concat) checkpoint: {ckpt3_path}")

    ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)
    ckpt2 = torch.load(ckpt2_path, map_location=DEVICE)
    ckpt3 = torch.load(ckpt3_path, map_location=DEVICE)

    # Encoders
    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)

    # Route heads (L,N,I from step-1)
    if "L" in ckpt1: route_heads["L"].load_state_dict(ckpt1["L"], strict=False)
    if "N" in ckpt1: route_heads["N"].load_state_dict(ckpt1["N"], strict=False)
    if "I" in ckpt1: route_heads["I"].load_state_dict(ckpt1["I"], strict=False)

    # Bimodal heads and fusion (from step-2)
    if "LN" in ckpt2: route_heads["LN"].load_state_dict(ckpt2["LN"], strict=False)
    if "LI" in ckpt2: route_heads["LI"].load_state_dict(ckpt2["LI"], strict=False)
    if "NI" in ckpt2: route_heads["NI"].load_state_dict(ckpt2["NI"], strict=False)

    if "fusion_LN" in ckpt2: fusion["LN"].load_state_dict(ckpt2["fusion_LN"], strict=False)
    if "fusion_LI" in ckpt2: fusion["LI"].load_state_dict(ckpt2["fusion_LI"], strict=False)
    if "fusion_NI" in ckpt2: fusion["NI"].load_state_dict(ckpt2["fusion_NI"], strict=False)

    # From step-3: gate, final head, optional LNI fusion/head
    gate_mode = ckpt3.get("gate_mode", "learned")
    loss_gate_alpha = float(ckpt3.get("loss_gate_alpha", 4.0))
    l2norm_each = bool(ckpt3.get("l2norm_each", False))

    if "gate" in ckpt3:
        gate_net.load_state_dict(ckpt3["gate"], strict=False)
    elif "gate_net" in ckpt3:
        gate_net.load_state_dict(ckpt3["gate_net"], strict=False)
    else:
        if gate_mode == "learned":
            raise KeyError("Step-3 checkpoint missing 'gate'/'gate_net' state dict but gate_mode='learned'.")

    if "final_head" in ckpt3:
        final_head.load_state_dict(ckpt3["final_head"], strict=False)
    else:
        raise KeyError("Step-3 checkpoint missing 'final_head' state dict.")

    if "fusion_LNI" in ckpt3:
        try:
            fusion["LNI"].load_state_dict(ckpt3["fusion_LNI"], strict=False)
        except Exception:
            pass
    if "LNI_head" in ckpt3:
        try:
            route_heads["LNI"].load_state_dict(ckpt3["LNI_head"], strict=False)
        except Exception:
            pass

    print(f"[{task}] Loaded: {os.path.basename(ckpt1_path)}, "
          f"{os.path.basename(ckpt2_path)}, {os.path.basename(ckpt3_path)} "
          f"(gate_mode={gate_mode}, alpha={loss_gate_alpha}, l2norm={l2norm_each})")

    return gate_mode, loss_gate_alpha, l2norm_each


def evaluate(task: str) -> dict:
    assert task in TASK_MAP, f"Unknown task '{task}'"
    t_idx = TASK_MAP[task]

    # Collate with sens placeholders
    _base_collate = collate_fn_factory(t_idx)
    def collate_fn_eval(batch):
        xL, notes, imgs, y = _base_collate(batch)
        sens = [{} for _ in batch]  
        return xL, notes, imgs, y, sens

    # Data
    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    test_ds = ICUStayDataset(ROOT, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn_eval,
        pin_memory=("cuda" in str(DEVICE) and torch.cuda.is_available()),
    )

    behrt, bbert, imgenc, fusion, route_heads, gate_net, final_head = _build_stack(n_tasks=1)
    gate_mode, loss_gate_alpha, l2norm_each = _load_checkpoints(
        task, behrt, bbert, imgenc, fusion, route_heads, gate_net, final_head
    )

    behrt.eval(); bbert.eval(); imgenc.eval()
    for k in fusion: fusion[k].eval()
    for k in route_heads: route_heads[k].eval()
    gate_net.eval(); final_head.eval()

    y_all, p_all = [], []
    sum_route_gates = torch.zeros(7, device=DEVICE)
    total_examples = 0

    eddi_sum = 0.0
    eddi_batches = 0

    with torch.no_grad():
        for xL, notes_list, imgs, y, sens in tqdm(test_loader, desc=f"Evaluating [{task}]", dynamic_ncols=True):
            xL   = xL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y    = y.to(DEVICE)
            B    = y.size(0)

            # Unimodal embeddings
            zL = behrt(xL)
            zN = bbert(notes_list)
            zI = imgenc(imgs)
            z  = {"L": zL, "N": zN, "I": zI}

            # Route embeddings
            routes_emb: Dict[str, torch.Tensor] = make_route_inputs(z, fusion)

            # Modality availability
            masks = build_masks(xL, notes_list, imgs)
            avail = route_availability_mask(masks, batch_size=B, device=xL.device)

            # Gates
            if gate_mode == "uniform":
                gates = avail / (avail.sum(dim=1, keepdim=True).clamp_min(1.0))
            elif gate_mode == "loss_based":
                # Per-route logits -> per-sample BCE_i 
                per_route_losses = []
                for r in ROUTES:
                    log_r = route_heads[r](routes_emb[r])
                    l_i = F.binary_cross_entropy_with_logits(log_r, y, reduction="none").squeeze(1)  
                    per_route_losses.append(l_i)
                Lmat = torch.stack(per_route_losses, dim=1)  
                masked_logits = (-float(loss_gate_alpha) * Lmat) + torch.log(avail + 1e-12)
                gates = torch.softmax(masked_logits, dim=1)
            else:  
                gates = gate_net({"L": zL, "N": zN, "I": zI}, masks=masks)
                gates = gates / (gates.sum(dim=1, keepdim=True).clamp_min(1e-6))

            # Concat weighted routes and predict
            x_cat, Zw = concat_routes(routes_emb, gates=gates, l2norm=l2norm_each)
            ylogits = final_head(x_cat)
            probs = torch.sigmoid(ylogits)

            # Collect
            y_all.append(y.detach().cpu().numpy())
            p_all.append(probs.detach().cpu().numpy())

            # Aggregate gates
            sum_route_gates += gates.sum(dim=0)  
            total_examples += B

            # EDDI on final predictions
            eddi = compute_eddi_final(sens, ylogits, y)
            if eddi.isfinite():
                eddi_sum += float(eddi)
                eddi_batches += 1

    Y = np.concatenate(y_all, axis=0) if y_all else np.zeros((0, 1), dtype=np.float32)
    P = np.concatenate(p_all, axis=0) if p_all else np.zeros((0, 1), dtype=np.float32)

    metrics = {"task": task}
    try:
        metrics["auroc"] = float(roc_auc_score(Y[:, 0], P[:, 0]))
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["ap"] = float(average_precision_score(Y[:, 0], P[:, 0]))
    except Exception:
        metrics["ap"] = float("nan")

    # Mean route gate weights 
    if total_examples > 0:
        mean_route_gates = (sum_route_gates / total_examples).detach().cpu().numpy()
        metrics["mean_route_gates"] = {ROUTES[i]: float(mean_route_gates[i]) for i in range(len(ROUTES))}
        block_means = {}
        idx_map = {r: i for i, r in enumerate(ROUTES)}
        for blk, names in BLOCKS.items():
            idx = [idx_map[n] for n in names]
            block_means[blk] = float(mean_route_gates[idx].mean())
        metrics["mean_block_gates"] = block_means
    else:
        metrics["mean_route_gates"] = {r: 0.0 for r in ROUTES}
        metrics["mean_block_gates"] = {"uni": 0.0, "bi": 0.0, "tri": 0.0}

    metrics["eddi_mean"] = (eddi_sum / eddi_batches) if eddi_batches > 0 else float("nan")
    return metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate embedding-level fusion model for a single task.")
    ap.add_argument("--task", type=str, default=getattr(CFG, "task_name", "mort"),
                    choices=list(TASK_MAP.keys()), help="Which task checkpoint to evaluate.")
    ap.add_argument("--data_root", type=str, default=CFG.data_root, help="Root path with MIMIC-IV data.")
    ap.add_argument("--ckpt_root", type=str, default=CFG.ckpt_root, help="Where checkpoints were saved.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.data_root != CFG.data_root:
        CFG.data_root = args.data_root
    if args.ckpt_root != CFG.ckpt_root:
        CFG.ckpt_root = args.ckpt_root

    out = evaluate(args.task)
    print(json.dumps(out, indent=2))
