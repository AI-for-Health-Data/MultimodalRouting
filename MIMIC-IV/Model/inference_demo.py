from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from env_config import CFG, DEVICE, ROUTES, BLOCKS
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


def parse_args():
    ap = argparse.ArgumentParser(description="Embedding-level fusion inference demo")
    ap.add_argument("--task", type=str, default=getattr(CFG, "task_name", "mort"),
                    choices=list(TASK_MAP.keys()),
                    help="Which single-task checkpoint set to load.")
    ap.add_argument("--data_root", type=str, default=CFG.data_root,
                    help="Root containing MIMIC-IV parquet data.")
    ap.add_argument("--ckpt_root", type=str, default=CFG.ckpt_root,
                    help="Where checkpoints are saved.")
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--show_k", type=int, default=5,
                    help="How many patients to print from the first batch.")
    ap.add_argument("--inspect_idx", type=int, default=0,
                    help="Index within the first batch to show detailed gates.")
    return ap.parse_args()

def build_masks(xL: torch.Tensor, notes_list, imgs: torch.Tensor) -> dict:
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

def build_stack():
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
    gate_net     = RouteGateNet(d=CFG.d, hidden=4 * 256, p_drop=CFG.dropout, use_masks=True).to(DEVICE)
    final_head   = FinalConcatHead(d=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE)
    return behrt, bbert, imgenc, fusion, route_heads, gate_net, final_head


def load_checkpoints(task: str,
                     behrt, bbert, imgenc,
                     fusion: Dict[str, torch.nn.Module],
                     route_heads: Dict[str, torch.nn.Module],
                     gate_net: RouteGateNet,
                     final_head: FinalConcatHead) -> Tuple[str, float, bool]:

    c1 = os.path.join(CFG.ckpt_root, f"{task}_step1_unimodal.pt")
    c2 = os.path.join(CFG.ckpt_root, f"{task}_step2_bimodal.pt")
    c3 = os.path.join(CFG.ckpt_root, f"{task}_step3_concat_gate.pt")

    if not os.path.exists(c1):
        raise FileNotFoundError(f"Missing Step-1 checkpoint: {c1}")
    if not os.path.exists(c2):
        raise FileNotFoundError(f"Missing Step-2 checkpoint: {c2}")
    if not os.path.exists(c3):
        raise FileNotFoundError(f"Missing Step-3 checkpoint: {c3}")

    ckpt1 = torch.load(c1, map_location=DEVICE)
    ckpt2 = torch.load(c2, map_location=DEVICE)
    ckpt3 = torch.load(c3, map_location=DEVICE)

    # Encoders
    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)

    if "L" in ckpt1:  route_heads["L"].load_state_dict(ckpt1["L"], strict=False)
    if "N" in ckpt1:  route_heads["N"].load_state_dict(ckpt1["N"], strict=False)
    if "I" in ckpt1:  route_heads["I"].load_state_dict(ckpt1["I"], strict=False)
    if "LN" in ckpt2: route_heads["LN"].load_state_dict(ckpt2["LN"], strict=False)
    if "LI" in ckpt2: route_heads["LI"].load_state_dict(ckpt2["LI"], strict=False)
    if "NI" in ckpt2: route_heads["NI"].load_state_dict(ckpt2["NI"], strict=False)

    if "fusion_LN" in ckpt2: fusion["LN"].load_state_dict(ckpt2["fusion_LN"], strict=False)
    if "fusion_LI" in ckpt2: fusion["LI"].load_state_dict(ckpt2["fusion_LI"], strict=False)
    if "fusion_NI" in ckpt2: fusion["NI"].load_state_dict(ckpt2["fusion_NI"], strict=False)

    gate_mode      = ckpt3.get("gate_mode", "learned")
    loss_gate_alpha = float(ckpt3.get("loss_gate_alpha", 4.0))
    l2norm_each     = bool(ckpt3.get("l2norm_each", False))

    if "gate" in ckpt3:
        gate_net.load_state_dict(ckpt3["gate"], strict=False)
    elif "gate_net" in ckpt3:
        gate_net.load_state_dict(ckpt3["gate_net"], strict=False)
    else:
        if gate_mode == "learned":
            raise KeyError("Step-3 checkpoint missing 'gate'/'gate_net' but gate_mode='learned'.")

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

    print(f"[{task}] Loaded: {os.path.basename(c1)}, {os.path.basename(c2)}, {os.path.basename(c3)} "
          f"(gate_mode={gate_mode}, alpha={loss_gate_alpha}, l2norm={l2norm_each})")
    return gate_mode, loss_gate_alpha, l2norm_each

@torch.no_grad()
def _compute_gates(gate_mode: str,
                   routes_emb: Dict[str, torch.Tensor],
                   route_heads: Dict[str, torch.nn.Module],
                   gate_net: RouteGateNet,
                   z_unimodal: Dict[str, torch.Tensor],
                   y: torch.Tensor,
                   masks: Dict[str, torch.Tensor],
                   loss_gate_alpha: float) -> torch.Tensor:
    B = y.size(0)
    avail = route_availability_mask(masks, batch_size=B, device=y.device)
    if gate_mode == "uniform":
        return avail / (avail.sum(dim=1, keepdim=True).clamp_min(1.0))
    if gate_mode == "learned":
        g = gate_net(z_unimodal, masks=masks)
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
def run_inference_demo(loader: DataLoader,
                       behrt, bbert, imgenc,
                       fusion, route_heads, gate_net, final_head,
                       task: str, show_k: int, inspect_idx: int,
                       gate_mode: str, loss_gate_alpha: float, l2norm_each: bool):
    behrt.eval(); bbert.eval(); imgenc.eval()
    for k in fusion: fusion[k].eval()
    for k in route_heads: route_heads[k].eval()
    gate_net.eval(); final_head.eval()

    # Grab first batch
    xL, notes_list, imgs, y, sens = next(iter(loader))
    xL   = xL.to(DEVICE)
    imgs = imgs.to(DEVICE)
    y    = y.to(DEVICE)

    # Unimodal embeddings
    zL = behrt(xL)
    zN = bbert(notes_list)
    zI = imgenc(imgs)
    z  = {"L": zL, "N": zN, "I": zI}

    # Build routes + masks
    routes_emb = make_route_inputs(z, fusion)
    masks = build_masks(xL, notes_list, imgs)

    # Gates
    gates = _compute_gates(gate_mode, routes_emb, route_heads, gate_net, z, y, masks, loss_gate_alpha)

    # Concat & predict
    x_cat, Zw = concat_routes(routes_emb, gates=gates, l2norm=l2norm_each)
    ylogits = final_head(x_cat)
    probs = torch.sigmoid(ylogits)

    B = probs.size(0)
    show_k = min(show_k, B)
    print(f"\nPredicted probabilities for task '{task}' (first {show_k} patients):")
    print(probs[:show_k].flatten().cpu().numpy())

    i = min(inspect_idx, B - 1)
    print(f"\n--- Route gates for sample index {i} ---")
    gate_i = gates[i].cpu().numpy()
    gate_dict = {ROUTES[j]: float(gate_i[j]) for j in range(len(ROUTES))}
    idx_map = {r: j for j, r in enumerate(ROUTES)}
    block_means = {}
    for blk, names in BLOCKS.items():
        idx = [idx_map[n] for n in names]
        block_means[blk] = float(torch.tensor(gate_i[idx]).mean().item())

    print("Final prob:", float(probs[i].item()))
    print("Block mean gates:", {k: round(v, 4) for k, v in block_means.items()})
    print("Route gates (sorted):")
    for name, w in sorted(gate_dict.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:>3}: {w:.4f}")

    return {
        "probs": probs, "ylogits": ylogits, "gates": gates,
        "routes_raw": routes_emb, "routes_weighted": {r: Zw[:, j, :] for j, r in enumerate(ROUTES)},
        "y_true": y, "sens": sens,
    }


def main():
    args = parse_args()

    if args.data_root != CFG.data_root:
        CFG.data_root = args.data_root
    if args.ckpt_root != CFG.ckpt_root:
        CFG.ckpt_root = args.ckpt_root

    behrt, bbert, imgenc, fusion, route_heads, gate_net, final_head = build_stack()
    gate_mode, loss_gate_alpha, l2norm_each = load_checkpoints(
        args.task, behrt, bbert, imgenc, fusion, route_heads, gate_net, final_head
    )

    tidx = TASK_MAP[args.task]
    base_collate = collate_fn_factory(tidx)
    def collate_fn(batch):
        xL, notes, imgs, y = base_collate(batch)
        sens = [{} for _ in batch]  
        return xL, notes, imgs, y, sens

    root = os.path.join(CFG.data_root, "MIMIC-IV")
    test_ds = ICUStayDataset(root, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=("cuda" in str(DEVICE) and torch.cuda.is_available()),
    )

    _ = run_inference_demo(
        test_loader,
        behrt, bbert, imgenc, fusion, route_heads, gate_net, final_head,
        task=args.task,
        show_k=args.show_k,
        inspect_idx=args.inspect_idx,
        gate_mode=gate_mode,
        loss_gate_alpha=loss_gate_alpha,
        l2norm_each=l2norm_each,
    )


if __name__ == "__main__":
    main()
