from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from env_config import CFG, DEVICE, ROUTES, BLOCKS
from encoders import EncoderConfig, build_encoders
from routing_and_heads import (
    build_fusions,
    RouteGateNet,
    FinalConcatHead,
    forward_emb_concat,
)
from train_step1_unimodal import ICUStayDataset, collate_fn


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
    fusion    = build_fusions(d=CFG.d, p_drop=CFG.dropout)
    gate_net  = RouteGateNet(d=CFG.d, hidden=4 * 256, p_drop=CFG.dropout, use_masks=True).to(DEVICE)
    final_head = FinalConcatHead(d=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE)
    return behrt, bbert, imgenc, fusion, gate_net, final_head


def load_checkpoints(task: str,
                     behrt, bbert, imgenc,
                     fusion: Dict[str, torch.nn.Module],
                     gate_net: RouteGateNet,
                     final_head: FinalConcatHead) -> None:

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

    # Pairwise fusions from Step-2
    if "fusion_LN" in ckpt2: fusion["LN"].load_state_dict(ckpt2["fusion_LN"], strict=False)
    if "fusion_LI" in ckpt2: fusion["LI"].load_state_dict(ckpt2["fusion_LI"], strict=False)
    if "fusion_NI" in ckpt2: fusion["NI"].load_state_dict(ckpt2["fusion_NI"], strict=False)

    # Step-3: gate + final head 
    if "gate" in ckpt3:
        gate_net.load_state_dict(ckpt3["gate"], strict=False)
    elif "gate_net" in ckpt3:
        gate_net.load_state_dict(ckpt3["gate_net"], strict=False)
    else:
        raise KeyError("Step-3 checkpoint missing 'gate'/'gate_net' state dict.")

    if "final_head" in ckpt3:
        final_head.load_state_dict(ckpt3["final_head"], strict=False)
    else:
        raise KeyError("Step-3 checkpoint missing 'final_head' state dict.")

    if "fusion_LNI" in ckpt3:
        try:
            fusion["LNI"].load_state_dict(ckpt3["fusion_LNI"], strict=False)
        except Exception:
            pass

    print(f"[{task}] Loaded: {os.path.basename(c1)}, {os.path.basename(c2)}, {os.path.basename(c3)}")

@torch.no_grad()
def run_inference_demo(loader: DataLoader,
                       behrt, bbert, imgenc,
                       fusion, gate_net, final_head,
                       task: str, show_k: int = 5, inspect_idx: int = 0):
    behrt.eval(); bbert.eval(); imgenc.eval()
    for k in fusion: fusion[k].eval()
    gate_net.eval(); final_head.eval()

    xL, notes_list, imgs, y, sens = next(iter(loader))
    xL   = xL.to(DEVICE)
    imgs = imgs.to(DEVICE)
    y    = y.to(DEVICE)  

    # Unimodal embeddings
    zL = behrt(xL)
    zN = bbert(notes_list)
    zI = imgenc(imgs)
    z  = {"L": zL, "N": zN, "I": zI}

    masks = build_masks(xL, notes_list, imgs)

    ylogits, gates, routes_raw, routes_weighted = forward_emb_concat(
        z_unimodal=z,
        fusion=fusion,
        final_head=final_head,
        gate_net=gate_net,
        masks=masks,
        l2norm_each=False,
    )
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
        "probs": probs,                    
        "ylogits": ylogits,                
        "gates": gates,                    
        "routes_raw": routes_raw,          
        "routes_weighted": routes_weighted,
        "y_true": y,                       
        "sens": sens,                      
    }


@torch.no_grad()
def run_inference_loader(loader: DataLoader,
                         behrt, bbert, imgenc,
                         fusion, gate_net, final_head):
    behrt.eval(); bbert.eval(); imgenc.eval()
    for k in fusion: fusion[k].eval()
    gate_net.eval(); final_head.eval()

    all_probs, all_logits = [], []
    all_gates = []

    for xL, notes_list, imgs, y, sens in loader:
        xL   = xL.to(DEVICE)
        imgs = imgs.to(DEVICE)

        zL = behrt(xL)
        zN = bbert(notes_list)
        zI = imgenc(imgs)
        z  = {"L": zL, "N": zN, "I": zI}

        masks = build_masks(xL, notes_list, imgs)

        ylogits, gates, _, _ = forward_emb_concat(
            z_unimodal=z,
            fusion=fusion,
            final_head=final_head,
            gate_net=gate_net,
            masks=masks,
            l2norm_each=False,
        )
        probs = torch.sigmoid(ylogits)

        all_probs.append(probs.cpu())
        all_logits.append(ylogits.cpu())
        all_gates.append(gates.cpu())

    probs  = torch.cat(all_probs, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()
    gates  = torch.cat(all_gates, dim=0).numpy()

    return {
        "probs": probs,     
        "logits": logits,   
        "gates": gates,     
        "routes": ROUTES,
        "blocks": list(BLOCKS.keys()),
    }


def main():
    args = parse_args()

    if args.data_root != CFG.data_root:
        CFG.data_root = args.data_root
    if args.ckpt_root != CFG.ckpt_root:
        CFG.ckpt_root = args.ckpt_root

    behrt, bbert, imgenc, fusion, gate_net, final_head = build_stack()
    load_checkpoints(args.task, behrt, bbert, imgenc, fusion, gate_net, final_head)

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
        behrt, bbert, imgenc, fusion, gate_net, final_head,
        task=args.task,
        show_k=args.show_k,
        inspect_idx=args.inspect_idx,
    )


if __name__ == "__main__":
    main()
