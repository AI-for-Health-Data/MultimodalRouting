# Freeze unimodal heads and train bimodal heads (LN, LI, NI) to capture residual signal beyond unimodal.

from __future__ import annotations
import os, argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from env_config import CFG, DEVICE
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead, build_fusions
from train_step1_unimodal import ICUStayDataset, collate_fn, COL_MAP

def is_cuda_device(dev) -> bool:
    return torch.cuda.is_available() and (
        (isinstance(dev, torch.device) and dev.type == "cuda")
        or (isinstance(dev, str) and "cuda" in dev)
    )

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="mort", choices=list(COL_MAP.keys()))
    args = ap.parse_args()
    TASK  = args.task
    TCOL  = COL_MAP[TASK]

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

    route_heads: Dict[str, RouteHead] = {
        r: RouteHead(d_in=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE)
        for r in ["L", "N", "I", "LN", "LI", "NI"]
    }

    # Pairwise fusion modules (produce d-dim features)
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)

    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    train_ds = ICUStayDataset(ROOT, split="train")
    val_ds   = ICUStayDataset(ROOT, split="val")

    # Optional: scalar pos_weight for class imbalance on the chosen task
    try:
        y_train_np = train_ds.labels[[TCOL]].values.astype("float32").reshape(-1)
        pos = float((y_train_np > 0.5).sum())
        neg = float(len(y_train_np) - pos)
        pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=DEVICE)
    except Exception:
        pos_weight = None

    IS_CUDA = is_cuda_device(DEVICE)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=IS_CUDA,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=IS_CUDA,
    )

    # Load Step 1 (encoders + unimodal heads) 
    ckpt1_path = os.path.join(CFG.ckpt_root, f"{TASK}_step1_unimodal.pt")
    ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)

    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)
    route_heads["L"].load_state_dict(ckpt1["L"], strict=False)
    route_heads["N"].load_state_dict(ckpt1["N"], strict=False)
    route_heads["I"].load_state_dict(ckpt1["I"], strict=False)
    print(f"[{TASK}] Loaded Step 1 weights from {ckpt1_path}")

    # Freeze encoders + unimodal heads
    for k in ["L", "N", "I"]:
        set_requires_grad(route_heads[k], False)
    behrt.eval(); bbert.eval(); imgenc.eval()
    set_requires_grad(behrt, False)
    set_requires_grad(bbert, False)
    set_requires_grad(imgenc, False)

    # Trainable params: pairwise fusions + bimodal heads 
    params_bi = []
    for k in ["LN", "LI", "NI"]:
        params_bi += list(fusion[k].parameters())
        params_bi += list(route_heads[k].parameters())

    opt = torch.optim.AdamW(params_bi, lr=CFG.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=IS_CUDA)

    best_val = float("inf")

    for epoch in range(CFG.max_epochs_bi):
        for k in ["LN","LI","NI"]:
            route_heads[k].train()
        behrt.eval(); bbert.eval(); imgenc.eval()
        route_heads["L"].eval(); route_heads["N"].eval(); route_heads["I"].eval()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_bi} [BI:{TASK}]", dynamic_ncols=True)
        running = 0.0; n_steps = 0

        for xL, notes_list, imgs, y, sens in pbar:
            # y is [B,1] from collate_fn
            xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
            imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
            y    = y.to(DEVICE, non_blocking=IS_CUDA)

            opt.zero_grad(set_to_none=True)

            # Encoders (frozen)
            with torch.no_grad():
                zL = behrt(xL)
                zN = bbert(notes_list)
                zI = imgenc(imgs)

            # Trainable part (fusions + heads) under AMP
            with torch.autocast(
                device_type=("cuda" if IS_CUDA else "cpu"),
                dtype=(torch.float16 if IS_CUDA else torch.bfloat16),
                enabled=True,
            ):
                zLN = fusion["LN"](zL, zN)
                zLI = fusion["LI"](zL, zI)
                zNI = fusion["NI"](zN, zI)

                logits_LN = route_heads["LN"](zLN)  # [B,1]
                logits_LI = route_heads["LI"](zLI)  # [B,1]
                logits_NI = route_heads["NI"](zNI)  # [B,1]

                bce_kw = {"pos_weight": pos_weight} if (pos_weight is not None) else {}
                loss = (
                    F.binary_cross_entropy_with_logits(logits_LN, y, **bce_kw)
                  + F.binary_cross_entropy_with_logits(logits_LI, y, **bce_kw)
                  + F.binary_cross_entropy_with_logits(logits_NI, y, **bce_kw)
                ) / 3.0

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params_bi, max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss); n_steps += 1
            pbar.set_postfix(loss=f"{running / n_steps:.4f}")

        for k in ["LN","LI","NI"]:
            route_heads[k].eval()

        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for xL, notes_list, imgs, y, sens in val_loader:
                xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
                imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
                y    = y.to(DEVICE, non_blocking=IS_CUDA)

                zL = behrt(xL)
                zN = bbert(notes_list)
                zI = imgenc(imgs)

                zLN = fusion["LN"](zL, zN)
                zLI = fusion["LI"](zL, zI)
                zNI = fusion["NI"](zN, zI)

                bce_kw = {"pos_weight": pos_weight} if (pos_weight is not None) else {}
                lval = (
                    F.binary_cross_entropy_with_logits(route_heads["LN"](zLN), y, **bce_kw)
                  + F.binary_cross_entropy_with_logits(route_heads["LI"](zLI), y, **bce_kw)
                  + F.binary_cross_entropy_with_logits(route_heads["NI"](zNI), y, **bce_kw)
                ) / 3.0

                val_loss += float(lval); n_val += 1

        val_loss /= max(n_val, 1)
        print(f"[BI:{TASK}] Val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(CFG.ckpt_root, exist_ok=True)
            ckpt_path = os.path.join(CFG.ckpt_root, f"{TASK}_step2_bimodal.pt")
            torch.save(
                {
                    "LN": route_heads["LN"].state_dict(),
                    "LI": route_heads["LI"].state_dict(),
                    "NI": route_heads["NI"].state_dict(),
                    "fusion_LN": fusion["LN"].state_dict(),
                    "fusion_LI": fusion["LI"].state_dict(),
                    "fusion_NI": fusion["NI"].state_dict(),
                    "best_val": best_val,
                    "task": TASK,
                    "cfg": vars(CFG),
                },
                ckpt_path,
            )
            print(f"[{TASK}] Saved best bimodal heads + fusion -> {ckpt_path}")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
