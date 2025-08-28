# Freeze unimodal heads and train bimodal heads (LN, LI, NI) to capture residual signal beyond unimodal.

from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from env_config import CFG, DEVICE
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead, build_fusions
from train_step1_unimodal import ICUStayDataset, collate_fn


def is_cuda_device(dev) -> bool:
    return (
        torch.cuda.is_available()
        and (
            (isinstance(dev, torch.device) and dev.type == "cuda")
            or (isinstance(dev, str) and "cuda" in dev)
        )
    )


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def main():
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

    # Route heads for unimodal + bimodal
    route_heads: Dict[str, RouteHead] = {
        r: RouteHead(d_in=CFG.d, n_tasks=3, p_drop=CFG.dropout).to(DEVICE)
        for r in ["L", "N", "I", "LN", "LI", "NI"]
    }

    # Pairwise fusion modules (produce d-dim features)
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)  

    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    train_ds = ICUStayDataset(ROOT, split="train")
    val_ds   = ICUStayDataset(ROOT, split="val")

    IS_CUDA = is_cuda_device(DEVICE)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=IS_CUDA,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=IS_CUDA,
    )

    # Load Step 1 (encoders + unimodal heads) 
    ckpt1_path = os.path.join(CFG.ckpt_root, "step1_unimodal.pt")
    ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)

    # Use strict=False for robustness across minor shape/name changes
    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)
    route_heads["L"].load_state_dict(ckpt1["L"], strict=False)
    route_heads["N"].load_state_dict(ckpt1["N"], strict=False)
    route_heads["I"].load_state_dict(ckpt1["I"], strict=False)
    print(f"Loaded Step 1 weights from {ckpt1_path}")

    # Freeze encoders + unimodal heads 
    set_requires_grad(route_heads["L"], False)
    set_requires_grad(route_heads["N"], False)
    set_requires_grad(route_heads["I"], False)

    behrt.eval(); bbert.eval(); imgenc.eval()
    set_requires_grad(behrt, False)
    set_requires_grad(bbert, False)
    set_requires_grad(imgenc, False)

    # Trainable params: pairwise fusions + bimodal heads 
    params_bi = []
    for k in ["LN", "LI", "NI"]:
        if k not in fusion:
            raise KeyError(f'Fusion block "{k}" missing from build_fusions().')
        params_bi += list(fusion[k].parameters())
        params_bi += list(route_heads[k].parameters())

    opt = torch.optim.AdamW(params_bi, lr=CFG.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=IS_CUDA)

    best_val = float("inf")

    for epoch in range(CFG.max_epochs_bi):
        route_heads["LN"].train(); route_heads["LI"].train(); route_heads["NI"].train()
        # Keep encoders + unimodal heads frozen/eval
        behrt.eval(); bbert.eval(); imgenc.eval()
        route_heads["L"].eval(); route_heads["N"].eval(); route_heads["I"].eval()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_bi} [BI]", dynamic_ncols=True)
        running = 0.0; n_steps = 0

        for xL, notes_list, imgs, y, sens in pbar:
            xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
            imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
            y    = y.to(DEVICE, non_blocking=IS_CUDA)

            opt.zero_grad(set_to_none=True)

            # Encoders (frozen): compute embeddings without grads
            with torch.no_grad():
                zL = behrt(xL)            
                zN = bbert(notes_list)    
                zI = imgenc(imgs)         

            # Trainable part (fusions + heads) under AMP
            with torch.autocast(
                device_type=("cuda" if IS_CUDA else "cpu"),
                dtype=(torch.float16 if IS_CUDA else torch.bfloat16),
                enabled=True
            ):
                zLN = fusion["LN"](zL, zN)   
                zLI = fusion["LI"](zL, zI)   
                zNI = fusion["NI"](zN, zI)   

                logits_LN = route_heads["LN"](zLN)  
                logits_LI = route_heads["LI"](zLI)  
                logits_NI = route_heads["NI"](zNI)  

                loss = (
                    F.binary_cross_entropy_with_logits(logits_LN, y)
                  + F.binary_cross_entropy_with_logits(logits_LI, y)
                  + F.binary_cross_entropy_with_logits(logits_NI, y)
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

                lval = (
                    F.binary_cross_entropy_with_logits(route_heads["LN"](zLN), y)
                  + F.binary_cross_entropy_with_logits(route_heads["LI"](zLI), y)
                  + F.binary_cross_entropy_with_logits(route_heads["NI"](zNI), y)
                ) / 3.0

                val_loss += float(lval); n_val += 1

        val_loss /= max(n_val, 1)
        print(f"[BI] Val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(CFG.ckpt_root, exist_ok=True)
            ckpt_path = os.path.join(CFG.ckpt_root, "step2_bimodal.pt")

            torch.save(
                {
                    # Heads
                    "LN": route_heads["LN"].state_dict(),
                    "LI": route_heads["LI"].state_dict(),
                    "NI": route_heads["NI"].state_dict(),
                    # Pairwise fusion weights 
                    "fusion_LN": fusion["LN"].state_dict(),
                    "fusion_LI": fusion["LI"].state_dict(),
                    "fusion_NI": fusion["NI"].state_dict(),
                    # Meta
                    "best_val": best_val,
                    "cfg": vars(CFG),
                },
                ckpt_path,
            )
            print(f"Saved best bimodal heads + fusion -> {ckpt_path}")


if __name__ == "__main__":
    main()
