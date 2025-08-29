# Step 3 (single-task): Freeze unimodal & bimodal heads, train the trimodal head (LNI) and the learned-gate router.
# Fairness: FAME-style sign-agnostic EDDI on FINAL predictions (size-weighted MAD across subgroups), added to BCE.

from __future__ import annotations

import os
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from env_config import CFG, DEVICE, ROUTES, BLOCKS
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead, build_fusions, LearnedGateRouter
from train_step1_unimodal import ICUStayDataset, collate_fn

def _is_cuda(dev) -> bool:
    return (
        torch.cuda.is_available()
        and (
            (isinstance(dev, torch.device) and dev.type == "cuda")
            or (isinstance(dev, str) and "cuda" in dev)
        )
    )


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag
# MIMIC-IV/Model/train_step3_trimodal_router.py
# Step 3 (single-task): Freeze unimodal & bimodal heads, train the trimodal head (LNI) and the learned-gate router.
# Fairness: FAME-style sign-agnostic EDDI on FINAL predictions (size-weighted MAD across subgroups), added to BCE.

from __future__ import annotations

import os
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from env_config import CFG, DEVICE, ROUTES, BLOCKS
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead, build_fusions, LearnedGateRouter
from train_step1_unimodal import ICUStayDataset, collate_fn


# -----------------------------
# Utilities
# -----------------------------
def _is_cuda(dev) -> bool:
    return (
        torch.cuda.is_available()
        and (
            (isinstance(dev, torch.device) and dev.type == "cuda")
            or (isinstance(dev, str) and "cuda" in dev)
        )
    )


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


# -----------------------------
# EDDI (FAME-style, sign-agnostic)
# -----------------------------
def _group_index(batch_groups: List[dict], key: str, device) -> Dict[str, torch.Tensor]:
    """Map subgroup -> tensor of indices for that subgroup (on device)."""
    buckets: Dict[str, List[int]] = {}
    for i, meta in enumerate(batch_groups):
        g = str(meta.get(key, "UNK"))
        buckets.setdefault(g, []).append(i)
    return {g: torch.tensor(ix, device=device, dtype=torch.long) for g, ix in buckets.items()}


def eddi_sign_agnostic(errors: torch.Tensor, batch_groups: List[dict], keys: List[str]) -> torch.Tensor:
    """
    Sign-agnostic EDDI:
      - errors: per-sample absolute errors (|p - y|), shape [B]
      - for each sensitive key:
          disparity = size-weighted mean absolute deviation of subgroup means from overall mean
      - return mean disparity across keys (0 if no valid groups)
    """
    if errors.numel() == 0:
        return torch.tensor(0.0, device=errors.device)

    overall = errors.mean()
    accum = 0.0
    nkeys = 0
    for k in keys:
        g2ix = _group_index(batch_groups, k, errors.device)
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

def eddi_final_from_logits(ylogits: torch.Tensor, y: torch.Tensor, sens: List[dict]) -> torch.Tensor:
    """
    Compute EDDI on FINAL predictions (single-task).
      ylogits: [B, 1] logits
      y:       [B, 1] labels in {0,1}
    """
    probs = torch.sigmoid(ylogits).squeeze(1)  # [B]
    err = (probs - y.squeeze(1)).abs()         # [B]
    return eddi_sign_agnostic(err, sens, getattr(CFG, "sensitive_keys", []))

def _ensure_fusions_step3():
    """
    Ensure fusion dict has LN/LI/NI/LNI on DEVICE. If build_fusions already did, just move to device.
    """
    global fusion
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)
    for k in fusion:
        fusion[k] = fusion[k].to(DEVICE)


@torch.no_grad()
def _frozen_routes_from_unimodal(z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Given z = {"L":[B,d], "N":[B,d], "I":[B,d]}, return frozen route logits for L, N, I, LN, LI, NI (each [B,1]).
    LNI is excluded here (it is trainable in this step).
    """
    out: Dict[str, torch.Tensor] = {}
    # Unimodal
    out["L"] = route_heads["L"](z["L"])
    out["N"] = route_heads["N"](z["N"])
    out["I"] = route_heads["I"](z["I"])
    # Bimodal via fusion
    zLN = fusion["LN"](z["L"], z["N"])
    zLI = fusion["LI"](z["L"], z["I"])
    zNI = fusion["NI"](z["N"], z["I"])
    out["LN"] = route_heads["LN"](zLN)
    out["LI"] = route_heads["LI"](zLI)
    out["NI"] = route_heads["NI"](zNI)
    return out


def build_modality_masks(xL: torch.Tensor, notes_list: List[List[str]], imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Build modality-availability masks {L,N,I} with shape [B,1] in {0,1}, passed to the router.
    """
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)

    # N present if there exists a non-empty note string
    mN_list = [
        1.0 if (isinstance(notes, list) and any((isinstance(t, str) and len(t.strip()) > 0) for t in notes)) else 0.0
        for notes in notes_list
    ]
    mN = torch.tensor(mN_list, device=xL.device, dtype=torch.float32).unsqueeze(1)

    # I present if not all-zero placeholder
    with torch.no_grad():
        mI_vals = (imgs.abs().flatten(1).sum(dim=1) > 0).float()
    mI = mI_vals.to(xL.device).unsqueeze(1)

    return {"L": mL, "N": mN, "I": mI}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=getattr(CFG, "task", "mort"),
                        choices=["mort", "pe", "ph"], help="Train Step-3 for a single task/label")
    parser.add_argument("--lambda_fair", type=float, default=getattr(CFG, "lambda_fair", 0.0),
                        help="Weight for EDDI fairness loss on final predictions")
    args = parser.parse_args()

    TASK = args.task
    TCOL = TASK 

    # Build encoders
    global behrt, bbert, imgenc, route_heads, fusion, router
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

    # Route heads 
    route_heads = {
        r: RouteHead(d_in=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE)
        for r in ROUTES
    }

    # Fusions + Router
    _ensure_fusions_step3()
    router = LearnedGateRouter(
        routes=ROUTES,
        blocks=BLOCKS,
        d=CFG.d,
        n_tasks=1,
        hidden=1024,
        p_drop=CFG.dropout,
        use_masks=True,
        temperature=1.0,
    ).to(DEVICE)

    # Data
    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    train_ds = ICUStayDataset(ROOT, split="train")
    val_ds   = ICUStayDataset(ROOT, split="val")

    from torch.utils.data import DataLoader
    IS_CUDA = _is_cuda(DEVICE)
    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=IS_CUDA
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=IS_CUDA
    )

    # Class imbalance: pos_weight for the chosen task
    try:
        y_train_np = train_ds.labels[[TCOL]].values.astype("float32").reshape(-1)
        pos = float((y_train_np > 0.5).sum()); neg = float(len(y_train_np) - pos)
        pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=DEVICE)
    except Exception:
        pos_weight = None

    ckpt1_path = os.path.join(CFG.ckpt_root, f"{TASK}_step1_unimodal.pt")
    ckpt2_path = os.path.join(CFG.ckpt_root, f"{TASK}_step2_bimodal.pt")
    ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)
    ckpt2 = torch.load(ckpt2_path, map_location=DEVICE)

    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)

    route_heads["L"].load_state_dict(ckpt1["L"], strict=False)
    route_heads["N"].load_state_dict(ckpt1["N"], strict=False)
    route_heads["I"].load_state_dict(ckpt1["I"], strict=False)

    route_heads["LN"].load_state_dict(ckpt2["LN"], strict=False)
    route_heads["LI"].load_state_dict(ckpt2["LI"], strict=False)
    route_heads["NI"].load_state_dict(ckpt2["NI"], strict=False)

    if "fusion_LN" in ckpt2: fusion["LN"].load_state_dict(ckpt2["fusion_LN"], strict=False)
    if "fusion_LI" in ckpt2: fusion["LI"].load_state_dict(ckpt2["fusion_LI"], strict=False)
    if "fusion_NI" in ckpt2: fusion["NI"].load_state_dict(ckpt2["fusion_NI"], strict=False)

    print(f"[{TASK}] Loaded Step 1 & Step 2 from {ckpt1_path}, {ckpt2_path}")

    # Freeze SMRO components
    FINETUNE_ENCODERS     = False
    TRAIN_TRIMODAL_FUSION = True  

    for r in ["L", "N", "I", "LN", "LI", "NI"]:
        set_requires_grad(route_heads[r], False)
        route_heads[r].eval()

    set_requires_grad(behrt,  FINETUNE_ENCODERS); behrt.train(FINETUNE_ENCODERS)
    set_requires_grad(bbert,  FINETUNE_ENCODERS); bbert.train(FINETUNE_ENCODERS)
    set_requires_grad(imgenc, FINETUNE_ENCODERS); imgenc.train(FINETUNE_ENCODERS)

    # Trainable LNI head + router
    route_heads["LNI"].train()
    set_requires_grad(fusion["LNI"], TRAIN_TRIMODAL_FUSION)
    fusion["LNI"].train(TRAIN_TRIMODAL_FUSION)

    router.train()

    params = list(route_heads["LNI"].parameters()) + list(router.parameters())
    if TRAIN_TRIMODAL_FUSION:
        params += list(fusion["LNI"].parameters())
    if FINETUNE_ENCODERS:
        params += list(behrt.parameters()) + list(bbert.parameters()) + list(imgenc.parameters())

    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=1e-2)

    amp_enabled = _is_cuda(DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # BCE loss (single task); include pos_weight if available
    bce_fn = (lambda logits, target: F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight if pos_weight is not None else None
    ))

    best_val = float("inf")

    for epoch in range(CFG.max_epochs_tri):
        if not FINETUNE_ENCODERS:
            behrt.eval(); bbert.eval(); imgenc.eval()
        else:
            behrt.train(); bbert.train(); imgenc.train()

        fusion["LNI"].train(TRAIN_TRIMODAL_FUSION)
        route_heads["LNI"].train()
        router.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_tri} [TRI:{TASK}]", dynamic_ncols=True)
        running = 0.0; n_steps = 0

        for xL, notes_list, imgs, y_all, sens in pbar:
            y = y_all[:, [ {"mort":0, "pe":1, "ph":2}[TASK] ]]

            xL   = xL.to(DEVICE, non_blocking=amp_enabled)
            imgs = imgs.to(DEVICE, non_blocking=amp_enabled)
            y    = y.to(DEVICE, non_blocking=amp_enabled)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # Unimodal embeddings (encoders possibly frozen)
                if FINETUNE_ENCODERS:
                    zL = behrt(xL); zN = bbert(notes_list); zI = imgenc(imgs)
                else:
                    with torch.no_grad():
                        zL = behrt(xL); zN = bbert(notes_list); zI = imgenc(imgs)

                z = {"L": zL, "N": zN, "I": zI}

                # Frozen route logits + trainable LNI
                route_logits = _frozen_routes_from_unimodal(z)
                zLNI = fusion["LNI"](z["L"], z["N"], z["I"])
                route_logits["LNI"] = route_heads["LNI"](zLNI)  

                # Modality masks -> router
                masks = build_modality_masks(xL, notes_list, imgs)

                # Router -> final logits
                ylogits, route_w, block_w, block_logits = router(z, route_logits, masks=masks)  

                # Loss = BCE + lambda_fair * EDDI(final)
                bce  = bce_fn(ylogits, y)
                fair = eddi_final_from_logits(ylogits, y, sens)
                loss = bce + float(args.lambda_fair) * fair

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss); n_steps += 1
            pbar.set_postfix(loss=f"{running / n_steps:.4f}", bce=f"{float(bce):.4f}", fair=f"{float(fair):.4f}")

        behrt.eval(); bbert.eval(); imgenc.eval()
        fusion["LNI"].eval(); route_heads["LNI"].eval(); router.eval()

        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for xL, notes_list, imgs, y_all, sens in val_loader:
                y = y_all[:, [ {"mort":0, "pe":1, "ph":2}[TASK] ]]
                xL = xL.to(DEVICE, non_blocking=amp_enabled)
                imgs = imgs.to(DEVICE, non_blocking=amp_enabled)
                y = y.to(DEVICE, non_blocking=amp_enabled)

                zL = behrt(xL); zN = bbert(notes_list); zI = imgenc(imgs)
                z  = {"L": zL, "N": zN, "I": zI}

                route_logits = _frozen_routes_from_unimodal(z)
                zLNI = fusion["LNI"](z["L"], z["N"], z["I"])
                route_logits["LNI"] = route_heads["LNI"](zLNI)

                masks = build_modality_masks(xL, notes_list, imgs)

                ylogits, route_w, block_w, block_logits = router(z, route_logits, masks=masks)

                bce  = bce_fn(ylogits, y)
                fair = eddi_final_from_logits(ylogits, y, sens)
                lval = bce + float(args.lambda_fair) * fair

                val_loss += float(lval); n_val += 1

        val_loss /= max(n_val, 1)
        print(f"[TRI:{TASK}] Val loss: {val_loss:.4f}")

        # Save best router + LNI (+ LNI fusion)
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(CFG.ckpt_root, exist_ok=True)
            save_obj = {
                "router": router.state_dict(),
                "LNI": route_heads["LNI"].state_dict(),
                "best_val": best_val,
                "cfg": vars(CFG),
                "finetune_encoders": FINETUNE_ENCODERS,
                "train_trimodal_fusion": TRAIN_TRIMODAL_FUSION,
            }
            try:
                save_obj["fusion_LNI"] = fusion["LNI"].state_dict()
            except Exception:
                pass

            ckpt_out = os.path.join(CFG.ckpt_root, f"{TASK}_step3_trimodal_router.pt")
            torch.save(save_obj, ckpt_out)
            print(f"Saved best trimodal+router -> {ckpt_out}")


if __name__ == "__main__":
    main()
