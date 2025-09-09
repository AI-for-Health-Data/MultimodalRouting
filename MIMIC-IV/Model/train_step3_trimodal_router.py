from __future__ import annotations

import os
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from env_config import CFG, DEVICE, ROUTES
from encoders import EncoderConfig, build_encoders
from routing_and_heads import (
    RouteHead,
    build_fusions,
    FinalConcatHead,
    RouteGateNet,          
    route_availability_mask,
    make_route_inputs,
    concat_routes,
)

from PIL import Image
from torchvision.transforms import functional as VF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _cxr_to_tensor_medfuse(pil_img):
    pil_img = pil_img.convert("L")
    pil_img = VF.resize(pil_img, 256, antialias=True)
    pil_img = VF.center_crop(pil_img, 224)
    x = VF.to_tensor(pil_img)
    x = x.repeat(3, 1, 1)
    x = VF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x

def _load_first_cxr(paths):
    if not paths:
        return torch.zeros(3, 224, 224)
    p = paths[0]  
    try:
        img = Image.open(p)
        return _cxr_to_tensor_medfuse(img)
    except Exception:
        return torch.zeros(3, 224, 224)
        
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

def eddi_final_from_logits(ylogits: torch.Tensor, y: torch.Tensor, sens: List[dict]) -> torch.Tensor:
    probs = torch.sigmoid(ylogits).squeeze(1)  
    err = (probs - y.squeeze(1)).abs()         
    return eddi_sign_agnostic(err, sens, getattr(CFG, "sensitive_keys", []))


def build_modality_masks(
    xL: torch.Tensor,
    notes_list: List[List[str]],
    imgs: torch.Tensor
) -> Dict[str, torch.Tensor]:
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)
    mN_list = [
        1.0 if (isinstance(notes, list) and any((isinstance(t, str) and len(t.strip()) > 0) for t in notes)) else 0.0
        for notes in notes_list
    ]
    mN = torch.tensor(mN_list, device=xL.device, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        mI_vals = (imgs.abs().flatten(1).sum(dim=1) > 0).float()
    mI = mI_vals.to(xL.device).unsqueeze(1)
    return {"L": mL, "N": mN, "I": mI}


from train_step1_unimodal import ICUStayDataset, pad_or_trim_struct

def _is_cuda(dev) -> bool:
    return torch.cuda.is_available() and (("cuda" in str(dev)) or (isinstance(dev, torch.device) and dev.type == "cuda"))

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def main():
    ap = argparse.ArgumentParser(description="Step-3: Embedding-level fusion with loss-based gates + final concat head")
    ap.add_argument("--task", type=str, default=getattr(CFG, "task_name", "mort"), choices=["mort", "pe", "ph"])
    ap.add_argument("--lambda_fair", type=float, default=getattr(CFG, "lambda_fair", 0.0),
                    help="Weight for EDDI fairness term on final predictions")
    ap.add_argument("--gamma", type=float, default=getattr(CFG, "gamma", 1.0),
                    help="Mix: gamma * BCE(final) + (1-gamma) * EDDI(final)")
    ap.add_argument("--gate_mode", type=str, default=getattr(CFG, "route_gate_mode", "loss_based"),
                    choices=["loss_based", "learned", "uniform"],
                    help="How to compute route gates")
    ap.add_argument("--loss_gate_alpha", type=float, default=getattr(CFG, "loss_gate_alpha", 4.0),
                    help="Alpha for loss-based gates: softmax(-alpha * BCE_i)")
    ap.add_argument("--l2norm_each", action="store_true", default=getattr(CFG, "l2norm_each", False),
                    help="L2-normalize each route embedding before gating/concat")
    ap.add_argument("--train_lni_fusion", action="store_true", default=True,
                    help="If set, train the LNI fusion block")
    ap.add_argument("--train_lni_head_aux", action="store_true", default=True,
                    help="If set, train the LNI head with a small aux loss so its BCE is meaningful for gates")
    ap.add_argument("--aux_lni_weight", type=float, default=0.05,
                    help="Weight for the auxiliary LNI BCE (if train_lni_head_aux)")
    args = ap.parse_args()

    TASK = args.task

    TASK_MAP = {"mort": 0, "pe": 1, "ph": 2}
    tidx = TASK_MAP[TASK]

    def collate_fn(batch):
        T_len = CFG.structured_seq_len
        F_dim = CFG.structured_n_feats

        xL = torch.stack([pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0)

        notes = [
            b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            for b in batch
        ]

        imgs = torch.stack([_load_first_cxr(b["image_paths"]) for b in batch], dim=0)

        y_all = torch.stack([b["y"] for b in batch], dim=0)
        y = y_all[:, tidx].unsqueeze(1).to(torch.float32)

        sens = [{} for _ in batch]  
        return xL, notes, imgs, y, sens

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

    # Fusion blocks (bimodal + trimodal)
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)

    # Route heads for all 7 routes (logits used only for: (a) aux train of LNI, (b) computing loss-based gates)
    route_heads: Dict[str, RouteHead] = {
        r: RouteHead(d_in=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE) for r in ROUTES
    }

    # Final concat head over 7 * d concat
    final_head = FinalConcatHead(d=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE)

    gate_net = None
    if args.gate_mode == "learned":
        gate_net = RouteGateNet(d=CFG.d, hidden=4 * 256, p_drop=CFG.dropout, use_masks=True).to(DEVICE)

    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    train_ds = ICUStayDataset(ROOT, split="train")
    val_ds   = ICUStayDataset(ROOT, split="val")

    from torch.utils.data import DataLoader
    IS_CUDA = _is_cuda(DEVICE)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=IS_CUDA)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=IS_CUDA)

    # Class imbalance (pos_weight)
    try:
        y_train_np = train_ds.labels[[TASK]].values.astype("float32").reshape(-1)
        pos = float((y_train_np > 0.5).sum())
        neg = float(len(y_train_np) - pos)
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

    # Load bimodal fusion blocks if available
    if "fusion_LN" in ckpt2: fusion["LN"].load_state_dict(ckpt2["fusion_LN"], strict=False)
    if "fusion_LI" in ckpt2: fusion["LI"].load_state_dict(ckpt2["fusion_LI"], strict=False)
    if "fusion_NI" in ckpt2: fusion["NI"].load_state_dict(ckpt2["fusion_NI"], strict=False)

    print(f"[{TASK}] Loaded encoders/heads/fusions from {ckpt1_path} & {ckpt2_path}")

    for k in ["LN", "LI", "NI"]:
        set_requires_grad(fusion[k], False)
        fusion[k].eval()

    # Encoders frozen
    for m in (behrt, bbert, imgenc):
        set_requires_grad(m, False)
        m.eval()

    # All existing route heads frozen (used to compute per-route BCE for gates)
    for r in ["L", "N", "I", "LN", "LI", "NI"]:
        set_requires_grad(route_heads[r], False)
        route_heads[r].eval()

    set_requires_grad(fusion["LNI"], bool(args.train_lni_fusion))
    fusion["LNI"].train(args.train_lni_fusion)

    if args.train_lni_head_aux:
        route_heads["LNI"].train()
    else:
        set_requires_grad(route_heads["LNI"], False)
        route_heads["LNI"].eval()

    final_head.train()

    if gate_net is not None:
        gate_net.train()

    params = list(final_head.parameters())
    if args.train_lni_fusion:
        params += list(fusion["LNI"].parameters())
    if args.train_lni_head_aux:
        params += list(route_heads["LNI"].parameters())
    if gate_net is not None:
        params += list(gate_net.parameters())

    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=1e-2)
    amp_enabled = IS_CUDA and (str(getattr(CFG, "precision_amp", "auto")).lower() != "off")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def bce_logits(logits, target, reduction="mean"):
        return F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=(pos_weight if pos_weight is not None else None),
            reduction=reduction,
        )

    best_val = float("inf")


    for epoch in range(CFG.max_epochs_tri):
        final_head.train()
        fusion["LNI"].train(args.train_lni_fusion)
        if args.train_lni_head_aux:
            route_heads["LNI"].train()
        if gate_net is not None:
            gate_net.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_tri} [TRI:{TASK}]", dynamic_ncols=True)
        running = 0.0; n_steps = 0

        for xL, notes_list, imgs, y, sens in pbar:
            xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
            imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
            y    = y.to(DEVICE, non_blocking=IS_CUDA)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # Frozen encoders -> unimodal embeddings
                with torch.no_grad():
                    zL = behrt(xL)               
                    zN = bbert(notes_list)       
                    zI = imgenc(imgs)            
                z_dict = {"L": zL, "N": zN, "I": zI}

                # Build all 7 route embeddings (using fusion for LN/LI/NI/LNI)
                routes_emb: Dict[str, torch.Tensor] = make_route_inputs(z_dict, fusion)  

                # Per-route logits (for loss-based gates); all heads frozen except optional LNI
                logits_per_route: Dict[str, torch.Tensor] = {
                    "L":  route_heads["L"](routes_emb["L"]),
                    "N":  route_heads["N"](routes_emb["N"]),
                    "I":  route_heads["I"](routes_emb["I"]),
                    "LN": route_heads["LN"](routes_emb["LN"]),
                    "LI": route_heads["LI"](routes_emb["LI"]),
                    "NI": route_heads["NI"](routes_emb["NI"]),
                }
                # Trimodal head (trainable only if aux enabled)
                logits_per_route["LNI"] = route_heads["LNI"](routes_emb["LNI"])

                # Modality availability mask -> route availability
                masks = build_modality_masks(xL, notes_list, imgs)
                avail = route_availability_mask(masks, batch_size=xL.size(0), device=xL.device) 

                if args.gate_mode == "uniform":
                    gates = avail / (avail.sum(dim=1, keepdim=True).clamp_min(1.0))  
                elif args.gate_mode == "learned":
                    g_raw = gate_net({"L": zL, "N": zN, "I": zI}, masks=masks)  
                    gates = g_raw / (g_raw.sum(dim=1, keepdim=True).clamp_min(1e-6))
                else: 
                    per_route_losses = []
                    for r in ROUTES:
                        l_i = bce_logits(logits_per_route[r], y, reduction="none").squeeze(1)  
                        per_route_losses.append(l_i)
                    Lmat = torch.stack(per_route_losses, dim=1) 
                    alpha = float(args.loss_gate_alpha)
                    masked_logits = (-alpha * Lmat) + torch.log(avail + 1e-12)
                    gates = torch.softmax(masked_logits, dim=1)  

                x_cat, _ = concat_routes(routes_emb, gates=gates, l2norm=args.l2norm_each)  

                ylogits = final_head(x_cat)  

                bce_final  = bce_logits(ylogits, y, reduction="mean")
                eddi_final = eddi_final_from_logits(ylogits, y, sens)
                total = (float(args.gamma) * bce_final) + ((1.0 - float(args.gamma)) * (float(args.lambda_fair) * eddi_final))

                if args.train_lni_head_aux:
                    aux_lni = bce_logits(logits_per_route["LNI"], y, reduction="mean")
                    total = total + float(args.aux_lni_weight) * aux_lni
                else:
                    aux_lni = torch.tensor(0.0, device=y.device)

            scaler.scale(total).backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            running += float(total); n_steps += 1
            pbar.set_postfix(
                loss=f"{running / max(n_steps,1):.4f}",
                bce=f"{float(bce_final):.4f}",
                eddi=f"{float(eddi_final):.4f}",
                aux_lni=f"{float(aux_lni):.4f}",
            )


        final_head.eval()
        fusion["LNI"].eval()
        route_heads["LNI"].eval()
        if gate_net is not None:
            gate_net.eval()

        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for xL, notes_list, imgs, y, sens in val_loader:
                xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
                imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
                y    = y.to(DEVICE, non_blocking=IS_CUDA)

                zL = behrt(xL); zN = bbert(notes_list); zI = imgenc(imgs)
                z_dict = {"L": zL, "N": zN, "I": zI}

                routes_emb = make_route_inputs(z_dict, fusion)  

                logits_per_route = {
                    "L":  route_heads["L"](routes_emb["L"]),
                    "N":  route_heads["N"](routes_emb["N"]),
                    "I":  route_heads["I"](routes_emb["I"]),
                    "LN": route_heads["LN"](routes_emb["LN"]),
                    "LI": route_heads["LI"](routes_emb["LI"]),
                    "NI": route_heads["NI"](routes_emb["NI"]),
                    "LNI": route_heads["LNI"](routes_emb["LNI"]),
                }

                masks = build_modality_masks(xL, notes_list, imgs)
                avail = route_availability_mask(masks, batch_size=xL.size(0), device=xL.device)

                if args.gate_mode == "uniform":
                    gates = avail / (avail.sum(dim=1, keepdim=True).clamp_min(1.0))
                elif args.gate_mode == "learned":
                    g_raw = gate_net({"L": zL, "N": zN, "I": zI}, masks=masks)
                    gates = g_raw / (g_raw.sum(dim=1, keepdim=True).clamp_min(1e-6))
                else:
                    per_route_losses = []
                    for r in ROUTES:
                        l_i = bce_logits(logits_per_route[r], y, reduction="none").squeeze(1)
                        per_route_losses.append(l_i)
                    Lmat = torch.stack(per_route_losses, dim=1)
                    alpha = float(args.loss_gate_alpha)
                    masked_logits = (-alpha * Lmat) + torch.log(avail + 1e-12)
                    gates = torch.softmax(masked_logits, dim=1)

                x_cat, _ = concat_routes(routes_emb, gates=gates, l2norm=args.l2norm_each)
                ylogits = final_head(x_cat)

                bce_final  = bce_logits(ylogits, y, reduction="mean")
                eddi_final = eddi_final_from_logits(ylogits, y, sens)
                lval = (float(args.gamma) * bce_final) + ((1.0 - float(args.gamma)) * (float(args.lambda_fair) * eddi_final))


                if args.train_lni_head_aux:
                    lval = lval + float(args.aux_lni_weight) * bce_logits(logits_per_route["LNI"], y, reduction="mean")

                val_loss += float(lval); n_val += 1

        val_loss /= max(n_val, 1)
        print(f"[TRI:{TASK}] Val loss: {val_loss:.4f}")


        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(CFG.ckpt_root, exist_ok=True)
            save_obj = {
                "final_head": final_head.state_dict(),
                "best_val": best_val,
                "cfg": vars(CFG),
                "task": TASK,
                "gate_mode": args.gate_mode,
                "loss_gate_alpha": float(args.loss_gate_alpha),
                "l2norm_each": bool(args.l2norm_each),
                "gamma": float(args.gamma),
                "lambda_fair": float(args.lambda_fair),
            }
            if args.train_lni_fusion:
                save_obj["fusion_LNI"] = fusion["LNI"].state_dict()
            if args.train_lni_head_aux:
                save_obj["LNI_head"] = route_heads["LNI"].state_dict()
            if gate_net is not None:
                save_obj["gate_net"] = gate_net.state_dict()

      
            ckpt_out = os.path.join(CFG.ckpt_root, f"{TASK}_step3_concat_gate.pt")
            torch.save(save_obj, ckpt_out)
            print(f"Saved best model -> {ckpt_out}")


if __name__ == "__main__":
    main()
