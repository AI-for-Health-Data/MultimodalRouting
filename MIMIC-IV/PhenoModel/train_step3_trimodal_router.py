from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from env_config import CFG, DEVICE, ROUTES
from encoders import EncoderConfig, build_encoders
from routing_and_heads import (
    RouteHead,
    build_fusions,
    RouteGateNet,
    route_availability_mask,
    make_route_inputs,
    concat_routes,                
    FinalConcatHead,               
    LearnedClasswiseGateNet,      
    compute_loss_based_classwise_gates,  
    FinalConcatHeadClasswise,      
    concat_routes_classwise,       
    build_capsule_inputs,
)
from PhenoModel.capsule_layers import CapsuleFC   


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
        
def build_image_transform(split: str) -> T.Compose:
    split = str(split).lower()
    if split == "train":
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(256),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=10, translate=(0.05, 0.05),
                scale=(0.95, 1.05), shear=5,
            ),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def pad_or_trim_struct(x: torch.Tensor, T_len: int, F_dim: int) -> torch.Tensor:
    t = x.shape[0]
    if t >= T_len:
        return x[-T_len:]
    pad = torch.zeros(T_len - t, F_dim, dtype=x.dtype)
    return torch.cat([pad, x], dim=0)

def load_cxr_tensor(paths: List[str], tfms: T.Compose) -> torch.Tensor:
    if not paths:
        return torch.zeros(3, 224, 224)
    p = paths[-1]
    try:
        with Image.open(p) as img:
            return tfms(img)
    except Exception:
        return torch.zeros(3, 224, 224)

def _extract_ph_vector(row_df: pd.DataFrame, K: int) -> np.ndarray:
    ph_cols = [c for c in row_df.columns if c.startswith("ph_")]
    if len(ph_cols) >= 1:
        vals = row_df[ph_cols].values[0].astype(np.float32)
        out = np.zeros((K,), dtype=np.float32)
        m = min(K, vals.shape[0])
        out[:m] = vals[:m]
        return out

    if "ph" in row_df.columns:
        v = row_df["ph"].values[0]
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                v = []
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            out = np.zeros((K,), dtype=np.float32)
            m = min(K, arr.shape[0])
            out[:m] = arr[:m]
            return out
        out = np.zeros((K,), dtype=np.float32)
        try:
            out[0] = float(v)
        except Exception:
            pass
        return out

    return np.zeros((K,), dtype=np.float32)

class ICUStayDataset(Dataset):
    def __init__(self, root: str, split: str = "train", K: int = 25):
        super().__init__()
        self.root = root
        self.split = split
        self.K = K
        self.img_tfms = build_image_transform(split)

        with open(os.path.join(root, "splits.json")) as f:
            splits = json.load(f)
        self.ids = list(splits[split])

        self.struct = pd.read_parquet(os.path.join(root, "structured_24h.parquet"))
        self.notes  = pd.read_parquet(os.path.join(root, "notes_24h.parquet"))
        self.images = pd.read_parquet(os.path.join(root, "images_24h.parquet"))
        self.labels = pd.read_parquet(os.path.join(root, "labels.parquet"))

        base_cols = {"stay_id", "hour"}
        self.feat_cols = [c for c in self.struct.columns if c not in base_cols]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = self.ids[idx]

        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].to_numpy(dtype=np.float32)
        xs = torch.from_numpy(xs_np)

        notes_list = self.notes[self.notes.stay_id == stay_id].text.tolist()

        df_i = self.images[self.images.stay_id == stay_id]
        img_paths = df_i.image_path.tolist()[-1:]

        row_lbl = self.labels[self.labels.stay_id == stay_id]
        y_np = _extract_ph_vector(row_lbl, self.K)
        y = torch.tensor(y_np, dtype=torch.float32)

        return {
            "stay_id": stay_id,
            "x_struct": xs,
            "notes_list": notes_list,
            "image_paths": img_paths,
            "y": y,
        }

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
    probs = torch.sigmoid(ylogits)
    err_vec = (probs - y).abs().mean(dim=1)
    return eddi_sign_agnostic(err_vec, sens, getattr(CFG, "sensitive_keys", []))

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

def _is_cuda(dev) -> bool:
    return torch.cuda.is_available() and (("cuda" in str(dev)) or (isinstance(dev, torch.device) and dev.type == "cuda"))

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag

def compute_pos_weight_ph(train_ds: ICUStayDataset, K: int) -> Optional[torch.Tensor]:
    """Compute per-class pos_weight for multi-label BCE: neg/pos for each k."""
    try:
        ph_cols = [c for c in train_ds.labels.columns if c.startswith("ph_")]
        if len(ph_cols) >= 1:
            Y = train_ds.labels.loc[train_ds.labels.stay_id.isin(train_ds.ids), ph_cols].values.astype("float32")
            if Y.shape[1] != K:
                Ttmp = np.zeros((Y.shape[0], K), dtype="float32")
                m = min(K, Y.shape[1]); Ttmp[:, :m] = Y[:, :m]; Y = Ttmp
        else:
            rows = train_ds.labels.loc[train_ds.labels.stay_id.isin(train_ds.ids)]
            Y_list = []
            for _, r in rows.iterrows():
                Y_list.append(_extract_ph_vector(r.to_frame().T, K))
            Y = np.stack(Y_list, axis=0) if len(Y_list) > 0 else np.zeros((1, K), dtype="float32")

        pos = Y.sum(axis=0) + 1e-6
        neg = (Y.shape[0] - pos) + 1e-6
        w = torch.tensor(neg / pos, dtype=torch.float32, device=DEVICE)
        return w
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Step-3: Trimodal router for phenotypes (7-route; supports classwise gating)")

    # gating modes 
    ap.add_argument("--gate_mode", type=str, default=str(getattr(CFG, "route_gate_mode", "loss_based_classwise")),
                    choices=[
                        "uniform",                 
                        "learned",                
                        "loss_based",              
                        "learned_classwise",       
                        "loss_based_classwise",    
                    ],
                    help="Choose sample-level or per-phenotype (classwise) routing.")
    ap.add_argument("--loss_gate_alpha", type=float, default=float(getattr(CFG, "loss_gate_alpha", 4.0)),
                    help="Alpha for softmax(-alpha * BCE). Used by loss_based(_classwise).")

    # modeling knobs 
    ap.add_argument("--l2norm_each", action="store_true", default=bool(getattr(CFG, "l2norm_each", False)),
                    help="L2-normalize each route embedding before gating/concat.")

    # fairness mix
    ap.add_argument("--lambda_fair", type=float, default=float(getattr(CFG, "lambda_fair", 0.0)),
                    help="Weight for EDDI fairness on final predictions.")
    ap.add_argument("--gamma", type=float, default=float(getattr(CFG, "gamma", 1.0)),
                    help="Mix: gamma * BCE(final) + (1-gamma) * EDDI(final).")

    # Trainable pieces
    ap.add_argument("--train_lni_fusion", action="store_true", default=True,
                    help="Train the LNI fusion block.")
    ap.add_argument("--train_lni_head_aux", action="store_true", default=True,
                    help="Add auxiliary BCE on the LNI head so its logits are meaningful.")
    ap.add_argument("--aux_lni_weight", type=float, default=0.05,
                    help="Weight for the auxiliary LNI BCE.")

    # Fusion choices
    ap.add_argument("--feature_mode", type=str,
                    default=str(getattr(CFG, "feature_mode", "concat")),
                    choices=["rich", "concat"],
                    help="Feature construction for MLP fusion.")
    ap.add_argument("--bi_fusion_mode", type=str,
                    default=str(getattr(CFG, "bi_fusion_mode", "mlp")),
                    choices=["mlp", "attn"],
                    help="Bimodal fusion type for LN/LI/NI (used only for route embeddings here).")
    ap.add_argument("--bi_layers", type=int, default=int(getattr(CFG, "bi_layers", 2)),
                    help="Attention layers for bi_fusion_mode=attn.")
    ap.add_argument("--bi_heads", type=int, default=int(getattr(CFG, "bi_heads", 4)),
                    help="Attention heads for bi_fusion_mode=attn.")
    ap.add_argument("--tri_fusion_mode", type=str,
                    default=str(getattr(CFG, "tri_fusion_mode", "mlp")),
                    choices=["mlp", "attn"],
                    help="Trimodal fusion type for LNI (trained in Step-3).")
    ap.add_argument("--tri_layers", type=int, default=int(getattr(CFG, "tri_layers", 2)),
                    help="Attention layers for tri_fusion_mode=attn.")
    ap.add_argument("--tri_heads", type=int, default=int(getattr(CFG, "tri_heads", 4)),
                    help="Attention heads for tri_fusion_mode=attn.")

    args = ap.parse_args()

    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    K = int(getattr(CFG, "num_phenotypes", 25))

    train_ds = ICUStayDataset(ROOT, split="train", K=K)
    val_ds   = ICUStayDataset(ROOT, split="val",   K=K)

    pos_weight = compute_pos_weight_ph(train_ds, K)

    IS_CUDA = _is_cuda(DEVICE)
    gen = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers,
        collate_fn=collate_fn_factory(K, img_tfms=train_ds.img_tfms),
        pin_memory=IS_CUDA, generator=gen, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers,
        collate_fn=collate_fn_factory(K, img_tfms=val_ds.img_tfms),
        pin_memory=IS_CUDA, generator=gen, drop_last=False,
    )

    # encoders 
    behrt, bbert, imgenc = build_encoders(
        EncoderConfig(
            d=CFG.d,
            dropout=CFG.dropout,
            structured_seq_len=CFG.structured_seq_len,
            structured_n_feats=CFG.structured_n_feats,
            text_model_name=CFG.text_model_name,
            text_max_len=CFG.max_text_len,
            note_agg=getattr(CFG, "text_note_agg", "mean"),
            max_notes_concat=getattr(CFG, "max_notes_concat", 8),
            img_agg=getattr(CFG, "image_agg", "last"),
        )
    )

    # fusions 
    fusion = build_fusions(
        d=CFG.d,
        p_drop=CFG.dropout,
        feature_mode=args.feature_mode,
        bi_fusion_mode=args.bi_fusion_mode,
        tri_fusion_mode=args.tri_fusion_mode,
        bi_layers=args.bi_layers,
        bi_heads=args.bi_heads,
        tri_layers=args.tri_layers,
        tri_heads=args.tri_heads,
    )

    # route heads (all 7, each emits K logits) 
    route_heads: Dict[str, RouteHead] = {
        r: RouteHead(d_in=CFG.d, n_tasks=K, p_drop=CFG.dropout).to(DEVICE) for r in ROUTES
    }

    # final heads (sample-level and classwise)
    final_head_sample = FinalConcatHead(d=CFG.d, n_tasks=K, p_drop=CFG.dropout).to(DEVICE)
    final_head_cls    = FinalConcatHeadClasswise(d=CFG.d, p_drop=CFG.dropout).to(DEVICE)

    # gates 
    gate_net_sample = None
    gate_net_cls    = None
    if args.gate_mode == "learned":
        gate_net_sample = RouteGateNet(d=CFG.d, hidden=4 * 256, p_drop=CFG.dropout, use_masks=True).to(DEVICE)
    if args.gate_mode == "learned_classwise":
        gate_net_cls = LearnedClasswiseGateNet(d=CFG.d, n_tasks=K, hidden=1024, p_drop=CFG.dropout, use_masks=True).to(DEVICE)

    ckpt1_path = os.path.join(CFG.ckpt_root, "ph_step1_unimodal.pt")
    ckpt2_path = os.path.join(CFG.ckpt_root, "ph_step2_bimodal.pt")
    ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)
    ckpt2 = torch.load(ckpt2_path, map_location=DEVICE)

    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)

    # Unimodal heads
    route_heads["L"].load_state_dict(ckpt1["L"], strict=False)
    route_heads["N"].load_state_dict(ckpt1["N"], strict=False)
    route_heads["I"].load_state_dict(ckpt1["I"], strict=False)

    # Bimodal heads
    route_heads["LN"].load_state_dict(ckpt2["LN"], strict=False)
    route_heads["LI"].load_state_dict(ckpt2["LI"], strict=False)
    route_heads["NI"].load_state_dict(ckpt2["NI"], strict=False)

    # Bimodal fusion blocks
    if "fusion_LN" in ckpt2: fusion["LN"].load_state_dict(ckpt2["fusion_LN"], strict=False)
    if "fusion_LI" in ckpt2: fusion["LI"].load_state_dict(ckpt2["fusion_LI"], strict=False)
    if "fusion_NI" in ckpt2: fusion["NI"].load_state_dict(ckpt2["fusion_NI"], strict=False)

    print(f"[ph] Loaded encoders/heads/fusions from {ckpt1_path} & {ckpt2_path}")

    # freeze encoders + frozen fusions/heads except what we train in step-3 
    for m in (behrt, bbert, imgenc):
        set_requires_grad(m, False); m.eval()

    for k in ["LN", "LI", "NI"]:
        set_requires_grad(fusion[k], False); fusion[k].eval()

    for r in ["L", "N", "I", "LN", "LI", "NI"]:
        set_requires_grad(route_heads[r], False); route_heads[r].eval()

    # train the trimodal fusion if requested
    set_requires_grad(fusion["LNI"], bool(args.train_lni_fusion))
    fusion["LNI"].train(args.train_lni_fusion)

    # aux LNI head if requested
    if args.train_lni_head_aux:
        route_heads["LNI"].train()
    else:
        set_requires_grad(route_heads["LNI"], False); route_heads["LNI"].eval()

    # choose which final head to train based on gate_mode
    use_classwise = args.gate_mode in {"learned_classwise", "loss_based_classwise"}
    if use_classwise:
        final_head_cls.train()
        final_head_sample.eval(); set_requires_grad(final_head_sample, False)
    else:
        final_head_sample.train()
        final_head_cls.eval(); set_requires_grad(final_head_cls, False)

    # gate nets train/eval
    if gate_net_sample is not None:
        gate_net_sample.train()
    if gate_net_cls is not None:
        gate_net_cls.train()

    params: List[nn.Parameter] = []
    if use_classwise:
        params += list(final_head_cls.parameters())
    else:
        params += list(final_head_sample.parameters())

    if args.train_lni_fusion:
        params += list(fusion["LNI"].parameters())
    if args.train_lni_head_aux:
        params += list(route_heads["LNI"].parameters())
    if gate_net_sample is not None:
        params += list(gate_net_sample.parameters())
    if gate_net_cls is not None:
        params += list(gate_net_cls.parameters())

    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=getattr(CFG, "weight_decay", 1e-2))
    amp_enabled = IS_CUDA and (str(getattr(CFG, "precision_amp", "auto")).lower() != "off")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def bce_logits(logits: torch.Tensor, target: torch.Tensor, reduction="mean"):
        return F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=(pos_weight if pos_weight is not None else None),
            reduction=reduction,
        )

    best_val = float("inf")

    for epoch in range(CFG.max_epochs_tri):
        fusion["LNI"].train(args.train_lni_fusion)
        if args.train_lni_head_aux:
            route_heads["LNI"].train()
        else:
            route_heads["LNI"].eval()

        if use_classwise:
            final_head_cls.train()
            final_head_sample.eval()
        else:
            final_head_sample.train()
            final_head_cls.eval()

        if gate_net_sample is not None:
            gate_net_sample.train()
        if gate_net_cls is not None:
            gate_net_cls.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_tri} [TRI:ph|{args.tri_fusion_mode}|{args.gate_mode}]", dynamic_ncols=True)
        running = 0.0; n_steps = 0

        for xL, notes_list, imgs, y, sens in pbar:
            xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
            imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
            y    = y.to(DEVICE, non_blocking=IS_CUDA)  

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # Encoders (frozen)
                with torch.no_grad():
                    zL = behrt(xL)          
                    zN = bbert(notes_list)  
                    zI = imgenc(imgs)       
                z_dict = {"L": zL, "N": zN, "I": zI}

                # 7 route embeddings
                routes_emb: Dict[str, torch.Tensor] = make_route_inputs(z_dict, fusion) 

                # Per-route logits (K each)
                logits_per_route: Dict[str, torch.Tensor] = {
                    "L":   route_heads["L"](routes_emb["L"]),
                    "N":   route_heads["N"](routes_emb["N"]),
                    "I":   route_heads["I"](routes_emb["I"]),
                    "LN":  route_heads["LN"](routes_emb["LN"]),
                    "LI":  route_heads["LI"](routes_emb["LI"]),
                    "NI":  route_heads["NI"](routes_emb["NI"]),
                    "LNI": route_heads["LNI"](routes_emb["LNI"]),
                }  # each [B,K]

                # Availability
                masks = build_modality_masks(xL, notes_list, imgs)
                avail = route_availability_mask(masks, batch_size=xL.size(0), device=xL.device)  # [B,7]

                # GATES & FINAL 
                if use_classwise:
                    # gates_cls: [B,7,K]
                    if args.gate_mode == "learned_classwise":
                        gates_cls = gate_net_cls(z_dict, masks=masks)  # [B,7,K]
                    else:
                        # loss_based_classwise
                        gates_cls = compute_loss_based_classwise_gates(
                            logits_per_route, y, avail, alpha=float(args.loss_gate_alpha),
                            pos_weight=pos_weight
                        )  # [B,7,K]

                    # Concat classwise and final logits
                    X_flat, Zw_cls = concat_routes_classwise(
                        routes_emb, gates_classwise=gates_cls, l2norm=bool(args.l2norm_each)
                    )  # X_flat: [B*K, 7*d]
                    ylogits = final_head_cls(X_flat, B=xL.size(0), K=int(K))  # [B,K]

                else:
                    # SAMPLE-LEVEL 
                    if args.gate_mode == "uniform":
                        gates = avail / (avail.sum(dim=1, keepdim=True).clamp_min(1.0))  # [B,7]
                    elif args.gate_mode == "learned":
                        g_raw = gate_net_sample({"L": zL, "N": zN, "I": zI}, masks=masks)  # [B,7]
                        gates = g_raw / (g_raw.sum(dim=1, keepdim=True).clamp_min(1e-6))
                    else:
                        # loss_based (sample-level)
                        per_route_losses = []
                        for r in ROUTES:
                            l_elem = bce_logits(logits_per_route[r], y, reduction="none")  # [B,K]
                            l_mean = l_elem.mean(dim=1)                                     # [B]
                            per_route_losses.append(l_mean)
                        Lmat = torch.stack(per_route_losses, dim=1)  # [B,7]
                        alpha = float(args.loss_gate_alpha)
                        masked_logits = (-alpha * Lmat) + torch.log(avail + 1e-12)  # mask via -inf
                        gates = torch.softmax(masked_logits, dim=1)  # [B,7]

                    x_cat, _ = concat_routes(routes_emb, gates=gates, l2norm=bool(args.l2norm_each))  # [B, 7*d]
                    ylogits = final_head_sample(x_cat)                                               # [B,K]

                bce_final  = bce_logits(ylogits, y, reduction="mean")
                eddi_final = eddi_final_from_logits(ylogits, y, sens)
                total = (float(args.gamma) * bce_final) + ((1.0 - float(args.gamma)) * (float(args.lambda_fair) * eddi_final))

                if args.train_lni_head_aux:
                    aux_lni = bce_logits(logits_per_route["LNI"], y, reduction="mean")
                    total = total + float(args.aux_lni_weight) * aux_lni
                else:
                    aux_lni = torch.tensor(0.0, device=y.device)

            scaler.scale(total).backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=getattr(CFG, "grad_clip_norm", 1.0))
            scaler.step(opt)
            scaler.update()

            running += float(total); n_steps += 1
            pbar.set_postfix(
                loss=f"{running / max(n_steps,1):.4f}",
                bce=f"{float(bce_final):.4f}",
                eddi=f"{float(eddi_final):.4f}",
                aux_lni=f"{float(aux_lni):.4f}",
            )

        fusion["LNI"].eval()
        route_heads["LNI"].eval()
        if use_classwise:
            final_head_cls.eval()
        else:
            final_head_sample.eval()
        if gate_net_sample is not None:
            gate_net_sample.eval()
        if gate_net_cls is not None:
            gate_net_cls.eval()

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
                    "L":   route_heads["L"](routes_emb["L"]),
                    "N":   route_heads["N"](routes_emb["N"]),
                    "I":   route_heads["I"](routes_emb["I"]),
                    "LN":  route_heads["LN"](routes_emb["LN"]),
                    "LI":  route_heads["LI"](routes_emb["LI"]),
                    "NI":  route_heads["NI"](routes_emb["NI"]),
                    "LNI": route_heads["LNI"](routes_emb["LNI"]),
                }

                masks = build_modality_masks(xL, notes_list, imgs)
                avail = route_availability_mask(masks, batch_size=xL.size(0), device=xL.device)

                if use_classwise:
                    if args.gate_mode == "learned_classwise":
                        gates_cls = gate_net_cls(z_dict, masks=masks)  # [B,7,K]
                    else:
                        gates_cls = compute_loss_based_classwise_gates(
                            logits_per_route, y, avail, alpha=float(args.loss_gate_alpha),
                            pos_weight=pos_weight
                        )  # [B,7,K]

                    X_flat, Zw_cls = concat_routes_classwise(
                        routes_emb, gates_classwise=gates_cls, l2norm=bool(args.l2norm_each)
                    )
                    ylogits = final_head_cls(X_flat, B=xL.size(0), K=int(K))

                else:
                    if args.gate_mode == "uniform":
                        gates = avail / (avail.sum(dim=1, keepdim=True).clamp_min(1.0))
                    elif args.gate_mode == "learned":
                        g_raw = gate_net_sample({"L": zL, "N": zN, "I": zI}, masks=masks)
                        gates = g_raw / (g_raw.sum(dim=1, keepdim=True).clamp_min(1e-6))
                    else:
                        per_route_losses = []
                        for r in ROUTES:
                            l_elem = F.binary_cross_entropy_with_logits(
                                logits_per_route[r], y,
                                pos_weight=(pos_weight if pos_weight is not None else None),
                                reduction="none",
                            )
                            per_route_losses.append(l_elem.mean(dim=1))
                        Lmat = torch.stack(per_route_losses, dim=1)
                        alpha = float(args.loss_gate_alpha)
                        masked_logits = (-alpha * Lmat) + torch.log(avail + 1e-12)
                        gates = torch.softmax(masked_logits, dim=1)

                    x_cat, _ = concat_routes(routes_emb, gates=gates, l2norm=bool(args.l2norm_each))
                    ylogits = final_head_sample(x_cat)

                bce_final  = F.binary_cross_entropy_with_logits(
                    ylogits, y, pos_weight=(pos_weight if pos_weight is not None else None), reduction="mean"
                )
                eddi_final = eddi_final_from_logits(ylogits, y, sens)
                lval = (float(args.gamma) * bce_final) + ((1.0 - float(args.gamma)) * (float(args.lambda_fair) * eddi_final))

                if args.train_lni_head_aux:
                    lval = lval + float(args.aux_lni_weight) * F.binary_cross_entropy_with_logits(
                        logits_per_route["LNI"], y, pos_weight=(pos_weight if pos_weight is not None else None), reduction="mean"
                    )

                val_loss += float(lval); n_val += 1

        val_loss /= max(n_val, 1)
        print(f"[TRI:ph|{args.gate_mode}] Val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(CFG.ckpt_root, exist_ok=True)
            save_obj: Dict[str, Any] = {
                "best_val": best_val,
                "cfg": dict(CFG.__dict__) if hasattr(CFG, "__dict__") else {},
                "task": "ph",
                "num_phenotypes": K,
                "gate_mode": args.gate_mode,
                "loss_gate_alpha": float(args.loss_gate_alpha),
                "l2norm_each": bool(args.l2norm_each),
                "gamma": float(args.gamma),
                "lambda_fair": float(args.lambda_fair),
                "tri_fusion_mode": args.tri_fusion_mode,
                "tri_layers": int(args.tri_layers),
                "tri_heads": int(args.tri_heads),
            }
            if args.train_lni_fusion:
                save_obj["fusion_LNI"] = fusion["LNI"].state_dict()
            if args.train_lni_head_aux:
                save_obj["LNI_head"] = route_heads["LNI"].state_dict()

            # save the right final head + gate
            if use_classwise:
                save_obj["final_head_cls"] = final_head_cls.state_dict()
                if gate_net_cls is not None:
                    save_obj["gate_net_cls"] = gate_net_cls.state_dict()
            else:
                save_obj["final_head"] = final_head_sample.state_dict()
                if gate_net_sample is not None:
                    save_obj["gate_net"] = gate_net_sample.state_dict()

            ckpt_out = os.path.join(CFG.ckpt_root, "ph_step3_concat_gate.pt")
            torch.save(save_obj, ckpt_out)
            print(f"[ph] Saved best router -> {ckpt_out}")

if __name__ == "__main__":
    main()
