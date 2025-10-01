from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from env_config import CFG, DEVICE, ensure_dir
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead, build_fusions


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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step-2: Bimodal training (LN, LI, NI) for multi-label phenotyping"
    )
    ap.add_argument("--data_root", type=str, default=CFG.data_root,
                    help="Root that contains MIMIC-IV folder (parquets + splits.json).")
    ap.add_argument("--ckpt_root", type=str, default=CFG.ckpt_root,
                    help="Where to save checkpoints.")
    ap.add_argument("--bi_fusion_mode", type=str,
                    default=str(getattr(CFG, "bi_fusion_mode", "mlp")),
                    choices=["mlp", "attn"],
                    help="Bimodal fusion type for LN/LI/NI.")
    ap.add_argument("--feature_mode", type=str,
                    default=str(getattr(CFG, "feature_mode", "concat")),
                    choices=["rich", "concat"],
                    help="Feature construction for MLP fusion.")
    ap.add_argument("--bi_layers", type=int,
                    default=int(getattr(CFG, "bi_layers", 2)),
                    help="Attention blocks if bi_fusion_mode='attn'.")
    ap.add_argument("--bi_heads", type=int,
                    default=int(getattr(CFG, "bi_heads", 4)),
                    help="Attention heads if bi_fusion_mode='attn'.")
    return ap.parse_args()


def pad_or_trim_struct(x: torch.Tensor, T: int, F: int) -> torch.Tensor:
    t = x.shape[0]
    if t >= T:
        return x[-T:]
    pad = torch.zeros(T - t, F, dtype=x.dtype)
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
        if vals.shape[0] != K:
            out = np.zeros((K,), dtype=np.float32)
            m = min(K, vals.shape[0])
            out[:m] = vals[:m]
            return out
        return vals.astype(np.float32)

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

        # Structured: 48h window binned into 24 rows (2h bins)
        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].to_numpy(dtype=np.float32)
        xs = torch.from_numpy(xs_np)

        # Notes: list of pre-chunked texts 
        notes_list = self.notes[self.notes.stay_id == stay_id].text.tolist()

        # Images: exporter already picked ONE selected path per stay -> use last
        df_i = self.images[self.images.stay_id == stay_id]
        img_paths = df_i.image_path.tolist()[-1:]

        # Phenotypes K-dim vector
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


def collate_fn_factory(K: int, img_tfms: T.Compose):
    def _collate(batch: List[Dict[str, Any]]):
        T_len, F_dim = CFG.structured_seq_len, CFG.structured_n_feats

        xL_batch = torch.stack(
            [pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0
        )

        notes_batch = [
            b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            for b in batch
        ]

        imgs_batch = torch.stack(
            [load_cxr_tensor(b["image_paths"], img_tfms) for b in batch], dim=0
        )

        y_batch = torch.stack([b["y"] for b in batch], dim=0).to(torch.float32) 
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)

        return xL_batch, notes_batch, imgs_batch, y_batch
    return _collate


def set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def is_cuda() -> bool:
    return torch.cuda.is_available() and ("cuda" in str(DEVICE))

def compute_pos_weight_ph(train_ds: ICUStayDataset, K: int) -> Optional[torch.Tensor]:
    try:
        ph_cols = [c for c in train_ds.labels.columns if c.startswith("ph_")]
        if len(ph_cols) >= 1:
            Y = train_ds.labels.loc[train_ds.labels.stay_id.isin(train_ds.ids), ph_cols].values.astype("float32")
            if Y.shape[1] != K:
                # pad/trim
                Ttmp = np.zeros((Y.shape[0], K), dtype="float32")
                m = min(K, Y.shape[1])
                Ttmp[:, :m] = Y[:, :m]
                Y = Ttmp
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
    args = parse_args()
    ROOT = os.path.join(args.data_root, "MIMIC-IV")
    K = int(getattr(CFG, "num_phenotypes", 25))

    # Build encoders (will be frozen here)
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

    # Build pairwise fusion modules (LN, LI, NI)
    fusion = build_fusions(
        d=CFG.d,
        p_drop=CFG.dropout,
        feature_mode=args.feature_mode,
        bi_fusion_mode=args.bi_fusion_mode,
        bi_layers=args.bi_layers,
        bi_heads=args.bi_heads,
    )

    # Three bimodal heads (each outputs K logits)
    route_heads: Dict[str, RouteHead] = {
        r: RouteHead(d_in=CFG.d, n_tasks=K, p_drop=CFG.dropout).to(DEVICE)
        for r in ["LN", "LI", "NI"]
    }

    train_ds = ICUStayDataset(ROOT, split="train", K=K)
    val_ds   = ICUStayDataset(ROOT, split="val",   K=K)

    pos_weight = compute_pos_weight_ph(train_ds, K)  

    IS_CUDA = is_cuda()
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

    # Load Step-1 encoders & freeze
    ckpt1_path = os.path.join(args.ckpt_root, "ph_step1_unimodal.pt")
    ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)
    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)
    print(f"[ph] Loaded Step-1 encoders from {ckpt1_path}")

    set_requires_grad(behrt, False)
    set_requires_grad(bbert, False)
    set_requires_grad(imgenc, False)
    behrt.eval(); bbert.eval(); imgenc.eval()
    print("Encoders frozen.")

    # Optimizer (fusions + heads only)
    params_bi = (
        list(fusion["LN"].parameters())
        + list(fusion["LI"].parameters())
        + list(fusion["NI"].parameters())
        + list(route_heads["LN"].parameters())
        + list(route_heads["LI"].parameters())
        + list(route_heads["NI"].parameters())
    )
    opt = torch.optim.AdamW(params_bi, lr=CFG.lr, weight_decay=getattr(CFG, "weight_decay", 1e-2))

    amp_enabled = IS_CUDA and (str(getattr(CFG, "precision_amp", "auto")).lower() != "off")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val = float("inf")

    # Train
    for epoch in range(CFG.max_epochs_bi):
        for k in ["LN", "LI", "NI"]:
            fusion[k].train()
            route_heads[k].train()
        behrt.eval(); bbert.eval(); imgenc.eval()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_bi} [BI:ph|{args.bi_fusion_mode}]", dynamic_ncols=True)
        running = 0.0; n_steps = 0

        for xL, notes_list, imgs, y in pbar:
            xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
            imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
            y    = y.to(DEVICE, non_blocking=IS_CUDA)     

            opt.zero_grad(set_to_none=True)

            # Encoders (frozen)
            with torch.no_grad():
                zL = behrt(xL)               
                zN = bbert(notes_list)       
                zI = imgenc(imgs)            

            # Trainable fusion + heads
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                zLN = fusion["LN"](zL, zN)   
                zLI = fusion["LI"](zL, zI)   
                zNI = fusion["NI"](zN, zI)  

                log_LN = route_heads["LN"](zLN)  
                log_LI = route_heads["LI"](zLI)  
                log_NI = route_heads["NI"](zNI)  

                bce_kw = {"pos_weight": pos_weight} if (pos_weight is not None) else {}
                loss = (
                    F.binary_cross_entropy_with_logits(log_LN, y, **bce_kw)
                    + F.binary_cross_entropy_with_logits(log_LI, y, **bce_kw)
                    + F.binary_cross_entropy_with_logits(log_NI, y, **bce_kw)
                ) / 3.0

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params_bi, max_norm=getattr(CFG, "grad_clip_norm", 1.0))
            scaler.step(opt)
            scaler.update()

            running += float(loss); n_steps += 1
            pbar.set_postfix(loss=f"{running / max(n_steps,1):.4f}")

        # Validation
        for k in ["LN", "LI", "NI"]:
            fusion[k].eval()
            route_heads[k].eval()

        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for xL, notes_list, imgs, y in val_loader:
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
        print(f"[BI:ph|{args.bi_fusion_mode}] Val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ensure_dir(args.ckpt_root)
            ckpt_path = os.path.join(args.ckpt_root, "ph_step2_bimodal.pt")
            torch.save(
                {
                    "LN": route_heads["LN"].state_dict(),
                    "LI": route_heads["LI"].state_dict(),
                    "NI": route_heads["NI"].state_dict(),
                    "fusion_LN": fusion["LN"].state_dict(),
                    "fusion_LI": fusion["LI"].state_dict(),
                    "fusion_NI": fusion["NI"].state_dict(),
                    "best_val": best_val,
                    "task": "ph",
                    "cfg": dict(CFG.__dict__) if hasattr(CFG, "__dict__") else {},
                    "num_phenotypes": K,
                    "bi_fusion_mode": args.bi_fusion_mode,
                    "feature_mode": args.feature_mode,
                    "bi_layers": args.bi_layers,
                    "bi_heads": args.bi_heads,
                },
                ckpt_path,
            )
            print(f"[ph] Saved best bimodal heads + fusions -> {ckpt_path}")


if __name__ == "__main__":
    main()
