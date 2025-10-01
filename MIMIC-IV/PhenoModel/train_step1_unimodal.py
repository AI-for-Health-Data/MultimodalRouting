from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from env_config import CFG, DEVICE, ensure_dir
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead


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
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5,
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
        description="Step-1 Unimodal pretraining (L, N, I) for phenotype task"
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default=CFG.data_root,
        help="Root path that contains MIMIC-IV folder (parquets + splits.json).",
    )
    ap.add_argument(
        "--ckpt_root",
        type=str,
        default=CFG.ckpt_root,
        help="Where to save checkpoints.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=CFG.num_workers,
        help="DataLoader workers.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=CFG.batch_size,
        help="Batch size.",
    )
    return ap.parse_args()


def _natural_sort_key(col: str) -> Tuple[int, ...]:
    try:
        base, idx = col.split("_")
        return (int(idx),)
    except Exception:
        return (10**9,)

def _parse_pheno_value(v, K: int) -> np.ndarray:
    if v is None:
        return np.zeros(K, dtype="float32")
    if isinstance(v, (list, tuple, np.ndarray)):
        v = np.array(v, dtype="float32")
        if v.ndim == 1 and v.size == K:
            return (v > 0).astype("float32")
        if v.ndim == 1 and v.size > 0 and v.dtype.kind in {"i", "u"}:
            out = np.zeros(K, dtype="float32")
            out[np.clip(v, 0, K - 1).astype(int)] = 1.0
            return out
        return np.zeros(K, dtype="float32")
    if isinstance(v, str):
        try:
            arr = json.loads(v)
            return _parse_pheno_value(arr, K)
        except Exception:
            return np.zeros(K, dtype="float32")
    try:
        s = float(v)
        return np.zeros(K, dtype="float32")
    except Exception:
        return np.zeros(K, dtype="float32")


class ICUStayDataset(Dataset):
    def __init__(self, root: str, split: str = "train", K: Optional[int] = None):
        super().__init__()
        self.root = root
        self.split = split
        self.img_tfms = build_image_transform(split)

        with open(os.path.join(root, "splits.json"), "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.ids: List[int] = list(splits[split])

        self.struct = pd.read_parquet(os.path.join(root, "structured_24h.parquet"))
        self.notes  = pd.read_parquet(os.path.join(root, "notes_24h.parquet"))
        self.images = pd.read_parquet(os.path.join(root, "images_24h.parquet"))
        self.labels = pd.read_parquet(os.path.join(root, "labels.parquet"))

        base_cols = {"stay_id", "hour"}
        self.feat_cols: List[str] = [c for c in self.struct.columns if c not in base_cols]

        # Phenotype layout
        self.K: int = int(K if K is not None else getattr(CFG, "num_phenotypes", 25))
        self.has_single_ph_col: bool = "ph" in self.labels.columns
        if not self.has_single_ph_col:
            self.ph_cols: List[str] = sorted(
                [c for c in self.labels.columns if c.startswith("ph_")],
                key=_natural_sort_key
            )
            assert len(self.ph_cols) == self.K, (
                f"Found {len(self.ph_cols)} ph_* columns but K={self.K}."
            )

    def __len__(self) -> int:
        return len(self.ids)

    def _get_y_row(self, stay_id: int) -> torch.Tensor:
        if self.has_single_ph_col:
            v = self.labels.loc[self.labels.stay_id == stay_id, "ph"].values
            v = v[0] if len(v) > 0 else None
            y_np = _parse_pheno_value(v, self.K)
        else:
            row = self.labels.loc[self.labels.stay_id == stay_id, self.ph_cols].values
            if len(row) == 0:
                y_np = np.zeros(self.K, dtype="float32")
            else:
                y_np = row[0].astype("float32")
        return torch.tensor(y_np, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict:
        stay_id = self.ids[idx]

        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].to_numpy(dtype=np.float32)
        xs = torch.from_numpy(xs_np)  

        notes_list = self.notes[self.notes.stay_id == stay_id].text.tolist()
        notes_list = [str(t) if t is not None else "" for t in notes_list]

        df_i = self.images[self.images.stay_id == stay_id]
        img_paths = df_i.image_path.tolist()[-1:]  
        y = self._get_y_row(stay_id)

        return {
            "stay_id": stay_id,
            "x_struct": xs,          
            "notes_list": notes_list,
            "image_paths": img_paths,
            "y": y,                  
        }


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

def collate_fn_factory(img_tfms: T.Compose):
    def _collate(batch: List[Dict]):
        T_len, F_dim = CFG.structured_seq_len, CFG.structured_n_feats

        xL_batch = torch.stack(
            [pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0
        )  

        notes_batch: List[List[str]] = [
            b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            for b in batch
        ]

        imgs_batch = torch.stack(
            [load_cxr_tensor(b["image_paths"], img_tfms) for b in batch], dim=0
        )  

        y_batch = torch.stack([b["y"] for b in batch], dim=0)  

        return xL_batch, notes_batch, imgs_batch, y_batch

    return _collate


def main():
    args = parse_args()
    ROOT = os.path.join(args.data_root, "MIMIC-IV")
    K = int(getattr(CFG, "num_phenotypes", 25))

    behrt, bbert, imgenc = build_encoders(
        EncoderConfig(
            d=CFG.d,
            dropout=CFG.dropout,
            structured_seq_len=CFG.structured_seq_len,
            structured_n_feats=CFG.structured_n_feats,
            structured_layers=CFG.structured_layers,
            structured_heads=CFG.structured_heads,
            structured_pool=CFG.structured_pool,
            text_model_name=CFG.text_model_name,
            text_max_len=CFG.max_text_len,
            note_agg=CFG.text_note_agg,
            max_notes_concat=CFG.max_notes_concat,
            img_agg=CFG.image_agg,
        )
    )

    # Unimodal heads (L, N, I): K logits each (multi-label phenotypes)
    route_heads: Dict[str, RouteHead] = {
        r: RouteHead(d_in=CFG.d, n_tasks=K, p_drop=CFG.dropout).to(DEVICE)
        for r in ["L", "N", "I"]
    }


    train_ds = ICUStayDataset(ROOT, split="train", K=K)
    val_ds   = ICUStayDataset(ROOT, split="val",   K=K)

    try:
        if "ph" in train_ds.labels.columns:
            # parse each row's 'ph' to multi-hot
            arr = []
            for _, row in train_ds.labels.iterrows():
                arr.append(_parse_pheno_value(row["ph"], K))
            Y = np.stack(arr, axis=0)  
        else:
            Y = train_ds.labels[[c for c in train_ds.labels.columns if c.startswith("ph_")]] \
                    .values.astype("float32")  
        pos = Y.sum(axis=0)  
        neg = np.maximum(0.0, Y.shape[0] - pos)
        pos_weight_vec = torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=DEVICE) 
    except Exception:
        pos_weight_vec = None

    IS_CUDA = torch.cuda.is_available() and (
        ("cuda" in str(DEVICE)) or (isinstance(DEVICE, torch.device) and DEVICE.type == "cuda")
    )

    gen = torch.Generator()
    gen.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_factory(train_ds.img_tfms),
        pin_memory=IS_CUDA,
        generator=gen,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_factory(val_ds.img_tfms),
        pin_memory=IS_CUDA,
        generator=gen,
        drop_last=False,
    )

    params = (
        list(behrt.parameters())
        + list(bbert.parameters())
        + list(imgenc.parameters())
        + list(route_heads["L"].parameters())
        + list(route_heads["N"].parameters())
        + list(route_heads["I"].parameters())
    )
    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=CFG.weight_decay)

    amp_flag = IS_CUDA and (str(getattr(CFG, "precision_amp", "auto")).lower() != "off")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_flag)

    best_val = float("inf")

    for epoch in range(CFG.max_epochs_uni):
        behrt.train(); bbert.train(); imgenc.train()
        if getattr(bbert, "hf_available", False) and bbert.bert is not None:
            bbert.bert.eval()
        for k in ["L", "N", "I"]:
            route_heads[k].train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_uni} [UNI:ph]", dynamic_ncols=True)
        running = 0.0; n_steps = 0

        for xL, notes_list, imgs, y in pbar:
            xL   = xL.to(DEVICE, non_blocking=IS_CUDA)          
            imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)       
            y    = y.to(DEVICE, non_blocking=IS_CUDA)           

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_flag):
                # Encoders -> embeddings
                zL = behrt(xL)          
                zN = bbert(notes_list)  
                zI = imgenc(imgs)       

                # Unimodal heads -> K logits each
                logits_L = route_heads["L"](zL)  
                logits_N = route_heads["N"](zN)  
                logits_I = route_heads["I"](zI)  

                # BCE loss (multi-label) averaged across L/N/I
                bce_kwargs = {"pos_weight": pos_weight_vec} if (pos_weight_vec is not None) else {}
                loss_L = F.binary_cross_entropy_with_logits(logits_L, y, **bce_kwargs)
                loss_N = F.binary_cross_entropy_with_logits(logits_N, y, **bce_kwargs)
                loss_I = F.binary_cross_entropy_with_logits(logits_I, y, **bce_kwargs)
                loss = (loss_L + loss_N + loss_I) / 3.0

            scaler.scale(loss).backward()
            if CFG.grad_clip_norm and CFG.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=CFG.grad_clip_norm)
            scaler.step(opt)
            scaler.update()

            running += float(loss); n_steps += 1
            pbar.set_postfix(loss=f"{running / max(n_steps,1):.4f}")

        behrt.eval(); bbert.eval(); imgenc.eval()
        for k in ["L", "N", "I"]:
            route_heads[k].eval()

        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for xL, notes_list, imgs, y in val_loader:
                xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
                imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
                y    = y.to(DEVICE, non_blocking=IS_CUDA)

                with torch.cuda.amp.autocast(enabled=amp_flag):
                    zL = behrt(xL)
                    zN = bbert(notes_list)
                    zI = imgenc(imgs)

                    logits_L = route_heads["L"](zL)
                    logits_N = route_heads["N"](zN)
                    logits_I = route_heads["I"](zI)

                    bce_kwargs = {"pos_weight": pos_weight_vec} if (pos_weight_vec is not None) else {}
                    lval = (
                        F.binary_cross_entropy_with_logits(logits_L, y, **bce_kwargs)
                        + F.binary_cross_entropy_with_logits(logits_N, y, **bce_kwargs)
                        + F.binary_cross_entropy_with_logits(logits_I, y, **bce_kwargs)
                    ) / 3.0

                val_loss += float(lval); n_val += 1

        val_loss /= max(n_val, 1)
        print(f"[ph] Val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ensure_dir(args.ckpt_root)
            ckpt_path = os.path.join(args.ckpt_root, "ph_step1_unimodal.pt")
            torch.save(
                {
                    "behrt": behrt.state_dict(),
                    "bbert": bbert.state_dict(),
                    "imgenc": imgenc.state_dict(),
                    "L": route_heads["L"].state_dict(),
                    "N": route_heads["N"].state_dict(),
                    "I": route_heads["I"].state_dict(),
                    "best_val": best_val,
                    "task": "ph",
                    "num_phenotypes": K,
                    "cfg": dict(CFG.__dict__),  
                },
                ckpt_path,
            )
            print(f"Saved best unimodal checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
