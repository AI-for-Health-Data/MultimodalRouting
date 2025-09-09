from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

import torch
import torchvision.transforms as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as VF  

from env_config import CFG, DEVICE, ensure_dir
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead


TASK_MAP = {"mort": 0, "pe": 1, "ph": 2}
COL_MAP  = {"mort": "mort", "pe": "pe", "ph": "ph"}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def parse_args():
    ap = argparse.ArgumentParser(description="Step-1 Unimodal pretraining (L, N, I) for a single task")
    ap.add_argument(
        "--task",
        type=str,
        default=getattr(CFG, "task_name", "mort"),
        choices=list(TASK_MAP.keys()),
        help="Which label to train (single-task).",
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
    return ap.parse_args()


class ICUStayDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root

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

        # Structured: 24h time-series 
        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].to_numpy(dtype=np.float32)
        xs = torch.from_numpy(xs_np)  

        # Notes: list of texts within first 24h
        notes_list = self.notes[self.notes.stay_id == stay_id].text.tolist()

        # Images: list of image paths 
        img_paths = self.images[self.images.stay_id == stay_id].image_path.tolist()

        row = self.labels[self.labels.stay_id == stay_id][["mort", "pe", "ph"]].values[0].astype(np.float32)
        y = torch.tensor(row, dtype=torch.float32)  

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

IMG_TFMS = TF.Compose([
    TF.Resize((224, 224)),
    TF.ToTensor(),
])

def _cxr_to_tensor_medfuse(pil_img: Image.Image) -> torch.Tensor:
    pil_img = pil_img.convert("L")
    pil_img = VF.resize(pil_img, 256, antialias=True)
    pil_img = VF.center_crop(pil_img, 224)
    x = VF.to_tensor(pil_img)        
    x = x.repeat(3, 1, 1)            
    x = VF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x


def load_cxr_tensor(paths: List[str]) -> torch.Tensor:
    if not paths:
        return torch.zeros(3, 224, 224)
    p = paths[0]  
    try:
        img = Image.open(p)
        return _cxr_to_tensor_medfuse(img)
    except Exception:
        return torch.zeros(3, 224, 224)
        
def collate_fn_factory(tidx: int):
    def _collate(batch):
        T_len = CFG.structured_seq_len
        F_dim = CFG.structured_n_feats

        xL_batch = torch.stack([pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0)

        notes_batch = [
            b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            for b in batch
        ]

        # MedFuse: last CXR, grayscale->3ch, 256->224, ImageNet normalize
        imgs_batch = torch.stack([load_cxr_tensor(b["image_paths"]) for b in batch], dim=0)  

        # extract single task from y
        y_all = torch.stack([b["y"] for b in batch], dim=0)  
        y_batch = y_all[:, tidx].unsqueeze(1).to(torch.float32)  

        return xL_batch, notes_batch, imgs_batch, y_batch
    return _collate


def main():
    args  = parse_args()
    TASK  = args.task
    TIDX  = TASK_MAP[TASK]
    TCOL  = COL_MAP[TASK]
    ROOT  = os.path.join(args.data_root, "MIMIC-IV")

    # Build encoders with CFG (shared d); ImageEncoder uses MedFuse-style head in encoders.py
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

    # Unimodal heads (L, N, I) — single binary logit each for selected task
    route_heads: Dict[str, RouteHead] = {
        r: RouteHead(d_in=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE)
        for r in ["L", "N", "I"]
    }

    train_ds = ICUStayDataset(ROOT, split="train")
    val_ds   = ICUStayDataset(ROOT, split="val")

    # Compute scalar pos_weight for class imbalance on the selected task
    try:
        y_train_np = train_ds.labels[[TCOL]].values.astype("float32").reshape(-1)
        pos = float((y_train_np > 0.5).sum())
        neg = float(len(y_train_np) - pos)
        pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=DEVICE)
    except Exception:
        pos_weight = None

    IS_CUDA = torch.cuda.is_available() and (("cuda" in str(DEVICE)) or (isinstance(DEVICE, torch.device) and DEVICE.type == "cuda"))

    gen = torch.Generator()
    gen.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn_factory(TIDX),
        pin_memory=IS_CUDA,
        generator=gen,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn_factory(TIDX),
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
    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=1e-2)

    amp_enabled = IS_CUDA and (str(getattr(CFG, "precision_amp", "auto")).lower() != "off")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val = float("inf")

    for epoch in range(CFG.max_epochs_uni):
        behrt.train(); bbert.train(); imgenc.train()
        # Keep HF BERT in eval so dropout stays off
        if getattr(bbert, "hf_available", False) and bbert.bert is not None:
            bbert.bert.eval()
        for k in ["L", "N", "I"]:
            route_heads[k].train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_uni} [UNI:{TASK}]", dynamic_ncols=True)
        running = 0.0; n_steps = 0

        for xL, notes_list, imgs, y in pbar:
            xL   = xL.to(DEVICE, non_blocking=IS_CUDA)
            imgs = imgs.to(DEVICE, non_blocking=IS_CUDA)
            y    = y.to(DEVICE, non_blocking=IS_CUDA)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # Encoders -> embeddings
                zL = behrt(xL)                 
                zN = bbert(notes_list)         
                zI = imgenc(imgs)              

                # Unimodal heads -> single-logit per sample
                logits_L = route_heads["L"](zL)  
                logits_N = route_heads["N"](zN)  
                logits_I = route_heads["I"](zI)  

                # BCE over the three unimodal routes (mean)
                bce_kwargs = {"pos_weight": pos_weight} if (pos_weight is not None) else {}
                loss_L = F.binary_cross_entropy_with_logits(logits_L, y, **bce_kwargs)
                loss_N = F.binary_cross_entropy_with_logits(logits_N, y, **bce_kwargs)
                loss_I = F.binary_cross_entropy_with_logits(logits_I, y, **bce_kwargs)
                loss = (loss_L + loss_N + loss_I) / 3.0

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    zL = behrt(xL)
                    zN = bbert(notes_list)
                    zI = imgenc(imgs)

                    logits_L = route_heads["L"](zL)
                    logits_N = route_heads["N"](zN)
                    logits_I = route_heads["I"](zI)

                    bce_kwargs = {"pos_weight": pos_weight} if (pos_weight is not None) else {}
                    lval = (
                        F.binary_cross_entropy_with_logits(logits_L, y, **bce_kwargs)
                        + F.binary_cross_entropy_with_logits(logits_N, y, **bce_kwargs)
                        + F.binary_cross_entropy_with_logits(logits_I, y, **bce_kwargs)
                    ) / 3.0

                val_loss += float(lval); n_val += 1

        val_loss /= max(n_val, 1)
        print(f"[{TASK}] Val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ensure_dir(args.ckpt_root)
            ckpt_path = os.path.join(args.ckpt_root, f"{TASK}_step1_unimodal.pt")
            torch.save(
                {
                    "behrt": behrt.state_dict(),
                    "bbert": bbert.state_dict(),
                    "imgenc": imgenc.state_dict(),
                    "L": route_heads["L"].state_dict(),
                    "N": route_heads["N"].state_dict(),
                    "I": route_heads["I"].state_dict(),
                    "best_val": best_val,
                    "task": TASK,
                    "cfg": vars(CFG),
                },
                ckpt_path,
            )
            print(f"Saved best unimodal checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
