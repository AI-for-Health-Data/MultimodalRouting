from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF

from env_config import CFG, DEVICE, ensure_dir
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead, build_fusions


TASK_MAP = {"mort": 0, "pe": 1, "ph": 2}
COL_MAP  = {"mort": "mort", "pe": "pe", "ph": "ph"}


def parse_args():
    ap = argparse.ArgumentParser(description="Step-2 bimodal training (LN, LI, NI) for a single task")
    ap.add_argument(
        "--task", type=str, default=getattr(CFG, "task_name", "mort"),
        choices=list(TASK_MAP.keys()),
        help="Which label to train (single-task)."
    )
    ap.add_argument(
        "--data_root", type=str, default=CFG.data_root,
        help="Root path that contains MIMIC-IV folder (parquets + splits.json)."
    )
    ap.add_argument(
        "--ckpt_root", type=str, default=CFG.ckpt_root,
        help="Where to save checkpoints."
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

        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].to_numpy(dtype=np.float32)
        xs = torch.from_numpy(xs_np)  

        notes_list = self.notes[self.notes.stay_id == stay_id].text.tolist()
        img_paths  = self.images[self.images.stay_id == stay_id].image_path.tolist()

        row = self.labels[self.labels.stay_id == stay_id][["mort", "pe", "ph"]].values[0].astype(np.float32)
        y = torch.tensor(row, dtype=torch.float32)  

        return {"stay_id": stay_id, "x_struct": xs, "notes_list": notes_list, "image_paths": img_paths, "y": y}


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

def load_first_image(paths: List[str]) -> torch.Tensor:
    if not paths:
        return torch.zeros(3, 224, 224)
    p = paths[0]
    try:
        img = Image.open(p).convert("RGB")
        return IMG_TFMS(img)
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

        imgs_batch = torch.stack([load_first_image(b["image_paths"]) for b in batch], dim=0)

        y_all   = torch.stack([b["y"] for b in batch], dim=0)  
        y_batch = y_all[:, tidx].unsqueeze(1).to(torch.float32)  

        return xL_batch, notes_batch, imgs_batch, y_batch
    return _collate


def set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def is_cuda() -> bool:
    return torch.cuda.is_available() and ("cuda" in str(DEVICE))


def main():
    args  = parse_args()
    TASK  = args.task
    TIDX  = TASK_MAP[TASK]
    TCOL  = COL_MAP[TASK]
    ROOT  = os.path.join(args.data_root, "MIMIC-IV")

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

    # Pairwise fusion modules + BIMODAL heads (single-logit each)
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)
    route_heads: Dict[str, RouteHead] = {
        r: RouteHead(d_in=CFG.d, n_tasks=1, p_drop=CFG.dropout).to(DEVICE)
        for r in ["LN", "LI", "NI"]
    }

    train_ds = ICUStayDataset(ROOT, split="train")
    val_ds   = ICUStayDataset(ROOT, split="val")

    # Class imbalance pos_weight
    try:
        y_train_np = train_ds.labels[[TCOL]].values.astype("float32").reshape(-1)
        pos = float((y_train_np > 0.5).sum())
        neg = float(len(y_train_np) - pos)
        pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=DEVICE)
    except Exception:
        pos_weight = None

    IS_CUDA = is_cuda()
    gen = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers,
        collate_fn=collate_fn_factory(TIDX), pin_memory=IS_CUDA, generator=gen, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers,
        collate_fn=collate_fn_factory(TIDX), pin_memory=IS_CUDA, generator=gen, drop_last=False
    )


    ckpt1_path = os.path.join(args.ckpt_root, f"{TASK}_step1_unimodal.pt")
    ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)

    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)
    print(f"[{TASK}] Loaded Step-1 encoders from {ckpt1_path}")

    # Freeze encoders
    set_requires_grad(behrt, False)
    set_requires_grad(bbert, False)
    set_requires_grad(imgenc, False)
    behrt.eval(); bbert.eval(); imgenc.eval()
    print("Encoders frozen.")

    # Trainable params: pairwise fusions + bimodal heads
    params_bi = (
        list(fusion["LN"].parameters())
        + list(fusion["LI"].parameters())
        + list(fusion["NI"].parameters())
        + list(route_heads["LN"].parameters())
        + list(route_heads["LI"].parameters())
        + list(route_heads["NI"].parameters())
    )
    opt = torch.optim.AdamW(params_bi, lr=CFG.lr, weight_decay=1e-2)

    amp_enabled = IS_CUDA and (str(getattr(CFG, "precision_amp", "auto")).lower() != "off")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val = float("inf")


    for epoch in range(CFG.max_epochs_bi):
        for k in ["LN", "LI", "NI"]:
            fusion[k].train()
            route_heads[k].train()
        behrt.eval(); bbert.eval(); imgenc.eval()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_bi} [BI:{TASK}]", dynamic_ncols=True)
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
            torch.nn.utils.clip_grad_norm_(params_bi, max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss); n_steps += 1
            pbar.set_postfix(loss=f"{running / max(n_steps,1):.4f}")


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
        print(f"[BI:{TASK}] Val loss: {val_loss:.4f}")


        if val_loss < best_val:
            best_val = val_loss
            ensure_dir(args.ckpt_root)
            ckpt_path = os.path.join(args.ckpt_root, f"{TASK}_step2_bimodal.pt")
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
            print(f"[{TASK}] Saved best bimodal heads + fusions -> {ckpt_path}")


if __name__ == "__main__":
    main()
