from __future__ import annotations
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Project modules
from env_config import CFG, DEVICE, load_cfg, autocast_context, ensure_dir
from encoders import (
    BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder,
    EncoderConfig, build_encoders,
)
from routing_and_heads import (
    build_fusions, build_route_heads,
    forward_mixture_of_logits, 
)


TASK_MAP = {"mort": 0}
COL_MAP  = {"mort": "mort"}

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
    ap = argparse.ArgumentParser(description="Mortality (binary) with concat-only fusion + mixture-of-logits routing")
    ap.add_argument("--task", type=str, default=getattr(CFG, "task_name", "mort"),
                    choices=list(TASK_MAP.keys()), help="Which label to train (single-task).")
    ap.add_argument("--data_root", type=str, default=CFG.data_root,
                    help="Root path that contains MIMIC-IV folder (parquets + splits.json).")
    ap.add_argument("--ckpt_root", type=str, default=CFG.ckpt_root,
                    help="Where to save checkpoints.")
    ap.add_argument("--epochs", type=int, default=max(1, getattr(CFG, "max_epochs_tri", 5)))
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=CFG.num_workers)
    ap.add_argument("--finetune_text", action="store_true",
                    help="If set, unfreeze Bio_ClinicalBERT; otherwise it stays eval/frozen.")
    ap.add_argument("--resume", type=str, default="",
                    help="Path to checkpoint (.pt) to resume from.")
    return ap.parse_args()


class ICUStayDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root
        self.split = split
        self.img_tfms = build_image_transform(split)

        with open(os.path.join(root, "splits.json")) as f:
            splits = json.load(f)
        self.ids = list(splits[split])

        struct_fp = os.path.join(root, "structured_24h.parquet")
        notes_fp  = os.path.join(root, "notes_24h.parquet")
        images_fp = os.path.join(root, "images_24h.parquet")
        labels_fp = os.path.join(root, "labels_mort.parquet")

        if not os.path.exists(labels_fp):
            raise FileNotFoundError(
                f"Missing {labels_fp}. Create it with ['stay_id','mort'] "
                f"(use hospital_expire_flag)."
            )

        self.struct = pd.read_parquet(struct_fp)
        self.notes  = pd.read_parquet(notes_fp)  if os.path.exists(notes_fp)  else pd.DataFrame(columns=["stay_id","text"])
        self.images = pd.read_parquet(images_fp) if os.path.exists(images_fp) else pd.DataFrame(columns=["stay_id","image_path"])
        self.labels = pd.read_parquet(labels_fp)

        base_cols = {"stay_id", "hour"}
        self.feat_cols = [c for c in self.struct.columns if c not in base_cols]

        if hasattr(CFG, "structured_n_feats"):
            assert len(self.feat_cols) == CFG.structured_n_feats, \
                f"CFG.structured_n_feats={CFG.structured_n_feats}, but found {len(self.feat_cols)} features in parquet."

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = self.ids[idx]

        # Structured: up to 24 rows (2h bins across 48h), sorted by hour
        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].astype("float32").fillna(0.0).to_numpy()
        xs = torch.from_numpy(xs_np) 

        # Notes: list of text snippets within 0–48h
        notes_list: List[str] = []
        if not self.notes.empty:
            notes_list = self.notes[self.notes.stay_id == stay_id].text.dropna().astype(str).tolist()

        # Images: take last path if present
        img_paths: List[str] = []
        if not self.images.empty:
            df_i = self.images[self.images.stay_id == stay_id]
            img_paths = df_i.image_path.dropna().astype(str).tolist()[-1:]  

        # Mortality label
        lab_row = self.labels.loc[self.labels.stay_id == stay_id, ["mort"]]
        y = torch.tensor([0.0], dtype=torch.float32) if lab_row.empty \
            else torch.tensor(lab_row.values[0].astype("float32"), dtype=torch.float32)

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
        # normalized zero image (already zeros post-normalization is fine)
        return torch.zeros(3, 224, 224)
    p = paths[-1]
    try:
        with Image.open(p) as img:
            return tfms(img)
    except Exception:
        return torch.zeros(3, 224, 224)


def collate_fn_factory(tidx: int, img_tfms: T.Compose):
    def _collate(batch: List[Dict]):
        T_len, F_dim = CFG.structured_seq_len, CFG.structured_n_feats

        xL_batch = torch.stack(
            [pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0
        )  

        # timestep mask: a row is "valid" if any feature != 0
        mL_batch = (xL_batch.abs().sum(dim=2) > 0).float()  

        notes_batch: List[List[str]] = [
            b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            for b in batch
        ]

        imgs_batch = torch.stack(
            [load_cxr_tensor(b["image_paths"], img_tfms) for b in batch], dim=0
        )  

        # Availability masks per unimodal stream
        L_mask = (mL_batch.sum(dim=1, keepdim=True) > 0).float()          
        N_mask = torch.tensor([[1.0 if len(b["notes_list"]) > 0 else 0.0] for b in batch], dtype=torch.float32)
        I_mask = torch.tensor([[1.0 if len(b["image_paths"]) > 0 else 0.0] for b in batch], dtype=torch.float32)

        y_all = torch.stack([b["y"] for b in batch], dim=0)  
        y_batch = y_all[:, TASK_MAP[CFG.task_name]].unsqueeze(1).to(torch.float32)

        masks = {"L": L_mask, "N": N_mask, "I": I_mask}
        return xL_batch, mL_batch, notes_batch, imgs_batch, y_batch, masks
    return _collate


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@torch.no_grad()
def evaluate_epoch(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    fusion: Dict[str, nn.Module],
    route_heads: Dict[str, nn.Module],
    loader: DataLoader,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns: (avg_loss, avg_acc, avg_route_weight) where avg_route_weight is dict per route
    """
    behrt.eval(); imgenc.eval()

    bce = nn.BCEWithLogitsLoss(reduction="mean")
    total_loss, total_correct, total, wsum = 0.0, 0, 0, torch.zeros(7, dtype=torch.float32)

    for xL, mL, notes, imgs, y, masks in loader:
        xL, mL = xL.to(DEVICE), mL.to(DEVICE)
        imgs = imgs.to(DEVICE)
        y = y.to(DEVICE)

        with autocast_context():
            # Unimodal embeddings (all -> [B,d])
            zL = behrt(xL, mask=mL)                    
            zN = bbert(notes)                           
            zI = imgenc(imgs)                           

            z = {"L": zL, "N": zN, "I": zI}
            masks_dev = {k: v.to(DEVICE) for k, v in masks.items()}

            final_logit, gates, _, _ = forward_mixture_of_logits(
                z_unimodal=z,
                fusion=fusion,
                route_heads=route_heads,
                masks=masks_dev,
            )
            loss = bce(final_logit, y)

        total_loss += loss.item() * y.size(0)
        prob = torch.sigmoid(final_logit)
        pred = (prob >= 0.5).long()
        total_correct += (pred == y.long()).sum().item()
        total += y.size(0)
        wsum += gates.detach().cpu().sum(dim=0)

    avg_loss = total_loss / max(1, total)
    avg_acc = total_correct / max(1, total)
    avg_w = (wsum / max(1, total)).tolist()
    route_names = ["L","N","I","LN","LI","NI","LNI"]
    route_weight_dict = {r: avg_w[i] for i, r in enumerate(route_names)}
    return avg_loss, avg_acc, route_weight_dict


def save_checkpoint(path: str, state: Dict):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str, behrt, bbert, imgenc, fusion, route_heads, optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    behrt.load_state_dict(ckpt["behrt"])
    bbert.load_state_dict(ckpt["bbert"])
    imgenc.load_state_dict(ckpt["imgenc"])
    for k in fusion.keys():
        fusion[k].load_state_dict(ckpt["fusion"][k])
    for k in route_heads.keys():
        route_heads[k].load_state_dict(ckpt["route_heads"][k])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))



def main():
    args = parse_args()
    load_cfg() 
  
    train_ds = ICUStayDataset(args.data_root, split="train")
    val_ds   = ICUStayDataset(args.data_root, split="val")
    test_ds  = ICUStayDataset(args.data_root, split="test")

    collate_train = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("train"))
    collate_eval  = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("val"))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_train
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_eval
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_eval
    )

    # Encoders
    enc_cfg = EncoderConfig(
        d=CFG.d, dropout=CFG.dropout,
        structured_seq_len=CFG.structured_seq_len,
        structured_n_feats=CFG.structured_n_feats,
        structured_layers=CFG.structured_layers,
        structured_heads=CFG.structured_heads,
        structured_pool="mean", 
        text_model_name=CFG.text_model_name,
        text_max_len=CFG.max_text_len,
        note_agg="mean",
        max_notes_concat=8,
        img_agg="last",
    )
    behrt, bbert, imgenc = build_encoders(enc_cfg, device=DEVICE)

    if not args.finetune_text:
        if getattr(bbert, "bert", None) is not None:
            for p in bbert.bert.parameters():
                p.requires_grad = False
            bbert.bert.eval()

    fusion = build_fusions(d=CFG.d, feature_mode="concat", p_drop=CFG.dropout)
    route_heads = build_route_heads(d=CFG.d, p_drop=CFG.dropout)

    params = list(behrt.parameters()) + list(bbert.parameters()) + list(imgenc.parameters())
    for k in fusion.keys(): params += list(fusion[k].parameters())
    for k in route_heads.keys(): params += list(route_heads[k].parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_acc = -1.0
    ckpt_dir = os.path.join(args.ckpt_root, "mort_concat_mol")
    ensure_dir(ckpt_dir)
    if args.resume and os.path.isfile(args.resume):
        print(f"[main] Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, behrt, bbert, imgenc, fusion, route_heads, optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    for epoch in range(start_epoch, args.epochs):
        behrt.train(); imgenc.train()
        if args.finetune_text and getattr(bbert, "bert", None) is not None:
            bbert.bert.train()

        total_loss, total_correct, total, wsum = 0.0, 0, 0, torch.zeros(7, dtype=torch.float32)

        for step, (xL, mL, notes, imgs, y, masks) in enumerate(train_loader):
            xL, mL = xL.to(DEVICE), mL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y = y.to(DEVICE)
            masks_dev = {k: v.to(DEVICE) for k, v in masks.items()}

            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                # Unimodal embeddings
                zL = behrt(xL, mask=mL)     
                zN = bbert(notes)           
                zI = imgenc(imgs)           
                z = {"L": zL, "N": zN, "I": zI}

                # Mixture-of-logits forward
                final_logit, gates, _, route_logits = forward_mixture_of_logits(
                    z_unimodal=z,
                    fusion=fusion,
                    route_heads=route_heads,
                    masks=masks_dev,
                )

                loss = bce(final_logit, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            prob = torch.sigmoid(final_logit)
            pred = (prob >= 0.5).long()
            total_correct += (pred == y.long()).sum().item()
            total += y.size(0)
            wsum += gates.detach().cpu().sum(dim=0)

            if (step + 1) % 50 == 0:
                avg_w = (wsum / max(1, total)).tolist()
                print(f"[epoch {epoch+1} step {step+1}] loss={total_loss/max(1,total):.4f} "
                      f"acc={total_correct/max(1,total):.4f} "
                      f"w(L,N,I,LN,LI,NI,LNI)={', '.join(f'{w:.3f}' for w in avg_w)}")

        # End epoch: train stats
        train_loss = total_loss / max(1, total)
        train_acc = total_correct / max(1, total)
        train_avg_w = (wsum / max(1, total)).tolist()
        print(f"[epoch {epoch+1}] TRAIN loss={train_loss:.4f} acc={train_acc:.4f} "
              f"avg_w={', '.join(f'{w:.3f}' for w in train_avg_w)}")

        # Validation
        val_loss, val_acc, val_w = evaluate_epoch(behrt, bbert, imgenc, fusion, route_heads, val_loader)
        print(f"[epoch {epoch+1}] VAL   loss={val_loss:.4f} acc={val_acc:.4f} "
              f"avg_w={', '.join(f'{k}:{v:.3f}' for k,v in val_w.items())}")

        # Save best
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        ckpt = {
            "epoch": epoch + 1,
            "behrt": behrt.state_dict(),
            "bbert": bbert.state_dict(),
            "imgenc": imgenc.state_dict(),
            "fusion": {k: v.state_dict() for k, v in fusion.items()},
            "route_heads": {k: v.state_dict() for k, v in route_heads.items()},
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc,
        }
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), ckpt)
        if is_best:
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), ckpt)
            print(f"[epoch {epoch+1}] Saved BEST checkpoint (acc={val_acc:.4f})")

    # Final test
    print("[main] Evaluating BEST checkpoint on TEST...")
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_path):
        _ = load_checkpoint(best_path, behrt, bbert, imgenc, fusion, route_heads, optimizer)
    test_loss, test_acc, test_w = evaluate_epoch(behrt, bbert, imgenc, fusion, route_heads, test_loader)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} "
          f"avg_w={', '.join(f'{k}:{v:.3f}' for k,v in test_w.items())}")


if __name__ == "__main__":
    main()
