from __future__ import annotations
import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from env_config import CFG, DEVICE, load_cfg, autocast_context, ensure_dir
from encoders import (
    BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder,
    EncoderConfig, build_encoders,
)
from routing_and_heads import (
    build_fusions,
    RoutePrimaryProjector,
    CapsuleMortalityHead,
    forward_capsule_from_routes,
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
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
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
    ap = argparse.ArgumentParser(description="Mortality (binary) with 7-route capsule routing")
    ap.add_argument("--task", type=str, default=getattr(CFG, "task_name", "mort"),
                    choices=list(TASK_MAP.keys()))
    ap.add_argument("--data_root", type=str, default=CFG.data_root,
                    help="Root path with parquets + splits.json")
    ap.add_argument("--ckpt_root", type=str, default=CFG.ckpt_root,
                    help="Where to save checkpoints")
    ap.add_argument("--epochs", type=int, default=max(1, getattr(CFG, "max_epochs_tri", 5)))
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=CFG.num_workers)
    ap.add_argument("--finetune_text", action="store_true",
                    help="Unfreeze Bio_ClinicalBERT if set.")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt).")
    return ap.parse_args()


class ICUStayDataset(Dataset):
    """
    Expects:
      - structured_24h.parquet with columns: stay_id, hour, <F features>
      - notes_24h.parquet     with columns: stay_id, text
      - images_24h.parquet    with columns: stay_id, image_path
      - labels_mort.parquet   with columns: stay_id, mort
      - splits.json           { "train": [...], "val": [...], "test": [...] }
    """
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
                f"Missing {labels_fp}. Create it with ['stay_id','mort'] (use hospital_expire_flag)."
            )

        self.struct = pd.read_parquet(struct_fp)
        self.notes  = pd.read_parquet(notes_fp)  if os.path.exists(notes_fp)  else pd.DataFrame(columns=["stay_id","text"])
        self.images = pd.read_parquet(images_fp) if os.path.exists(images_fp) else pd.DataFrame(columns=["stay_id","image_path"])
        self.labels = pd.read_parquet(labels_fp)

        base_cols = {"stay_id", "hour"}
        self.feat_cols = [c for c in self.struct.columns if c not in base_cols]

        # Sanity-check number of structured features (expect 17 for your setup)
        if hasattr(CFG, "structured_n_feats"):
            assert len(self.feat_cols) == CFG.structured_n_feats, \
                f"CFG.structured_n_feats={CFG.structured_n_feats}, but found {len(self.feat_cols)}."

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = self.ids[idx]

        # Structured sequence (up to 24 rows for 48h@2h bins), sorted by hour
        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].astype("float32").fillna(0.0).to_numpy()
        xs = torch.from_numpy(xs_np)  # [T,F]

        # Notes list in 0–48h
        notes_list: List[str] = []
        if not self.notes.empty:
            notes_list = self.notes[self.notes.stay_id == stay_id].text.dropna().astype(str).tolist()

        # Last image path if present
        img_paths: List[str] = []
        if not self.images.empty:
            df_i = self.images[self.images.stay_id == stay_id]
            img_paths = df_i.image_path.dropna().astype(str).tolist()[-1:]  # take last

        # Mortality label
        lab_row = self.labels.loc[self.labels.stay_id == stay_id, ["mort"]]
        y = torch.tensor([0.0], dtype=torch.float32) if lab_row.empty \
            else torch.tensor(lab_row.values[0].astype("float32"), dtype=torch.float32)

        return {
            "stay_id": stay_id,
            "x_struct": xs,         # [T,F] (e.g., [24,17])
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

def collate_fn_factory(tidx: int, img_tfms: T.Compose):
    """
    Returns batches:
      xL: [B,T,F], mL: [B,T], notes_batch: List[List[str]],
      imgs_batch: [B,3,224,224], y: [B,1]
    """
    def _collate(batch: List[Dict]):
        T_len, F_dim = CFG.structured_seq_len, CFG.structured_n_feats

        xL_batch = torch.stack(
            [pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0
        )  # [B,T,F]

        # timestep mask: a row is "valid" if any feature != 0
        mL_batch = (xL_batch.abs().sum(dim=2) > 0).float()  # [B,T]

        notes_batch: List[List[str]] = [
            b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            for b in batch
        ]

        imgs_batch = torch.stack(
            [load_cxr_tensor(b["image_paths"], img_tfms) for b in batch], dim=0
        )  # [B,3,224,224]

        y_all = torch.stack([b["y"] for b in batch], dim=0)  # [B,1]
        y_batch = y_all[:, TASK_MAP[CFG.task_name]].unsqueeze(1).to(torch.float32)  # [B,1]

        return xL_batch, mL_batch, notes_batch, imgs_batch, y_batch
    return _collate


@torch.no_grad()
def evaluate_epoch(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    fusion: Dict[str, nn.Module],
    projector: RoutePrimaryProjector,
    cap_head: CapsuleMortalityHead,
    loader: DataLoader,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns: (avg_loss, avg_acc, avg_primary_act_by_route)
    """
    behrt.eval(); imgenc.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()

    bce = nn.BCEWithLogitsLoss(reduction="mean")
    total_loss, total_correct, total = 0.0, 0, 0
    act_sum = torch.zeros(7, dtype=torch.float32)

    for xL, mL, notes, imgs, y in loader:
        xL, mL = xL.to(DEVICE), mL.to(DEVICE)
        imgs = imgs.to(DEVICE)
        y = y.to(DEVICE)

        with autocast_context():
            # Unimodal pooled embeddings
            zL = behrt(xL, mask=mL)   # [B,d]
            zN = bbert(notes)         # [B,d] (BERT CLS pooled over chunks/notes)
            zI = imgenc(imgs)         # [B,d]
            z = {"L": zL, "N": zN, "I": zI}

            # 7-route capsule inference
            final_logit, prim_acts, _ = forward_capsule_from_routes(
                z_unimodal=z,
                fusion=fusion,
                projector=projector,
                capsule_head=cap_head,
            )
            loss = bce(final_logit, y)

        total_loss += loss.item() * y.size(0)
        prob = torch.sigmoid(final_logit)
        pred = (prob >= 0.5).long()
        total_correct += (pred == y.long()).sum().item()
        total += y.size(0)

        act_sum += prim_acts.detach().cpu().sum(dim=0)

    avg_loss = total_loss / max(1, total)
    avg_acc = total_correct / max(1, total)
    avg_act = (act_sum / max(1, total)).tolist()
    route_names = ["L","N","I","LN","LI","NI","LNI"]
    avg_act_dict = {r: avg_act[i] for i, r in enumerate(route_names)}
    return avg_loss, avg_acc, avg_act_dict


def save_checkpoint(path: str, state: Dict):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_checkpoint(path: str, behrt, bbert, imgenc, fusion, projector, cap_head, optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    behrt.load_state_dict(ckpt["behrt"])
    bbert.load_state_dict(ckpt["bbert"])
    imgenc.load_state_dict(ckpt["imgenc"])
    for k in fusion.keys():
        fusion[k].load_state_dict(ckpt["fusion"][k])
    projector.load_state_dict(ckpt["projector"])
    cap_head.load_state_dict(ckpt["cap_head"])
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

    # Encoders (set structured_n_feats=17 and structured_seq_len=24 in CFG)
    enc_cfg = EncoderConfig(
        d=CFG.d, dropout=CFG.dropout,
        structured_seq_len=CFG.structured_seq_len,     # 24
        structured_n_feats=CFG.structured_n_feats,     # 17
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

    if not args.finetune_text and getattr(bbert, "bert", None) is not None:
        for p in bbert.bert.parameters():
            p.requires_grad = False
        bbert.bert.eval()

    # Fusion + Capsule bridge
    fusion = build_fusions(d=CFG.d, feature_mode=CFG.feature_mode, p_drop=CFG.dropout)
    projector = RoutePrimaryProjector(d_in=CFG.d, pc_dim=CFG.capsule_pc_dim)
    cap_head = CapsuleMortalityHead(
        pc_dim=CFG.capsule_pc_dim,
        mc_caps_dim=CFG.capsule_mc_caps_dim,
        num_routing=CFG.capsule_num_routing,
        dp=CFG.dropout,
        act_type=CFG.capsule_act_type,
        layer_norm=CFG.capsule_layer_norm,
        dim_pose_to_vote=CFG.capsule_dim_pose_to_vote,
    )

    params = list(behrt.parameters()) + list(bbert.parameters()) + list(imgenc.parameters())
    for k in fusion.keys(): params += list(fusion[k].parameters())
    params += list(projector.parameters()) + list(cap_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_acc = -1.0
    ckpt_dir = os.path.join(args.ckpt_root, "mort_capsule")
    ensure_dir(ckpt_dir)
    if args.resume and os.path.isfile(args.resume):
        print(f"[main] Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, behrt, bbert, imgenc, fusion, projector, cap_head, optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    printed_once = False

    for epoch in range(start_epoch, args.epochs):
        behrt.train(); imgenc.train()
        if args.finetune_text and getattr(bbert, "bert", None) is not None:
            bbert.bert.train()

        total_loss, total_correct, total = 0.0, 0, 0
        act_sum = torch.zeros(7, dtype=torch.float32)

        for step, (xL, mL, notes, imgs, y) in enumerate(train_loader):
            xL, mL = xL.to(DEVICE), mL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                # Unimodal pooled embeddings
                zL = behrt(xL, mask=mL)   # [B,d]
                zN = bbert(notes)         # [B,d]
                zI = imgenc(imgs)         # [B,d]
                z = {"L": zL, "N": zN, "I": zI}

                # One-time sanity print
                if not printed_once:
                    printed_once = True
                    print(f"[sanity] xL: {tuple(xL.shape)}, mL: {tuple(mL.shape)}, "
                          f"imgs: {tuple(imgs.shape)}, y: {tuple(y.shape)}")
                    print(f"[sanity] zL: {tuple(zL.shape)}, zN: {tuple(zN.shape)}, zI: {tuple(zI.shape)}")

                # Capsule forward
                final_logit, prim_acts, debug_dict = forward_capsule_from_routes(
                    z_unimodal=z, fusion=fusion, projector=projector, capsule_head=cap_head
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
            act_sum += prim_acts.detach().cpu().sum(dim=0)

            if (step + 1) % 50 == 0:
                avg_act = (act_sum / max(1, total)).tolist()
                print(f"[epoch {epoch+1} step {step+1}] "
                      f"loss={total_loss/max(1,total):.4f} "
                      f"acc={total_correct/max(1,total):.4f} "
                      f"avg_prim_act(L,N,I,LN,LI,NI,LNI)="
                      f"{', '.join(f'{a:.3f}' for a in avg_act)}")

        # End epoch stats
        train_loss = total_loss / max(1, total)
        train_acc = total_correct / max(1, total)
        train_avg_act = (act_sum / max(1, total)).tolist()
        print(f"[epoch {epoch+1}] TRAIN loss={train_loss:.4f} acc={train_acc:.4f} "
              f"avg_prim_act={', '.join(f'{a:.3f}' for a in train_avg_act)}")

        # Validation
        val_loss, val_acc, val_act = evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, val_loader)
        print(f"[epoch {epoch+1}] VAL   loss={val_loss:.4f} acc={val_acc:.4f} "
              f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k,v in val_act.items())}")

        # Save checkpoints
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        ckpt = {
            "epoch": epoch + 1,
            "behrt": behrt.state_dict(),
            "bbert": bbert.state_dict(),
            "imgenc": imgenc.state_dict(),
            "fusion": {k: v.state_dict() for k, v in fusion.items()},
            "projector": projector.state_dict(),
            "cap_head": cap_head.state_dict(),
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
        _ = load_checkpoint(best_path, behrt, bbert, imgenc, fusion, projector, cap_head, optimizer)
    test_loss, test_acc, test_act = evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, test_loader)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} "
          f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k,v in test_act.items())}")


if __name__ == "__main__":
    main()
