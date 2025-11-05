from __future__ import annotations
import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional
from contextlib import nullcontext

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
    CapsuleMortalityHead,            # NOTE: updated to output [B,2] logits
    forward_capsule_from_routes,     # returns (logits, prim_acts, route_embs [, routing_coef])
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
    ap = argparse.ArgumentParser(description="Mortality (binary) with 7-route capsule routing (2-class CE)")
    ap.add_argument("--task", type=str, default=getattr(CFG, "task_name", "mort"),
                    choices=list(TASK_MAP.keys()))
    ap.add_argument("--data_root", type=str, default=CFG.data_root)
    ap.add_argument("--ckpt_root", type=str, default=CFG.ckpt_root)
    ap.add_argument("--epochs", type=int, default=max(1, getattr(CFG, "max_epochs_tri", 5)))
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=CFG.num_workers)
    ap.add_argument("--finetune_text", action="store_true", help="Unfreeze Bio_ClinicalBERT if set.")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt).")

    # NEW:
    ap.add_argument("--log_every", type=int, default=300, help="Print training stats every N steps.")
    ap.add_argument("--precision", type=str, default="auto",
                    choices=["auto", "fp16", "bf16", "off"],
                    help="AMP precision on CUDA; 'off' disables AMP. On CPU, AMP is forced off to avoid dtype issues.")
    ap.add_argument("--peek_first_batch", action="store_true", default=True,
                    help="Print a small debug sample at the first batch.")
    ap.add_argument("--verbose_sanity", action="store_true", default=False,
                    help="Print extra sanity info (emb norms, route shapes) at the very start.")
    return ap.parse_args()



class ICUStayDataset(Dataset):
    """
    Expects under data_root:
      - structured_24h.parquet with columns: stay_id, hour, <F features> (F=17)
      - notes_24h.parquet     with either:
            * chunk_* columns (chunk_000..chunk_XXX) OR
            * a single 'notes_24h' column (string) OR a single 'text' column
      - images_24h.parquet    with columns: stay_id, image_path
      - labels_mort.parquet   with columns: stay_id, mort
      - splits.json           { "train": [...], "val": [...], "test": [...] }
    """
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"[ICUStayDataset] data root not found: {root}")
        self.root = root
        self.split = split
        self.img_tfms = build_image_transform(split)

        # required files
        req = ["splits.json", "labels_mort.parquet", "structured_24h.parquet"]
        missing = [p for p in req if not os.path.exists(os.path.join(root, p))]
        if missing:
            raise FileNotFoundError(
                f"[ICUStayDataset] missing files under {root}: {missing}\n"
                f"Expected: {', '.join(req)} plus optional notes_24h.parquet, images_24h.parquet"
            )

        # ids for this split
        with open(os.path.join(root, "splits.json")) as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"[ICUStayDataset] split '{split}' not in splits.json keys: {list(splits.keys())}")
        self.ids: List[int] = list(splits[split])

        # file paths
        struct_fp = os.path.join(root, "structured_24h.parquet")
        notes_fp  = os.path.join(root, "notes_24h.parquet")
        images_fp = os.path.join(root, "images_24h.parquet")
        labels_fp = os.path.join(root, "labels_mort.parquet")

        # load tables
        self.struct = pd.read_parquet(struct_fp)
        self.notes  = pd.read_parquet(notes_fp)  if os.path.exists(notes_fp)  else pd.DataFrame()
        self.images = pd.read_parquet(images_fp) if os.path.exists(images_fp) else pd.DataFrame()
        self.labels = pd.read_parquet(labels_fp)

        # structured features
        base_cols = {"stay_id", "hour"}
        self.feat_cols: List[str] = [c for c in self.struct.columns if c not in base_cols]

        # feature-count sanity
        if hasattr(CFG, "structured_n_feats"):
            assert len(self.feat_cols) == CFG.structured_n_feats, \
                f"CFG.structured_n_feats={CFG.structured_n_feats}, found {len(self.feat_cols)} in {struct_fp}"

        # notes columns
        self.note_col: Optional[str] = None
        self.chunk_cols: Optional[List[str]] = None
        if not self.notes.empty:
            if "text" in self.notes.columns:
                self.note_col = "text"
            elif "notes_24h" in self.notes.columns:
                self.note_col = "notes_24h"
            else:
                cols = [c for c in self.notes.columns if str(c).startswith("chunk_")]
                cols.sort()
                self.chunk_cols = cols if cols else None

        n_chunks = 0 if self.chunk_cols is None else len(self.chunk_cols)
        using = f" using {self.note_col}" if self.note_col else ""
        print(f"[dataset:{split}] notes columns -> note_col={self.note_col} chunk_cols={n_chunks}")
        print(
            f"[dataset:{split}] root={root} ids={len(self.ids)} "
            f"| struct rows={len(self.struct)} (F={len(self.feat_cols)}) "
            f"| notes rows={len(self.notes)} (chunks={n_chunks}{using}) "
            f"| images rows={len(self.images)} | labels rows={len(self.labels)}"
        )


    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = self.ids[idx]

        # Structured sequence (ensure fixed [T,F] ordering by hour)
        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].astype("float32").fillna(0.0).to_numpy()
        xs = torch.from_numpy(xs_np)  # [<=T,F]

        # Notes list (prefer chunk_* else fallback to single text column)
        notes_list: List[str] = []
        if not self.notes.empty:
            df_n = self.notes[self.notes.stay_id == stay_id]
            if not df_n.empty:
                if self.chunk_cols:  # chunk_* columns present
                    row = df_n.iloc[0]
                    for c in self.chunk_cols:
                        if c in row.index:
                            val = row[c]
                            if pd.notna(val) and str(val).strip():
                                notes_list.append(str(val))
                elif self.note_col:  # single text column present
                    notes_list = df_n[self.note_col].dropna().astype(str).tolist()

        # Image path(s): take last image
        img_paths: List[str] = []
        if not self.images.empty:
            df_i = self.images[self.images.stay_id == stay_id]
            img_paths = df_i.image_path.dropna().astype(str).tolist()[-1:]  # last only

        # Label (float in parquet, convert to long class index 0/1)
        lab_row = self.labels.loc[self.labels.stay_id == stay_id, ["mort"]]
        y = 0 if lab_row.empty else int(lab_row.values[0][0])
        y = torch.tensor(y, dtype=torch.long)  # [ ] scalar long

        return {
            "stay_id": stay_id,
            "x_struct": xs,         # [<=T, F]
            "notes_list": notes_list,
            "image_paths": img_paths,
            "y": y,                 # long scalar 0/1
        }

def pad_or_trim_struct(x: torch.Tensor, T: int, F: int) -> torch.Tensor:
    t = x.shape[0]
    if t >= T:
        return x[-T:]
    pad = torch.zeros(T - t, F, dtype=x.dtype)
    return torch.cat([pad, x], dim=0)

def load_cxr_tensor(paths: List[str], tfms: T.Compose, return_path: bool = False):
    """
    Loads the last image in `paths`, applies `tfms`, and optionally returns the chosen path.
    Returns:
        - if return_path=False: Tensor [3,224,224]
        - if return_path=True : (Tensor [3,224,224], str path or "<none>")
    """
    if not paths:
        tensor = torch.zeros(3, 224, 224)
        return (tensor, "<none>") if return_path else tensor

    p = paths[-1]
    try:
        with Image.open(p) as img:
            tensor = tfms(img)
    except Exception as e:
        print(f"[warn] failed to open image: {p} ({e}) -> returning zero tensor")
        tensor = torch.zeros(3, 224, 224)

    return (tensor, p) if return_path else tensor



def collate_fn_factory(tidx: int, img_tfms: T.Compose):
    """
    Returns batches:
      xL: [B,T,F], mL: [B,T], notes_batch: List[List[str]],
      imgs_batch: [B,3,224,224], y: [B] (long),
      dbg: dict with small debug info (ids, image paths)
    """
    first_print = {"done": False}

    def _collate(batch: List[Dict[str, Any]]):
        T_len, F_dim = CFG.structured_seq_len, CFG.structured_n_feats

        # Structured data
        xL_batch = torch.stack(
            [pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0
        )  # [B,T,F]
        mL_batch = (xL_batch.abs().sum(dim=2) > 0).float()  # [B,T]

        # Notes (always a list per sample)
        notes_batch: List[List[str]] = [
            b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            for b in batch
        ]

        # Images (take last path per stay; keep path for debugging)
        imgs_list, img_paths_list = [], []
        for b in batch:
            img_t, path = load_cxr_tensor(b["image_paths"], img_tfms, return_path=True)
            imgs_list.append(img_t)
            img_paths_list.append(path)
        imgs_batch = torch.stack(imgs_list, dim=0)  # [B,3,224,224]

        # Labels -> class indices [B] long (handles [1]-shaped float tensors)
        y_list = []
        for b in batch:
            # b["y"] expected shape [1] float {0.0,1.0}; fall back to int cast
            y_scalar = b["y"].view(-1)[0].item() if torch.is_tensor(b["y"]) else b["y"]
            y_list.append(int(round(float(y_scalar))))
        y_batch = torch.tensor(y_list, dtype=torch.long)  # [B]

        # Small debug payload
        dbg = {
            "stay_ids": [b["stay_id"] for b in batch],
            "img_paths": img_paths_list,
        }

        # One-time shape print
        if not first_print["done"]:
            first_print["done"] = True
            print(
                f"[collate] xL_batch: {tuple(xL_batch.shape)} "
                f"| mL_batch: {tuple(mL_batch.shape)} "
                f"| notes_batch: len={len(notes_batch)} "
                f"(ex first notes={len(notes_batch[0]) if len(notes_batch)>0 else 0}) "
                f"| imgs_batch: {tuple(imgs_batch.shape)} "
                f"| y_batch: {tuple(y_batch.shape)}"
            )

        return xL_batch, mL_batch, notes_batch, imgs_batch, y_batch, dbg

    return _collate


@torch.no_grad()
def pretty_print_small_batch(
    xL: torch.Tensor, mL: torch.Tensor, notes: List[List[str]],
    dbg: Dict[str, Any], k: int = 3
) -> None:
    """
    Prints 2-3 sample EHR rows, a short note snippet, and the CXR path.
    """
    B, T, F = xL.shape
    k = min(k, B)
    print("\n[sample-inspect] ---- Top few samples ----")
    for i in range(k):
        sid = dbg["stay_ids"][i] if "stay_ids" in dbg else "<id?>"
        nz_rows = (mL[i] > 0.5).nonzero(as_tuple=False).flatten().tolist()
        show_rows = nz_rows[:2] if nz_rows else []
        ehr_rows = []
        for r in show_rows:
            vec = xL[i, r].detach().cpu().numpy()
            ehr_rows.append(np.round(vec[:min(5, F)], 3).tolist())  # first 5 features
        note_text = ""
        if len(notes[i]) > 0:
            note_text = notes[i][0]
            note_text = (note_text[:120] + "…") if len(note_text) > 120 else note_text
        imgp = dbg.get("img_paths", ["<path?>"] * B)[i]
        print(f"  • stay_id={sid} | ehr_rows(first2->first5feats)={ehr_rows} | "
              f"notes[0][:120]=\"{note_text}\" | cxr='{imgp}'")
    print("[sample-inspect] ---------------------------\n")


@torch.no_grad()
def evaluate_epoch(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    fusion: Dict[str, nn.Module],
    projector: RoutePrimaryProjector,
    cap_head: CapsuleMortalityHead,
    loader: DataLoader,
    amp_ctx,  # <<< pass the autocast/nullcontext from main()
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluate one epoch.
    Returns: (avg_loss, avg_acc, avg_primary_act_by_route)
    """
    behrt.eval(); imgenc.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()

    ce = nn.CrossEntropyLoss(reduction="mean")
    total_loss, total_correct, total = 0.0, 0, 0
    act_sum = torch.zeros(7, dtype=torch.float32)  # keep on CPU for stability
    printed_once = False

    for xL, mL, notes, imgs, y, dbg in loader:
        # to device
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)   # [B] long

        with amp_ctx:
            # Unimodal pooled embeddings
            zL = behrt(xL, mask=mL)   # [B, d]
            zN = bbert(notes)         # [B, d]
            zI = imgenc(imgs)         # [B, d]
            z = {"L": zL, "N": zN, "I": zI}

            if not printed_once:
                printed_once = True
                print(f"[eval:unimodal] zL:{tuple(zL.shape)} zN:{tuple(zN.shape)} zI:{tuple(zI.shape)}")
                pretty_print_small_batch(xL, mL, notes, dbg, k=3)

            # 7-route capsule inference
            out = forward_capsule_from_routes(
                z_unimodal=z, fusion=fusion, projector=projector, capsule_head=cap_head
            )
            logits, prim_acts, route_embs = out[0], out[1], out[2]
            # routing_coef is optional (may be absent depending on head)
            routing_coef = out[3] if len(out) > 3 else None

            if printed_once:
                keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
                print(f"[eval:caps] logits:{tuple(logits.shape)} "
                      f"prim_acts:{tuple(prim_acts.shape)} routes -> {keys}")

            loss = ce(logits, y)      # CE: logits [B,2], y [B] long

        # accumulate on CPU
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)
        act_sum += prim_acts.detach().float().cpu().sum(dim=0)

    avg_loss = total_loss / max(1, total)
    avg_acc  = total_correct / max(1, total)
    avg_act  = (act_sum / max(1, total)).tolist()
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
    print(f"[ckpt] loaded epoch={ckpt.get('epoch', 0)} val_acc={ckpt.get('val_acc', -1):.4f}")
    return int(ckpt.get("epoch", 0))


def main():
    args = parse_args()
    load_cfg()
    print(f"[setup] DEVICE={DEVICE} | batch_size={args.batch_size} | epochs={args.epochs}")
    
    # AMP policy: enable ONLY on CUDA. On CPU => nullcontext (prevents bf16/float mismatches).
    use_cuda = (DEVICE == "cuda" and torch.cuda.is_available())
    if use_cuda:
        if args.precision == "fp16":
            amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        elif args.precision == "bf16":
            amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:  # "auto"
            amp_ctx = torch.cuda.amp.autocast()  # Let PyTorch choose
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        amp_ctx = nullcontext()
        scaler = torch.amp.GradScaler("cuda", enabled=False)  # off on CPU
    train_ds = ICUStayDataset(args.data_root, split="train")
    val_ds   = ICUStayDataset(args.data_root, split="val")
    test_ds  = ICUStayDataset(args.data_root, split="test")

    collate_train = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("train"))
    collate_eval  = collate_fn_factory(tidx=TASK_MAP[args.task], img_tfms=build_image_transform("val"))
    pin = use_cuda

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_train
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_eval
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_eval
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
    print(f"[encoders] d={CFG.d} | BEHRT out_dim={behrt.out_dim} | "
          f"BERT hidden={getattr(bbert, 'hidden', 'NA')}→out_dim={bbert.out_dim} | "
          f"IMG out_dim={getattr(imgenc.proj, 'out_features', 'NA')}")

    # Freeze text encoder by default
    if not args.finetune_text and getattr(bbert, "bert", None) is not None:
        for p in bbert.bert.parameters():
            p.requires_grad = False
        bbert.bert.eval()
        print("[encoders] Bio_ClinicalBERT frozen (feature extractor mode)")

    # Fusion + Capsule bridge
    fusion = build_fusions(d=CFG.d, feature_mode=CFG.feature_mode, p_drop=CFG.dropout)
    for k in fusion.keys():
        fusion[k].to(DEVICE)
    projector = RoutePrimaryProjector(d_in=CFG.d, pc_dim=CFG.capsule_pc_dim).to(DEVICE)

    #projector = RoutePrimaryProjector(d_in=CFG.d, pc_dim=CFG.capsule_pc_dim)
    cap_head = CapsuleMortalityHead(
        pc_dim=CFG.capsule_pc_dim,
        mc_caps_dim=CFG.capsule_mc_caps_dim,
        num_routing=CFG.capsule_num_routing,
        dp=CFG.dropout,
        act_type=CFG.capsule_act_type,
        layer_norm=CFG.capsule_layer_norm,
        dim_pose_to_vote=CFG.capsule_dim_pose_to_vote,
    ).to(DEVICE)
    print(f"[capsule] pc_dim={CFG.capsule_pc_dim} mc_caps_dim={CFG.capsule_mc_caps_dim} "
          f"iters={CFG.capsule_num_routing} act_type={CFG.capsule_act_type}")

    # Optimizer
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

    #scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    ce = nn.CrossEntropyLoss(reduction="mean")

    printed_once = False

    for epoch in range(start_epoch, args.epochs):
        behrt.train(); imgenc.train()
        if args.finetune_text and getattr(bbert, "bert", None) is not None:
            bbert.bert.train()

        total_loss, total_correct, total = 0.0, 0, 0
        act_sum = torch.zeros(7, dtype=torch.float32)

        for step, (xL, mL, notes, imgs, y, dbg) in enumerate(train_loader):
            xL, mL = xL.to(DEVICE), mL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y = y.to(DEVICE)                  # [B] long

            if (epoch == start_epoch) and (step == 0):
                # print 2-3 samples to verify inputs
                pretty_print_small_batch(xL, mL, notes, dbg, k=3)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                # Unimodal pooled embeddings
                zL = behrt(xL, mask=mL)       # [B,d]
                zN = bbert(notes)             # [B,d]
                zI = imgenc(imgs)             # [B,d]
                z = {"L": zL, "N": zN, "I": zI}

                if not printed_once:
                    printed_once = True
                    print(f"[sanity] xL: {tuple(xL.shape)} | mL: {tuple(mL.shape)} "
                          f"| imgs: {tuple(imgs.shape)} | y: {tuple(y.shape)}")
                    print(f"[sanity] zL: {tuple(zL.shape)} | zN: {tuple(zN.shape)} | zI: {tuple(zI.shape)}")
                    with torch.no_grad():
                        for i in range(min(3, zL.size(0))):
                            print(f"[emb-norms] i={i} ||zL||={zL[i].norm().item():.3f} "
                                  f"||zN||={zN[i].norm().item():.3f} ||zI||={zI[i].norm().item():.3f}")

                # Capsule forward (routes → projector → head)
                out = forward_capsule_from_routes(z_unimodal=z, fusion=fusion, projector=projector, capsule_head=cap_head)
                logits, prim_acts, route_embs = out[0], out[1], out[2]
                routing_coef = out[3] if len(out) > 3 else None

                if args.route_debug and routing_coef is not None and (step % 100 == 0):
                    _print_routing_mort(routing_coef, prim_acts, where=f"TRAIN@step{step}")


                if printed_once and step == 0:
                    keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
                    print(f"[sanity] routes -> {keys} | logits: {tuple(logits.shape)} "
                          f"| prim_acts: {tuple(prim_acts.shape)}")

                loss = ce(logits, y)          # CE for [B,2] logits vs [B] targets

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
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total += y.size(0)
            act_sum += prim_acts.detach().cpu().sum(dim=0)

            if args.log_every > 0 and ((step + 1) % args.log_every == 0):
                avg_act = (act_sum / max(1, total)).tolist()
                msg = (f"[epoch {epoch+1} step {step+1}] "
                       f"loss={total_loss/max(1,total):.4f} "
                       f"acc={total_correct/max(1,total):.4f} "
                       f"avg_prim_act(L,N,I,LN,LI,NI,LNI)="
                       f"{', '.join(f'{a:.3f}' for a in avg_act)}")
                if routing_coef is not None:
                    rc = routing_coef.detach().float().cpu()   # [B,7,2]
                    rc_mean = rc.mean(dim=0)                   # [7,2]
                    routes = ["L","N","I","LN","LI","NI","LNI"]
                    rc_str = " | ".join(
                        f"{r}:({rc_mean[i,0]:.3f},{rc_mean[i,1]:.3f})" for i, r in enumerate(routes)
                    )
                    msg += f" | [routing mean] {rc_str}"
                print(msg)

        # End epoch stats
        train_loss = total_loss / max(1, total)
        train_acc = total_correct / max(1, total)
        train_avg_act = (act_sum / max(1, total)).tolist()
        print(f"[epoch {epoch+1}] TRAIN loss={train_loss:.4f} acc={train_acc:.4f} "
              f"avg_prim_act={', '.join(f'{a:.3f}' for a in train_avg_act)}")

        # Validation
        val_loss, val_acc, val_act = evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, val_loader, amp_ctx)
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
    test_loss, test_acc, test_act = evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, test_loader, amp_ctx)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} "
          f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k,v in test_act.items())}")


if __name__ == "__main__":
    main()
