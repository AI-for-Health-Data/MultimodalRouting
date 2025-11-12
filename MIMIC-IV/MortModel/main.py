from __future__ import annotations

# --- Must set env vars BEFORE importing transformers/tokenizers/torchvision ---
import os as _os
_os.environ.setdefault("HF_HOME", _os.path.expanduser("~/.cache/huggingface"))
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
from torch import amp as torch_amp 
# For metrics & calibration plots
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg") # headless
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
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

# --- HuggingFace tokenizer (for pre-tokenizing note strings) ---
TOKENIZER = None
MAXLEN = 512
CHUNK_STRIDE = 128  # if you want sliding windows inside pretok (kept here for completeness)

def _chunk_long_ids(ids: list[int], attn: list[int], maxlen: int, stride: int):
    if len(ids) <= maxlen:
        return [ids], [attn]
    out_ids, out_attn = [], []
    step = max(1, maxlen - max(stride, 0))
    i = 0
    while i < len(ids):
        s = ids[i:i+maxlen]; a = attn[i:i+maxlen]
        out_ids.append(s); out_attn.append(a)
        if i + maxlen >= len(ids): break
        i += step
    return out_ids, out_attn

def pretok_batch_notes(batch_notes: list[list[str]]):
    """
    Clean literal [CLS]/[SEP] from stored chunks, then tokenize into
    per-patient dicts: {'input_ids': Long[S,L], 'attention_mask': Long[S,L]}.
    """
    global TOKENIZER, MAXLEN
    if TOKENIZER is None:
        raise RuntimeError("TOKENIZER not initialized; call main() which runs load_cfg() first.")
    MAXLEN = int(getattr(CFG, "max_text_len", 512))
    # 1) Clean up: avoid double special-tokens when we re-tokenize
    cleaned = []
    for texts in batch_notes:
        cleaned.append([
            t.replace("[CLS]", "").replace("[SEP]", "").strip()
            for t in texts if t and str(t).strip()
        ])

    out = []
    pad_id = TOKENIZER.pad_token_id or 0
    for texts in cleaned:
        if not texts:
            out.append({"input_ids": torch.zeros(0, MAXLEN, dtype=torch.long, device=DEVICE),
                        "attention_mask": torch.zeros(0, MAXLEN, dtype=torch.long, device=DEVICE)})
            continue

        all_ids, all_attn = [], []
        for t in texts:
            enc = TOKENIZER(t, truncation=True, max_length=MAXLEN, padding=False,
                            return_attention_mask=True, add_special_tokens=True)
            ids, attn = enc["input_ids"], enc["attention_mask"]
            ids_chunks, attn_chunks = _chunk_long_ids(ids, attn, MAXLEN, CHUNK_STRIDE)
            all_ids.extend(ids_chunks); all_attn.extend(attn_chunks)

        def _pad(x, L=MAXLEN, v=pad_id): return x + [v]*(L-len(x))
        ids_mat  = torch.tensor([_pad(ch) for ch in all_ids],  dtype=torch.long, device=DEVICE)
        attn_mat = torch.tensor([_pad(ch, MAXLEN, 0) for ch in all_attn], dtype=torch.long, device=DEVICE)
        out.append({"input_ids": ids_mat, "attention_mask": attn_mat})
    return out

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
    # in parse_args()
    ap.add_argument("--require_all_modalities", action="store_true", default=True,
                    help="Only keep stays that have structured + notes + image. Hard-require files & per-stay presence.")

    ap.add_argument("--data_root", type=str, default=CFG.data_root)
    ap.add_argument("--ckpt_root", type=str, default=CFG.ckpt_root)
    ap.add_argument("--epochs", type=int, default=max(1, getattr(CFG, "max_epochs_tri", 5)))
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=CFG.num_workers)
    ap.add_argument("--finetune_text", action="store_true", help="Unfreeze Bio_ClinicalBERT if set.")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt).")

    # Logging / precision
    ap.add_argument("--log_every", type=int, default=300, help="Print training stats every N steps.")
    ap.add_argument("--precision", type=str, default="auto",
                    choices=["auto", "fp16", "bf16", "off"],
                    help="AMP precision on CUDA; 'off' disables AMP. On CPU, AMP is forced off.")
    ap.add_argument("--peek_first_batch", action="store_true", default=True,
                    help="Print a small debug sample at the first batch.")
    ap.add_argument("--verbose_sanity", action="store_true", default=False,
                    help="Print extra sanity info (emb norms, route shapes) at the very start.")
    ap.add_argument("--route_debug", action="store_true",
                    help="Print routing coeffs/acts periodically.")

    # Calibration / plots
    ap.add_argument("--calib_bins", type=int, default=10, help="Bins for reliability diagram/ECE.")

    return ap.parse_args()


class ICUStayDataset(Dataset):
    """
    Strict tri-modal dataset:
      REQUIRED under data_root:
        - splits.json
        - labels_mort.parquet          (columns: stay_id, mort)
        - structured_24h.parquet       (columns: stay_id, hour, <17 features>)
        - notes_48h_chunks.parquet     (columns: stay_id, chunk_000..chunk_XXX)  <-- ONLY THIS for notes
        - images_24h.parquet           (columns: stay_id, image_path)            <-- ONLY THIS for images
    Keeps only stays that have: structured rows, >=1 non-empty chunk, >=1 image path, and a label.
    """
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"[ICUStayDataset] data root not found: {root}")
        self.root = root
        self.split = split
        self.img_tfms = build_image_transform(split)

        # --- Required files (exactly these) ---
        req_files = [
            "splits.json",
            "labels_mort.parquet",
            "structured_24h.parquet",
            "notes_48h_chunks.parquet",
            "images_24h.parquet",
        ]
        missing = [p for p in req_files if not os.path.exists(os.path.join(root, p))]
        if missing:
            raise FileNotFoundError(
                f"[ICUStayDataset] missing files under {root}: {missing}\n"
                f"Expected exactly: {', '.join(req_files)}"
            )

        # --- Load splits ---
        with open(os.path.join(root, "splits.json")) as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"[ICUStayDataset] split '{split}' not in splits.json keys: {list(splits.keys())}")
        split_ids: List[int] = list(splits[split])

        # --- Load tables ---
        struct_fp = os.path.join(root, "structured_24h.parquet")
        notes_fp  = os.path.join(root, "notes_48h_chunks.parquet")
        images_fp = os.path.join(root, "images_24h.parquet")
        labels_fp = os.path.join(root, "labels_mort.parquet")

        self.struct = pd.read_parquet(struct_fp)
        self.notes  = pd.read_parquet(notes_fp)
        self.images = pd.read_parquet(images_fp)
        self.labels = pd.read_parquet(labels_fp)

        # --- Structured feature columns ---
        base_cols = {"stay_id", "hour"}
        self.feat_cols: List[str] = [c for c in self.struct.columns if c not in base_cols]
        if hasattr(CFG, "structured_n_feats"):
            assert len(self.feat_cols) == CFG.structured_n_feats, \
                f"CFG.structured_n_feats={CFG.structured_n_feats}, found {len(self.feat_cols)} in {struct_fp}"

        # --- Notes: MUST be chunk_* columns only ---
        self.note_col: Optional[str] = None
        self.chunk_cols: List[str] = [c for c in self.notes.columns if str(c).startswith("chunk_")]
        self.chunk_cols.sort()
        if len(self.chunk_cols) == 0:
            raise ValueError(
                "[ICUStayDataset] notes_48h_chunks.parquet must contain at least one 'chunk_*' column."
            )

        # --- Strict tri-modal filtering ---
        ids_set = set(split_ids)

        # 1) has structured rows
        struct_ids = set(self.struct["stay_id"].unique().tolist())

        # 2) has >=1 non-empty chunk
        note_rows = self.notes.copy()
        any_text = np.zeros(len(note_rows), dtype=bool)
        for c in self.chunk_cols:
            any_text |= note_rows[c].fillna("").astype(str).str.strip().ne("")
        note_ids = set(note_rows.loc[any_text, "stay_id"].unique().tolist())

        # 3) has >=1 image path (non-empty)
        img_rows = self.images.copy()
        img_ids = set(
            img_rows.loc[img_rows["image_path"].fillna("").astype(str).str.strip().ne(""), "stay_id"]
            .unique().tolist()
        )

        # 4) has label row
        label_ids = set(self.labels["stay_id"].unique().tolist())

        keep_ids = ids_set & struct_ids & note_ids & img_ids & label_ids
        dropped = len(ids_set) - len(keep_ids)
        self.ids: List[int] = sorted(list(keep_ids))
        if len(self.ids) == 0:
            raise RuntimeError(f"[ICUStayDataset] After tri-modal filtering, split '{self.split}' is empty.")
        print(
            f"[dataset:{split}] strict tri-modal -> kept {len(self.ids)} / {len(ids_set)} (dropped {dropped})"
        )

        # --- Info print ---
        n_chunks = len(self.chunk_cols)
        print(f"[dataset:{split}] notes columns -> note_col=None chunk_cols={n_chunks}")
        print(
            f"[dataset:{split}] root={root} ids={len(self.ids)} "
            f"| struct rows={len(self.struct)} (F={len(self.feat_cols)}) "
            f"| notes rows={len(self.notes)} (chunks={n_chunks}) "
            f"| images rows={len(self.images)} | labels rows={len(self.labels)}"
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = self.ids[idx]

        # Structured sequence (fixed [T,F] by hour)
        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].astype("float32").fillna(0.0).to_numpy()
        xs = torch.from_numpy(xs_np)  # [<=T,F]

        # Notes list from chunk_* columns only
        notes_list: List[str] = []
        df_n = self.notes[self.notes.stay_id == stay_id]
        if not df_n.empty:
            row = df_n.iloc[0]
            for c in self.chunk_cols:
                if c in row.index:
                    val = row[c]
                    if pd.notna(val) and str(val).strip():
                        notes_list.append(str(val))

        # Images: use last image path (string)
        img_paths: List[str] = []
        df_i = self.images[self.images.stay_id == stay_id]
        if not df_i.empty:
            img_paths = df_i.image_path.dropna().astype(str).tolist()[-1:]  # last only

        # Label (long scalar 0/1)
        lab_row = self.labels.loc[self.labels.stay_id == stay_id, ["mort"]]
        y = 0 if lab_row.empty else int(lab_row.values[0][0])
        y = torch.tensor(y, dtype=torch.long)

        return {
            "stay_id": stay_id,
            "x_struct": xs,         # [<=T,F]
            "notes_list": notes_list,      # list[str] from chunks
            "image_paths": img_paths,      # list[str] (last image used)
            "y": y,                         # long 0/1
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
        notes_batch: List[List[str]] = []
        for b in batch:
            raw = b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            valid = [t for t in raw if str(t).strip()]
            assert len(valid) > 0, "[collate] tri-modal strict: empty notes_list for a sample"
            notes_batch.append(valid)

        # Images (take last path per stay; keep path for debugging)
        imgs_list, img_paths_list = [], []
        for b in batch:
            assert len(b["image_paths"]) > 0 and str(b["image_paths"][-1]).strip(), \
                "[collate] tri-modal strict: missing image path for a sample"
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
    loss_fn: nn.Module, 
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluate one epoch.
    Returns: (avg_loss, avg_acc, avg_primary_act_by_route)
    """
    behrt.eval(); imgenc.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()

    #ce = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    total_loss, total_correct, total = 0.0, 0, 0
    act_sum = torch.zeros(7, dtype=torch.float32)  # keep on CPU for stability

    printed_unimodal = False
    printed_caps_once = False
    rpt_every = int(getattr(CFG, "routing_print_every", 0) or 0)

    for bidx, (xL, mL, notes, imgs, y, dbg) in enumerate(loader):
        # to device
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)   # [B] long

        with amp_ctx:
            # Unimodal pooled embeddings
            zL = behrt(xL, mask=mL)   # [B, d]
            notes_tok = pretok_batch_notes(notes)
            zN = bbert(notes_tok)     # [B, d]
            zI = imgenc(imgs)         # [B, d]
            z = {"L": zL, "N": zN, "I": zI}

            if not printed_unimodal:
                printed_unimodal = True
                print(f"[eval:unimodal] zL:{tuple(zL.shape)} zN:{tuple(zN.shape)} zI:{tuple(zI.shape)}")
                pretty_print_small_batch(xL, mL, notes, dbg, k=3)

            # 7-route capsule inference
            out = forward_capsule_from_routes(
                z_unimodal=z, fusion=fusion, projector=projector, capsule_head=cap_head
            )
            logits, prim_acts, route_embs = out[0], out[1], out[2]
            # routing_coef is optional (may be absent depending on head)
            routing_coef = out[3] if len(out) > 3 else None

            # print once, and then every rpt_every batches if configured
            if (not printed_caps_once) or (rpt_every > 0 and ((bidx + 1) % rpt_every == 0)):
                printed_caps_once = True
                keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
                print(f"[eval:caps] logits:{tuple(logits.shape)} "
                      f"prim_acts:{tuple(prim_acts.shape)} routes -> {keys}")

            loss = loss_fn(logits, y)     # CE: logits [B,2], y [B] long

        # accumulate on CPU
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)
        act_sum += prim_acts.detach().float().cpu().sum(dim=0).squeeze(-1)

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

@torch.no_grad()
def collect_epoch_outputs(loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx):
    behrt.eval(); imgenc.eval()
    if getattr(bbert, "bert", None) is not None: bbert.bert.eval()
    y_true, p1, y_pred, ids = [], [], [], []
    ce = nn.CrossEntropyLoss(reduction="none")
    for xL, mL, notes, imgs, y, dbg in loader:
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)
        with amp_ctx:
            zL = behrt(xL, mask=mL)
            zN = bbert(pretok_batch_notes(notes))
            zI = imgenc(imgs)
            logits, _, _ = forward_capsule_from_routes(
                {"L": zL, "N": zN, "I": zI}, fusion, projector, cap_head
            )[:3]
        probs = torch.softmax(logits, dim=-1)[:, 1]
        y_true.append(y.detach().cpu())
        p1.append(probs.detach().cpu())
        y_pred.append(logits.argmax(dim=1).detach().cpu())
        ids += dbg.get("stay_ids", [])
    y_true = torch.cat(y_true).numpy()
    p1     = torch.cat(p1).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return y_true, p1, y_pred, ids

def epoch_metrics(y_true, p1, y_pred):
    out = {}
    try:
        out["AUROC"] = float(roc_auc_score(y_true, p1))
    except Exception:
        out["AUROC"] = float("nan")
    try:
        out["AUPRC"] = float(average_precision_score(y_true, p1))
    except Exception:
        out["AUPRC"] = float("nan")
    out["F1"]     = float(f1_score(y_true, y_pred))
    out["Recall"] = float(recall_score(y_true, y_pred))
    out["CM"]     = confusion_matrix(y_true, y_pred)
    return out

def expected_calibration_error(p, y, n_bins=10):
    p = np.asarray(p); y = np.asarray(y).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids = 0.5 * (bins[1:] + bins[:-1])
    ece = 0.0
    bconf, bacc, bcnt = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            bconf.append(0.0); bacc.append(0.0); bcnt.append(0)
            continue
        conf = float(p[m].mean()); acc = float((y[m] == 1).mean())
        w = m.mean()
        ece += w * abs(acc - conf)
        bconf.append(conf); bacc.append(acc); bcnt.append(int(m.sum()))
    return float(ece), mids, np.array(bconf), np.array(bacc), np.array(bcnt)

def reliability_plot(bin_centers, bin_conf, bin_acc, out_path):
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(bin_conf, bin_acc, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _print_routing_mort(routing_coef, prim_acts, where=""):
    # routing_coef: [B,7,2], prim_acts: [B,7,1]
    with torch.no_grad():
        beta = routing_coef * prim_acts  # broadcast to [B,7,2]
        bmean = beta.mean(dim=0).detach().cpu().numpy()  # [7,2]
        routes = ["L","N","I","LN","LI","NI","LNI"]
        msg = " | ".join(f"{r}:{bmean[i,0]:.3f}/{bmean[i,1]:.3f}" for i,r in enumerate(routes))
        print(f"[routing β] {where} {msg}")

def main():
    args = parse_args()
    load_cfg()

    # Initialize tokenizer/maxlen AFTER load_cfg and make them visible to pretok_batch_notes()
    global TOKENIZER, MAXLEN
    TOKENIZER = AutoTokenizer.from_pretrained(CFG.text_model_name)
    MAXLEN = int(getattr(CFG, "max_text_len", 512))

    print(f"[setup] DEVICE={DEVICE} | batch_size={args.batch_size} | epochs={args.epochs}")

    # AMP policy: enable ONLY on CUDA. On CPU => nullcontext
    use_cuda = (str(DEVICE).startswith("cuda") and torch.cuda.is_available())
    if use_cuda:
        if args.precision == "fp16":
            amp_ctx = torch_amp.autocast(device_type="cuda", dtype=torch.float16)
        elif args.precision == "bf16":
            amp_ctx = torch_amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:  # "auto" -> let PyTorch pick the best dtype for the device
            amp_ctx = torch_amp.autocast(device_type="cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        amp_ctx = nullcontext()
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    train_ds = ICUStayDataset(args.data_root, split="train")
    val_ds   = ICUStayDataset(args.data_root, split="val")
    test_ds  = ICUStayDataset(args.data_root, split="test")

    from torch.utils.data import WeightedRandomSampler

    # Prevalence on the *kept* train IDs (after tri-modal filtering)
    train_label_df = train_ds.labels.merge(
        pd.DataFrame({"stay_id": train_ds.ids}), on="stay_id"
    )
    pos = int(train_label_df["mort"].sum())
    neg = len(train_label_df) - pos
    pos_ratio = (neg / max(1, pos))  # e.g., ~4–10x


    # Build per-sample weights matching the loader’s order (train_ds.ids)
    y_by_id = {int(r.stay_id): int(r.mort) for _, r in train_label_df.iterrows()}
    weights = [pos_ratio if y_by_id[sid] == 1 else 1.0 for sid in train_ds.ids]

    # Use sampler (balanced batches); do NOT also weight the loss
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # ---- Unweighted CE (since sampler already balances) ----
    ce = nn.CrossEntropyLoss(reduction="mean")
    print("[loss] using UNWEIGHTED CE with class-balanced sampler")

    collate_train = collate_fn_factory(tidx=TASK_MAP[args.task],
                                       img_tfms=build_image_transform("train"))
    collate_eval  = collate_fn_factory(tidx=TASK_MAP[args.task],
                                       img_tfms=build_image_transform("val"))
    pin = use_cuda

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,           # <-- use sampler
        shuffle=False,             # <-- must be False when sampler is set
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_train,
        drop_last=False            # (optional) True for perfectly even steps
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
                notes_tok = pretok_batch_notes(notes)
                zN = bbert(notes_tok)             # [B,d]
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

                if getattr(args, "route_debug", False) and routing_coef is not None and (step % 100 == 0):
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
            act_sum += prim_acts.detach().cpu().sum(dim=0).squeeze(-1)

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

        # Validation: loss/acc/avg acts
        val_loss, val_acc, val_act = evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, val_loader, amp_ctx, ce)
        print(f"[epoch {epoch+1}] VAL loss={val_loss:.4f} acc={val_acc:.4f} "
            f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k,v in val_act.items())}")


        # Validation: full metrics (AUROC/AUPRC/F1/Recall) + ECE + reliability plot
        y_true, p1, y_pred, _ = collect_epoch_outputs(val_loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx)
        m = epoch_metrics(y_true, p1, y_pred)
        print(f"[epoch {epoch+1}] VAL AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f} F1={m['F1']:.4f} Recall={m['Recall']:.4f}")
        print(f"[epoch {epoch+1}] VAL Confusion Matrix:\n{m['CM']}")
        ece, centers, bconf, bacc, bcnt = expected_calibration_error(p1, y_true, n_bins=args.calib_bins)
        print(f"[epoch {epoch+1}] VAL ECE({args.calib_bins} bins) = {ece:.4f}")
        rel_path = os.path.join(ckpt_dir, f"reliability_val_epoch{epoch+1:03d}.png")
        reliability_plot(centers, bconf, bacc, out_path=rel_path)
        print(f"[epoch {epoch+1}] Saved reliability diagram → {rel_path}")


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
        _ = torch.load(best_path, map_location="cpu") # peek to avoid silent failure
        _ = torch.load(best_path, map_location="cpu") # consistent load for existence
        # Reload properly via helper (ensures optimizer not needed for test)
        ckpt = torch.load(best_path, map_location="cpu")
        behrt.load_state_dict(ckpt["behrt"]) ; bbert.load_state_dict(ckpt["bbert"]) ; imgenc.load_state_dict(ckpt["imgenc"])
        for k in fusion.keys():
            fusion[k].load_state_dict(ckpt["fusion"][k])
        projector.load_state_dict(ckpt["projector"]) ; cap_head.load_state_dict(ckpt["cap_head"])


    test_loss, test_acc, test_act = evaluate_epoch(behrt, bbert, imgenc, fusion, projector, cap_head, test_loader, amp_ctx, ce)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} "
        f"avg_prim_act={', '.join(f'{k}:{v:.3f}' for k,v in test_act.items())}")


    # Test metrics + calibration
    y_true_t, p1_t, y_pred_t, _ = collect_epoch_outputs(test_loader, behrt, bbert, imgenc, fusion, projector, cap_head, amp_ctx)    
    mt = epoch_metrics(y_true_t, p1_t, y_pred_t)
    print(f"[TEST] AUROC={mt['AUROC']:.4f} AUPRC={mt['AUPRC']:.4f} F1={mt['F1']:.4f} Recall={mt['Recall']:.4f}")
    print(f"[TEST] Confusion Matrix:\n{mt['CM']}")
    ece_t, centers_t, bconf_t, bacc_t, bcnt_t = expected_calibration_error(p1_t, y_true_t, n_bins=args.calib_bins)
    print(f"[TEST] ECE({args.calib_bins} bins) = {ece_t:.4f}")
    rel_test_path = os.path.join(ckpt_dir, "reliability_test.png")
    reliability_plot(centers_t, bconf_t, bacc_t, out_path=rel_test_path)
    print(f"[TEST] Saved reliability diagram → {rel_test_path}")




if __name__ == "__main__":
    main()
