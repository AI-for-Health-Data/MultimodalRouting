from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import torch.nn.functional as F
from torch.utils.data import DataLoader

from env_config import CFG, DEVICE, ensure_dir, set_deterministic, set_global_seed
from encoders import (
    EncoderConfig,
    build_encoders,
    encode_pooled_modalities,
)
from routing_and_heads import (
    ROUTES,
    build_fusions,
    LearnedClasswiseGateNet,
    FinalConcatHeadClasswise,
    build_route_heads,
    per_class_route_embeddings_learned,
    per_class_route_embeddings_loss_based,
    route_availability_mask,
)

# Phenotype set (25 labels)
PHENO_25 = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
]

# Utilities
def try_torch_bincount_weight(y: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Compute positive class weights per label: (N - pos) / max(pos,1).
    y: [N,K] {0,1} float or bool.
    """
    with torch.no_grad():
        if y.numel() == 0:
            return None
        K = y.size(1)
        pos = y.sum(dim=0).clamp(min=0.0)
        N = y.size(0)
        # Avoid div by zero; if pos==0 -> weight=N (very large), cap mildly
        w = (N - pos) / (pos.clamp(min=1.0))
        # Make sure it's >=1 and finite
        w = torch.where(torch.isfinite(w), w, torch.ones_like(w))
        w = torch.clamp(w, min=1.0, max=100.0)
        if (w <= 0).any():
            return None
        return w

def sigmoid_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Basic multilabel metrics (macro AUROC is optional without sklearn).
    Returns macro-averaged precision-at-0.5 and recall-at-0.5 and F1.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        eps = 1e-7
        tp = (preds * targets).sum(0)
        fp = ((preds == 1) & (targets == 0)).float().sum(0)
        fn = ((preds == 0) & (targets == 1)).float().sum(0)

        prec_k = tp / (tp + fp + eps)
        rec_k = tp / (tp + fn + eps)
        f1_k = 2 * prec_k * rec_k / (prec_k + rec_k + eps)

        return {
            "macro_precision@0.5": prec_k.mean().item(),
            "macro_recall@0.5": rec_k.mean().item(),
            "macro_f1@0.5": f1_k.mean().item(),
        }

# Image utilities
def default_img_transform(img: Image.Image) -> torch.Tensor:
    # Minimal transform to 224x224 and to tensor [0,1]
    img = img.convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1)) 
    return torch.from_numpy(arr)

class DicomToJpgIndex:

    def __init__(self, image_filenames: Union[str, Path]):
        p = Path(image_filenames)
        self.map: Dict[str, str] = {}
        if not p.exists():
            print(f"[warn] IMAGE_FILENAMES '{p}' not found. Will expect 'jpg_path' in dataset.")
            return
        if p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
            if "dicom_id" in df.columns and "jpg_path" in df.columns:
                self.map = dict(zip(df["dicom_id"].astype(str), df["jpg_path"].astype(str)))
        else:
            with open(p, "r") as f:
                for line in f:
                    path_str = line.strip()
                    if not path_str:
                        continue
                    stem = Path(path_str).stem
                    self.map[stem] = path_str

    def get(self, dicom_id: Union[str, int]) -> Optional[str]:
        return self.map.get(str(dicom_id), None)


class PhenoTriplesDataset(tud.Dataset):
    def __init__(
        self,
        parquet_path: Union[str, Path],
        img_root: Union[str, Path],
        image_filenames: Optional[Union[str, Path]] = None,
        pheno_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        self.img_root = Path(img_root)
        self.idx = DicomToJpgIndex(image_filenames) if image_filenames else None

        # label columns
        all_cols = list(self.df.columns)
        pheno_list = pheno_names or PHENO_25
        self.label_cols = [c for c in pheno_list if c in all_cols]
        if len(self.label_cols) == 0:
            raise RuntimeError(f"No phenotype columns found in {parquet_path}.")

        # heuristic columns
        self.has_dicom = "dicom_id" in all_cols
        self.has_jpg_path = "jpg_path" in all_cols
        self.has_notes_chunks = "note_chunks" in all_cols
        self.has_note_text = "note_text" in all_cols
        self.has_ehr_tensor = "ehr_tensor" in all_cols
        self.has_ehr_mask = "ehr_mask" in all_cols

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (img_tensor [3,224,224], mask [1])
        If missing, returns zeros and mask=0.
        """
        path: Optional[str] = None
        if self.has_jpg_path and isinstance(row["jpg_path"], str) and row["jpg_path"]:
            path = row["jpg_path"]
        elif self.has_dicom and self.idx:
            path = self.idx.get(row["dicom_id"])
        if path is None:
            return torch.zeros(3, 224, 224), torch.zeros(1)
        p = self.img_root / path if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            return torch.zeros(3, 224, 224), torch.zeros(1)
        try:
            img = Image.open(p)
            x = default_img_transform(img)
            return x, torch.ones(1)
        except Exception:
            return torch.zeros(3, 224, 224), torch.zeros(1)

    def _load_notes(self, row) -> Tuple[List[str], torch.Tensor]:
        if self.has_notes_chunks and isinstance(row["note_chunks"], list) and len(row["note_chunks"]) > 0:
            return list(map(str, row["note_chunks"])), torch.ones(1)
        if self.has_note_text and isinstance(row["note_text"], str) and row["note_text"].strip():
            return [row["note_text"]], torch.ones(1)
        return [], torch.zeros(1)

    def _load_ehr(self, row) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.has_ehr_tensor and isinstance(row["ehr_tensor"], (list, np.ndarray)):
            arr = np.asarray(row["ehr_tensor"], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            xL = torch.from_numpy(arr)  # [T,F]
            if self.has_ehr_mask and isinstance(row["ehr_mask"], (list, np.ndarray)):
                m = torch.from_numpy(np.asarray(row["ehr_mask"], dtype=np.float32))
            else:
                m = torch.ones(xL.size(0), dtype=torch.float32)
            return xL, m
        Fdim = CFG.structured_n_feats
        xL = torch.zeros(1, Fdim)
        m = torch.zeros(1)
        return xL, m

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # labels -> float32 {0,1}
        y = torch.tensor(row[self.label_cols].values.astype(np.float32))  # [K]

        # modalities
        xI, mI = self._load_image(row)           # [3,224,224], [1]
        notes, mN = self._load_notes(row)        # list[str], [1]
        xL, mL = self._load_ehr(row)             # [T,F], [T]

        masks = {
            "L": mL.max().view(1),  
            "N": mN.view(1),
            "I": mI.view(1),
        }
        sample = {
            "xL": xL,
            "mL": mL,
            "notes": notes,
            "img": xI,
            "masks": masks,
            "y": y,
        }
        return sample

def pad_collate(batch: List[Dict]):
    B = len(batch)
    # EHR pad
    T_max = max(item["xL"].size(0) for item in batch)
    Fdim = batch[0]["xL"].size(1)
    xL = torch.zeros(B, T_max, Fdim)
    mL = torch.zeros(B, T_max)
    for i, item in enumerate(batch):
        t = item["xL"].size(0)
        xL[i, :t] = item["xL"]
        mL[i, :t] = item["mL"]

    # Notes as list-of-lists
    notes = [item["notes"] for item in batch]

    # Images -> [B,3,224,224]
    imgs = torch.stack([item["img"] for item in batch], dim=0)

    # Route availability masks -> per-modality [B,1]
    mL_av = torch.stack([item["masks"]["L"] for item in batch], dim=0)
    mN_av = torch.stack([item["masks"]["N"] for item in batch], dim=0)
    mI_av = torch.stack([item["masks"]["I"] for item in batch], dim=0)
    masks = {"L": mL_av, "N": mN_av, "I": mI_av}

    y = torch.stack([item["y"] for item in batch], dim=0)  # [B,K]
    return xL, mL, notes, imgs, masks, y


class PhenoModel(nn.Module):
    """
    End-to-end:
      encoders -> unimodal z -> fusion -> per-class gates -> Zw_cls [B,7,K,d] -> concat -> logits [B,K]
    Supports:
      gate_mode = "learned" (default)  OR  gate_mode = "loss_based"
    """
    def __init__(self, cfg: EncoderConfig, K: int, gate_mode: str = "learned"):
        super().__init__()
        self.K = int(K)
        self.gate_mode = gate_mode
        # encoders
        self.behrt, self.bbert, self.imgenc = build_encoders(cfg)
        # fusions
        self.fusions = build_fusions(
            d=cfg.d,
            p_drop=cfg.dropout,
            feature_mode=CFG.feature_mode,
            bi_fusion_mode=CFG.bi_fusion_mode,
            tri_fusion_mode=CFG.tri_fusion_mode,
        )
        # classwise head
        self.head_cls = FinalConcatHeadClasswise(d=cfg.d, p_drop=cfg.dropout)
        # gates
        if gate_mode == "learned":
            self.cls_gate = LearnedClasswiseGateNet(d=cfg.d, n_tasks=K, p_drop=cfg.dropout)
            self.route_heads = None
        elif gate_mode == "loss_based":
            self.cls_gate = None
            self.route_heads = build_route_heads(d=cfg.d, n_tasks=K, p_drop=cfg.dropout)
        else:
            raise ValueError("gate_mode must be {'learned','loss_based'}")

    def forward(
        self,
        xL: torch.Tensor, mL: torch.Tensor,
        notes: List[List[str]],
        imgs: torch.Tensor,
        masks: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        alpha: float = 4.0,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Encode
        z = encode_pooled_modalities(self.behrt, self.bbert, self.imgenc, xL, notes, imgs, mL) 

        # Per-class 7-route embeddings
        if self.gate_mode == "learned":
            Zw_cls, gates_cls, route_embs = per_class_route_embeddings_learned(
                z_unimodal=z, fusion=self.fusions, gate_net_classwise=self.cls_gate, masks=masks,
                l2norm_each=CFG.l2norm_each
            )
            X_flat, _ = self._concat_for_head(Zw_cls)   # [B*K,7*d]
            logits = self.head_cls(X_flat, B=labels.size(0) if labels is not None else Zw_cls.size(0), K=self.K)
            out = {"logits": logits, "Zw_cls": Zw_cls, "gates_cls": gates_cls, "route_embs": route_embs}
            return out

        else:  # loss_based
            assert labels is not None, "loss_based gates require labels in forward()"
            Zw_cls, gates_cls, route_embs, logits_routes = per_class_route_embeddings_loss_based(
                z_unimodal=z, fusion=self.fusions, route_heads=self.route_heads, y=labels,
                masks=masks, alpha=alpha, pos_weight=pos_weight, l2norm_each=CFG.l2norm_each
            )
            X_flat, _ = self._concat_for_head(Zw_cls)
            logits = self.head_cls(X_flat, B=labels.size(0), K=self.K)
            out = {
                "logits": logits,
                "Zw_cls": Zw_cls,
                "gates_cls": gates_cls,
                "route_embs": route_embs,
                "logits_routes": logits_routes,
            }
            return out

    @staticmethod
    def _concat_for_head(Zw_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zw_cls: [B,7,K,d] -> [B*K, 7*d]
        """
        B, R, K, d = Zw_cls.shape
        X = Zw_cls.permute(0, 2, 1, 3).contiguous().view(B, K, R * d)  # [B,K,7*d]
        X_flat = X.view(B * K, R * d)
        return X_flat, X

def step_batch(
    model: PhenoModel,
    batch: Tuple,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    gate_mode: str,
    pos_weight: Optional[torch.Tensor],
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    (xL, mL, notes, imgs, masks, y) = batch
    xL = xL.to(DEVICE)
    mL = mL.to(DEVICE)
    imgs = imgs.to(DEVICE)
    y = y.to(DEVICE).float()
    masks = {k: v.to(DEVICE) for k, v in masks.items()}
    pos_w = pos_weight.to(DEVICE) if pos_weight is not None else None

    out = model(xL, mL, notes, imgs, masks, labels=y if gate_mode == "loss_based" else None, alpha=alpha, pos_weight=pos_w)
    logits = out["logits"]  # [B,K]
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
        optimizer.step()

    return loss.detach(), {"logits": logits.detach(), "targets": y.detach(), **out}

def run_epoch(dl: DataLoader, model: PhenoModel, train: bool, criterion, optimizer, gate_mode, pos_weight, alpha):
    model.train(mode=train)
    total_loss = 0.0
    n = 0
    all_logits = []
    all_targets = []
    for batch in dl:
        loss, out = step_batch(model, batch, criterion, optimizer if train else None, gate_mode, pos_weight, alpha)
        B = out["targets"].size(0)
        total_loss += loss.item() * B
        n += B
        all_logits.append(out["logits"].cpu())
        all_targets.append(out["targets"].cpu())
    avg_loss = total_loss / max(1, n)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = sigmoid_metrics(logits, targets)
    return avg_loss, metrics

def build_argparser():
    p = argparse.ArgumentParser("PhenoModel — per-class 7-route embeddings training")
    p.add_argument("--base", type=str, default="phenomodel", help="folder with *_triples_img_exist.parquet")
    p.add_argument("--splits", type=str, nargs="+", default=["train", "validate", "test"])
    p.add_argument("--image-root", type=str, default=os.environ.get("CXR_JPG_ROOT", "mimic-cxr-jpg/2.1.0"))
    p.add_argument("--image-filenames", type=str, default=os.environ.get("IMAGE_FILENAMES", "IMAGE_FILENAMES"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=CFG.lr)
    p.add_argument("--weight-decay", type=float, default=CFG.weight_decay)
    p.add_argument("--gate-mode", type=str, default="learned", choices=["learned", "loss_based"])
    p.add_argument("--alpha", type=float, default=CFG.loss_gate_alpha, help="alpha for loss-based gates")
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--num-workers", type=int, default=CFG.num_workers)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--log-gates", action="store_true", help="save gates snapshots .npy on val")
    return p

def main():
    args = build_argparser().parse_args()
    set_global_seed(args.seed)
    set_deterministic(CFG.deterministic)
    ensure_dir(args.save_dir)

    # Datasets
    base = Path(args.base)
    split2file = {sp: base / f"paired_pheno_{sp}_triples_img_exist.parquet" for sp in args.splits}
    for sp, f in split2file.items():
        if not f.exists():
            raise FileNotFoundError(f"Missing split parquet: {f}")

    ds_train = PhenoTriplesDataset(split2file["train"], args.image_root, args.image_filenames, PHENO_25)
    ds_val = PhenoTriplesDataset(split2file.get("validate", split2file["train"]), args.image_root, args.image_filenames, PHENO_25)
    ds_test = PhenoTriplesDataset(split2file.get("test", split2file["train"]), args.image_root, args.image_filenames, PHENO_25)

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=True)

    # Compute pos_weight on train
    with torch.no_grad():
        ys = []
        for _, _, _, _, _, y in dl_train:
            ys.append(y)
        Y_train = torch.cat(ys, dim=0)
        pos_weight = try_torch_bincount_weight(Y_train)

    # Build model
    enc_cfg = EncoderConfig(
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
    K = len(ds_train.label_cols)
    model = PhenoModel(enc_cfg, K=K, gate_mode=args.gate_mode).to(DEVICE)

    # Optimizer / Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None)

    best_val_f1 = -1.0
    ckpt_path = Path(args.save_dir) / f"pheno_{args.gate_mode}_best.pt"

    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss, tr_metrics = run_epoch(dl_train, model, True, criterion, optimizer, args.gate_mode, pos_weight, args.alpha)
        # Validate
        va_loss, va_metrics = run_epoch(dl_val, model, False, criterion, None, args.gate_mode, pos_weight, args.alpha)

        print(f"[{epoch:03d}] "
              f"train loss {tr_loss:.4f} f1 {tr_metrics['macro_f1@0.5']:.4f} | "
              f"val loss {va_loss:.4f} f1 {va_metrics['macro_f1@0.5']:.4f}")

        # Save best
        if va_metrics["macro_f1@0.5"] > best_val_f1:
            best_val_f1 = va_metrics["macro_f1@0.5"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                    "cfg": vars(CFG),
                    "enc_cfg": enc_cfg.__dict__,
                    "label_cols": ds_train.label_cols,
                },
                ckpt_path,
            )
            print(f"  -> saved best to {ckpt_path} (val macro_f1@0.5={best_val_f1:.4f})")

        # Optional: quick gate snapshot on val (first batch)
        if args.log_gates:
            model.eval()
            with torch.no_grad():
                for batch in dl_val:
                    _, out = step_batch(model, batch, criterion, None, args.gate_mode, pos_weight, args.alpha)
                    if "gates_cls" in out:
                        gates = out["gates_cls"].cpu().numpy()  # [B,7,K]
                        np.save(Path(args.save_dir) / f"gates_epoch{epoch}.npy", gates)
                    break

    # Test with best model
    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(blob["model"])
        print(f"[load best] epoch={blob['epoch']} val_f1={blob['val_f1']:.4f}")

    te_loss, te_metrics = run_epoch(dl_test, model, False, criterion, None, args.gate_mode, pos_weight, args.alpha)
    print(f"[test] loss {te_loss:.4f}  macro_f1@0.5 {te_metrics['macro_f1@0.5']:.4f}  "
          f"macro_precision@0.5 {te_metrics['macro_precision@0.5']:.4f}  "
          f"macro_recall@0.5 {te_metrics['macro_recall@0.5']:.4f}")

if __name__ == "__main__":
    main()
