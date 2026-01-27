from __future__ import annotations
import os
import math
import time
import json
import random
import argparse
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from env_config import CFG, DEVICE, load_cfg, apply_cli_overrides, Config  
import main as main_mod 
from encoders import build_encoders, encode_unimodal_pooled 
from transformers import AutoTokenizer

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device) for x in obj)
    return obj

@torch.no_grad()
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def get_batch_label(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)) and len(batch) >= 5:
        y = batch[4]
    elif isinstance(batch, dict):
        y = batch.get("y_mort", batch.get("mortality", batch.get("y", None)))
        if y is None:
            raise KeyError("Batch missing label key")
    else:
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    if not torch.is_tensor(y):
        y = torch.tensor(y)

    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    return y.float()

class LateFusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.1, num_layers: int = 2) -> None:
        super().__init__()
        assert num_layers >= 1
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LateFusionModel(nn.Module):
    def __init__(
        self,
        encoder_cfg: Any,
        dropout: float = 0.1,
        head_hidden_dim: int = 512,
        head_layers: int = 2,
        add_presence_flags: bool = True,
        freeze_encoders: bool = False,
    ) -> None:
        super().__init__()

        self.behrt, self.bbert, self.imgenc = build_encoders(encoder_cfg)
        def _get_dim(enc, fallback_name: str) -> Optional[int]:
            for attr in ["out_dim", "hidden_dim", "d_model", "embed_dim", "dim"]:
                if hasattr(enc, attr):
                    v = getattr(enc, attr)
                    if isinstance(v, int) and v > 0:
                        return v
            if hasattr(enc, "config") and hasattr(enc.config, "hidden_size"):
                v = enc.config.hidden_size
                if isinstance(v, int) and v > 0:
                    return v
            return None

        dL = _get_dim(self.behrt, "L") if self.behrt is not None else None
        dN = _get_dim(self.bbert, "N") if self.bbert is not None else None
        dI = _get_dim(self.imgenc, "I") if self.imgenc is not None else None

        for name, val in [("dL", dL), ("dN", dN), ("dI", dI)]:
            if val is None:
                for attr in [
                    "d_l", "d_n", "d_i",
                    "behrt_dim", "bbert_dim", "img_dim",
                    "structured_dim", "text_dim", "image_dim",
                ]:
                    if hasattr(encoder_cfg, attr):
                        v = getattr(encoder_cfg, attr)
                        if isinstance(v, int) and v > 0:
                            if name == "dL":
                                dL = v
                            elif name == "dN":
                                dN = v
                            elif name == "dI":
                                dI = v
                            break

        if self.behrt is not None and dL is None:
            raise RuntimeError("Could not infer pooled dim for BEHRT encoder. Add it to EncoderConfig (e.g., d_l).")
        if self.bbert is not None and dN is None:
            raise RuntimeError("Could not infer pooled dim for BioClinicalBERT encoder. Add it to EncoderConfig (e.g., d_n).")
        if self.imgenc is not None and dI is None:
            raise RuntimeError("Could not infer pooled dim for Image encoder. Add it to EncoderConfig (e.g., d_i).")

        self.dL = int(dL or 0)
        self.dN = int(dN or 0)
        self.dI = int(dI or 0)

        self.add_presence_flags = add_presence_flags
        in_dim = self.dL + self.dN + self.dI + (3 if add_presence_flags else 0)
        self.head = LateFusionHead(
            in_dim=in_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            num_layers=head_layers,
        )

        if freeze_encoders:
            for m in [self.behrt, self.bbert, self.imgenc]:
                if m is None:
                    continue
                for p in m.parameters():
                    p.requires_grad = False

    def _presence_from_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "has_L" in batch and "has_N" in batch and "has_I" in batch:
            hasL = batch["has_L"]
            hasN = batch["has_N"]
            hasI = batch["has_I"]
            if not torch.is_tensor(hasL): hasL = torch.tensor(hasL)
            if not torch.is_tensor(hasN): hasN = torch.tensor(hasN)
            if not torch.is_tensor(hasI): hasI = torch.tensor(hasI)
            return hasL.float(), hasN.float(), hasI.float()

        B = None
        for k in ["xL", "notes_batch", "imgs", "mL"]:
            if k in batch and torch.is_tensor(batch[k]):
                B = batch[k].shape[0]
                break
            if k in batch and isinstance(batch[k], dict):
                for v in batch[k].values():
                    if torch.is_tensor(v):
                        B = v.shape[0]
                        break
            if B is not None:
                break
        if B is None:
            raise RuntimeError("Could not infer batch size for presence flags.")

        device = next(self.head.parameters()).device
        hasL = torch.ones(B, device=device) if batch.get("xL", None) is not None else torch.zeros(B, device=device)
        hasN = torch.ones(B, device=device) if batch.get("notes_batch", None) is not None else torch.zeros(B, device=device)
        hasI = torch.ones(B, device=device) if batch.get("imgs", None) is not None else torch.zeros(B, device=device)
        return hasL, hasN, hasI

    def forward(self, batch) -> torch.Tensor:
        if isinstance(batch, (tuple, list)) and len(batch) >= 5:
            xL, mL, notes_batch, imgs = batch[0], batch[1], batch[2], batch[3]

            Tcfg = int(getattr(CFG, "structured_seq_len", xL.shape[1]))
            if xL is not None and torch.is_tensor(xL) and xL.ndim == 3 and xL.shape[1] != Tcfg:
                xL = xL[:, :Tcfg, :]
            if mL is not None and torch.is_tensor(mL) and mL.ndim == 2 and mL.shape[1] != Tcfg:
                mL = mL[:, :Tcfg]

            batch_dict = {"xL": xL, "mL": mL, "notes_batch": notes_batch, "imgs": imgs}

        elif isinstance(batch, dict):
            batch_dict = batch
            xL = batch_dict.get("xL", None)
            mL = batch_dict.get("mL", None)

            if torch.is_tensor(xL) and xL.ndim == 3:
                Tcfg = int(getattr(CFG, "structured_seq_len", xL.shape[1]))
                if xL.shape[1] != Tcfg:
                    batch_dict["xL"] = xL[:, :Tcfg, :]
            if torch.is_tensor(mL) and mL.ndim == 2:
                Tcfg = int(getattr(CFG, "structured_seq_len", mL.shape[1]))
                if mL.shape[1] != Tcfg:
                    batch_dict["mL"] = mL[:, :Tcfg]
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        Tcfg = int(getattr(CFG, "structured_seq_len", 0) or 0)
        if Tcfg > 0:
            xL = batch_dict.get("xL", None)
            mL = batch_dict.get("mL", None)

            if torch.is_tensor(xL) and xL.ndim == 3 and xL.shape[1] > Tcfg:
                batch_dict["xL"] = xL[:, :Tcfg, :]

            if torch.is_tensor(mL) and mL.ndim == 2 and mL.shape[1] > Tcfg:
                batch_dict["mL"] = mL[:, :Tcfg]

        pooled = encode_unimodal_pooled(
            self.behrt, self.bbert, self.imgenc,
            batch_dict.get("xL", None),
            batch_dict.get("notes_batch", None),
            batch_dict.get("imgs", None),
            batch_dict.get("mL", None),
        )

        device = next(self.head.parameters()).device
        B = batch_dict["xL"].shape[0]

        hasL = torch.ones(B, device=device)
        hasN = torch.ones(B, device=device)
        hasI = torch.ones(B, device=device)

        zL = pooled["L"]
        zN = pooled["N"]
        zI = pooled["I"]

        feats = [zL, zN, zI]
        if self.add_presence_flags:
            feats += [hasL.unsqueeze(-1), hasN.unsqueeze(-1), hasI.unsqueeze(-1)]
        return self.head(x)

@torch.no_grad()
def binary_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_score), dtype=np.float64) + 1.0

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1

    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_pos_weight_from_loader(loader: DataLoader, device: torch.device, max_batches: int = 200) -> Optional[torch.Tensor]:
    n_pos = 0
    n_neg = 0
    seen = 0
    for batch in loader:
        y = get_batch_label(batch).detach().cpu().numpy()
        n_pos += int((y == 1).sum())
        n_neg += int((y == 0).sum())
        seen += 1
        if seen >= max_batches:
            break
    if n_pos == 0:
        return None
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)


def run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    pos_weight: Optional[torch.Tensor],
    grad_clip: float = 0.0,
    log_every: int = 50,
) -> Dict[str, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

    losses = []
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        y = get_batch_label(batch)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(batch)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

        if log_every > 0 and step % log_every == 0:
            dt = time.time() - t0
            print(f"[train] step={step:6d} loss={np.mean(losses[-log_every:]):.4f} ({dt:.1f}s)")
            t0 = time.time()

    return {"loss": float(np.mean(losses)) if losses else float("nan")}


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_y = []
    losses = []
    criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = to_device(batch, device)
        y = get_batch_label(batch)
        logits = model(batch)
        loss = criterion(logits, y)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        losses.append(float(loss.detach().cpu().item()))

    if not all_logits:
        return {"loss": float("nan"), "auc": float("nan"), "brier": float("nan")}

    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0)
    prob = sigmoid_np(logits)

    auc = binary_auc_roc(y, prob)
    brier = binary_brier(y, prob)

    return {
        "loss": float(np.mean(losses)),
        "auc": float(auc),
        "brier": float(brier),
    }


def build_loaders_from_project(cfg_unused=None):
    data_root = os.path.abspath(os.path.expanduser(getattr(CFG, "data_root", "./data")))
    train_ds = main_mod.ICUStayDataset(root=data_root, split="train")
    val_ds   = main_mod.ICUStayDataset(root=data_root, split="val")
    test_ds  = main_mod.ICUStayDataset(root=data_root, split="test") if os.path.exists(os.path.join(data_root, "splits.json")) else None

    train_tfms = main_mod.build_image_transform("train")
    eval_tfms  = main_mod.build_image_transform("val")

    collate_train = main_mod.collate_fn_factory(train_tfms)
    collate_eval  = main_mod.collate_fn_factory(eval_tfms)

    bs = int(getattr(CFG, "batch_size", 16))
    nw = int(getattr(CFG, "num_workers", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_train,
        worker_init_fn=main_mod.seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_eval,
        worker_init_fn=main_mod.seed_worker,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_eval,
            worker_init_fn=main_mod.seed_worker,
        )

    return train_loader, val_loader, test_loader

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="mortality")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--ckpt_root", type=str, default=None)  
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--precision", type=str, default="auto") 
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head_hidden_dim", type=int, default=512)
    ap.add_argument("--head_layers", type=int, default=2)
    ap.add_argument("--add_presence_flags", action="store_true")
    ap.add_argument("--no_presence_flags", action="store_true")
    ap.add_argument("--freeze_encoders", action="store_true")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--pos_weight_auto", action="store_true")
    ap.add_argument("--save_path", type=str, default="latefusion_mortality.pt")
    ap.add_argument("--config_json", type=str, default="", help="Optional: path to a JSON file to override Config fields.")
    return ap.parse_args()


def apply_config_overrides(cfg: Config, config_json_path: str) -> Config:
    if not config_json_path:
        return cfg
    with open(config_json_path, "r") as f:
        overrides = json.load(f)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            print(f"[warn] Config has no attribute '{k}' (override ignored)")
    return cfg


def main() -> None:
    args = parse_args()

    load_cfg()
    apply_cli_overrides(args)

    if main_mod.TOKENIZER is None:
        text_model = CFG.text_model_name

        main_mod.TOKENIZER = AutoTokenizer.from_pretrained(text_model, use_fast=True)

    cfg = Config()  
    cfg = apply_config_overrides(cfg, args.config_json)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    train_loader, val_loader, test_loader = build_loaders_from_project()
    print("[info] dataloaders ready:",
          f"train={len(train_loader)} val={len(val_loader)} test={(len(test_loader) if test_loader is not None else None)}")

    encoder_cfg = CFG  
    add_presence = args.add_presence_flags and (not args.no_presence_flags)
    if args.no_presence_flags:
        add_presence = False

    model = LateFusionModel(
        encoder_cfg=encoder_cfg,
        dropout=args.dropout,
        head_hidden_dim=args.head_hidden_dim,
        head_layers=args.head_layers,
        add_presence_flags=add_presence,
        freeze_encoders=args.freeze_encoders,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type == "cuda") else None

    # pos_weight
    pos_weight = None
    if args.pos_weight_auto:
        pos_weight = compute_pos_weight_from_loader(train_loader, device=device)
        print(f"[info] pos_weight={pos_weight.detach().cpu().numpy().tolist() if pos_weight is not None else None}")

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch}/{args.epochs} =====")

        tr = run_one_epoch_train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            pos_weight=pos_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
        )
        va = run_eval(model=model, loader=val_loader, device=device)

        print(f"[epoch {epoch}] train_loss={tr['loss']:.4f} | val_loss={va['loss']:.4f} val_auc={va['auc']:.4f} val_brier={va['brier']:.4f}")

        if not math.isnan(va["auc"]) and va["auc"] > best_val_auc:
            best_val_auc = va["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[best] new best val_auc={best_val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    if test_loader is not None:
        te = run_eval(model=model, loader=test_loader, device=device)
        print(f"\n[final test] loss={te['loss']:.4f} auc={te['auc']:.4f} brier={te['brier']:.4f}")

    # Save
    save_obj = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "best_val_auc": best_val_auc,
        "seed": args.seed,
    }
    torch.save(save_obj, args.save_path)
    print(f"[saved] {args.save_path}")


if __name__ == "__main__":
    main()
