from __future__ import annotations
import os
import time
import json
import math
import random
import argparse
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import main as main_mod  
from env_config import CFG, load_cfg, apply_cli_overrides, Config, ensure_dir
from main import epoch_metrics, find_best_thresholds
from encoders import EncoderConfig, build_encoders, encode_unimodal_pooled
from transformers import AutoTokenizer

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

def get_batch_label(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)) and len(batch) >= 5:
        y = batch[4]
    elif isinstance(batch, dict):
        y = batch.get("y", batch.get("y_pheno", None))
        if y is None:
            raise KeyError("Batch missing phenotype label key (expected 'y' or 'y_pheno').")
    else:
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    if not torch.is_tensor(y):
        y = torch.tensor(y)

    if y.ndim == 1:
        y = y.unsqueeze(1)
    return y.float()

class LateFusionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_labels: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        assert num_layers >= 1
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, num_labels))  # (B, K)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, K)


class LateFusionModel(nn.Module):
    def __init__(
        self,
        encoder_cfg: EncoderConfig,
        num_labels: int,
        dropout: float = 0.1,
        head_hidden_dim: int = 512,
        head_layers: int = 2,
        add_presence_flags: bool = False,  
        freeze_encoders: bool = False,
    ):
        super().__init__()

        self.behrt, self.bbert, self.imgenc = build_encoders(encoder_cfg)
        self.num_labels = int(num_labels)
        self.add_presence_flags = bool(add_presence_flags)

        self.head: Optional[LateFusionHead] = None
        self._head_kwargs = dict(
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

    @torch.no_grad()
    def init_head_from_batch(self, batch, device: torch.device) -> None:
        if self.head is not None:
            return

        if not (isinstance(batch, (tuple, list)) and len(batch) >= 4):
            raise TypeError("Expected tuple batch from your collate_fn_factory.")

        xL, mL, notes_batch, imgs = batch[0], batch[1], batch[2], batch[3]

        pooled = encode_unimodal_pooled(
            behrt=self.behrt,
            bbert=self.bbert,
            imgenc=self.imgenc,
            xL=xL,
            notes_batch=notes_batch,
            imgs=imgs,
            mL=mL,
        )
        zL, zN, zI = pooled["L"], pooled["N"], pooled["I"]
        dL, dN, dI = int(zL.shape[-1]), int(zN.shape[-1]), int(zI.shape[-1])

        in_dim = dL + dN + dI + (3 if self.add_presence_flags else 0)
        self.head = LateFusionHead(in_dim=in_dim, num_labels=self.num_labels, **self._head_kwargs).to(device)

        print(f"[latefusion] inferred dL,dN,dI={dL},{dN},{dI} -> head_in={in_dim} K={self.num_labels}")

    def forward(self, batch) -> torch.Tensor:
        assert self.head is not None, "Call init_head_from_batch() once before training."

        if not (isinstance(batch, (tuple, list)) and len(batch) >= 4):
            raise TypeError("Expected tuple batch from your collate_fn_factory.")

        xL, mL, notes_batch, imgs = batch[0], batch[1], batch[2], batch[3]
        B = xL.shape[0]

        pooled = encode_unimodal_pooled(
            behrt=self.behrt,
            bbert=self.bbert,
            imgenc=self.imgenc,
            xL=xL,
            notes_batch=notes_batch,
            imgs=imgs,
            mL=mL,
        )
        zL, zN, zI = pooled["L"], pooled["N"], pooled["I"]

        feats = [zL, zN, zI]
        if self.add_presence_flags:
            device = zL.device
            feats += [
                torch.ones(B, device=device).unsqueeze(-1),
                torch.ones(B, device=device).unsqueeze(-1),
                torch.ones(B, device=device).unsqueeze(-1),
            ]

        x = torch.cat(feats, dim=-1)
        return self.head(x)

def compute_pos_weight_per_label(
    loader: DataLoader,
    device: torch.device,
    K: int,
    max_batches: int = 200,
) -> torch.Tensor:
    pos = np.zeros(K, dtype=np.float64)
    tot = 0
    seen = 0

    for batch in loader:
        y = get_batch_label(batch).detach().cpu().numpy()  # (B,K)
        pos += y.sum(axis=0)
        tot += y.shape[0]
        seen += 1
        if seen >= max_batches:
            break

    neg = tot - pos
    pw = neg / (pos + 1e-6)
    max_pw = float(getattr(CFG, "pos_weight_max", 20.0))
    pw = np.clip(pw, 1.0, max_pw)

    return torch.tensor(pw, dtype=torch.float32, device=device)  # (K,)

def run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor],
    grad_clip: float,
    log_every: int,
    use_bf16: bool,
) -> Dict[str, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    losses = []
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        y = get_batch_label(batch)
        optimizer.zero_grad(set_to_none=True)

        if use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(batch)  # (B,K)
                loss = criterion(logits, y)
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
def run_eval_pheno(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thr: Optional[np.ndarray],
    use_bf16: bool,
) -> Dict[str, Any]:
    model.eval()
    all_logits = []
    all_y = []
    losses = []
    criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = to_device(batch, device)
        y = get_batch_label(batch)

        if use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(batch)
                loss = criterion(logits, y)
        else:
            logits = model(batch)
            loss = criterion(logits, y)

        all_logits.append(logits.detach().float().cpu())
        all_y.append(y.detach().float().cpu())
        losses.append(float(loss.detach().cpu().item()))

    logits = torch.cat(all_logits, dim=0).numpy()  # (N,K)
    y_true = torch.cat(all_y, dim=0).numpy()       # (N,K)
    p = 1.0 / (1.0 + np.exp(-logits))              # (N,K)

    if thr is None:
        y_pred = (p >= 0.5).astype(float)
    else:
        y_pred = (p >= thr[np.newaxis, :]).astype(float)

    m = epoch_metrics(y_true, p, y_pred)

    return {
        "loss": float(np.mean(losses)),
        "y_true": y_true,
        "p": p,
        "y_pred": y_pred,
        "metrics": m,
    }

def build_loaders_from_project(data_root_override: Optional[str] = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = data_root_override if (data_root_override is not None and str(data_root_override).strip() != "") else CFG.data_root
    data_root = os.path.abspath(os.path.expanduser(root))
    print(f"[data] using data_root={data_root}")

    train_ds = main_mod.ICUStayDataset(root=data_root, split="train")
    val_ds   = main_mod.ICUStayDataset(root=data_root, split="val")
    test_ds  = main_mod.ICUStayDataset(root=data_root, split="test")

    train_tfms = main_mod.build_image_transform("train")
    eval_tfms  = main_mod.build_image_transform("val")

    collate_train = main_mod.collate_fn_factory(tidx=0, img_tfms=train_tfms)
    collate_eval  = main_mod.collate_fn_factory(tidx=0, img_tfms=eval_tfms)

    bs = int(getattr(CFG, "batch_size", 16))
    nw = int(getattr(CFG, "num_workers", 4))

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin,
        drop_last=False, collate_fn=collate_train,
        worker_init_fn=main_mod.seed_worker,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin,
        drop_last=False, collate_fn=collate_eval,
        worker_init_fn=main_mod.seed_worker,
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin,
        drop_last=False, collate_fn=collate_eval,
        worker_init_fn=main_mod.seed_worker,
        persistent_workers=(nw > 0),
    )
    return train_loader, val_loader, test_loader

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, default="pheno")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--ckpt_root", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head_hidden_dim", type=int, default=512)
    ap.add_argument("--head_layers", type=int, default=2)
    ap.add_argument("--freeze_encoders", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=300)
    ap.add_argument("--pos_weight_auto", action="store_true")
    ap.add_argument("--save_path", type=str, default="latefusion_pheno.pt")
    ap.add_argument("--config_json", type=str, default="")

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
    if getattr(main_mod, "TOKENIZER", None) is None:
        main_mod.TOKENIZER = AutoTokenizer.from_pretrained(CFG.text_model_name, use_fast=True)

    _ = apply_config_overrides(Config(), args.config_json)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} precision={args.precision}")

    use_bf16 = (device.type == "cuda" and args.precision == "bf16")

    print(f"[debug] args.data_root={args.data_root} | CFG.data_root={CFG.data_root}")
    train_loader, val_loader, test_loader = build_loaders_from_project(args.data_root)

    # infer K
    K = int(getattr(train_loader.dataset, "num_labels", 25))
    print(f"[info] phenotypes K={K}")

    encoder_cfg = EncoderConfig(
        d=int(getattr(CFG, "d", 256)),
        structured_seq_len=int(getattr(CFG, "structured_seq_len", 256)),
        structured_n_feats=int(getattr(CFG, "structured_n_feats", 61)),
        structured_layers=int(getattr(CFG, "structured_layers", 2)),
        structured_heads=int(getattr(CFG, "structured_heads", 8)),
        structured_pool=str(getattr(CFG, "structured_pool", "mean")),  
        text_model_name=str(getattr(CFG, "text_model_name", "emilyalsentzer/Bio_ClinicalBERT")),
        bert_chunk_bs=int(getattr(CFG, "bert_chunk_bs", 8)),
        note_agg=str(getattr(CFG, "note_agg", "mean")).lower(),
        vision_backbone=str(getattr(CFG, "image_model_name", "resnet34")),
        vision_pretrained=True,
    )

    model = LateFusionModel(
        encoder_cfg=encoder_cfg,
        num_labels=K,
        dropout=float(args.dropout),
        head_hidden_dim=int(args.head_hidden_dim),
        head_layers=int(args.head_layers),
        add_presence_flags=False,
        freeze_encoders=bool(args.freeze_encoders),
    ).to(device)

    first_batch = next(iter(train_loader))
    first_batch = to_device(first_batch, device)
    model.init_head_from_batch(first_batch, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # pos_weight
    pos_weight = None
    if args.pos_weight_auto:
        pos_weight = compute_pos_weight_per_label(train_loader, device=device, K=K)
        print("[info] pos_weight (first 10):", pos_weight.detach().cpu().numpy()[:10].round(3).tolist())

    # tracking
    best_thr = np.full(K, 0.5, dtype=np.float32)
    best_val_auroc = -1.0
    best_state = None

    # training
    for epoch in range(1, int(args.epochs) + 1):
        print(f"\n===== epoch {epoch}/{args.epochs} =====")
        tr = run_one_epoch_train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            grad_clip=float(args.grad_clip),
            log_every=int(args.log_every),
            use_bf16=use_bf16,
        )

        va = run_eval_pheno(
            model=model,
            loader=val_loader,
            device=device,
            thr=(best_thr if epoch > 1 else None),
            use_bf16=use_bf16,
        )
        m = va["metrics"]

        print(
            f"[epoch {epoch}] train_loss={tr['loss']:.4f} | val_loss={va['loss']:.4f} | "
            f"AUROC={float(m['AUROC']):.4f} AUPRC={float(m['AUPRC']):.4f} "
            f"F1={float(m['F1']):.4f} R={float(m['Recall']):.4f} P={float(m['Precision']):.4f}"
        )

        best_thr = find_best_thresholds(va["y_true"], va["p"], n_steps=50).astype(np.float32)
        if float(m["AUROC"]) > best_val_auroc:
            best_val_auroc = float(m["AUROC"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[best] new best val AUROC={best_val_auroc:.4f}")

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    te = run_eval_pheno(model=model, loader=test_loader, device=device, thr=best_thr, use_bf16=use_bf16)
    mt = te["metrics"]
    print(
        f"\n[final test] loss={te['loss']:.4f} | "
        f"AUROC={float(mt['AUROC']):.4f} AUPRC={float(mt['AUPRC']):.4f} "
        f"F1={float(mt['F1']):.4f} R={float(mt['Recall']):.4f} P={float(mt['Precision']):.4f}"
    )

    # save
    ensure_dir(os.path.dirname(args.save_path) or ".")
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "best_val_auroc": best_val_auroc,
            "best_thr": best_thr,
        },
        args.save_path,
    )
    print(f"[saved] {args.save_path}")


if __name__ == "__main__":
    main()
