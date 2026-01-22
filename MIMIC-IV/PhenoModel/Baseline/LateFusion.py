from __future__ import annotations
import os as _os
_os.environ.setdefault("HF_HOME", _os.path.expanduser("~/.cache/huggingface"))
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import os, json, argparse, random
import numpy as np
import pandas as pd
from dataclasses import asdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp as torch_amp
from transformers import AutoTokenizer

from env_config import CFG, DEVICE, load_cfg, ensure_dir, apply_cli_overrides, get_pheno_name

from encoders import EncoderConfig, build_encoders
from encoders import encode_unimodal_pooled  # returns {"L": zL, "N": zN, "I": zI}
from main import (
    ICUStayDataset,
    collate_fn_factory,
    seed_worker,
    epoch_metrics,
    find_best_thresholds,
    build_image_transform,
)

class LateFusionHead(nn.Module):
    """
    Late fusion = fuse only pooled embeddings at the very end.
    """
    def __init__(self, dL: int, dN: int, dI: int, num_labels: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        d_in = int(dL + dN + dI)
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
        z = torch.cat([zL, zN, zI], dim=-1)
        return self.net(z)


def parse_args():
    ap = argparse.ArgumentParser("Late-fusion phenotype prediction")
    ap.add_argument("--task", type=str, default=getattr(CFG, "task_name", "pheno"))
    ap.add_argument("--data_root", type=str, default=getattr(CFG, "data_root", "./data"))
    ap.add_argument("--ckpt_root", type=str, default=getattr(CFG, "ckpt_root", "./ckpts"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=getattr(CFG, "batch_size", 16))
    ap.add_argument("--lr", type=float, default=getattr(CFG, "lr", 2e-4))
    ap.add_argument("--weight_decay", type=float, default=getattr(CFG, "weight_decay", 1e-4))
    ap.add_argument("--num_workers", type=int, default=getattr(CFG, "num_workers", 4))
    ap.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "off"])
    ap.add_argument("--finetune_text", action="store_true")
    ap.add_argument("--encoder_warmup_epochs", type=int, default=int(getattr(CFG, "encoder_warmup_epochs", 2)))
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--resume", type=str, default="")
    return ap.parse_args()


def grads_are_finite(param_list):
    for p in param_list:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


@torch.no_grad()
def eval_epoch_late_fusion(
    behrt, bbert, imgenc,
    head: nn.Module,
    loader,
    amp_ctx_enc,
    loss_fn,
    thr: Optional[np.ndarray] = None,
):
    behrt.eval()
    imgenc.eval()
    if getattr(bbert, "bert", None) is not None:
        bbert.bert.eval()
    head.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0
    num_samples = 0

    all_y = []
    all_p = []

    for xL, mL, notes, imgs, y, dbg in loader:
        xL = xL.to(DEVICE, non_blocking=True)
        mL = mL.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with amp_ctx_enc:
            pooled = encode_unimodal_pooled(
                behrt=behrt,
                bbert=bbert,
                imgenc=imgenc,
                xL=xL,
                notes_batch=notes,
                imgs=imgs,
                mL=mL,
            )
            zL, zN, zI = pooled["L"], pooled["N"], pooled["I"]

            logits = head(zL.float(), zN.float(), zI.float())
            loss = loss_fn(logits, y.float())

        total_loss += float(loss.item()) * y.size(0)
        probs = torch.sigmoid(logits)

        if thr is None:
            pred = (probs >= 0.5).float()
        else:
            thr_t = torch.tensor(thr, device=probs.device, dtype=probs.dtype).view(1, -1)
            pred = (probs >= thr_t).float()

        total_correct += (pred == y.float()).sum().item()
        total += y.numel()
        num_samples += y.size(0)

        all_y.append(y.detach().cpu())
        all_p.append(probs.detach().cpu())

    avg_loss = total_loss / max(1, num_samples)
    avg_acc = total_correct / max(1, total)

    y_true = torch.cat(all_y, dim=0).numpy()
    p = torch.cat(all_p, dim=0).numpy()
    return avg_loss, avg_acc, y_true, p


def save_checkpoint(path: str, state: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def main():
    import env_config as E
    E.load_cfg()
    args = parse_args()
    apply_cli_overrides(args)

    global CFG, DEVICE
    CFG = E.CFG
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("[forced] DEVICE =", DEVICE)
    print("[env_config] CFG:", json.dumps(asdict(CFG), indent=2))

    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(CFG.text_model_name, local_files_only=True)

    use_cuda = (str(DEVICE).startswith("cuda") and torch.cuda.is_available())
    precision = str(args.precision).lower()
    use_amp = use_cuda and (precision != "off")
    if use_amp:
        if precision == "fp16":
            amp_ctx_enc = torch_amp.autocast(device_type="cuda", dtype=torch.float16)
        elif precision == "bf16":
            amp_ctx_enc = torch_amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            amp_ctx_enc = torch_amp.autocast(device_type="cuda")
    else:
        amp_ctx_enc = nullcontext()

    from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=(use_amp and precision in {"auto", "fp16"}))
    print(f"[amp] use_amp={use_amp} precision={precision} scaler_enabled={scaler.is_enabled()}")

    train_ds = ICUStayDataset(args.data_root, split="train")
    val_ds   = ICUStayDataset(args.data_root, split="val")
    test_ds  = ICUStayDataset(args.data_root, split="test")

    print("\n[DATA CHECK] data_root =", args.data_root)
    expected = [
        "splits.json",
        "xehr_haru17_2h_76.parquet",
        "notes_fullstay_radiology_TEXTCHUNKS_11230.parquet",
        "images.parquet",
        "labels_pheno.parquet",
    ]
    for fn in expected:
        fp = os.path.join(args.data_root, fn)
        print(f"[DATA CHECK] exists={os.path.isfile(fp)} -> {fp}")

    print("\n[DATA CHECK] #ids train/val/test:", len(train_ds.ids), len(val_ds.ids), len(test_ds.ids))
    print("[DATA CHECK] num_labels:", train_ds.num_labels)
    print("[DATA CHECK] label_cols (first 10):", train_ds.label_cols[:10])
    print("[DATA CHECK] label_cols count:", len(train_ds.label_cols))

    inter_tv = set(train_ds.ids).intersection(set(val_ds.ids))
    inter_tt = set(train_ds.ids).intersection(set(test_ds.ids))
    inter_vt = set(val_ds.ids).intersection(set(test_ds.ids))
    print("\n[DATA CHECK] leakage intersections sizes:")
    print("  train∩val:", len(inter_tv))
    print("  train∩test:", len(inter_tt))
    print("  val∩test:", len(inter_vt))

    # Peek a few samples through __getitem__ (this checks your internal joins)
    print("\n[DATA CHECK] sampling 3 items from train_ds")
    for i in [0, 1, 2]:
        sample = train_ds[i]
        # ICUStayDataset likely returns something like (xL, mL, notes, imgs, y, dbg) OR a dict
        if isinstance(sample, dict):
            keys = list(sample.keys())
            print(f"  sample[{i}] dict keys:", keys)
            if "y" in sample:
                print("    y shape:", np.array(sample["y"]).shape, "sum:", float(np.array(sample["y"]).sum()))
        else:
            try:
                xL, mL, notes, imgs, y, dbg = sample
                print(f"  sample[{i}] xL={getattr(xL,'shape',None)} mL={getattr(mL,'shape',None)} "
                      f"notes_type={type(notes)} imgs={getattr(imgs,'shape',None)} y={getattr(y,'shape',None)} "
                      f"dbg_keys={list(dbg.keys()) if isinstance(dbg, dict) else type(dbg)}")
                if hasattr(y, "sum"):
                    print("    y.sum:", float(y.sum().item()) if torch.is_tensor(y) else float(np.sum(y)))
                if isinstance(dbg, dict):
                    for k in ["stay_id", "hadm_id", "subject_id"]:
                        if k in dbg:
                            print(f"    dbg[{k}] =", dbg[k])
            except Exception as e:
                print(f"  sample[{i}] could not unpack sample format. type={type(sample)} err={e}")

    print("\n[DATA CHECK] one batch from train_loader (after loaders are created)")

    num_labels = int(train_ds.num_labels)
    print(f"[labels] num_labels={num_labels} (EXPECT 25 for your setting)")
    if num_labels != 25:
        print("[WARN] num_labels != 25. Late fusion will still be correct, but you should verify labels_pheno.parquet columns.")

    tri_ids = set(train_ds.ids)
    train_label_df = (
        train_ds.labels[train_ds.labels["stay_id"].isin(tri_ids)]
        .loc[:, ["stay_id"] + train_ds.label_cols]
        .drop_duplicates(subset=["stay_id"], keep="first")
    )
    N_train = len(train_label_df)
    pos_counts = train_label_df[train_ds.label_cols].sum(axis=0).values
    neg_counts = N_train - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    max_pw = float(getattr(CFG, "pos_weight_max", 20.0))
    pos_weight = np.clip(pos_weight, 1.0, max_pw)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_tensor)

    collate_train = collate_fn_factory(tidx=0, img_tfms=build_image_transform("train"))
    collate_eval  = collate_fn_factory(tidx=0, img_tfms=build_image_transform("val"))

    pin = use_cuda
    g_train = torch.Generator().manual_seed(int(CFG.seed) + 123)
    g_eval  = torch.Generator().manual_seed(int(CFG.seed) + 456)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_train, drop_last=False,
        worker_init_fn=seed_worker, generator=g_train,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    batch = next(iter(train_loader))
    xL, mL, notes, imgs, y, dbg = batch
    print("[BATCH CHECK] xL:", xL.shape, xL.dtype)
    print("[BATCH CHECK] mL:", mL.shape, mL.dtype, "mask_sum:", int(mL.sum().item()))
    print("[BATCH CHECK] imgs:", imgs.shape, imgs.dtype)
    print("[BATCH CHECK] y:", y.shape, y.dtype, "pos_per_label(first 10):", y.float().mean(0)[:10].tolist())
    print("[BATCH CHECK] notes type:", type(notes))
    if isinstance(dbg, dict):
        print("[BATCH CHECK] dbg keys:", list(dbg.keys()))

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_eval,
        worker_init_fn=seed_worker, generator=g_eval,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate_eval,
        worker_init_fn=seed_worker, generator=g_eval,
        persistent_workers=(args.num_workers > 0),
    )

    enc_cfg = EncoderConfig(
        d=getattr(CFG, "d", 256),
        structured_seq_len=getattr(CFG, "structured_seq_len", 256),
        structured_n_feats=getattr(CFG, "structured_n_feats", 61),
        structured_layers=getattr(CFG, "structured_layers", 2),
        structured_heads=getattr(CFG, "structured_heads", 8),

        text_model_name=getattr(CFG, "text_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        bert_chunk_bs=getattr(CFG, "bert_chunk_bs", 8),
        note_agg=str(getattr(CFG, "note_agg", "mean")).lower(),

        vision_backbone=getattr(CFG, "image_model_name", "resnet34"),
        vision_pretrained=True,
    )
    behrt, bbert, imgenc = build_encoders(enc_cfg, device=DEVICE)

    if not getattr(CFG, "finetune_text", False) and getattr(bbert, "bert", None) is not None and not args.finetune_text:
        for p in bbert.bert.parameters():
            p.requires_grad = False
        bbert.bert.eval()
        print("[encoders] Bio_ClinicalBERT frozen (feature extractor mode)")
    else:
        print("[encoders] Bio_ClinicalBERT finetuning enabled")

    dL = int(behrt.out_dim)
    dN = int(bbert.out_dim)
    # your ImageEncoder likely has proj.out_features as embedding dim
    dI = int(imgenc.proj.out_features) if hasattr(imgenc, "proj") else int(getattr(imgenc, "out_dim", dL))

    head = LateFusionHead(dL=dL, dN=dN, dI=dI, num_labels=num_labels,
                          hidden=int(getattr(CFG, "latefusion_hidden", 256)),
                          dropout=float(getattr(CFG, "dropout", 0.2))).to(DEVICE)

    encoder_warmup_epochs = int(getattr(args, "encoder_warmup_epochs", 2))
    enc_params = [p for p in behrt.parameters() if p.requires_grad] + [p for p in imgenc.parameters() if p.requires_grad]
    if getattr(CFG, "finetune_text", False) or args.finetune_text:
        enc_params += [p for p in bbert.parameters() if p.requires_grad]

    head_params = [p for p in head.parameters() if p.requires_grad]
    params = enc_params + head_params

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr, "weight_decay": args.weight_decay, "name": "enc"},
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay, "name": "head"},
        ]
    )

    ckpt_dir = os.path.join(args.ckpt_root, "pheno_latefusion")
    ensure_dir(ckpt_dir)

    best_val_auroc = -float("inf")
    best_thr = np.full(num_labels, 0.5, dtype=np.float32)

    # TRAIN
    for epoch in range(args.epochs):
        # warmup encoders
        enc_lr = 0.0 if epoch < encoder_warmup_epochs else args.lr
        optimizer.param_groups[0]["lr"] = enc_lr
        optimizer.param_groups[1]["lr"] = args.lr

        behrt.train()
        imgenc.train()
        head.train()
        if getattr(bbert, "bert", None) is not None:
            if getattr(CFG, "finetune_text", False) or args.finetune_text:
                bbert.bert.train()
            else:
                bbert.bert.eval()

        total_loss = 0.0
        total = 0
        num_samples = 0

        for step, (xL, mL, notes, imgs, y, dbg) in enumerate(train_loader):
            xL = xL.to(DEVICE, non_blocking=True)
            mL = mL.to(DEVICE, non_blocking=True)
            imgs = imgs.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx_enc:
                pooled = encode_unimodal_pooled(
                    behrt=behrt, bbert=bbert, imgenc=imgenc,
                    xL=xL, notes_batch=notes, imgs=imgs, mL=mL,
                )
                zL, zN, zI = pooled["L"], pooled["N"], pooled["I"]
                logits = head(zL.float(), zN.float(), zI.float())
                loss = bce(logits, y.float())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(getattr(CFG, "grad_clip_norm", 1.0)))
                if not grads_are_finite(params):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(getattr(CFG, "grad_clip_norm", 1.0)))
                if not grads_are_finite(params):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            total += y.numel()
            num_samples += y.size(0)

            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                print(f"[epoch {epoch+1} step {step+1}] train_loss={total_loss/max(1,num_samples):.4f}")

        print(f"[epoch {epoch+1}] TRAIN loss={total_loss/max(1,num_samples):.4f}")

        val_loss, val_acc, y_true, p = eval_epoch_late_fusion(
            behrt, bbert, imgenc, head,
            val_loader, amp_ctx_enc, bce,
            thr=(best_thr if epoch > 0 else None),
        )
        y_pred = (p >= 0.5).astype(float)
        m = epoch_metrics(y_true, p, y_pred)
        print(f"[epoch {epoch+1}] VAL loss={val_loss:.4f} acc@0.5={val_acc:.4f} AUROC_macro={m['AUROC']:.4f} AUPRC_macro={m['AUPRC']:.4f}")

        best_thr = find_best_thresholds(y_true, p, n_steps=50).astype(np.float32)

        if float(m["AUROC"]) > best_val_auroc:
            best_val_auroc = float(m["AUROC"])
            ckpt = {
                "epoch": epoch + 1,
                "behrt": behrt.state_dict(),
                "bbert": bbert.state_dict(),
                "imgenc": imgenc.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_thr": torch.from_numpy(best_thr),
                "val_auroc": best_val_auroc,
            }
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), ckpt)
            print(f"[epoch {epoch+1}] Saved BEST (AUROC={best_val_auroc:.4f})")

        # always save last
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), {
            "epoch": epoch + 1,
            "behrt": behrt.state_dict(),
            "bbert": bbert.state_dict(),
            "imgenc": imgenc.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_thr": torch.from_numpy(best_thr),
            "val_auroc": float(m["AUROC"]),
        })

    # TEST best
    print("[main] Loading BEST and evaluating on TEST...")
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        behrt.load_state_dict(ckpt["behrt"])
        bbert.load_state_dict(ckpt["bbert"])
        imgenc.load_state_dict(ckpt["imgenc"])
        head.load_state_dict(ckpt["head"])
        best_thr = ckpt["best_thr"].detach().cpu().numpy().astype(np.float32)

    test_loss, test_acc, y_true_t, p_t = eval_epoch_late_fusion(
        behrt, bbert, imgenc, head,
        test_loader, amp_ctx_enc, bce,
        thr=best_thr,
    )
    y_pred_t = (p_t >= best_thr[np.newaxis, :]).astype(float)
    mt = epoch_metrics(y_true_t, p_t, y_pred_t)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} AUROC_macro={mt['AUROC']:.4f} AUPRC_macro={mt['AUPRC']:.4f} F1_macro={mt['F1']:.4f}")

if __name__ == "__main__":
    main()
