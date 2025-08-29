# Apply trained unimodal (BEHRT / BioClinicalBERT / ResNet34),
# then trained bimodal fusions (LN/LI/NI), then trimodal (LNI),
# and save ALL route embeddings to a single NPZ per split.

import os
import argparse
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from env_config import CFG, DEVICE
from encoders import EncoderConfig, build_encoders
from routing_and_heads import build_fusions
from train_step1_unimodal import ICUStayDataset, collate_fn

@torch.no_grad()
def _batch_embeddings(
    behrt, bbert, imgenc, fusion: Dict[str, torch.nn.Module],
    xL: torch.Tensor, notes_list: List[List[str]], imgs: torch.Tensor
) -> Dict[str, torch.Tensor]:
    # Unimodal
    zL = behrt(xL)
    zN = bbert(notes_list)
    zI = imgenc(imgs)
    # Bimodal
    zLN = fusion["LN"](zL, zN)
    zLI = fusion["LI"](zL, zI)
    zNI = fusion["NI"](zN, zI)
    # Trimodal
    zLNI = fusion["LNI"](zL, zN, zI)
    return {"L": zL, "N": zN, "I": zI, "LN": zLN, "LI": zLI, "NI": zNI, "LNI": zLNI}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--limit_batches", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    # Build encoders (unimodal)
    behrt, bbert, imgenc = build_encoders(EncoderConfig(
        d=CFG.d,
        dropout=CFG.dropout,
        structured_seq_len=CFG.structured_seq_len,
        structured_n_feats=CFG.structured_n_feats,
        text_model_name=CFG.text_model_name,
        text_max_len=CFG.max_text_len,
        note_agg="mean",
        max_notes_concat=8,
        img_agg="last",
    ))

    # Build fusions (LN/LI/NI/LNI)
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)

    # Load trained weights (SMRO)
    ckpt1 = torch.load(os.path.join(CFG.ckpt_root, "step1_unimodal.pt"), map_location=DEVICE)
    behrt.load_state_dict(ckpt1["behrt"], strict=False)
    bbert.load_state_dict(ckpt1["bbert"], strict=False)
    imgenc.load_state_dict(ckpt1["imgenc"], strict=False)

    ckpt2_path = os.path.join(CFG.ckpt_root, "step2_bimodal.pt")
    if os.path.exists(ckpt2_path):
        ckpt2 = torch.load(ckpt2_path, map_location=DEVICE)
        fusion["LN"].load_state_dict(ckpt2["fusion_LN"], strict=False)
        fusion["LI"].load_state_dict(ckpt2["fusion_LI"], strict=False)
        fusion["NI"].load_state_dict(ckpt2["fusion_NI"], strict=False)

    ckpt3_path = os.path.join(CFG.ckpt_root, "step3_trimodal_router.pt")
    if os.path.exists(ckpt3_path):
        ckpt3 = torch.load(ckpt3_path, map_location=DEVICE)
        if "fusion_LNI" in ckpt3:
            fusion["LNI"].load_state_dict(ckpt3["fusion_LNI"], strict=False)

    behrt.eval(); bbert.eval(); imgenc.eval()
    for k in ["LN","LI","NI","LNI"]:
        fusion[k].eval()

    # Data
    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    ds = ICUStayDataset(ROOT, split=args.split)
    loader = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    # Accumulate
    embs = {r: [] for r in ["L","N","I","LN","LI","NI","LNI"]}
    stay_ids: List[int] = []

    with torch.no_grad():
        for b_idx, (xL, notes_list, imgs, y, sens) in enumerate(loader):
            xL   = xL.to(DEVICE)
            imgs = imgs.to(DEVICE)

            Z = _batch_embeddings(behrt, bbert, imgenc, fusion, xL, notes_list, imgs)
            for r in embs:
                embs[r].append(Z[r].cpu().numpy())

            # keep dataset order
            start = b_idx * CFG.batch_size
            end = min((b_idx + 1) * CFG.batch_size, len(ds))
            stay_ids.extend(ds.ids[start:end])

            if args.limit_batches is not None and (b_idx + 1) >= args.limit_batches:
                break

    for r in embs:
        embs[r] = np.vstack(embs[r]) if embs[r] else np.zeros((0, CFG.d), dtype=np.float32)

    out_dir = args.out_dir or os.path.join(CFG.ckpt_root, f"embeddings_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "route_embeddings.npz")
    np.savez(out_path, stay_ids=np.asarray(stay_ids, dtype=np.int64), **embs)
    print(f"[OK] Saved: {out_path}")
    print("Keys:", ["stay_ids"] + list(embs.keys()), "| shapes:", {k: v.shape for k, v in embs.items()})

if __name__ == "__main__":
    main()
