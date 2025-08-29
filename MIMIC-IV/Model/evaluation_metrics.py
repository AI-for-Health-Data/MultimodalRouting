import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

from env_config import CFG, DEVICE, ROUTES, BLOCKS, TASKS
from encoders import EncoderConfig, build_encoders
from routing_and_heads import RouteHead, build_fusions, LearnedGateRouter
from train_step1_unimodal import ICUStayDataset, collate_fn

def _build_stack():
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

    # Route heads 
    C = len(TASKS)
    route_heads = {
        r: RouteHead(d_in=CFG.d, n_tasks=C, p_drop=CFG.dropout).to(DEVICE)
        for r in ROUTES
    }

    # Fusion modules
    fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)

    # Router
    router = LearnedGateRouter(
        routes=ROUTES,
        blocks=BLOCKS,
        d=CFG.d,
        n_tasks=C,
        hidden=1024,
        p_drop=CFG.dropout,
        use_masks=True,
        temperature=1.0,
    ).to(DEVICE)

    return behrt, bbert, imgenc, route_heads, fusion, router


def _load_checkpoints(behrt, bbert, imgenc, route_heads, fusion, router):
    # Step 1 (unimodal) + Step 2 (bimodal) + Step 3 (trimodal + router)
    ckpt1_path = os.path.join(CFG.ckpt_root, "step1_unimodal.pt")
    ckpt2_path = os.path.join(CFG.ckpt_root, "step2_bimodal.pt")
    ckpt3_path = os.path.join(CFG.ckpt_root, "step3_trimodal_router.pt")

    if os.path.exists(ckpt1_path):
        ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)
        behrt.load_state_dict(ckpt1["behrt"], strict=False)
        bbert.load_state_dict(ckpt1["bbert"], strict=False)
        imgenc.load_state_dict(ckpt1["imgenc"], strict=False)
        route_heads["L"].load_state_dict(ckpt1["L"], strict=False)
        route_heads["N"].load_state_dict(ckpt1["N"], strict=False)
        route_heads["I"].load_state_dict(ckpt1["I"], strict=False)
        print("Loaded Step 1 (unimodal) weights.")

    if os.path.exists(ckpt2_path):
        ckpt2 = torch.load(ckpt2_path, map_location=DEVICE)
        route_heads["LN"].load_state_dict(ckpt2["LN"], strict=False)
        route_heads["LI"].load_state_dict(ckpt2["LI"], strict=False)
        route_heads["NI"].load_state_dict(ckpt2["NI"], strict=False)
        print("Loaded Step 2 (bimodal) weights.")

    if os.path.exists(ckpt3_path):
        ckpt3 = torch.load(ckpt3_path, map_location=DEVICE)
        if "router" in ckpt3:
            router.load_state_dict(ckpt3["router"], strict=False)
        if "LNI" in ckpt3:
            route_heads["LNI"].load_state_dict(ckpt3["LNI"], strict=False)
        if "fusion_LNI" in ckpt3 and "LNI" in fusion:
            try:
                fusion["LNI"].load_state_dict(ckpt3["fusion_LNI"], strict=False)
            except Exception:
                pass
        print("Loaded Step 3 (trimodal + router) weights.")

def compute_eddi_final(batch_groups, ylogits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if getattr(CFG, "lambda_fair", 0.0) <= 0.0:
        return torch.tensor(0.0, device=y.device)

    probs = torch.sigmoid(ylogits)             
    err   = (probs - y).abs().mean(dim=1)      

    penalty = 0.0
    ncount  = 0
    for key in CFG.sensitive_keys:
        group_to_ix = {}
        for i, meta in enumerate(batch_groups):
            g = str(meta.get(key, "UNK"))
            group_to_ix.setdefault(g, []).append(i)

        all_mean = err.mean()
        accum = 0.0
        total = 0
        for _, ix in group_to_ix.items():
            ix_t = torch.tensor(ix, device=y.device, dtype=torch.long)
            gmean = err[ix_t].mean() if len(ix_t) > 0 else all_mean
            accum = accum + (gmean - all_mean).abs() * len(ix_t)
            total += len(ix_t)

        if total > 0:
            penalty = penalty + accum / total
            ncount += 1

    if ncount == 0:
        return torch.tensor(0.0, device=y.device)
    return penalty / ncount

def build_masks(xL: torch.Tensor, notes_list, imgs: torch.Tensor) -> dict:
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)

    # N present if any non-empty note
    mN_list = []
    for notes in notes_list:
        present = 1.0 if (isinstance(notes, list) and any((isinstance(t, str) and len(t.strip()) > 0) for t in notes)) else 0.0
        mN_list.append(present)
    mN = torch.tensor(mN_list, device=xL.device, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        mI_vals = (imgs.abs().flatten(1).sum(dim=1) > 0).float()
    mI = mI_vals.to(xL.device).unsqueeze(1)

    return {"L": mL, "N": mN, "I": mI}


def route_logits_from_embeddings(z: dict, route_heads: dict, fusion: dict) -> dict:
    """Compute per-route logits from unimodal embeddings and fusion blocks."""
    out = {
        "L": route_heads["L"](z["L"]),
        "N": route_heads["N"](z["N"]),
        "I": route_heads["I"](z["I"]),
    }
    zLN = fusion["LN"](z["L"], z["N"])
    zLI = fusion["LI"](z["L"], z["I"])
    zNI = fusion["NI"](z["N"], z["I"])
    out["LN"]  = route_heads["LN"](zLN)
    out["LI"]  = route_heads["LI"](zLI)
    out["NI"]  = route_heads["NI"](zNI)
    zLNI = fusion["LNI"](z["L"], z["N"], z["I"])
    out["LNI"] = route_heads["LNI"](zLNI)
    return out

def evaluate():
    ROOT = os.path.join(CFG.data_root, "MIMIC-IV")
    test_ds = ICUStayDataset(ROOT, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    behrt, bbert, imgenc, route_heads, fusion, router = _build_stack()
    _load_checkpoints(behrt, bbert, imgenc, route_heads, fusion, router)

    behrt.eval(); bbert.eval(); imgenc.eval()
    for r in ROUTES: route_heads[r].eval()
    for k in ["LN","LI","NI","LNI"]: fusion[k].eval()
    router.eval()

    C = len(TASKS)
    y_all, p_all = [], []

    sum_route_w = torch.zeros(C, 7, device=DEVICE)
    sum_block_w = torch.zeros(C, 3, device=DEVICE)
    total_examples = 0

    eddi_sum = 0.0
    eddi_batches = 0

    with torch.no_grad():
        for xL, notes_list, imgs, y, sens in tqdm(test_loader, desc="Evaluating", dynamic_ncols=True):
            xL   = xL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y    = y.to(DEVICE)
            B    = y.size(0)

            # embeddings
            zL = behrt(xL)
            zN = bbert(notes_list)
            zI = imgenc(imgs)
            z  = {"L": zL, "N": zN, "I": zI}

            # per-route logits
            route_logits = route_logits_from_embeddings(z, route_heads, fusion)

            # masks
            masks = build_masks(xL, notes_list, imgs)

            # router -> final logits + weights
            ylogits, route_w, block_w, _ = router(z, route_logits, masks=masks)

            # predictions
            probs = torch.sigmoid(ylogits)
            y_all.append(y.detach().cpu().numpy())
            p_all.append(probs.detach().cpu().numpy())

            # weights
            sum_route_w += route_w.sum(dim=0)
            sum_block_w += block_w.sum(dim=0)
            total_examples += B

            # fairness on FINAL predictions (matches training)
            eddi = compute_eddi_final(sens, ylogits, y)
            if eddi.isfinite():
                eddi_sum += float(eddi)
                eddi_batches += 1

    Y = np.concatenate(y_all, axis=0) if y_all else np.zeros((0, C))
    P = np.concatenate(p_all, axis=0) if p_all else np.zeros((0, C))

    metrics = {}
    for i, t in enumerate(TASKS):
        # AUROC
        try:
            metrics[f"{t}_auroc"] = float(roc_auc_score(Y[:, i], P[:, i]))
        except Exception:
            metrics[f"{t}_auroc"] = float("nan")
        # Average Precision
        try:
            metrics[f"{t}_ap"] = float(average_precision_score(Y[:, i], P[:, i]))
        except Exception:
            metrics[f"{t}_ap"] = float("nan")

    # Mean weights (per task & overall)
    if total_examples > 0:
        mean_route_w_per_task = (sum_route_w / total_examples).detach().cpu().numpy()
        mean_block_w_per_task = (sum_block_w / total_examples).detach().cpu().numpy()

        metrics["route_weights_per_task"] = {
            TASKS[t]: {ROUTES[r]: float(mean_route_w_per_task[t, r]) for r in range(len(ROUTES))}
            for t in range(C)
        }
        metrics["block_weights_per_task"] = {
            TASKS[t]: {
                "uni": float(mean_block_w_per_task[t, 0]),
                "bi":  float(mean_block_w_per_task[t, 1]),
                "tri": float(mean_block_w_per_task[t, 2]),
            }
            for t in range(C)
        }
        metrics["mean_route_weights_overall"] = {
            ROUTES[r]: float(mean_route_w_per_task[:, r].mean()) for r in range(len(ROUTES))
        }
        metrics["mean_block_weights_overall"] = {
            "uni": float(mean_block_w_per_task[:, 0].mean()),
            "bi":  float(mean_block_w_per_task[:, 1].mean()),
            "tri": float(mean_block_w_per_task[:, 2].mean()),
        }
    else:
        metrics["route_weights_per_task"] = {}
        metrics["block_weights_per_task"] = {}
        metrics["mean_route_weights_overall"] = {r: 0.0 for r in ROUTES}
        metrics["mean_block_weights_overall"] = {"uni": 0.0, "bi": 0.0, "tri": 0.0}

    metrics["eddi_mean"] = (eddi_sum / eddi_batches) if eddi_batches > 0 else float("nan")
    return metrics


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))
