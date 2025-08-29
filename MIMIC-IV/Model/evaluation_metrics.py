# Computes per-task AUROC / AP, optional fairness (EDDI), and exports mean route/block weights.

# This evaluation runs the encoders to get z_L, z_N, z_I, builds all 7 route logits
# using the *fusion modules* for LN/LI/NI/LNI, runs the learned gate to get per-task
# route/block weights, and computes metrics.

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

# Ensure fusion modules exist (LN, LI, NI, LNI)
def _ensure_fusions_eval():
    """
    Makes sure a global `fusion` dict with keys {"LN","LI","NI","LNI"} exists on DEVICE.
    If not present, builds lightweight fallbacks that were used in Steps 2/3.
    Also tries to load LNI fusion weights from step3 checkpoint if available.
    """
    global fusion
    if "fusion" in globals() and isinstance(fusion, dict):
        needed = {"LN","LI","NI","LNI"}
        if needed.issubset(set(fusion.keys())):
            for k in needed:
                fusion[k] = fusion[k].to(DEVICE)
            return

    class _PairwiseFusion(nn.Module):
        def __init__(self, d: int, hidden: int = None, p_drop: float = 0.1):
            super().__init__()
            hidden = hidden or (2 * d)
            self.net = nn.Sequential(
                nn.LayerNorm(2 * d),
                nn.Linear(2 * d, hidden),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden, d),
            )
            self.gate = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.Sigmoid(),
            )
        def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
            h = torch.cat([za, zb], dim=-1) 
            f = self.net(h)                  
            base = (za + zb) / 2.0
            g = self.gate(base)
            return g * f + (1.0 - g) * base

    class _TriFusion(nn.Module):
        def __init__(self, d: int, hidden: int = None, p_drop: float = 0.1):
            super().__init__()
            hidden = hidden or (3 * d)
            self.net = nn.Sequential(
                nn.LayerNorm(3 * d),
                nn.Linear(3 * d, hidden),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden, d),
            )
            self.gate = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.Sigmoid(),
            )
        def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
            h = torch.cat([zL, zN, zI], dim=-1)  
            f = self.net(h)                     
            base = (zL + zN + zI) / 3.0
            g = self.gate(base)
            return g * f + (1.0 - g) * base

    fusion = {
        "LN":  _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE),
        "LI":  _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE),
        "NI":  _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE),
        "LNI": _TriFusion     (d=CFG.d, p_drop=CFG.dropout).to(DEVICE),
    }

    step3_path = os.path.join(CFG.ckpt_root, "step3_trimodal_router.pt")
    if os.path.exists(step3_path):
        try:
            ckpt3 = torch.load(step3_path, map_location=DEVICE)
            if "fusion_LNI" in ckpt3:
                fusion["LNI"].load_state_dict(ckpt3["fusion_LNI"])
                print("Loaded fusion['LNI'] weights from step3_trimodal_router.pt")
        except Exception:
            pass

_ensure_fusions_eval()

ROOT = os.path.join(CFG.data_root, "MIMIC-IV")  # or INSPECT
test_ds = ICUStayDataset(ROOT, split="test")
test_loader = DataLoader(
    test_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    collate_fn=collate_fn,
    pin_memory=True
)

def compute_eddi(batch_groups: list[dict], route_logits: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    if getattr(CFG, "lambda_fair", 0.0) <= 0.0:
        return torch.tensor(0.0, device=y.device)
    # per-route per-example mean absolute error across tasks
    err = {r: torch.sigmoid(route_logits[r]).sub(y).abs().mean(dim=1) for r in route_logits}  
    penalty = 0.0
    ncount = 0
    for key in CFG.sensitive_keys:
        group_to_ix = {}
        for i, meta in enumerate(batch_groups):
            g = str(meta.get(key, "UNK"))
            group_to_ix.setdefault(g, []).append(i)
        for r in route_logits:
            all_mean = err[r].mean()
            accum = 0.0
            total = 0
            for g, ix in group_to_ix.items():
                ix_t = torch.tensor(ix, device=y.device, dtype=torch.long)
                gmean = err[r][ix_t].mean() if len(ix_t) > 0 else all_mean
                accum = accum + (gmean - all_mean).abs() * len(ix_t)
                total += len(ix_t)
            if total > 0:
                penalty = penalty + accum / total
                ncount += 1
    if ncount == 0:
        return torch.tensor(0.0, device=y.device)
    return penalty / ncount

def build_masks(xL: torch.Tensor, notes_list: list[list[str]], imgs: torch.Tensor) -> dict[str, torch.Tensor]:
    # L: assume present
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)

    # N: present if at least one note string exists and is non-empty
    mN_list = []
    for notes in notes_list:
        present = 1.0 if (isinstance(notes, list) and any((isinstance(t, str) and len(t.strip()) > 0) for t in notes)) else 0.0
        mN_list.append(present)
    mN = torch.tensor(mN_list, device=xL.device, dtype=torch.float32).unsqueeze(1)

    # I: present if the image is not the all-zeros placeholder
    with torch.no_grad():
        mI_vals = (imgs.abs().flatten(1).sum(dim=1) > 0).float()
    mI = mI_vals.to(xL.device).unsqueeze(1)

    return {"L": mL, "N": mN, "I": mI}

def route_logits_from_embeddings(z: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Given unimodal embeddings z = {"L","N","I"} each [B,d],
    produce per-route logits using route heads:
      - L,N,I directly from their heads
      - LN,LI,NI via pairwise fusion blocks
      - LNI via trimodal fusion block
    Returns dict r -> logits [B, C]
    """
    # Unimodal
    out = {
        "L": route_heads["L"](z["L"]),
        "N": route_heads["N"](z["N"]),
        "I": route_heads["I"](z["I"]),
    }
    # Pairwise fused embeddings
    zLN = fusion["LN"](z["L"], z["N"])
    zLI = fusion["LI"](z["L"], z["I"])
    zNI = fusion["NI"](z["N"], z["I"])
    out["LN"] = route_heads["LN"](zLN)
    out["LI"] = route_heads["LI"](zLI)
    out["NI"] = route_heads["NI"](zNI)
    # Trimodal fused embedding
    zLNI = fusion["LNI"](z["L"], z["N"], z["I"])
    out["LNI"] = route_heads["LNI"](zLNI)
    return out

# Evaluate 
def evaluate(loader: DataLoader) -> dict:
    behrt.eval(); bbert.eval(); imgenc.eval()
    for r in ROUTES: route_heads[r].eval()
    for k in ["LN","LI","NI","LNI"]:
        fusion[k].eval()
    router.eval()

    y_all = []
    p_all = []

    # aggregate mean route / block weights (per task) without storing all batches
    C = len(TASKS)
    sum_route_w = torch.zeros(C, 7, device=DEVICE)  
    sum_block_w = torch.zeros(C, 3, device=DEVICE)
    total_examples = 0

    # fairness
    eddi_sum = 0.0
    eddi_batches = 0

    with torch.no_grad():
        for xL, notes_list, imgs, y, sens in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
            xL   = xL.to(DEVICE)         
            imgs = imgs.to(DEVICE)       
            y    = y.to(DEVICE)          
            B    = y.size(0)

            # embeddings
            zL = behrt(xL)
            zN = bbert(notes_list)   
            zI = imgenc(imgs)

            z = {"L": zL, "N": zN, "I": zI}

            # logits for each route (use fusion for multi-modal routes)
            route_logits = route_logits_from_embeddings(z)  

            # masks for router
            masks = build_masks(xL, notes_list, imgs)

            # router → final logits + per-task weights
            ylogits, route_w, block_w, _ = router(z, route_logits, masks=masks)  

            # predictions
            probs = torch.sigmoid(ylogits)  
            y_all.append(y.detach().cpu().numpy())
            p_all.append(probs.detach().cpu().numpy())

            # accumulate weights
            sum_route_w += route_w.sum(dim=0)   
            sum_block_w += block_w.sum(dim=0)   
            total_examples += B

            # fairness (batch-wise)
            eddi = compute_eddi(sens, route_logits, y)
            if eddi.isfinite():
                eddi_sum += float(eddi)
                eddi_batches += 1

    # stack labels and preds
    Y = np.concatenate(y_all, axis=0) if y_all else np.zeros((0, C))
    P = np.concatenate(p_all, axis=0) if p_all else np.zeros((0, C))

    # metrics
    metrics: dict[str, float | dict] = {}
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

    # mean weights (per task & overall)
    if total_examples > 0:
        mean_route_w_per_task = (sum_route_w / total_examples).detach().cpu().numpy()  
        mean_block_w_per_task = (sum_block_w / total_examples).detach().cpu().numpy()  

        # per-task dictionaries
        metrics["route_weights_per_task"] = {
            TASKS[t]: {ROUTES[r]: float(mean_route_w_per_task[t, r]) for r in range(len(ROUTES))}
            for t in range(C)
        }
        metrics["block_weights_per_task"] = {
            TASKS[t]: {"uni": float(mean_block_w_per_task[t, 0]),
                       "bi":  float(mean_block_w_per_task[t, 1]),
                       "tri": float(mean_block_w_per_task[t, 2])}
            for t in range(C)
        }

        # overall means (averaged across tasks)
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

    # fairness (EDDI)
    if eddi_batches > 0:
        metrics["eddi_mean"] = eddi_sum / eddi_batches
    else:
        metrics["eddi_mean"] = float("nan")

    return metrics

metrics = evaluate(test_loader)
print(json.dumps(metrics, indent=2))
