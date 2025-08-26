# Freeze unimodal & bimodal heads, train the trimodal head (LNI) *and* the learned-gate router.
# The router blends per-route and per-block contributions with per-task weights computed from (z_L, z_N, z_I).

import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

if "train_loader" not in globals() or "val_loader" not in globals():
    try:
        ROOT = os.path.join(CFG.data_root, "MIMIC-IV")  # or INSPECT
        train_ds = ICUStayDataset(ROOT, split="train")
        val_ds   = ICUStayDataset(ROOT, split="val")

        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_ds,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        print("Re-created train/val loaders for Step 3.")
    except NameError as e:
        raise RuntimeError("Missing dataset/collate definitions. Please run Step 1 first.") from e

# Load Step 1 & Step 2 weights (encoders + heads)
ckpt1 = torch.load(os.path.join(CFG.ckpt_root, "step1_unimodal.pt"), map_location=DEVICE)
ckpt2 = torch.load(os.path.join(CFG.ckpt_root, "step2_bimodal.pt"), map_location=DEVICE)

behrt.load_state_dict(ckpt1["behrt"])
bbert.load_state_dict(ckpt1["bbert"])
imgenc.load_state_dict(ckpt1["imgenc"])

route_heads["L"].load_state_dict(ckpt1["L"])
route_heads["N"].load_state_dict(ckpt1["N"])
route_heads["I"].load_state_dict(ckpt1["I"])

route_heads["LN"].load_state_dict(ckpt2["LN"])
route_heads["LI"].load_state_dict(ckpt2["LI"])
route_heads["NI"].load_state_dict(ckpt2["NI"])

print("Loaded Step 1 & Step 2 weights.")

def _ensure_fusions_step3():
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
    print("Built fallback fusion modules for LN, LI, NI, LNI.")

_ensure_fusions_step3()

# -----------------------------
# Freeze unimodal + bimodal heads; train only LNI head (and the router)
# Encoders: kept frozen by default; toggle FINETUNE_ENCODERS=True to fine-tune.
# -----------------------------
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

for r in ["L", "N", "I", "LN", "LI", "NI"]:
    set_requires_grad(route_heads[r], False)

FINETUNE_ENCODERS       = False  
TRAIN_TRIMODAL_FUSION   = True   

set_requires_grad(behrt,  FINETUNE_ENCODERS)
set_requires_grad(bbert,  FINETUNE_ENCODERS)
set_requires_grad(imgenc, FINETUNE_ENCODERS)
set_requires_grad(fusion["LNI"], TRAIN_TRIMODAL_FUSION)

behrt.train(FINETUNE_ENCODERS); bbert.train(FINETUNE_ENCODERS); imgenc.train(FINETUNE_ENCODERS)
for r in ["L","N","I","LN","LI","NI"]:
    route_heads[r].eval()  # frozen heads
fusion["LN"].eval(); fusion["LI"].eval(); fusion["NI"].eval()
fusion["LNI"].train(TRAIN_TRIMODAL_FUSION)

route_heads["LNI"].train()
router.train()

# Optimizer: train LNI head + router (+ encoders if FINETUNE_ENCODERS, + LNI fusion if TRAIN_TRIMODAL_FUSION)
params = list(route_heads["LNI"].parameters()) + list(router.parameters())
if TRAIN_TRIMODAL_FUSION:
    params += list(fusion["LNI"].parameters())
if FINETUNE_ENCODERS:
    params += list(behrt.parameters()) + list(bbert.parameters()) + list(imgenc.parameters())

opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

# -----------------------------
# Fairness penalty (EDDI) — FINAL PREDICTION ONLY
#   - Computes disparity over per-example absolute errors using the final probabilities:
#       err_i = mean_c | sigmoid(ylogits_i,c) - y_i,c |
#   - For each sensitive key, computes size-weighted mean absolute deviation of group means
#     from the overall mean; averages across sensitive keys.
# -----------------------------
def compute_eddi_final(batch_groups: list[dict], ylogits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        for g, ix in group_to_ix.items():
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

# compute all *frozen* route logits (L, N, I, LN, LI, NI)
@torch.no_grad()
def _frozen_routes_from_unimodal(z: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Inputs:
      z: {"L":[B,d], "N":[B,d], "I":[B,d]}
    Returns:
      dict with logits for frozen routes: L, N, I, LN, LI, NI (each [B, C])
      (LNI is excluded here; it's trained below with gradients.)
    """
    out = {}
    # Unimodal
    out["L"] = route_heads["L"](z["L"])
    out["N"] = route_heads["N"](z["N"])
    out["I"] = route_heads["I"](z["I"])
    # Bimodal (use fusion modules)
    zLN = fusion["LN"](z["L"], z["N"])
    zLI = fusion["LI"](z["L"], z["I"])
    zNI = fusion["NI"](z["N"], z["I"])
    out["LN"] = route_heads["LN"](zLN)
    out["LI"] = route_heads["LI"](zLI)
    out["NI"] = route_heads["NI"](zNI)
    return out

# Training loop
best_val = float("inf")

for epoch in range(CFG.max_epochs_tri):
    if not FINETUNE_ENCODERS:
        behrt.eval(); bbert.eval(); imgenc.eval()
    else:
        behrt.train(); bbert.train(); imgenc.train()

    fusion["LNI"].train(TRAIN_TRIMODAL_FUSION)
    route_heads["LNI"].train()
    router.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_tri} [TRI]", dynamic_ncols=True)
    running = 0.0; n_steps = 0

    for xL, notes_list, imgs, y, sens in pbar:
        xL   = xL.to(DEVICE)       
        imgs = imgs.to(DEVICE)    
        y    = y.to(DEVICE)        

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            # Unimodal embeddings 
            if FINETUNE_ENCODERS:
                zL = behrt(xL)                 
                zN = bbert(notes_list)         
                zI = imgenc(imgs)              
            else:
                with torch.no_grad():
                    zL = behrt(xL)
                    zN = bbert(notes_list)
                    zI = imgenc(imgs)

            z = {"L": zL, "N": zN, "I": zI}

            # Route logits (frozen for L/N/I/LN/LI/NI; trainable for LNI) 
            route_logits = _frozen_routes_from_unimodal(z)  # no_grad for frozen ones

            # LNI fused embedding (trainable if TRAIN_TRIMODAL_FUSION=True)
            zLNI = fusion["LNI"](z["L"], z["N"], z["I"])
            route_logits["LNI"] = route_heads["LNI"](zLNI)  # with grad

            # Router gating (per-task route & block weights) 
            ylogits, route_w, block_w, block_logits = router(z, route_logits, masks=None)  

            # Loss = BCE + lambda_fair * EDDI_final (on final predictions only) 
            bce  = F.binary_cross_entropy_with_logits(ylogits, y)
            fair = compute_eddi_final(sens, ylogits, y)  
            loss = bce + getattr(CFG, "lambda_fair", 0.0) * fair

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        running += float(loss); n_steps += 1
        pbar.set_postfix(loss=f"{running / n_steps:.4f}", bce=f"{float(bce):.4f}", fair=f"{float(fair):.4f}")

    # Validation
    behrt.eval(); bbert.eval(); imgenc.eval()
    fusion["LNI"].eval()
    route_heads["LNI"].eval()
    router.eval()

    val_loss = 0.0; n_val = 0
    with torch.no_grad():
        for xL, notes_list, imgs, y, sens in val_loader:
            xL   = xL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y    = y.to(DEVICE)

            zL = behrt(xL)
            zN = bbert(notes_list)
            zI = imgenc(imgs)
            z  = {"L": zL, "N": zN, "I": zI}

            # frozen routes
            route_logits = _frozen_routes_from_unimodal(z)

            # trimodal route (uses fusion)
            zLNI = fusion["LNI"](z["L"], z["N"], z["I"])
            route_logits["LNI"] = route_heads["LNI"](zLNI)

            # Final prediction via router
            ylogits, route_w, block_w, block_logits = router(z, route_logits, masks=None)

            bce  = F.binary_cross_entropy_with_logits(ylogits, y)
            fair = compute_eddi_final(sens, ylogits, y)   # <-- final-only fairness
            lval = bce + getattr(CFG, "lambda_fair", 0.0) * fair

            val_loss += float(lval); n_val += 1

    val_loss /= max(n_val, 1)
    print(f"[TRI] Val loss: {val_loss:.4f}")

    # Save best: router + LNI (+ optionally fusion LNI config)
    if val_loss < best_val:
        best_val = val_loss
        os.makedirs(CFG.ckpt_root, exist_ok=True)
        save_obj = {
            "router": router.state_dict(),
            "LNI": route_heads["LNI"].state_dict(),
            "best_val": best_val,
            "cfg": vars(CFG),
            "finetune_encoders": FINETUNE_ENCODERS,
            "train_trimodal_fusion": TRAIN_TRIMODAL_FUSION,
        }
        # Save LNI fusion weights if present
        if fusion is not None and "LNI" in fusion:
            try:
                save_obj["fusion_LNI"] = fusion["LNI"].state_dict()
            except Exception:
                pass

        torch.save(save_obj, os.path.join(CFG.ckpt_root, "step3_trimodal_router.pt"))
        print("Saved best trimodal+router -> checkpoints/step3_trimodal_router.pt")
