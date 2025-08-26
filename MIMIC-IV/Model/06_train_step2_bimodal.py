# Freeze unimodal heads and train bimodal heads (LN, LI, NI) to capture residual signal beyond unimodal.

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
        print("Re-created train/val loaders for Step 2.")
    except NameError as e:
        raise RuntimeError("Missing dataset/collate definitions. Please run the Step 1 notebook first.") from e

# Load Step 1 checkpoints (encoders + unimodal heads) 
ckpt1 = torch.load(os.path.join(CFG.ckpt_root, "step1_unimodal.pt"), map_location=DEVICE)
behrt.load_state_dict(ckpt1["behrt"])
bbert.load_state_dict(ckpt1["bbert"])
imgenc.load_state_dict(ckpt1["imgenc"])
route_heads["L"].load_state_dict(ckpt1["L"])
route_heads["N"].load_state_dict(ckpt1["N"])
route_heads["I"].load_state_dict(ckpt1["I"])
print("Loaded Step 1 weights.")

# Ensure fusion modules exist (from Notebook 04). If not, build minimal fallbacks. 
def _ensure_fusions():
    global fusion
    if "fusion" in globals() and isinstance(fusion, dict):
        # Make sure required keys are present and on the right device
        needed = {"LN","LI","NI"}
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

    fusion_local = {
        "LN": _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE),
        "LI": _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE),
        "NI": _PairwiseFusion(d=CFG.d, p_drop=CFG.dropout).to(DEVICE),
    }
    fusion = fusion_local
    print("Built fallback fusion modules for LN, LI, NI.")

_ensure_fusions()

# Freeze unimodal heads (we keep encoders frozen in the update graph using no_grad below) 
def set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

set_requires_grad(route_heads["L"], False)
set_requires_grad(route_heads["N"], False)
set_requires_grad(route_heads["I"], False)

# We will compute encoder outputs under torch.no_grad() so encoders do not update.
behrt.eval(); bbert.eval(); imgenc.eval()

# Optimizer: only the bimodal heads 
params_bi = list(route_heads["LN"].parameters()) \
          + list(route_heads["LI"].parameters()) \
          + list(route_heads["NI"].parameters())

opt = torch.optim.AdamW(params_bi, lr=CFG.lr, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

best_val = float("inf")

for epoch in range(CFG.max_epochs_bi):
    # Train bimodal heads only
    route_heads["LN"].train()
    route_heads["LI"].train()
    route_heads["NI"].train()

    # Keep encoders + unimodal heads frozen/eval
    behrt.eval(); bbert.eval(); imgenc.eval()
    route_heads["L"].eval(); route_heads["N"].eval(); route_heads["I"].eval()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_bi} [BI]", dynamic_ncols=True)
    running = 0.0; n_steps = 0

    for xL, notes_list, imgs, y, sens in pbar:
        xL   = xL.to(DEVICE)        
        imgs = imgs.to(DEVICE)      
        y    = y.to(DEVICE)         

        opt.zero_grad(set_to_none=True)

        # Encoders (frozen): compute embeddings without building grads 
        with torch.no_grad():
            zL = behrt(xL)                  
            zN = bbert(notes_list)          
            zI = imgenc(imgs)               

        # Bimodal fused embeddings (each in R^d) 
        zLN = fusion["LN"](zL, zN)          
        zLI = fusion["LI"](zL, zI)          
        zNI = fusion["NI"](zN, zI)          

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = {
                "LN": route_heads["LN"](zLN),  # [B, 3]
                "LI": route_heads["LI"](zLI),  # [B, 3]
                "NI": route_heads["NI"](zNI),  # [B, 3]
            }
            # Mean BCE across the three bimodal routes
            loss = (
                F.binary_cross_entropy_with_logits(logits["LN"], y)
              + F.binary_cross_entropy_with_logits(logits["LI"], y)
              + F.binary_cross_entropy_with_logits(logits["NI"], y)
            ) / 3.0

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(params_bi, max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        running += float(loss); n_steps += 1
        pbar.set_postfix(loss=f"{running / n_steps:.4f}")

    # Validation 
    for k in ["LN","LI","NI"]:
        route_heads[k].eval()

    val_loss = 0.0; n_val = 0
    with torch.no_grad():
        for xL, notes_list, imgs, y, sens in val_loader:
            xL   = xL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y    = y.to(DEVICE)

            # encoders frozen
            zL = behrt(xL)
            zN = bbert(notes_list)
            zI = imgenc(imgs)

            # fused bimodal embeddings
            zLN = fusion["LN"](zL, zN)
            zLI = fusion["LI"](zL, zI)
            zNI = fusion["NI"](zN, zI)

            lval = (
                F.binary_cross_entropy_with_logits(route_heads["LN"](zLN), y)
              + F.binary_cross_entropy_with_logits(route_heads["LI"](zLI), y)
              + F.binary_cross_entropy_with_logits(route_heads["NI"](zNI), y)
            ) / 3.0

            val_loss += float(lval); n_val += 1

    val_loss /= max(n_val, 1)
    print(f"[BI] Val loss: {val_loss:.4f}")

    # Save best bimodal heads 
    if val_loss < best_val:
        best_val = val_loss
        os.makedirs(CFG.ckpt_root, exist_ok=True)
        torch.save(
            {
                "LN": route_heads["LN"].state_dict(),
                "LI": route_heads["LI"].state_dict(),
                "NI": route_heads["NI"].state_dict(),
                "best_val": best_val,
                "cfg": vars(CFG),
            },
            os.path.join(CFG.ckpt_root, "step2_bimodal.pt"),
        )
        print("Saved best bimodal heads -> checkpoints/step2_bimodal.pt")
