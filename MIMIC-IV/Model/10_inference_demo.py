# Load checkpoints, run per-patient predictions, and display route/block weights
# using the *learned-gate* router (per-task route/block weights).

import os, json
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


if "ROUTES" not in globals():
    ROUTES = ["L", "N", "I", "LN", "LI", "NI", "LNI"]
if "TASKS" not in globals():
    TASKS = ["mort", "pe", "ph"]
if "BLOCKS" not in globals():
    BLOCKS = {"uni": ["L", "N", "I"], "bi": ["LN", "LI", "NI"], "tri": ["LNI"]}
if "DEVICE" not in globals():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


ckpt1 = torch.load(os.path.join(CFG.ckpt_root, "step1_unimodal.pt"), map_location=DEVICE)
ckpt2 = torch.load(os.path.join(CFG.ckpt_root, "step2_bimodal.pt"), map_location=DEVICE)
ckpt3 = torch.load(os.path.join(CFG.ckpt_root, "step3_trimodal_router.pt"), map_location=DEVICE)

# encoders
behrt.load_state_dict(ckpt1["behrt"])
bbert.load_state_dict(ckpt1["bbert"])
imgenc.load_state_dict(ckpt1["imgenc"])

# heads
for key in ["L","N","I"]:
    route_heads[key].load_state_dict(ckpt1[key])
for key in ["LN","LI","NI"]:
    route_heads[key].load_state_dict(ckpt2[key])
route_heads["LNI"].load_state_dict(ckpt3["LNI"])

# router (learned gate)
router.load_state_dict(ckpt3["router"])

# eval mode
behrt.eval(); bbert.eval(); imgenc.eval()
for r in ROUTES: route_heads[r].eval()
router.eval()

def _ensure_test_loader():
    """Build test_loader if not present in globals() using the same logic as Notebook 05."""
    if "test_loader" in globals():
        return globals()["test_loader"]

    import json, pandas as pd, numpy as np
    from torch.utils.data import Dataset
    import torchvision.transforms as T

    class ICUStayDataset(Dataset):
        def __init__(self, root: str, split: str = "test"):
            super().__init__()
            self.root = root
            with open(os.path.join(root, "splits.json")) as f:
                splits = json.load(f)
            self.ids = list(splits[split])

            self.struct = pd.read_parquet(os.path.join(root, "structured_24h.parquet"))
            self.notes  = pd.read_parquet(os.path.join(root, "notes_24h.parquet"))
            self.images = pd.read_parquet(os.path.join(root, "images_24h.parquet"))
            self.labels = pd.read_parquet(os.path.join(root, "labels.parquet"))
            self.sensitive = pd.read_parquet(os.path.join(root, "sensitive.parquet"))

            base_cols = {"stay_id", "hour"}
            self.feat_cols = [c for c in self.struct.columns if c not in base_cols]

        def __len__(self): return len(self.ids)

        def __getitem__(self, idx: int):
            stay_id = self.ids[idx]
            df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
            xs_np = df_s[self.feat_cols].to_numpy(dtype=np.float32)
            xs = torch.from_numpy(xs_np)

            # notes: list of strings for this stay_id
            notes_list = self.notes[self.notes.stay_id == stay_id].text.tolist()

            # images: list of paths (we'll load the first, like in 05)
            img_paths = self.images[self.images.stay_id == stay_id].image_path.tolist()

            # labels: (mort, pe, ph) in that order
            row_y = self.labels[self.labels.stay_id == stay_id][["mort", "pe", "ph"]].values[0].astype(np.float32)
            y = torch.from_numpy(row_y)

            # sensitive/meta dict 
            sens = self.sensitive[self.sensitive.stay_id == stay_id].iloc[0].to_dict()

            return {"stay_id": stay_id, "x_struct": xs, "notes_list": notes_list,
                    "image_paths": img_paths, "y": y, "sens": sens}

    def pad_or_trim_struct(x: torch.Tensor, T: int, F: int) -> torch.Tensor:
        t = x.shape[0]
        if t >= T: return x[-T:]
        pad = torch.zeros(T - t, F, dtype=x.dtype)
        return torch.cat([pad, x], dim=0)

    IMG_TFMS = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    def load_first_image(paths: list[str]) -> torch.Tensor:
        if not paths:
            return torch.zeros(3, 224, 224)
        p = paths[0]
        try:
            from PIL import Image
            img = Image.open(p).convert("RGB")
            return IMG_TFMS(img)
        except Exception:
            return torch.zeros(3, 224, 224)

    def collate_fn(batch):
        B = len(batch)
        T = CFG.structured_seq_len
        F = CFG.structured_n_feats

        xL_batch = torch.stack([pad_or_trim_struct(b["x_struct"], T, F) for b in batch], dim=0)  

        # ensure list-of-strings per patient
        notes_batch = [
            b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])]
            for b in batch
        ]

        imgs_batch  = torch.stack([load_first_image(b["image_paths"]) for b in batch], dim=0)  
        y_batch     = torch.stack([b["y"] for b in batch], dim=0)  
        sens_batch  = [b["sens"] for b in batch]
        return xL_batch, notes_batch, imgs_batch, y_batch, sens_batch

    ROOT_local = os.path.join(CFG.data_root, "MIMIC-IV")  # or INSPECT
    test_ds_local = ICUStayDataset(ROOT_local, split="test")
    test_loader_local = DataLoader(
        test_ds_local,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    globals()["test_loader"] = test_loader_local
    return test_loader_local

test_loader = _ensure_test_loader()

def build_masks(xL, notes_list, imgs):
    B = xL.size(0)
    mL = torch.ones(B, 1, device=xL.device)

    mN_list = []
    for notes in notes_list:
        present = 1.0 if (isinstance(notes, list) and any((isinstance(t, str) and len(t.strip()) > 0) for t in notes)) else 0.0
        mN_list.append(present)
    mN = torch.tensor(mN_list, device=xL.device, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        mI_vals = (imgs.abs().flatten(1).sum(dim=1) > 0).float()
    mI = mI_vals.to(xL.device).unsqueeze(1)
    return {"L": mL, "N": mN, "I": mI}

def embeddings_from_batch(xL, notes_list, imgs):
    zL = behrt(xL)
    zN = bbert(notes_list)
    zI = imgenc(imgs)
    return {"L": zL, "N": zN, "I": zI}

def all_route_logits(z):
    # Use fusion modules for multi-modal routes; each route head expects d-dim input.
    out = {
        "L": route_heads["L"](z["L"]),
        "N": route_heads["N"](z["N"]),
        "I": route_heads["I"](z["I"]),
    }
    zLN = fusion["LN"](z["L"], z["N"])
    zLI = fusion["LI"](z["L"], z["I"])
    zNI = fusion["NI"](z["N"], z["I"])
    zLNI = fusion["LNI"](z["L"], z["N"], z["I"])
    out["LN"] = route_heads["LN"](zLN)
    out["LI"] = route_heads["LI"](zLI)
    out["NI"] = route_heads["NI"](zNI)
    out["LNI"] = route_heads["LNI"](zLNI)
    return out

route_index = {r: i for i, r in enumerate(ROUTES)}
block_names = ["uni", "bi", "tri"]

@torch.no_grad()
def run_inference_demo(loader: DataLoader, show_k: int = 5, inspect_idx: int = 0):
    xL, notes_list, imgs, y, sens = next(iter(loader))
    xL   = xL.to(DEVICE)
    imgs = imgs.to(DEVICE)
    y    = y.to(DEVICE)

    z = embeddings_from_batch(xL, notes_list, imgs)
    masks = build_masks(xL, notes_list, imgs)
    route_logits = all_route_logits(z)

    # learned-gate router returns: ylogits, route_w [B,C,7], block_w [B,C,3], block_logits [B,3,C]
    ylogits, route_w, block_w, block_logits = router(z, route_logits, masks=masks)
    probs = torch.sigmoid(ylogits)  # [B, C]

    print("Predicted probabilities (", ", ".join(TASKS), ") for first", show_k, "patients:")
    print(probs[:show_k].cpu().numpy())

    i = min(inspect_idx, probs.size(0)-1)  
    print(f"\n--- Per-task route & block weights for sample index {i} ---")
    for t_idx, t_name in enumerate(TASKS):
        rw = {r: float(route_w[i, t_idx, route_index[r]]) for r in ROUTES}
        bw = {block_names[b]: float(block_w[i, t_idx, b]) for b in range(3)}

        # sort routes by weight (desc) for display
        top_routes = sorted(rw.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\nTask = {t_name}")
        print("  Final prob:", float(probs[i, t_idx]))
        print("  Block weights:", {k: round(v, 4) for k, v in bw.items()})
        print("  Top routes by weight:")
        for name, w in top_routes:
            print(f"    {name:>3}: {w:.4f}")

    print("\nBlock logits (before block weighting) for that sample:")
    for t_idx, t_name in enumerate(TASKS):
        print(f"  {t_name}: uni={float(block_logits[i,0,t_idx]):+.4f}, "
              f"bi={float(block_logits[i,1,t_idx]):+.4f}, "
              f"tri={float(block_logits[i,2,t_idx]):+.4f}")

    return {
        "probs": probs,
        "ylogits": ylogits,
        "route_w": route_w,
        "block_w": block_w,
        "block_logits": block_logits,
        "route_logits": route_logits,
        "masks": masks,
        "z": z,
        "y_true": y,
        "sens": sens,
    }

@torch.no_grad()
def run_inference_loader(loader: DataLoader):
    behrt.eval(); bbert.eval(); imgenc.eval()
    for r in ROUTES: route_heads[r].eval()
    router.eval()

    all_probs = []
    all_logits = []
    all_targets = []
    all_route_w = []
    all_block_w = []

    for xL, notes_list, imgs, y, sens in loader:
        xL   = xL.to(DEVICE)
        imgs = imgs.to(DEVICE)
        y    = y.to(DEVICE)

        z = embeddings_from_batch(xL, notes_list, imgs)
        masks = build_masks(xL, notes_list, imgs)
        route_logits = all_route_logits(z)
        ylogits, route_w, block_w, _ = router(z, route_logits, masks=masks)
        probs = torch.sigmoid(ylogits)

        all_probs.append(probs.cpu())
        all_logits.append(ylogits.cpu())
        all_targets.append(y.cpu())
        all_route_w.append(route_w.cpu())
        all_block_w.append(block_w.cpu())

    if not all_probs:
        raise RuntimeError("Loader yielded no batches.")

    probs = torch.cat(all_probs, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    route_w = torch.cat(all_route_w, dim=0).numpy()   
    block_w = torch.cat(all_block_w, dim=0).numpy()   

    return {
        "probs": probs,
        "logits": logits,
        "targets": targets,
        "route_w": route_w,
        "block_w": block_w,
        "tasks": TASKS,
        "routes": ROUTES,
        "blocks": ["uni", "bi", "tri"],
    }


_ = run_inference_demo(test_loader, show_k=5, inspect_idx=0)
