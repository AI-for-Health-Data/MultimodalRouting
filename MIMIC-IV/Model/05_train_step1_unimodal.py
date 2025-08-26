# Trains: BEHRT (structured), BioClinicalBERT (notes), Image encoder + unimodal route heads L, N, I

import os, json, math, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class ICUStayDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        stay_id = self.ids[idx]

        # Structured: 24h time-series -> [T, F] (sorted by hour)
        df_s = self.struct[self.struct.stay_id == stay_id].sort_values("hour")
        xs_np = df_s[self.feat_cols].to_numpy(dtype=np.float32)  
        xs = torch.from_numpy(xs_np) 

        # Notes: list of texts within first 24h (kept *separately*; encoder will chunk per-note)
        notes_list = self.notes[self.notes.stay_id == stay_id].text.tolist()

        # Images: list of image paths (we load 1st for unimodal L/N/I training efficiency)
        img_paths = self.images[self.images.stay_id == stay_id].image_path.tolist()

        # Labels: [mort, pe, ph]
        row_y = self.labels[self.labels.stay_id == stay_id][["mort", "pe", "ph"]].values[0].astype(np.float32)
        y = torch.from_numpy(row_y)

        # Sensitive metadata dict
        sens = self.sensitive[self.sensitive.stay_id == stay_id].iloc[0].to_dict()

        return {
            "stay_id": stay_id,
            "x_struct": xs,           
            "notes_list": notes_list, 
            "image_paths": img_paths, 
            "y": y,                   
            "sens": sens,
        }


#  Collate: pad/stack structured, stack images, keep notes as List[List[str]] 
def pad_or_trim_struct(x: torch.Tensor, T: int, F: int) -> torch.Tensor:
    t = x.shape[0]
    if t >= T:
        return x[-T:]
    pad = torch.zeros(T - t, F, dtype=x.dtype)
    return torch.cat([pad, x], dim=0)

from PIL import Image
import torchvision.transforms as TF

IMG_TFMS = TF.Compose([
    TF.Resize((224, 224)),
    TF.ToTensor(),
])

def load_first_image(paths: list[str]) -> torch.Tensor:
    if not paths:
        return torch.zeros(3, 224, 224)
    p = paths[0]
    try:
        img = Image.open(p).convert("RGB")
        return IMG_TFMS(img)
    except Exception:
        return torch.zeros(3, 224, 224)

def collate_fn(batch):
    B = len(batch)
    T_len = CFG.structured_seq_len
    F_dim = CFG.structured_n_feats

    xL_batch = torch.stack([pad_or_trim_struct(b["x_struct"], T_len, F_dim) for b in batch], dim=0) 
    notes_batch = [b["notes_list"] if isinstance(b["notes_list"], list) else [str(b["notes_list"])] for b in batch]  
    imgs_batch  = torch.stack([load_first_image(b["image_paths"]) for b in batch], dim=0)  
    y_batch     = torch.stack([b["y"] for b in batch], dim=0)  
    sens_batch  = [b["sens"] for b in batch]

    return xL_batch, notes_batch, imgs_batch, y_batch, sens_batch


ROOT = os.path.join(CFG.data_root, "MIMIC-IV")  # (or INSPECT)
train_ds = ICUStayDataset(ROOT, split="train")
val_ds   = ICUStayDataset(ROOT, split="val")

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

params = (
    list(behrt.parameters())
    + list(bbert.parameters())
    + list(imgenc.parameters())
    + list(route_heads["L"].parameters())
    + list(route_heads["N"].parameters())
    + list(route_heads["I"].parameters())
)
opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

best_val = float("inf")

for epoch in range(CFG.max_epochs_uni):
    behrt.train(); bbert.train(); imgenc.train()
    route_heads["L"].train(); route_heads["N"].train(); route_heads["I"].train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.max_epochs_uni} [UNI]", dynamic_ncols=True)
    running = 0.0; n_steps = 0

    for xL, notes_list, imgs, y, sens in pbar:
        xL   = xL.to(DEVICE)          
        imgs = imgs.to(DEVICE)        
        y    = y.to(DEVICE)           

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            # Encoders -> embeddings
            zL = behrt(xL)                     
            zN = bbert(notes_list)             
            zI = imgenc(imgs)                  

            # Unimodal heads -> logits
            logits_L = route_heads["L"](zL)    
            logits_N = route_heads["N"](zN)    
            logits_I = route_heads["I"](zI)    

            # BCE over the three unimodal routes (mean)
            loss_L = F.binary_cross_entropy_with_logits(logits_L, y)
            loss_N = F.binary_cross_entropy_with_logits(logits_N, y)
            loss_I = F.binary_cross_entropy_with_logits(logits_I, y)
            loss = (loss_L + loss_N + loss_I) / 3.0

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        running += float(loss); n_steps += 1
        pbar.set_postfix(loss=f"{running / n_steps:.4f}")

    behrt.eval(); bbert.eval(); imgenc.eval()
    route_heads["L"].eval(); route_heads["N"].eval(); route_heads["I"].eval()

    val_loss = 0.0; n_val = 0
    with torch.no_grad():
        for xL, notes_list, imgs, y, sens in val_loader:
            xL   = xL.to(DEVICE)
            imgs = imgs.to(DEVICE)
            y    = y.to(DEVICE)

            zL = behrt(xL)
            zN = bbert(notes_list)
            zI = imgenc(imgs)

            logits_L = route_heads["L"](zL)
            logits_N = route_heads["N"](zN)
            logits_I = route_heads["I"](zI)

            lval = (
                F.binary_cross_entropy_with_logits(logits_L, y)
                + F.binary_cross_entropy_with_logits(logits_N, y)
                + F.binary_cross_entropy_with_logits(logits_I, y)
            ) / 3.0

            val_loss += float(lval); n_val += 1

    val_loss /= max(n_val, 1)
    print(f"Val loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        os.makedirs(CFG.ckpt_root, exist_ok=True)
        torch.save(
            {
                "behrt": behrt.state_dict(),
                "bbert": bbert.state_dict(),
                "imgenc": imgenc.state_dict(),
                "L": route_heads["L"].state_dict(),
                "N": route_heads["N"].state_dict(),
                "I": route_heads["I"].state_dict(),
                "best_val": best_val,
                "cfg": vars(CFG),
            },
            os.path.join(CFG.ckpt_root, "step1_unimodal.pt"),
        )
        print("Saved best unimodal checkpoint -> checkpoints/step1_unimodal.pt")
