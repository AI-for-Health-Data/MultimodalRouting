from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from routing import MMRouting  
from models.encoders import (
    BEHRTLabEncoder,  
    BioClinBERTEncoder,
    ImageCXREncoder, #If needed to be changed(Zhongjie)
)
from models.routes import RouteMLP
from data.icustay_dataset import ICUStayDataset
from utils.fairness import compute_eddi  # returns scalar overall EDDI

# Hyper‑parameters
DEFAULTS: Dict[str, Any] = {
    "batch_size": 32,
    "hidden": 256,
    "lr": 3e‑4,
    "weight_decay": 1e‑4,
    "epochs": {"uni": 4, "bi": 4, "tri": 8},
    "ema_beta": 0.9,
    "lambda_eddi": 0.5,
    "router_alpha": 5.0,
    "grad_clip": 1.0,
}

class FAMEPlusPlus(nn.Module):
    TASKS = ("mortality", "pe", "ph")
    ROUTES = ("L", "N", "I", "LN", "LI", "NI", "LNI")

    def __init__(self, hidden: int, alpha: float):
        super().__init__()
        # encoders 
        self.enc_L = BEHRTLabEncoder(out_dim=hidden)
        self.enc_N = BioClinBERTEncoder(out_dim=hidden)
        self.enc_I = ImageCXREncoder(out_dim=hidden)
        # route heads 
        self.heads = nn.ModuleDict({
            r: RouteMLP(in_dim=hidden * len(r), out_dim=len(self.TASKS))
            for r in self.ROUTES
        })
        # router 
        self.router = MMRouting(alpha=alpha)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        L = self.enc_L(batch["lab_feats"], batch["demo"])
        N = self.enc_N(batch["note_ids"], batch["note_attn"])
        I = self.enc_I(batch["image"])
        Z = {
            "L": L,
            "N": N,
            "I": I,
            "LN": torch.cat([L, N], -1),
            "LI": torch.cat([L, I], -1),
            "NI": torch.cat([N, I], -1),
            "LNI": torch.cat([L, N, I], -1),
        }
        route_logits = torch.stack([self.heads[r](Z[r]) for r in self.ROUTES], 1)
        return route_logits  # [B,7,3]

# Trainer 
class Trainer:
    def __init__(self, cfg: Dict[str, Any], out: Path, device: torch.device):
        self.cfg = cfg
        self.device = device
        out.mkdir(parents=True, exist_ok=True)
        self.out = out

        # data
        self.loaders = {
            split: DataLoader(
                ICUStayDataset(split=split),
                batch_size=cfg["batch_size"],
                shuffle=(split == "train"),
                num_workers=4,
            )
            for split in ("train", "val", "test")
        }

        self.model = FAMEPlusPlus(cfg["hidden"], cfg["router_alpha"]).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.crit = nn.BCEWithLogitsLoss()
        # EMA route loss buffer
        self.ema = torch.zeros(7, device=device)

    def _forward_batch(self, batch: Dict[str, torch.Tensor]):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits_r = self.model(batch)  # [B,7,3]
        y = batch["labels"].float()
        per_route_loss = torch.stack([self.crit(logits_r[:, i, :], y) for i in range(7)])
        fused, _, _ = self.model.router(logits_r, self.ema)
        eddi = compute_eddi(y.cpu().numpy(), torch.sigmoid(fused).cpu().numpy(), batch["sens"].cpu().numpy())
        loss = self.crit(fused, y) + self.cfg["lambda_eddi"] * eddi
        return loss, per_route_loss

    def _train_phase(self, routes_trainable: Sequence[str], n_epochs: int):
        # freeze / unfreeze only the route heads
        for name, p in self.model.named_parameters():
            if name.startswith("heads"):
                p.requires_grad = any(rt in name for rt in routes_trainable)
        for ep in range(1, n_epochs + 1):
            self.model.train()
            for batch in self.loaders["train"]:
                self.opt.zero_grad(set_to_none=True)
                loss, route_loss = self._forward_batch(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.opt.step()
                # update EMA
                self.ema = self.ema * self.cfg["ema_beta"] + route_loss.detach() * (1 - self.cfg["ema_beta"])
            print(f"{routes_trainable} | epoch {ep}/{n_epochs} | loss {loss.item():.4f}")

    def train(self):
        print(" Stage 1 (uni)…")
        self._train_phase(["L", "N", "I"], self.cfg["epochs"]["uni"])
        print(" Stage 2 (bi)…")
        self._train_phase(["LN", "LI", "NI"], self.cfg["epochs"]["bi"])
        print(" Stage 3 (tri)…")
        self._train_phase(["LNI"], self.cfg["epochs"]["tri"])
        torch.save(self.model.state_dict(), self.out / "famepp_final.pt")

    def evaluate(self):
        self.model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in self.loaders["test"]:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                fused, _, _ = self.model.router(self.model(batch), self.ema)
                ys.append(batch["labels"].cpu())
                ps.append(torch.sigmoid(fused).cpu())
        y = torch.cat(ys)
        p = torch.cat(ps)
        metrics = {
            t: {
                "auroc": roc_auc_score(y[:, i], p[:, i]),
                "auprc": average_precision_score(y[:, i], p[:, i]),
                "f1": f1_score(y[:, i], (p[:, i] > 0.5).int()),
            }
            for i, t in enumerate(self.model.TASKS)
        }
        (self.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs") / dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cfg", type=Path, help="JSON file with hyper‑parameters")
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    if args.cfg and args.cfg.exists():
        cfg.update(json.loads(args.cfg.read_text()))

    trainer = Trainer(cfg, args.out, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train()
    trainer.evaluate()
