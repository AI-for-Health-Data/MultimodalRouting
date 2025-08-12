from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from routing import MMRouting
from models.encoders import (
    BEHRTLabEncoder,
    BioClinBERTEncoder,
    ImageCXREncoder,  # If needed to be changed (Zhongjie)
)
from models.routes import RouteMLP
from data.icustay_dataset import ICUStayDataset
from utils.fairness import compute_eddi  # NOTE: returns numpy scalar (non-differentiable)

# Hyper‑parameters
DEFAULTS: Dict[str, Any] = {
    "batch_size": 32,
    "hidden": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
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
        # encoders (shared across all routes)
        self.enc_L = BEHRTLabEncoder(out_dim=hidden)
        self.enc_N = BioClinBERTEncoder(out_dim=hidden)
        self.enc_I = ImageCXREncoder(out_dim=hidden)

        # route heads (unique per route)
        self.heads = nn.ModuleDict({
            r: RouteMLP(in_dim=hidden * len(r), out_dim=len(self.TASKS))
            for r in self.ROUTES
        })
        # deterministic router (no trainable params)
        self.router = MMRouting(alpha=alpha)
        
        self.route_to_idx = {r: i for i, r in enumerate(self.ROUTES)}
        self.blocks = {
            "uni": (self.route_to_idx["L"], self.route_to_idx["N"], self.route_to_idx["I"]),
            "bi":  (self.route_to_idx["LN"], self.route_to_idx["LI"], self.route_to_idx["NI"]),
            "tri": (self.route_to_idx["LNI"],),
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encoders
        L = self.enc_L(batch["lab_feats"], batch.get("demo", None))
        N = self.enc_N(batch["note_ids"], batch["note_attn"])
        I = self.enc_I(batch["image"])

        # Per-route concatenations
        Z = {
            "L": L,
            "N": N,
            "I": I,
            "LN": torch.cat([L, N], dim=-1),
            "LI": torch.cat([L, I], dim=-1),
            "NI": torch.cat([N, I], dim=-1),
            "LNI": torch.cat([L, N, I], dim=-1),
        }

        route_logits = torch.stack([self.heads[r](Z[r]) for r in self.ROUTES], dim=1)
        return route_logits

# Trainer 
class Trainer:
    def __init__(self, cfg: Dict[str, Any], out: Path, device: torch.device):
        self.cfg = cfg
        self.device = device
        out.mkdir(parents=True, exist_ok=True)
        self.out = out

        # Data
        self.loaders = {
            split: DataLoader(
                ICUStayDataset(split=split),
                batch_size=cfg["batch_size"],
                shuffle=(split == "train"),
                num_workers=4,
                pin_memory=True,
            )
            for split in ("train", "val", "test")
        }

        # Model/optim
        self.model = FAMEPlusPlus(cfg["hidden"], cfg["router_alpha"]).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(),
                                     lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
        self.crit = nn.BCEWithLogitsLoss()

        # EMA buffer of per-route losses
        self.ema = torch.zeros(7, device=device)

        # Convenience indices
        self.idx_uni = torch.tensor([0, 1, 2], device=device)
        self.idx_bi  = torch.tensor([3, 4, 5], device=device)
        self.idx_tri = torch.tensor([6], device=device)

    @staticmethod
    def _weighted_block_sum(route_logits: torch.Tensor,
                            route_w: torch.Tensor,
                            idxs: torch.Tensor) -> torch.Tensor:
        weighted = route_logits * route_w.view(1, -1, 1)
        return weighted[:, idxs, :].sum(dim=1)

    def _masked_ema(self, phase: str) -> torch.Tensor:
        ema = self.ema.clone()
        INF = 1e6
        if phase == "uni":
            # mask bi + tri
            ema[3] = INF; ema[4] = INF; ema[5] = INF; ema[6] = INF
        elif phase == "bi":
            # mask tri only
            ema[6] = INF
        elif phase == "tri":
            pass
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return ema

    def _fused_logits_with_stopgrad(self,
                                    route_logits: torch.Tensor,
                                    route_w: torch.Tensor,
                                    block_w: torch.Tensor,
                                    phase: str) -> torch.Tensor:

        # Per-block weighted sums
        uni = self._weighted_block_sum(route_logits, route_w, self.idx_uni)  
        bi  = self._weighted_block_sum(route_logits, route_w, self.idx_bi)   
        tri = self._weighted_block_sum(route_logits, route_w, self.idx_tri)  

        w_uni, w_bi, w_tri = block_w  # scalars

        if phase == "uni":
            # Only unimodal contributes
            fused = w_uni * uni
        elif phase == "bi":
            # Stop-grad on uni; learn residual in bi
            fused = w_uni * uni.detach() + w_bi * bi
        elif phase == "tri":
            # Stop-grad on uni+bi; learn residual in tri
            fused = w_uni * uni.detach() + w_bi * bi.detach() + w_tri * tri
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return fused

    def _forward_batch(self, batch: Dict[str, torch.Tensor], phase: str) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits_r = self.model(batch)  
        y = batch["labels"].float()

        # Per-route BCE 
        per_route_loss = torch.stack([self.crit(logits_r[:, i, :], y) for i in range(7)])

        # Compute router weights from (masked) EMA so inactive blocks get ~0 weight
        masked_ema = self._masked_ema(phase)
        # Call router to get weights; we will not use its fused output for the loss
        _fused_unused, route_w, block_w = self.model.router(logits_r, masked_ema)

        # Build fused logits with MRO-style stop-gradient at the block level
        fused = self._fused_logits_with_stopgrad(logits_r, route_w, block_w, phase)

        # Main prediction loss
        pred_loss = self.crit(fused, y)

        # fairness penalty: current compute_eddi is numpy -> no gradients
        with torch.no_grad():
            eddi_np = compute_eddi(
                y.cpu().numpy(),
                torch.sigmoid(fused).cpu().numpy(),
                batch["sens"].cpu().numpy()
            )
        loss = pred_loss + self.cfg["lambda_eddi"] * float(eddi_np)  # constant wrt params

        return loss, per_route_loss

    def _set_trainable_heads(self, routes_trainable: Sequence[str]) -> None:
        active = set(routes_trainable)
        for name, p in self.model.named_parameters():
            if name.startswith("heads."):
                route_name = name.split(".", 2)[1]
                p.requires_grad = (route_name in active)
        for m in (self.model.enc_L, self.model.enc_N, self.model.enc_I):
            for p in m.parameters():
                p.requires_grad = True

    def _train_phase(self, routes_trainable: Sequence[str], n_epochs: int, phase: str):
        self._set_trainable_heads(routes_trainable)

        for ep in range(1, n_epochs + 1):
            self.model.train()
            for batch in self.loaders["train"]:
                self.opt.zero_grad(set_to_none=True)

                loss, per_route_loss = self._forward_batch(batch, phase)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.opt.step()

                self.ema = self.ema * self.cfg["ema_beta"] + per_route_loss.detach() * (1.0 - self.cfg["ema_beta"])

            print(f"{phase.upper()} {routes_trainable} | epoch {ep}/{n_epochs} | loss {loss.item():.4f}")

    def train(self):
        print(" Stage 1 (uni)…")
        self._train_phase(["L", "N", "I"], self.cfg["epochs"]["uni"], phase="uni")
        print(" Stage 2 (bi)…")
        self._train_phase(["LN", "LI", "NI"], self.cfg["epochs"]["bi"], phase="bi")
        print(" Stage 3 (tri)…")
        self._train_phase(["LNI"], self.cfg["epochs"]["tri"], phase="tri")
        torch.save(self.model.state_dict(), self.out / "famepp_final.pt")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ys, ps = [], []
        for batch in self.loaders["test"]:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # At test time, use the router’s full fusion (all routes active, no stop-grad)
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
    parser.add_argument("--cfg", type=Path, help="JSON file with hyper-parameters")
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    if args.cfg and args.cfg.exists():
        cfg.update(json.loads(args.cfg.read_text()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, args.out, device)
    trainer.train()
    trainer.evaluate()
