from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from routing import MMRouting
from models.encoders import BEHRTLabEncoder, BioClinBERTEncoder, ImageCXREncoder
from models.routes import RouteMLP
from data.icustay_dataset import ICUStayDataset


# Hyper-parameters
DEFAULTS: Dict[str, Any] = {
    "batch_size": 32,
    "hidden": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": {"uni": 4, "bi": 4, "tri": 8},
    "grad_clip": 1.0,
    # Fairness
    "lambda_eddi": 0.5,              
    "fairness_detach_router": True,  
}

# Model
class FAMEPlusPlus(nn.Module):
    TASKS = ("mortality", "pe", "ph")
    ROUTES = ("L", "N", "I", "LN", "LI", "NI", "LNI")

    def __init__(self, hidden: int):
        super().__init__()
        # Encoders (shared across all routes)
        self.enc_L = BEHRTLabEncoder(out_dim=hidden)
        self.enc_N = BioClinBERTEncoder(out_dim=hidden)
        self.enc_I = ImageCXREncoder(out_dim=hidden)

        # Route heads (unique per route)
        self.heads = nn.ModuleDict({
            r: RouteMLP(in_dim=hidden * len(r), out_dim=len(self.TASKS))
            for r in self.ROUTES
        })

        # Trainable per-instance gate over 7 routes + 3 blocks 
        self.router = MMRouting(feat_dim=hidden * 3)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Encoders
        L = self.enc_L(batch["lab_feats"], batch.get("demo", None))  
        N = self.enc_N(batch["note_ids"], batch["note_attn"])        
        I = self.enc_I(batch["image"])                               

        # Per-route concatenations
        Z = {
            "L": L, "N": N, "I": I,
            "LN": torch.cat([L, N], dim=-1),
            "LI": torch.cat([L, I], dim=-1),
            "NI": torch.cat([N, I], dim=-1),
            "LNI": torch.cat([L, N, I], dim=-1),
        }

        route_logits = torch.stack([self.heads[r](Z[r]) for r in self.ROUTES], dim=1)  # [B,7,C]
        return route_logits, L, N, I

# Differentiable fairness 
def fairness_loss_soft_eq_odds(
    logits: torch.Tensor,           
    labels: torch.Tensor,           
    sens: torch.Tensor,             
    eps: float = 1e-6,
) -> torch.Tensor:

    B, C = logits.shape
    p = torch.sigmoid(logits)  # [B,C]

    unique_groups = torch.unique(sens)
    G = unique_groups.numel()

    # One-hot group mask M: [B,G]
    M = torch.stack([(sens == g).float() for g in unique_groups], dim=1)  

    # Expand for class-wise aggregation
    M_bg1 = M.unsqueeze(2)         
    y_b1c = labels.unsqueeze(1)    
    p_b1c = p.unsqueeze(1)         

    # Positives/negatives per group & class
    pos_gc = (M_bg1 * y_b1c).sum(dim=0)                 
    neg_gc = (M_bg1 * (1.0 - y_b1c)).sum(dim=0)         

    # Soft counts
    tp_gc = (M_bg1 * y_b1c * p_b1c).sum(dim=0)          
    fp_gc = (M_bg1 * (1.0 - y_b1c) * p_b1c).sum(dim=0)  

    tpr_gc = tp_gc / (pos_gc + eps)                      
    fpr_gc = fp_gc / (neg_gc + eps)                      

    # Disparity = variance across groups (unbiased=False to keep gradients stable)
    var_tpr_c = tpr_gc.var(dim=0, unbiased=False)        
    var_fpr_c = fpr_gc.var(dim=0, unbiased=False)        

    return (var_tpr_c + var_fpr_c).mean()                

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
        self.model = FAMEPlusPlus(cfg["hidden"]).to(device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        self.crit = nn.BCEWithLogitsLoss()

    def _set_trainable_heads(self, routes_trainable: Sequence[str]) -> None:
        active = set(routes_trainable)
        # Route heads: only those for this stage
        for name, p in self.model.named_parameters():
            if name.startswith("heads."):
                route_name = name.split(".", 2)[1]
                p.requires_grad = (route_name in active)
        # Encoders: always trainable across stages
        for m in (self.model.enc_L, self.model.enc_N, self.model.enc_I):
            for p in m.parameters():
                p.requires_grad = True
        # Router (gate): trainable in all stages
        for p in self.model.router.parameters():
            p.requires_grad = True

    @staticmethod
    def _block_sums_with_weights(
        route_logits: torch.Tensor,      
        route_w: torch.Tensor,           
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (uni, bi, tri) = weighted sums of route logits per block."""
        B = route_logits.size(0)
        weighted = route_logits * route_w.view(B, 7, 1)
        uni = weighted[:, [0, 1, 2], :].sum(dim=1)   
        bi  = weighted[:, [3, 4, 5], :].sum(dim=1)   
        tri = weighted[:, [6],      :].sum(dim=1)    
        return uni, bi, tri

    def _fused_with_stage_and_detach(
        self,
        route_logits: torch.Tensor,      
        route_w: torch.Tensor,           
        block_w: torch.Tensor,           
        phase: str,
        detach_gate: bool,               
        stopgrad_lower_blocks: bool,     
    ) -> torch.Tensor:
        B = route_logits.size(0)

        rw = route_w.detach() if detach_gate else route_w
        bw = block_w.detach() if detach_gate else block_w

        uni, bi, tri = self._block_sums_with_weights(route_logits, rw)

        w_uni = bw[:, 0].view(B, 1)
        w_bi  = bw[:, 1].view(B, 1)
        w_tri = bw[:, 2].view(B, 1)

        phase = (phase or "eval").lower()
        if phase == "uni":
            fused = w_uni * uni
        elif phase == "bi":
            fused = w_uni * (uni.detach() if stopgrad_lower_blocks else uni) + w_bi * bi
        elif phase == "tri":
            uni_term = uni.detach() if stopgrad_lower_blocks else uni
            bi_term  = bi.detach()  if stopgrad_lower_blocks else bi
            fused = w_uni * uni_term + w_bi * bi_term + w_tri * tri
        elif phase == "eval":
            fused = w_uni * uni + w_bi * bi + w_tri * tri
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return fused

    def _forward_batch(self, batch: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Route scores + encoder embeddings for gating
        route_logits, L, N, I = self.model(batch)  

        # Trainable gate + sMRO fusion (stop-grad in main loss)
        fused_main, route_w, block_w = self.model.router(
            route_logits, L, N, I, stage=phase
        )  

        # Prediction loss
        y = batch["labels"].float()
        pred_loss = self.crit(fused_main, y)

        # Differentiable fairness loss (optional)
        l_fair = torch.tensor(0.0, device=self.device)
        lam = float(self.cfg.get("lambda_eddi", 0.0))
        if lam > 0.0 and "sens" in batch:
            fused_for_fair = self._fused_with_stage_and_detach(
                route_logits=route_logits,
                route_w=route_w,
                block_w=block_w,
                phase=phase,
                detach_gate=bool(self.cfg.get("fairness_detach_router", True)),
                stopgrad_lower_blocks=True,   # keep sMRO semantics for fairness as well
            )
            l_fair = fairness_loss_soft_eq_odds(
                logits=fused_for_fair, labels=y, sens=batch["sens"]
            )

        loss = pred_loss + lam * l_fair
        return loss

    def _train_phase(self, routes_trainable: Sequence[str], n_epochs: int, phase: str):
        self._set_trainable_heads(routes_trainable)

        for ep in range(1, n_epochs + 1):
            self.model.train()
            for batch in self.loaders["train"]:
                self.opt.zero_grad(set_to_none=True)
                loss = self._forward_batch(batch, phase)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.opt.step()
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
            route_logits, L, N, I = self.model(batch)
            fused, _, _ = self.model.router(route_logits, L, N, I, stage="eval")
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
