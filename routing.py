from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional
import torch
import torch.nn as nn

__all__ = ["MMRouting", "InteractionAttributor"]

# 1. Deterministic routing layer
class MMRouting(nn.Module):
    """Performance‑based fusion of seven interaction routes."""

    ROUTE_LABELS: List[str] = ["L", "N", "I", "LN", "LI", "NI", "LNI"]
    BLOCKS: Dict[str, Tuple[int, ...]] = {
        "uni": (0, 1, 2),
        "bi": (3, 4, 5),
        "tri": (6,),
    }

    def __init__(self, *, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer("_route_weights", torch.full((7,), 1.0 / 7.0))
        self.register_buffer("_block_weights", torch.full((3,), 1.0 / 3.0))
        
        # Hold last per-block contributions
        self._last_uni: Optional[torch.Tensor] = None
        self._last_bi:  Optional[torch.Tensor] = None
        self._last_tri: Optional[torch.Tensor] = None

    @staticmethod
    def _softmax_neg_loss(losses: torch.Tensor, alpha: float) -> torch.Tensor:
        scaled = torch.exp(-alpha * losses)
        return scaled / (scaled.sum(dim=-1, keepdim=False) + 1e-8)

    def _compute_route_weights(self, route_losses: torch.Tensor) -> torch.Tensor:
        return self._softmax_neg_loss(route_losses, self.alpha)

    def _compute_block_weights(self, route_losses: torch.Tensor) -> torch.Tensor:
        # Average route losses within each block, then softmax(-alpha * loss)
        block_losses = torch.stack([
            route_losses[list(self.BLOCKS[b])].mean() for b in ("uni", "bi", "tri")
        ])
        return self._softmax_neg_loss(block_losses, self.alpha)

    def _within_block_contribs(
        self,
        route_logits: torch.Tensor,  
        route_w: torch.Tensor        
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-block contributions (uni/bi/tri) as weighted sums of their routes.
        """
        B, R, C = route_logits.shape
        if R != 7:
            raise ValueError(f"Expected 7 routes; got {R}")

        # Reshape for broadcasting weights over [B, routes, C]
        rw = route_w.view(1, -1, 1)  
        weighted = route_logits * rw

        uni = weighted[:, self.BLOCKS["uni"], :].sum(dim=1)  
        bi  = weighted[:, self.BLOCKS["bi"],  :].sum(dim=1)  
        tri = weighted[:, self.BLOCKS["tri"], :].sum(dim=1)  

        return uni, bi, tri

    def forward(
        self,
        route_logits: torch.Tensor,   
        route_losses: torch.Tensor,   
        *,
        stage: Optional[str] = None   
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if route_logits.ndim != 3 or route_logits.shape[1] != 7:
            raise ValueError(f"route_logits must be [B,7,C], got {tuple(route_logits.shape)}")
        if route_losses.numel() != 7:
            raise ValueError("route_losses must have 7 elements.")

        # Treat incoming losses as constants for gating (no grad through weights)
        route_losses = route_losses.detach().to(route_logits)

        # 1. Compute route weights (within-block)
        route_w = self._compute_route_weights(route_losses)  
        self._route_weights.copy_(route_w.detach())

        # 2. Compute block weights (for logging; used only in eval fusion)
        block_w = self._compute_block_weights(route_losses)  
        self._block_weights.copy_(block_w.detach())

        # 3. Per-block contributions (using route weights only)
        uni, bi, tri = self._within_block_contribs(route_logits, route_w)
        self._last_uni, self._last_bi, self._last_tri = uni, bi, tri

        # 4. Fuse depending on mode
        if stage is None or stage == "eval":
            # Backward-compatible eval: apply block weights across [uni,bi,tri]
            bw = block_w.view(1, -1, 1)  
            final = torch.stack([uni, bi, tri], dim=1) * bw  
            final_logits = final.sum(dim=1)                 
            return final_logits, route_w, block_w

        stage = stage.lower()
        if stage not in {"uni", "bi", "tri"}:
            raise ValueError(f"Invalid stage '{stage}'. Use None/'eval' or 'uni'/'bi'/'tri'.")

        if stage == "uni":
            final_logits = uni
        elif stage == "bi":
            final_logits = uni.detach() + bi
        else:  
            final_logits = (uni + bi).detach() + tri

        return final_logits, route_w, block_w

    @property
    def route_weights(self) -> torch.Tensor:
        return self._route_weights.clone()

    @property
    def block_weights(self) -> torch.Tensor:
        return self._block_weights.clone()

    @property
    def last_uni(self) -> Optional[torch.Tensor]:
        return self._last_uni

    @property
    def last_bi(self) -> Optional[torch.Tensor]:
        return self._last_bi

    @property
    def last_tri(self) -> Optional[torch.Tensor]:
        return self._last_tri

# 2) Post-hoc interaction attribution (unchanged API)
class InteractionAttributor:
    """
    Model-agnostic, post-hoc attribution of unimodal (UC), bimodal (BI), and
    trimodal (TI) interactions via Monte-Carlo shuffling/marginalization.

    Expect f(L, N, I) -> logits  (same shape as model outputs).
    """

    def __init__(self, f: Callable[..., torch.Tensor], *, n_mc: int = 20, device=None):
        self.f = f
        self.n_mc = n_mc
        self.device = device or torch.device("cpu")

    @staticmethod
    def _permute_except(x: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
        """Return a tensor where rows NOT selected by keep_idx are permuted."""
        idx = torch.arange(x.size(0), device=x.device)
        perm = idx[~keep_idx].clone()
        perm = perm[torch.randperm(perm.numel(), device=x.device)]
        idx[~keep_idx] = perm
        return x[idx]

    def _expectation(self, L: torch.Tensor, N: torch.Tensor, I: torch.Tensor,
                     hold: str) -> torch.Tensor:
        B = L.size(0)
        acc = 0
        for _ in range(self.n_mc):
            if hold == "L":
                N_s = N[torch.randperm(B, device=N.device)]
                I_s = I[torch.randperm(B, device=I.device)]
                acc += self.f(L, N_s, I_s)
            elif hold == "N":
                L_s = L[torch.randperm(B, device=L.device)]
                I_s = I[torch.randperm(B, device=I.device)]
                acc += self.f(L_s, N, I_s)
            elif hold == "I":
                L_s = L[torch.randperm(B, device=L.device)]
                N_s = N[torch.randperm(B, device=N.device)]
                acc += self.f(L_s, N_s, I)
            else:
                raise ValueError(hold)
        return acc / self.n_mc

    def compute_uc_bi_ti(
        self,
        L: torch.Tensor,
        N: torch.Tensor,
        I: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return UC, BI, TI tensors (same shape as f's logits).
        UC generalizes EMAP-style unimodal contributions to three modalities.
        """
        #   UC = E_{v,a} f(L,v,a) + E_{t,a} f(t,N,a) + E_{t,v} f(t,v,I) - 2 E_{t,v,a} f(t,v,a)
        f_Lva = self._expectation(L, N, I, hold="L")
        f_tNa = self._expectation(L, N, I, hold="N")
        f_tvI = self._expectation(L, N, I, hold="I")

        B = L.size(0)
        acc_full = 0
        for _ in range(self.n_mc):
            L_s = L[torch.randperm(B, device=L.device)]
            N_s = N[torch.randperm(B, device=N.device)]
            I_s = I[torch.randperm(B, device=I.device)]
            acc_full += self.f(L_s, N_s, I_s)
        f_tva = acc_full / self.n_mc

        UC = f_Lva + f_tNa + f_tvI - 2 * f_tva

        # BI: average over holding one modality fixed, removing UC
        def bi_term(keep: str):
            logits = self.f(L, N, I)
            uc_logits = (
                self._expectation(L, N, I, hold="L")
                + self._expectation(L, N, I, hold="N")
                + self._expectation(L, N, I, hold="I")
                - 2 * f_tva
            )
            return logits - uc_logits

        BI = (bi_term("L") + bi_term("N") + bi_term("I")) / 3.0
        TI = self.f(L, N, I) - UC - BI
        return UC, BI, TI

    def from_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        L = batch["lab_feats"].to(self.device)
        N = batch["note_feats"].to(self.device)
        I = batch["image_feats"].to(self.device)
        return self.compute_uc_bi_ti(L, N, I)
