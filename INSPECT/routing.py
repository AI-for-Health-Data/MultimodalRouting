from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn

__all__ = ["MMRouting", "InteractionAttributor"]


class MMRouting(nn.Module):
    """
    Deterministic fusion of seven modality‐interaction routes based on their losses.

    Keeps buffers for:
      • _route_weights (7 routes: L, N, I, LN, LI, NI, LNI)
      • _block_weights (3 blocks: uni-, bi-, tri-modal)

    At each forward pass you provide per-route losses; it computes softmax(-alpha * loss)
    to get route_weights, groups them into blocks (mean within uni/bi/tri), softmaxes again,
    and then fuses logits accordingly.
    """
    ROUTE_LABELS: List[str] = ["L", "N", "I", "LN", "LI", "NI", "LNI"]
    BLOCKS: Dict[str, Tuple[int, ...]] = {
        "uni": (0, 1, 2),
        "bi":  (3, 4, 5),
        "tri": (6,),
    }

    def __init__(self, *, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha: float = alpha
        # Buffers (not parameters) to store last‐used weights
        self.register_buffer("_route_weights", torch.full((7,), 1/7))
        self.register_buffer("_block_weights", torch.full((3,), 1/3))

    @staticmethod
    def _softmax_neg_loss(losses: Tensor, alpha: float) -> Tensor:
        """
        Compute normalized weights ∝ exp(-alpha * loss) over the last dimension.
        """
        scaled = torch.exp(-alpha * losses)
        return scaled / scaled.sum(dim=-1, keepdim=True)

    def _compute_route_weights(self, route_losses: Tensor) -> Tensor:
        """Softmax over the 7 route losses."""
        return self._softmax_neg_loss(route_losses, self.alpha)

    def _compute_block_weights(self, route_losses: Tensor) -> Tensor:
        """
        Aggregate losses within each block (uni/bi/tri), then softmax.
        """
        block_losses = torch.stack([
            route_losses[self.BLOCKS[block]].mean()
            for block in ("uni", "bi", "tri")
        ])
        return self._softmax_neg_loss(block_losses, self.alpha)

    def forward(
        self,
        route_logits: Tensor,
        route_losses: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if route_logits.size(1) != 7:
            raise ValueError(f"Expected 7 routes (got {route_logits.size(1)})")
        if route_losses.numel() != 7:
            raise ValueError(f"Expected route_losses.numel()==7 (got {route_losses.numel()})")

        route_losses = route_losses.to(route_logits.device)

        # compute & store weights
        route_w = self._compute_route_weights(route_losses)
        block_w = self._compute_block_weights(route_losses)
        self._route_weights.copy_(route_w.detach())
        self._block_weights.copy_(block_w.detach())

        # weight each route’s logit 
        weighted = route_logits * route_w.view(1, -1, 1)

        # sum within uni, bi, tri groups → each (B,C)
        uni  = weighted[:, self.BLOCKS["uni"],  :].sum(dim=1)
        bi   = weighted[:, self.BLOCKS["bi"],   :].sum(dim=1)
        tri  = weighted[:, self.BLOCKS["tri"],  :].sum(dim=1)

        # stack → (B,3,C), weight by block_w, then sum → (B,C)
        blocks = torch.stack([uni, bi, tri], dim=1) * block_w.view(1, -1, 1)
        final_logits = blocks.sum(dim=1)

        return final_logits, route_w, block_w

    @property
    def route_weights(self) -> Tensor:
        """Last-computed route weights (length 7)."""
        return self._route_weights.clone()

    @property
    def block_weights(self) -> Tensor:
        """Last-computed block weights (length 3)."""
        return self._block_weights.clone()


class InteractionAttributor:
    """
    Post-hoc attribution of unique (UC), bi-modal (BI), and tri-modal (TI)
    interactions via Monte Carlo permutation.

    Given features L, N, I and a model f(L,N,I)→logits, estimates:
      UC = unique contribution of each modality
      BI = average pairwise interaction residual
      TI = final remaining interaction
    """

    def __init__(
        self,
        f: Callable[[Tensor, Tensor, Tensor], Tensor],
        *,
        n_mc: int = 20,
        device: Optional[torch.device] = None,
    ):
        self.f = f
        self.n_mc: int = n_mc
        self.device = device or torch.device("cpu")

    def _expectation(
        self,
        L: Tensor,
        N: Tensor,
        I: Tensor,
        hold: str
    ) -> Tensor:
        """
        Monte Carlo estimate of E[f] when holding one modality fixed
        and permuting the other two.
        hold must be "L", "N", or "I".
        """
        B = L.size(0)
        acc = 0.0
        for _ in range(self.n_mc):
            if hold == "L":
                v = N[torch.randperm(B, device=N.device)]
                a = I[torch.randperm(B, device=I.device)]
                acc = acc + self.f(L, v, a)
            elif hold == "N":
                t = L[torch.randperm(B, device=L.device)]
                a = I[torch.randperm(B, device=I.device)]
                acc = acc + self.f(t, N, a)
            elif hold == "I":
                t = L[torch.randperm(B, device=L.device)]
                v = N[torch.randperm(B, device=N.device)]
                acc = acc + self.f(t, v, I)
            else:
                raise ValueError(f"Unknown hold key: {hold!r}")
        return acc / self.n_mc

    def compute_uc_bi_ti(
        self,
        L: Tensor,
        N: Tensor,
        I: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns three tensors (same shape as f’s output):
          UC — unique contributions
          BI — pairwise interaction residual (averaged)
          TI — triple interaction residual
        """
        # expectations holding each modality
        f_Lva = self._expectation(L, N, I, hold="L")
        f_tNa = self._expectation(L, N, I, hold="N")
        f_tvI = self._expectation(L, N, I, hold="I")

        # full permutation (permute all three)
        B = L.size(0)
        acc = 0.0
        for _ in range(self.n_mc):
            t = L[torch.randperm(B, device=L.device)]
            v = N[torch.randperm(B, device=N.device)]
            a = I[torch.randperm(B, device=I.device)]
            acc = acc + self.f(t, v, a)
        f_tva = acc / self.n_mc

        UC = f_Lva + f_tNa + f_tvI - 2 * f_tva

        # raw logits
        logits = self.f(L, N, I)
        # pairwise interaction = residual after removing UC, averaged
        BI = (logits - UC).mean(dim=0, keepdim=False)

        # triple interaction = leftover
        TI = logits - UC - BI
        return UC, BI, TI

    def from_batch(
        self,
        batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Extracts 'lab_feats', 'note_feats', 'image_feats' from batch,
        moves them to self.device, and computes UC/BI/TI.
        """
        L = batch["lab_feats"].to(self.device)
        N = batch["note_feats"].to(self.device)
        I = batch["image_feats"].to(self.device)
        return self.compute_uc_bi_ti(L, N, I)
