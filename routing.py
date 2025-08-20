from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MMRouting", "InteractionAttributor"]


# Small MLP helper used for the trainable gate
def _mlp(in_dim: int, out_dim: int, hidden: int = 256, p: float = 0.10) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden, bias=True),
        nn.GELU(),
        nn.Dropout(p),
        nn.Linear(hidden, out_dim, bias=True),
    )

# Trainable multimodal router (per-instance gating) with sMRO-style fusion
class MMRouting(nn.Module):

    ROUTE_LABELS: List[str] = ["L", "N", "I", "LN", "LI", "NI", "LNI"]
    BLOCKS: Dict[str, Tuple[int, ...]] = {"uni": (0, 1, 2), "bi": (3, 4, 5), "tri": (6,)}

    def __init__(
        self,
        *,
        feat_dim: int,              
        gate_hidden: int = 256,
        dropout_p: float = 0.10,
        strict_freeze_gate: bool = False,  
    ) -> None:
        super().__init__()

        # Per-instance gating heads over the shared context x = [L|N|I]
        self.route_gate = _mlp(feat_dim, 7, hidden=gate_hidden, p=dropout_p)
        self.block_gate = _mlp(feat_dim, 3, hidden=gate_hidden, p=dropout_p)

        self.strict_freeze_gate = strict_freeze_gate

        # Buffers for logging: batch-mean weights of last forward
        self.register_buffer("_route_w_mean", torch.full((7,), 1.0 / 7.0))
        self.register_buffer("_block_w_mean", torch.full((3,), 1.0 / 3.0))

        # Hold last per-block contributions
        self._last_uni: Optional[torch.Tensor] = None
        self._last_bi:  Optional[torch.Tensor] = None
        self._last_tri: Optional[torch.Tensor] = None

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return F.softmax(logits, dim=-1)
        if mask.dim() == 1:
            mask = mask.view(1, -1).expand_as(logits)
        # Use -1e9 to zero out masked entries after softmax
        masked = logits.masked_fill(mask == 0, -1e9)
        return F.softmax(masked, dim=-1)

    @staticmethod
    def _stage_masks(stage: Optional[str]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if stage is None or stage == "eval":
            return None, None
        s = stage.lower()
        if s == "uni":
            route_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0], dtype=torch.long)
            block_mask = torch.tensor([1, 0, 0], dtype=torch.long)
        elif s == "bi":
            route_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.long)
            block_mask = torch.tensor([1, 1, 0], dtype=torch.long)
        elif s == "tri":
            route_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
            block_mask = torch.tensor([1, 1, 1], dtype=torch.long)
        else:
            raise ValueError(f"Invalid stage '{stage}'")
        return route_mask, block_mask

    def forward(
        self,
        route_logits: torch.Tensor,   
        L: torch.Tensor,              
        N: torch.Tensor,              
        I: torch.Tensor,              
        *,
        stage: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if route_logits.ndim != 3 or route_logits.shape[1] != 7:
            raise ValueError(f"route_logits must be [B,7,C], got {tuple(route_logits.shape)}")

        B, _, C = route_logits.shape
        x = torch.cat([L, N, I], dim=-1)  

        # Build masks and move to correct device
        rmask, bmask = self._stage_masks(stage)
        device = route_logits.device
        if rmask is not None:
            rmask = rmask.to(device)
        if bmask is not None:
            bmask = bmask.to(device)

        # Trainable, per-instance weights
        route_w = self._masked_softmax(self.route_gate(x), rmask)  
        block_w = self._masked_softmax(self.block_gate(x), bmask)  

        # Save batch-mean weights for logging
        self._route_w_mean.copy_(route_w.detach().mean(dim=0))
        self._block_w_mean.copy_(block_w.detach().mean(dim=0))

        # Per-block contributions as weighted sums of route logits
        weighted = route_logits * route_w.view(B, 7, 1)
        uni = weighted[:, self.BLOCKS["uni"], :].sum(dim=1)  
        bi  = weighted[:, self.BLOCKS["bi"],  :].sum(dim=1)  
        tri = weighted[:, self.BLOCKS["tri"], :].sum(dim=1)  

        self._last_uni, self._last_bi, self._last_tri = uni, bi, tri

        # Block weights
        w_uni = block_w[:, 0].view(B, 1)
        w_bi  = block_w[:, 1].view(B, 1)
        w_tri = block_w[:, 2].view(B, 1)

        # sMRO-style fusion with stop-gradient at block boundaries
        if stage is None or stage == "eval":
            fused = w_uni * uni + w_bi * bi + w_tri * tri
        else:
            s = stage.lower()
            if s == "uni":
                fused = w_uni * uni
            elif s == "bi":
                if self.strict_freeze_gate:
                    fused = w_uni.detach() * uni.detach() + w_bi * bi
                else:
                    fused = w_uni * uni.detach() + w_bi * bi
            elif s == "tri":
                if self.strict_freeze_gate:
                    fused = (
                        w_uni.detach() * uni.detach()
                        + w_bi.detach()  * bi.detach()
                        + w_tri * tri
                    )
                else:
                    fused = (
                        w_uni * uni.detach()
                        + w_bi  * bi.detach()
                        + w_tri * tri
                    )
            else:
                raise ValueError(f"Invalid stage '{stage}'")

        return fused, route_w, block_w

    @property
    def route_weights_mean(self) -> torch.Tensor:
        """Mean route weights over the most recent batch: [7]."""
        return self._route_w_mean.clone()

    @property
    def block_weights_mean(self) -> torch.Tensor:
        """Mean block weights over the most recent batch: [3] (uni, bi, tri)."""
        return self._block_w_mean.clone()

    @property
    def last_uni(self) -> Optional[torch.Tensor]:
        """Last per-batch unimodal contribution logits [B, C]."""
        return self._last_uni

    @property
    def last_bi(self) -> Optional[torch.Tensor]:
        """Last per-batch bimodal contribution logits [B, C]."""
        return self._last_bi

    @property
    def last_tri(self) -> Optional[torch.Tensor]:
        """Last per-batch trimodal contribution logits [B, C]."""
        return self._last_tri


#  Post-hoc interaction attribution 
class InteractionAttributor:
    def __init__(self, f: Callable[..., torch.Tensor], *, n_mc: int = 20, device=None):
        self.f = f
        self.n_mc = n_mc
        self.device = device or torch.device("cpu")

    @staticmethod
    def _permute(x: torch.Tensor) -> torch.Tensor:
        idx = torch.randperm(x.size(0), device=x.device)
        return x[idx]

    def _E_all(self, L, N, I):
        # E_{t,v,a} f(t,v,a)
        acc = 0
        for _ in range(self.n_mc):
            acc += self.f(self._permute(L), self._permute(N), self._permute(I))
        return acc / self.n_mc

    def _E_hold_L(self, L, N, I):
        # E_{v,a} f(L, v, a)
        acc = 0
        for _ in range(self.n_mc):
            acc += self.f(L, self._permute(N), self._permute(I))
        return acc / self.n_mc

    def _E_hold_N(self, L, N, I):
        # E_{t,a} f(t, N, a)
        acc = 0
        for _ in range(self.n_mc):
            acc += self.f(self._permute(L), N, self._permute(I))
        return acc / self.n_mc

    def _E_hold_I(self, L, N, I):
        # E_{t,v} f(t, v, I)
        acc = 0
        for _ in range(self.n_mc):
            acc += self.f(self._permute(L), self._permute(N), I)
        return acc / self.n_mc

    def _E_keep_LN(self, L, N, I):
        # E_{a} f(L, N, a)  (permute only I)
        acc = 0
        for _ in range(self.n_mc):
            acc += self.f(L, N, self._permute(I))
        return acc / self.n_mc

    def _E_keep_LI(self, L, N, I):
        # E_{v} f(L, v, I)  (permute only N)
        acc = 0
        for _ in range(self.n_mc):
            acc += self.f(L, self._permute(N), I)
        return acc / self.n_mc

    def _E_keep_NI(self, L, N, I):
        # E_{t} f(t, N, I)  (permute only L)
        acc = 0
        for _ in range(self.n_mc):
            acc += self.f(self._permute(L), N, I)
        return acc / self.n_mc

    @torch.no_grad()
    def compute_uc_bi_ti(self, L: torch.Tensor, N: torch.Tensor, I: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. Grand mean
        G = self._E_all(L, N, I)                         

        # 2. Main effects
        EL_va = self._E_hold_L(L, N, I)                  
        EN_ta = self._E_hold_N(L, N, I)                  
        EI_tv = self._E_hold_I(L, N, I)                  

        u_L = EL_va - G
        u_N = EN_ta - G
        u_I = EI_tv - G
        UC  = u_L + u_N + u_I

        # 3. Pairwise interactions
        ELN_a = self._E_keep_LN(L, N, I)                 
        ELI_v = self._E_keep_LI(L, N, I)                 
        ENI_t = self._E_keep_NI(L, N, I)                 

        p_LN = ELN_a - G - u_L - u_N
        p_LI = ELI_v - G - u_L - u_I
        p_NI = ENI_t - G - u_N - u_I
        BI   = p_LN + p_LI + p_NI

        # 4. Trimodal (residual)
        f_obs = self.f(L, N, I)
        TI    = f_obs - (G + UC + BI)

        return UC, BI, TI

    def from_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        L = batch["lab_feats"].to(self.device)
        N = batch["note_feats"].to(self.device)
        I = batch["image_feats"].to(self.device)
        return self.compute_uc_bi_ti(L, N, I)

