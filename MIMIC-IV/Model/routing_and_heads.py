import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Sequence, Optional
from env_config import CFG, ROUTES, BLOCKS, DEVICE


class _MLP(nn.Module):
    """Generic MLP: [in] -> hidden... -> out, with LayerNorm + GELU + Dropout between layers."""
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[Sequence[int]] = None, p_drop: float = 0.1):
        super().__init__()
        hidden = list(hidden) if hidden is not None else [4 * out_dim, 2 * out_dim]
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [
                nn.LayerNorm(dims[i]),
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
                nn.Dropout(p_drop),
            ]
        layers += [
            nn.LayerNorm(dims[-2]),
            nn.Linear(dims[-2], dims[-1]),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairwiseFusion(nn.Module):
    """
    Multimodal MLP for bimodal fusion.
      - feature_mode='concat': x = [zA, zB]           -> MLP(in=2d, out=d)
      - feature_mode='rich':   x = [zA,zB,zA*zB,|Δ|]  -> MLP(in=4d, out=d)
    """
    def __init__(self, d: int, hidden: Optional[Sequence[int]] = None, p_drop: float = 0.1,
                 feature_mode: str = "rich"):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        self.d = d
        self.feature_mode = feature_mode
        in_dim = 2 * d if feature_mode == "concat" else 4 * d
        self.mlp = _MLP(in_dim, d, hidden=hidden, p_drop=p_drop)
        # small residual to help stability
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "concat":
            x = torch.cat([za, zb], dim=-1)
        else:  # 'rich'
            had = za * zb
            diff = (za - zb).abs()
            x = torch.cat([za, zb, had, diff], dim=-1)
        h = self.mlp(x)
        base = 0.5 * (za + zb)
        return h + self.res_scale * base


class TrimodalFusion(nn.Module):

    def __init__(self, d: int, hidden: Optional[Sequence[int]] = None, p_drop: float = 0.1,
                 feature_mode: str = "rich"):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        self.d = d
        self.feature_mode = feature_mode
        in_dim = 3 * d if feature_mode == "concat" else 7 * d
        self.mlp = _MLP(in_dim, d, hidden=hidden, p_drop=p_drop)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "concat":
            x = torch.cat([zL, zN, zI], dim=-1)
        else:  # 'rich'
            zLN = zL * zN
            zLI = zL * zI
            zNI = zN * zI
            zLNI = zL * zN * zI
            x = torch.cat([zL, zN, zI, zLN, zLI, zNI, zLNI], dim=-1)
        h = self.mlp(x)
        base = (zL + zN + zI) / 3.0
        return h + self.res_scale * base


def build_fusions(d: int,
                  p_drop: float = 0.1,
                  feature_mode: str = "rich",
                  hidden: Optional[Sequence[int]] = None) -> Dict[str, nn.Module]:
    """
    Build all fusion modules (bimodal + trimodal) using Multimodal-MLP blocks.

    Args:
      d: shared embedding size
      p_drop: dropout prob inside MLPs
      feature_mode: 'rich' (default) or 'concat'
      hidden: list of hidden sizes for MLPs, defaults to [4*d, 2*d]
    """
    return {
        "LN":  PairwiseFusion(d=d, hidden=hidden, p_drop=p_drop, feature_mode=feature_mode).to(DEVICE),
        "LI":  PairwiseFusion(d=d, hidden=hidden, p_drop=p_drop, feature_mode=feature_mode).to(DEVICE),
        "NI":  PairwiseFusion(d=d, hidden=hidden, p_drop=p_drop, feature_mode=feature_mode).to(DEVICE),
        "LNI": TrimodalFusion(d=d, hidden=hidden, p_drop=p_drop, feature_mode=feature_mode).to(DEVICE),
    }


@torch.no_grad()
def _safe_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone() if x.requires_grad is False else x


def make_route_inputs(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
) -> Dict[str, torch.Tensor]:
    """
    Build d-dim inputs for all 7 routes using unimodal embeddings and fusion modules.
    z: {"L":[B,d], "N":[B,d], "I":[B,d]}
    """
    zL, zN, zI = z["L"], z["N"], z["I"]
    return {
        "L":   _safe_clone(zL),
        "N":   _safe_clone(zN),
        "I":   _safe_clone(zI),
        "LN":  fusion["LN"](zL, zN),
        "LI":  fusion["LI"](zL, zI),
        "NI":  fusion["NI"](zN, zI),
        "LNI": fusion["LNI"](zL, zN, zI),
    }


class RouteHead(nn.Module):
    """
    Small MLP mapping a route-specific d-dim embedding to task logits.
    Default n_tasks=1 for single-task training.
    """
    def __init__(self, d_in: int, n_tasks: int = 1, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 2 * d_in),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(2 * d_in, n_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_route_heads(d: int, n_tasks: int = 1, p_drop: float = 0.1) -> Dict[str, 'RouteHead']:
    """Helper to construct all 7 route heads with consistent n_tasks."""
    return {
        r: RouteHead(d_in=d, n_tasks=n_tasks, p_drop=p_drop).to(DEVICE)
        for r in ROUTES
    }


class LearnedGateRouter(nn.Module):
    """
    Gating network over routes/blocks.
    Inputs:
      z_dict: {"L":[B,d], "N":[B,d], "I":[B,d]}
      route_logits_dict: {route:[B,C]} for routes in ROUTES
      masks (optional): {"L":[B,1], "N":[B,1], "I":[B,1]} (1=present, 0=missing)
    Outputs:
      ylogits:     [B,C]
      route_w:     [B,C,7] (masked softmax over routes)
      block_w:     [B,C,3] (masked softmax over blocks)
      block_logits:[B,3,C] logits for (uni, bi, tri)
    """
    def __init__(
        self,
        routes: List[str],
        blocks: Dict[str, List[str]],
        d: int,
        n_tasks: int = 1,
        hidden: int = 1024,
        p_drop: float = 0.1,
        use_masks: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert len(routes) == 7, "routes must be ['L','N','I','LN','LI','NI','LNI']"
        assert set(blocks.keys()) == {"uni", "bi", "tri"}
        self.routes = routes
        self.blocks = blocks
        self.n_tasks = n_tasks
        self.use_masks = use_masks
        self.temperature = temperature

        gate_in = 3 * d + (3 if use_masks else 0)
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_tasks * (7 + 3)),
        )

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(dtype=logits.dtype)
        keep = (mask.sum(dim=dim, keepdim=True) > 0).to(logits.dtype)
        safe_mask = mask * keep + (1.0 - keep)
        logits = logits + torch.log(safe_mask + 1e-12)
        return torch.softmax(logits, dim=dim)

    def _stack_gate_input(self, z_dict: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor] | None) -> torch.Tensor:
        zL, zN, zI = z_dict["L"], z_dict["N"], z_dict["I"]
        pieces = [zL, zN, zI]
        if self.use_masks:
            if masks is None:
                B = zL.size(0)
                m = torch.ones(B, 3, device=zL.device)
            else:
                m = torch.cat([masks["L"], masks["N"], masks["I"]], dim=1)
            pieces.append(m)
        return torch.cat(pieces, dim=1)

    def _route_availability_mask(self, masks: Dict[str, torch.Tensor] | None, B: int, device) -> torch.Tensor:
        if not self.use_masks:
            return torch.ones(B, 7, device=device)
        if masks is None:
            mL = mN = mI = torch.ones(B, 1, device=device)
        else:
            mL, mN, mI = masks["L"], masks["N"], masks["I"]
        req = {
            "L":   mL,
            "N":   mN,
            "I":   mI,
            "LN":  mL * mN,
            "LI":  mL * mI,
            "NI":  mN * mI,
            "LNI": mL * mN * mI,
        }
        return torch.cat([req[r] for r in self.routes], dim=1).clamp(0, 1)

    def _block_availability_mask(self, route_mask: torch.Tensor) -> torch.Tensor:
        device = route_mask.device
        def block_any(names: List[str]) -> torch.Tensor:
            idx = torch.tensor([self.routes.index(r) for r in names], device=device)
            return (route_mask.index_select(1, idx).sum(dim=1, keepdim=True) > 0).float()
        b_uni = block_any(self.blocks["uni"])
        b_bi  = block_any(self.blocks["bi"])
        b_tri = block_any(self.blocks["tri"])
        return torch.cat([b_uni, b_bi, b_tri], dim=1)

    def forward(
        self,
        z_dict: Dict[str, torch.Tensor],
        route_logits_dict: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = next(iter(route_logits_dict.values())).shape[0]
        C = self.n_tasks
        device = z_dict["L"].device

        # Gate outputs
        g_in = self._stack_gate_input(z_dict, masks)
        raw = self.gate(g_in).view(B, C, 7 + 3) / max(self.temperature, 1e-6)
        route_logits_gate = raw[:, :, :7]
        block_logits_gate = raw[:, :, 7:]

        # Availability masks
        route_mask = self._route_availability_mask(masks, B, device)
        block_mask = self._block_availability_mask(route_mask)

        # Masked softmax
        route_w = self._masked_softmax(route_logits_gate, route_mask.unsqueeze(1).expand(-1, C, -1), dim=2)
        block_w = self._masked_softmax(block_logits_gate, block_mask.unsqueeze(1).expand(-1, C, -1), dim=2)

        # Stack route logits in the same order as self.routes
        R = torch.cat([
            (route_logits_dict[r].unsqueeze(-1) if route_logits_dict[r].dim() == 2
             else route_logits_dict[r].unsqueeze(1).unsqueeze(-1))
            for r in self.routes
        ], dim=2)

        # Block-wise sums
        def sum_block(names: List[str]) -> torch.Tensor:
            idx = torch.tensor([self.routes.index(r) for r in names], device=R.device)
            R_blk = R.index_select(dim=2, index=idx)
            w_blk = route_w.index_select(dim=2, index=idx)
            return (w_blk * R_blk).sum(dim=2)

        y_uni = sum_block(self.blocks["uni"])
        y_bi  = sum_block(self.blocks["bi"])
        y_tri = sum_block(self.blocks["tri"])

        block_logits = torch.stack([y_uni, y_bi, y_tri], dim=1)
        block_logits = block_logits * block_mask.unsqueeze(-1)

        # Final blend across blocks
        y = (block_w.transpose(1, 2) * block_logits).sum(dim=1)
        return y, route_w, block_w, block_logits
