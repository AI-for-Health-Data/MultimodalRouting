from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import CFG, ROUTES, DEVICE



class _MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Optional[Sequence[int]] = None,
        p_drop: float = 0.1,
    ):
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
        layers += [nn.LayerNorm(dims[-2]), nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairwiseFusion(nn.Module):
    def __init__(
        self,
        d: int,
        hidden: Optional[Sequence[int]] = None,
        p_drop: float = 0.1,
        feature_mode: str = "rich",  # {"concat", "rich"}
    ):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        self.feature_mode = feature_mode
        in_dim = 2 * d if feature_mode == "concat" else 4 * d
        self.mlp = _MLP(in_dim, d, hidden=hidden, p_drop=p_drop)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "concat":
            x = torch.cat([za, zb], dim=-1)
        else:
            had = za * zb
            diff = (za - zb).abs()
            x = torch.cat([za, zb, had, diff], dim=-1)
        h = self.mlp(x)
        base = 0.5 * (za + zb)
        return h + self.res_scale * base


class TrimodalFusion(nn.Module):
    def __init__(
        self,
        d: int,
        hidden: Optional[Sequence[int]] = None,
        p_drop: float = 0.1,
        feature_mode: str = "rich",  
    ):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        in_dim = 3 * d if feature_mode == "concat" else 7 * d
        self.mlp = _MLP(in_dim, d, hidden=hidden, p_drop=p_drop)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "concat":
            x = torch.cat([zL, zN, zI], dim=-1)
        else:
            zLN = zL * zN
            zLI = zL * zI
            zNI = zN * zI
            zLNI = zL * zN * zI
            x = torch.cat([zL, zN, zI, zLN, zLI, zNI, zLNI], dim=-1)
        h = self.mlp(x)
        base = (zL + zN + zI) / 3.0
        return h + self.res_scale * base


def build_fusions(
    d: int,
    p_drop: float = 0.1,
    feature_mode: str = "rich",
    hidden: Optional[Sequence[int]] = None,
) -> Dict[str, nn.Module]:
    dev = torch.device(DEVICE)
    return {
        "LN":  PairwiseFusion(d=d, hidden=hidden, p_drop=p_drop, feature_mode=feature_mode).to(dev),
        "LI":  PairwiseFusion(d=d, hidden=hidden, p_drop=p_drop, feature_mode=feature_mode).to(dev),
        "NI":  PairwiseFusion(d=d, hidden=hidden, p_drop=p_drop, feature_mode=feature_mode).to(dev),
        "LNI": TrimodalFusion(d=d, hidden=hidden, p_drop=p_drop, feature_mode=feature_mode).to(dev),
    }


@torch.no_grad()
def _safe_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone() if x.requires_grad is False else x


def make_route_inputs(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
) -> Dict[str, torch.Tensor]:
    """Build embeddings for all 7 routes from unimodal z and fusion blocks."""
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


def build_route_heads(d: int, n_tasks: int = 1, p_drop: float = 0.1) -> Dict[str, RouteHead]:
    dev = torch.device(DEVICE)
    return {r: RouteHead(d_in=d, n_tasks=n_tasks, p_drop=p_drop).to(dev) for r in ROUTES}


def compute_route_logits(
    route_embs: Dict[str, torch.Tensor],
    route_heads: Dict[str, RouteHead],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for r, z in route_embs.items():
        out[r] = route_heads[r](z)
    return out


def route_availability_mask(
    masks: Optional[Dict[str, torch.Tensor]],
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    if masks is None:
        return torch.ones(batch_size, len(ROUTES), device=device)

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
    return torch.cat([req[r] for r in ROUTES], dim=1).clamp(0, 1)


class RouteGateNet(nn.Module):
    """
    Produces per-sample route weights (gates) from unimodal embeddings.
    """
    def __init__(self, d: int, hidden: int = 1024, p_drop: float = 0.1, use_masks: bool = True):
        super().__init__()
        self.use_masks = use_masks
        in_dim = 3 * d 
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, len(ROUTES)),  
        )

    def forward(self, z: Dict[str, torch.Tensor], masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x = torch.cat([z["L"], z["N"], z["I"]], dim=1)  
        logits = self.net(x)                             
        w = torch.softmax(logits, dim=1)                
        if self.use_masks and masks is not None:
            avail = route_availability_mask(masks, batch_size=x.size(0), device=x.device)  
            w = w * avail
            w = w / (w.sum(dim=1, keepdim=True).clamp_min(1e-6))
        return w  



class FinalConcatHead(nn.Module):
    """
    MLP over concatenated 7*d features (single- or multi-task).
    """
    def __init__(self, d: int, n_tasks: int = 1, hidden: Optional[Sequence[int]] = None, p_drop: float = 0.1):
        super().__init__()
        in_dim = 7 * d
        hidden = list(hidden) if hidden is not None else [4 * in_dim, 2 * in_dim]
        dims = [in_dim] + hidden + [n_tasks]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [
                nn.LayerNorm(dims[i]),
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
                nn.Dropout(p_drop),
            ]
        layers += [nn.LayerNorm(dims[-2]), nn.Linear(dims[-2], dims[-1])]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.dim() == 3:
            B, C, _ = x_cat.shape
            y = self.mlp(x_cat.reshape(B * C, -1))
            return y.view(B, C)
        return self.mlp(x_cat)  


def concat_routes(
    route_embs: Dict[str, torch.Tensor],
    gates: torch.Tensor,
    l2norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    B = next(iter(route_embs.values())).size(0)
    d = next(iter(route_embs.values())).size(1)
    Z = torch.stack([route_embs[r] for r in ROUTES], dim=1)  
    if l2norm:
        Z = F.normalize(Z, dim=2)
    Gw = gates.unsqueeze(-1)  
    Zw = Gw * Z               
    x_cat = Zw.reshape(B, 7 * d)
    return x_cat, Zw



@torch.no_grad()
def forward_emb_concat(
    z_unimodal: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
    final_head: FinalConcatHead,
    gate_net: RouteGateNet,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False,
):

    # 7 route embeddings from unimodal + fusion
    route_embs = make_route_inputs(z_unimodal, fusion)  
    # learned gates (respecting masks if provided)
    gates = gate_net(z_unimodal, masks=masks)           
    # concat weighted
    x_cat, Zw = concat_routes(route_embs, gates=gates, l2norm=l2norm_each)  
    # final logits
    logits = final_head(x_cat)  
    # map weighted back to dict
    routes_weighted = {r: Zw[:, i, :] for i, r in enumerate(ROUTES)}
    return logits, gates, route_embs, routes_weighted
