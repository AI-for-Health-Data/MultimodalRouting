from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import ROUTES, DEVICE


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
        feature_mode: str = "concat",  
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
        feature_mode: str = "concat",
    ):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        in_dim = 3 * d if feature_mode == "concat" else 7 * d
        self.mlp = _MLP(in_dim, d, hidden=hidden, p_drop=p_drop)
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.feature_mode = feature_mode

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
    feature_mode: str = "concat",
) -> Dict[str, nn.Module]:
    dev = torch.device(DEVICE)
    LN = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    LI = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    NI = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    LNI = TrimodalFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    return {"LN": LN, "LI": LI, "NI": NI, "LNI": LNI}


@torch.no_grad()
def _safe_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone() if x.requires_grad is False else x


def make_route_inputs(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
) -> Dict[str, torch.Tensor]:
    """
    Build embeddings for all 7 routes from unimodal z and fusion blocks.
    z keys: {"L","N","I"} → outputs {"L","N","I","LN","LI","NI","LNI"} each [B,d]
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


# Per-route mortality logits (classifiers)
class RouteHead(nn.Module):
    """
    Maps a d-dim route embedding to a **mortality logit** (binary).
    """
    def __init__(self, d_in: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 2 * d_in),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(2 * d_in, 1),     # 1 logit per route
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_route_heads(d: int, p_drop: float = 0.1) -> Dict[str, RouteHead]:
    dev = torch.device(DEVICE)
    return {r: RouteHead(d_in=d, p_drop=p_drop).to(dev) for r in ROUTES}


def _assert_same_dim(route_embs: Dict[str, torch.Tensor]) -> int:
    d_set = {t.size(1) for t in route_embs.values()}
    assert len(d_set) == 1, f"Route embedding dims differ: {d_set}"
    return next(iter(d_set))


def compute_route_logits(
    route_embs: Dict[str, torch.Tensor],
    route_heads: Dict[str, RouteHead],
) -> torch.Tensor:
    """
    Returns per-route mortality logits: [B, 7]
    """
    _ = _assert_same_dim(route_embs)
    logits = []
    for r in ROUTES:
        lr = route_heads[r](route_embs[r])  
        logits.append(lr)
    return torch.cat(logits, dim=1)  


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


def route_weights_from_logits(
    route_logits: torch.Tensor,                
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    B, R = route_logits.shape
    device = route_logits.device
    if masks is None:
        return torch.softmax(route_logits, dim=1)

    avail = route_availability_mask(masks, batch_size=B, device=device)  
    masked_scores = route_logits.masked_fill(avail < 0.5, float("-inf"))
    all_neg_inf = ~torch.isfinite(masked_scores).any(dim=1)
    if all_neg_inf.any():
        masked_scores[all_neg_inf] = 0.0
    w = torch.softmax(masked_scores, dim=1)
    w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return w / w_sum


class FinalConcatHead(nn.Module):
    def __init__(self, d: int, hidden: Optional[Sequence[int]] = None, p_drop: float = 0.1):
        super().__init__()
        in_dim = 7 * d
        hidden = list(hidden) if hidden is not None else [4 * in_dim, 2 * in_dim]
        dims = [in_dim] + hidden + [1]  
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
    gates: torch.Tensor,   # [B,7]
    l2norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    order = ROUTES
    Z_list = [route_embs[r] for r in order]

    B = Z_list[0].size(0)
    d_set = {z.size(1) for z in Z_list}
    assert len(d_set) == 1, f"Route embedding dims differ: {d_set}"
    d = next(iter(d_set))

    Z = torch.stack(Z_list, dim=1)  # [B,7,d]
    if l2norm:
        Z = F.normalize(Z, dim=2)

    R = len(order)
    assert gates.shape == (B, R), f"gates shape {tuple(gates.shape)} != {(B, R)}"
    Zw = gates.to(Z.dtype).unsqueeze(-1) * Z  # [B,7,d]

    x_cat = Zw.reshape(B, R * d)  # [B, 7*d]
    return x_cat, Zw


@torch.no_grad()
def forward_mixture_of_logits(
    z_unimodal: Dict[str, torch.Tensor],     
    fusion: Dict[str, nn.Module],            
    route_heads: Dict[str, RouteHead],      
    masks: Optional[Dict[str, torch.Tensor]] = None,  
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

    # Build all 7 route embeddings
    route_embs = make_route_inputs(z_unimodal, fusion)  
    _assert_same_dim(route_embs)                        

    # Per-route mortality logits
    route_logits = compute_route_logits(route_embs, route_heads)  

    # Weights from logits (masked softmax)
    gates = route_weights_from_logits(route_logits, masks=masks)  

    # Final mortality logit = weighted sum of per-route logits
    final_logit = (gates * route_logits).sum(dim=1, keepdim=True)  
    return final_logit, gates, route_embs, route_logits


@torch.no_grad()
def forward_emb_concat(
    z_unimodal: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
    final_head: FinalConcatHead,
    route_heads: Dict[str, RouteHead],       
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    route_embs = make_route_inputs(z_unimodal, fusion)
    _assert_same_dim(route_embs)

    scores = compute_route_logits(route_embs, route_heads)  
    gates  = route_weights_from_logits(scores, masks=masks)  

    x_cat, Zw = concat_routes(route_embs, gates=gates, l2norm=l2norm_each)
    logits = final_head(x_cat)  
    routes_weighted = {r: Zw[:, i, :] for i, r in enumerate(ROUTES)}
    return logits, gates, route_embs, routes_weighted
