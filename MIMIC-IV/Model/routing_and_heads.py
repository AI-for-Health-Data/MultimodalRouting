from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import CFG, ROUTES, BLOCKS, DEVICE


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: Optional[Sequence[int]] = None,
                 p_drop: float = 0.1):
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
    def __init__(self, d: int,
                 hidden: Optional[Sequence[int]] = None,
                 p_drop: float = 0.1,
                 feature_mode: str = "rich"):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        self.d = d
        self.feature_mode = feature_mode
        in_dim = 2 * d if feature_mode == "concat" else 4 * d
        self.mlp = _MLP(in_dim, d, hidden=hidden, p_drop=p_drop)
        # Residual to stabilize
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
    def __init__(self, d: int,
                 hidden: Optional[Sequence[int]] = None,
                 p_drop: float = 0.1,
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
        else:
            zLN  = zL * zN
            zLI  = zL * zI
            zNI  = zN * zI
            zLNI = zL * zN * zI
            x = torch.cat([zL, zN, zI, zLN, zLI, zNI, zLNI], dim=-1)
        h = self.mlp(x)
        base = (zL + zN + zI) / 3.0
        return h + self.res_scale * base


def build_fusions(d: int,
                  p_drop: float = 0.1,
                  feature_mode: str = "rich",
                  hidden: Optional[Sequence[int]] = None) -> Dict[str, nn.Module]:
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
    return {r: RouteHead(d_in=d, n_tasks=n_tasks, p_drop=p_drop).to(DEVICE) for r in ROUTES}



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
    def __init__(self, d: int, hidden: int = 4 * 256, p_drop: float = 0.1, use_masks: bool = True):
        super().__init__()
        self.use_masks = use_masks
        gate_in = 3 * d + (3 if use_masks else 0)
        self.net = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 7),
        )

    def forward(self,
                z_dict: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        zL, zN, zI = z_dict["L"], z_dict["N"], z_dict["I"]
        pieces = [zL, zN, zI]
        if self.use_masks:
            if masks is None:
                B = zL.size(0)
                m = torch.ones(B, 3, device=zL.device)
            else:
                m = torch.cat([masks["L"], masks["N"], masks["I"]], dim=1)
            pieces.append(m)
        x = torch.cat(pieces, dim=1)          
        g_raw = self.net(x)                    
        g = torch.sigmoid(g_raw)               
        avail = route_availability_mask(masks, batch_size=x.size(0), device=x.device)
        return g * avail


class FinalConcatHead(nn.Module):
    def __init__(self, d: int, n_tasks: int, hidden: Optional[Sequence[int]] = None, p_drop: float = 0.1):
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
        layers += [
            nn.LayerNorm(dims[-2]),
            nn.Linear(dims[-2], dims[-1]),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        return self.net(x_cat)  


def concat_routes(
    routes: Dict[str, torch.Tensor],
    gates: Optional[torch.Tensor] = None,
    l2norm: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    B = next(iter(routes.values())).size(0)
    d = next(iter(routes.values())).size(1)
    out_pieces: List[torch.Tensor] = []
    routed: Dict[str, torch.Tensor] = {}

    for i, r in enumerate(ROUTES):
        zi = routes[r]  
        if l2norm:
            zi = F.normalize(zi, dim=1)
        if gates is not None:
            gi = gates[:, i].unsqueeze(1)  
            zi = gi * zi
        routed[r] = zi
        out_pieces.append(zi)

    x_cat = torch.cat(out_pieces, dim=1) 
    return x_cat, routed


@torch.no_grad()
def build_availability_mask(masks: Optional[Dict[str, torch.Tensor]],
                            batch_size: int, device) -> torch.Tensor:
    return route_availability_mask(masks, batch_size, device)


def forward_emb_concat(
    z_unimodal: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
    final_head: FinalConcatHead,
    gate_net: Optional[RouteGateNet] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    l2norm_each: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    routes = make_route_inputs(z_unimodal, fusion)  

    if gate_net is not None and getattr(CFG, "use_gates", True):
        gates = gate_net(z_unimodal, masks=masks)  
    else:
        B = next(iter(routes.values())).size(0)
        gates = torch.ones(B, len(ROUTES), device=next(iter(routes.values())).device)

    x_cat, routed = concat_routes(routes, gates=gates, l2norm=l2norm_each)  

    logits = final_head(x_cat)  
    return logits, gates, routes, routed
