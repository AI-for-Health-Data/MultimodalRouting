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
        feature_mode: str = "rich",
    ):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        self.feature_mode = feature_mode
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
    feature_mode: str = "concat",
    bi_fusion_mode: str = "mlp",
    tri_fusion_mode: str = "mlp",

) -> Dict[str, nn.Module]:
    if bi_fusion_mode != "mlp" or tri_fusion_mode != "mlp":
        raise ValueError("This project uses MLP+concat fusion only. Set bi_fusion_mode='mlp' and tri_fusion_mode='mlp'.")

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
    If masks are provided, weights for unavailable routes are zeroed and renormalized.
    """
    def __init__(self, d: int, hidden: int = 1024, p_drop: float = 0.1, use_masks: bool = True):
        super().__init__()
        self.use_masks = use_masks
        in_dim = 3 * d  # concat of zL, zN, zI
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
    MLP over concatenated, gate-weighted 7*d route features -> n_tasks.
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
        return self.mlp(x_cat)


def concat_routes(
    route_embs: Dict[str, torch.Tensor],
    gates: torch.Tensor,
    l2norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    order = ROUTES
    Z_list = [route_embs[r] for r in order]

    B = Z_list[0].size(0)
    d_set = {z.size(1) for z in Z_list}
    assert len(d_set) == 1, f"Route embedding dims differ: {d_set}"
    d = next(iter(d_set))

    Z = torch.stack(Z_list, dim=1) 
    if l2norm:
        Z = F.normalize(Z, dim=2)

    R = len(order)
    assert gates.shape == (B, R), f"gates shape {tuple(gates.shape)} != {(B, R)}"
    Zw = gates.to(Z.dtype).unsqueeze(-1) * Z  

    x_cat = Zw.reshape(B, R * d)            
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
    """
    Build 7 route embeddings, compute gates, create weighted concat, and
    return logits plus useful intermediates.
    """
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

class LearnedClasswiseGateNet(nn.Module):
    """
    Produces per-sample, per-class route weights (gates) of shape [B, 7, K].
    Gates are masked by availability (if provided) and renormalized over routes for each class.
    """
    def __init__(self, d: int, n_tasks: int, hidden: int = 1024, p_drop: float = 0.1, use_masks: bool = True):
        super().__init__()
        self.use_masks = use_masks
        self.K = int(n_tasks)
        in_dim = 3 * d  # concat of zL, zN, zI
        # Produce raw logits per route *and* per class
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, len(ROUTES) * self.K),  
        )

    def forward(self, z: Dict[str, torch.Tensor], masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x = torch.cat([z["L"], z["N"], z["I"]], dim=1)        
        logits = self.net(x)                                  
        gates = logits.view(x.size(0), len(ROUTES), self.K)   
        gates = torch.softmax(gates, dim=1)                  
        if self.use_masks and masks is not None:
            avail = route_availability_mask(masks, batch_size=x.size(0), device=x.device) 
            avail = avail.unsqueeze(-1)                                                           
            gates = gates * avail
            denom = gates.sum(dim=1, keepdim=True).clamp_min(1e-6)                               
            gates = gates / denom
        return gates  # [B,7,K]


def compute_loss_based_classwise_gates(
    logits_per_route: Dict[str, torch.Tensor],  
    y: torch.Tensor,                            
    avail: torch.Tensor,                        
    alpha: float = 4.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Classwise loss-based gates as in your Step-3 (but per class):
        For each class k, gates_k = softmax_r( -alpha * BCE_r,k + log(avail_r) )
    Returns gates: [B,7,K]
    """
    B, K = y.shape
    losses: List[torch.Tensor] = []
    for r in ROUTES:
        # element-wise BCE per class
        l_elem = F.binary_cross_entropy_with_logits(
            logits_per_route[r], y, pos_weight=pos_weight, reduction="none"
        )  
        losses.append(l_elem)
    L = torch.stack(losses, dim=1)  # [B,7,K]

    # mask by availability (add log(avail)) and softmax across routes
    avail_log = torch.log(avail.clamp_min(1e-12)).unsqueeze(-1)  # [B,7,1]
    masked_logits = (-float(alpha) * L) + avail_log               # [B,7,K]
    gates = torch.softmax(masked_logits, dim=1)                   # [B,7,K]
    return gates


def concat_routes_classwise(
    route_embs: Dict[str, torch.Tensor],  
    gates_classwise: torch.Tensor,        
    l2norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    order = ROUTES
    Z_list = [route_embs[r] for r in order]      
    B = Z_list[0].size(0)
    d_set = {z.size(1) for z in Z_list}
    assert len(d_set) == 1, f"Route embedding dims differ: {d_set}"
    d = next(iter(d_set))
    R = len(order)

    Z = torch.stack(Z_list, dim=1)  # [B,7,d]
    if l2norm:
        Z = F.normalize(Z, dim=2)

    assert gates_classwise.shape[:2] == (B, R), f"gates_classwise shape {tuple(gates_classwise.shape)} bad"
    K = gates_classwise.size(2)

    # Weighted per class: [B,7,K] * [B,7,d] -> [B,7,K,d]
    Zw_cls = gates_classwise.unsqueeze(-1) * Z.unsqueeze(2)      # [B,7,K,d]

    # Concat 7*d per class -> [B,K,7*d]
    X = Zw_cls.permute(0, 2, 1, 3).contiguous().view(B, K, R * d)  # [B,K,7*d]
    # Flatten batch*class for a single MLP that outputs 1 logit per row
    X_flat = X.view(B * K, R * d)                                   # [B*K,7*d]
    return X_flat, Zw_cls


class FinalConcatHeadClasswise(nn.Module):
    """
    Per-class MLP over concatenated, classwise gate-weighted 7*d route features.
    Implemented by flattening [B,K,7*d] -> [B*K,7*d] -> mlp -> [B*K,1] -> reshape [B,K].
    """
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

    def forward(self, x_flat: torch.Tensor, B: int, K: int) -> torch.Tensor:
        y = self.mlp(x_flat)          
        y = y.view(B, K)             
        return y


@torch.no_grad()
def build_capsule_inputs(
    route_embs: Dict[str, torch.Tensor],
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    order = ROUTES
    Z_list = [route_embs[r] for r in order]
    pose = torch.stack(Z_list, dim=1)  # [B,7,d]

    if masks is None:
        act = torch.ones(pose.size(0), pose.size(1), device=pose.device, dtype=pose.dtype)
    else:
        avail = route_availability_mask(masks, batch_size=pose.size(0), device=pose.device)  
        act = avail.to(pose.dtype)

    return pose, act
