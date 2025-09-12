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

# Attention-based bimodal fusion (MulT-ish)
class _CrossAttnBlock(nn.Module):
    """One layer of bidirectional cross-attention with residuals + MLP."""
    def __init__(self, d: int, n_heads: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.a2b = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, dropout=p_drop, batch_first=True)
        self.b2a = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, dropout=p_drop, batch_first=True)
        self.norm_a1 = nn.LayerNorm(d)
        self.norm_b1 = nn.LayerNorm(d)
        self.ff_a = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(p_drop), nn.Linear(4*d, d))
        self.ff_b = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(p_drop), nn.Linear(4*d, d))
        self.drop = nn.Dropout(p_drop)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        q = self.norm_a1(xa); k = self.norm_b1(xb)
        a_ctx, _ = self.a2b(query=q, key=k, value=k, need_weights=False)
        xa = xa + self.drop(a_ctx)
        xa = xa + self.drop(self.ff_a(xa))
        q = self.norm_b1(xb); k = self.norm_a1(xa)
        b_ctx, _ = self.b2a(query=q, key=k, value=k, need_weights=False)
        xb = xb + self.drop(b_ctx)
        xb = xb + self.drop(self.ff_b(xb))
        return xa, xb


class CrossModalEncoder(nn.Module):
    """Tiny pairwise encoder producing a single d-dim vector via bidirectional cross-attn."""
    def __init__(self, d: int, n_layers: int = 2, n_heads: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([_CrossAttnBlock(d, n_heads, p_drop) for _ in range(n_layers)])
        self.pool = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, d))

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        xa, xb = za.unsqueeze(1), zb.unsqueeze(1)  
        for blk in self.blocks:
            xa, xb = blk(xa, xb)
        h = torch.cat([xa, xb], dim=-1)  
        return self.pool(h.squeeze(1))   

class TrimodalCrossEncoder(nn.Module):
    def __init__(self, d: int, n_layers: int = 2, n_heads: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttnBlock(d, n_heads, p_drop) for _ in range(n_layers)
        ])
        # fuse the three tokens after attention
        self.pool = nn.Sequential(
            nn.LayerNorm(3*d),
            nn.Linear(3*d, 4*d), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(4*d, d)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, zL, zN, zI):
        xL, xN, xI = zL.unsqueeze(1), zN.unsqueeze(1), zI.unsqueeze(1)  
        for blk in self.blocks:
            # round-robin: L↔N then L↔I then N↔I (order can be tuned)
            xL, xN = blk(xL, xN)
            xL, xI = blk(xL, xI)
            xN, xI = blk(xN, xI)
        h = torch.cat([xL, xN, xI], dim=-1).squeeze(1)  
        base = (zL + zN + zI) / 3.0
        return self.pool(h) + self.res_scale * base

def build_fusions(d, p_drop=0.1, feature_mode="rich",
                  bi_fusion_mode="mlp", tri_fusion_mode="mlp",
                  bi_layers=2, bi_heads=4, tri_layers=2, tri_heads=4):
    dev = torch.device(DEVICE)
    if bi_fusion_mode == "attn":
        LN = CrossModalEncoder(d, n_layers=bi_layers, n_heads=bi_heads, p_drop=p_drop).to(dev)
        LI = CrossModalEncoder(d, n_layers=bi_layers, n_heads=bi_heads, p_drop=p_drop).to(dev)
        NI = CrossModalEncoder(d, n_layers=bi_layers, n_heads=bi_heads, p_drop=p_drop).to(dev)
    else:
        LN = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
        LI = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
        NI = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)

    if tri_fusion_mode == "attn":
        LNI = TrimodalCrossEncoder(d, n_layers=tri_layers, n_heads=tri_heads, p_drop=p_drop).to(dev)
    else:
        LNI = TrimodalFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)

    return {"LN": LN, "LI": LI, "NI": NI, "LNI": LNI}



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
