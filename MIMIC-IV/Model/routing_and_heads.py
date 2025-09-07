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
        feature_mode: str = "rich",
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
        return self.net(x)  # [B, C]


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


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    logits = logits + torch.log(mask.to(dtype=logits.dtype) + 1e-12)
    return torch.softmax(logits, dim=dim)



class BCEWeightedEmbedCombiner(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        alpha: float = None,
        ema_decay: float = None,
        use_masks: bool = True,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = float(alpha if alpha is not None else getattr(CFG, "alpha", 4.0))
        self.ema_decay = float(ema_decay if ema_decay is not None else getattr(CFG, "ema_decay", 0.98))
        self.use_masks = use_masks

        ema_init = torch.zeros(n_tasks, len(ROUTES))
        self.register_buffer("ema_loss", ema_init)

        self.seen = False

    def _uniform_weights(self, B: int, device) -> torch.Tensor:
        w = torch.full((self.n_tasks, len(ROUTES)), 1.0 / len(ROUTES), device=device)
        return w.unsqueeze(0).expand(B, -1, -1).clone()

    @torch.no_grad()
    def update_ema_from_batch(
        self,
        route_logits: Dict[str, torch.Tensor],  
        y_true: torch.Tensor,                   
        pos_weight: Optional[torch.Tensor] = None,  
    ) -> torch.Tensor:

        device = y_true.device
        C = y_true.size(1)
        assert C == self.n_tasks, "y_true's C must equal combiner.n_tasks"

        batch_loss = torch.zeros(C, len(ROUTES), device=device)
        bce_kwargs = {}
        if pos_weight is not None:
            bce_kwargs["pos_weight"] = pos_weight.to(device)

        for j, r in enumerate(ROUTES):
            logits_r = route_logits[r]  
            l_r = F.binary_cross_entropy_with_logits(
                logits_r, y_true, reduction="none", **bce_kwargs
            ).mean(dim=0)
            batch_loss[:, j] = l_r


        if self.seen:
            self.ema_loss.mul_(self.ema_decay).add_(batch_loss * (1.0 - self.ema_decay))
        else:
            self.ema_loss.copy_(batch_loss)
            self.seen = True

        return batch_loss  

    def compute_weights(
        self,
        B: int,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        device=None,
    ) -> torch.Tensor:
        device = device or self.ema_loss.device
        if not self.seen:
            w = self._uniform_weights(B, device)
        else:
            logits_c7 = -self.alpha * self.ema_loss  
            w = logits_c7.unsqueeze(0).expand(B, -1, -1).clone()  

        if self.use_masks and masks is not None:
            avail = route_availability_mask(masks, batch_size=B, device=device)  
        else:
            avail = torch.ones(B, len(ROUTES), device=device)

        avail_bc7 = avail.unsqueeze(1).expand(-1, self.n_tasks, -1)
        w = _masked_softmax(w, mask=avail_bc7, dim=2)  
        return w  

    def combine_embeddings(
        self,
        route_embs: Dict[str, torch.Tensor],   
        route_w: torch.Tensor,                 
        l2norm_each: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = next(iter(route_embs.values())).size(0)
        d = next(iter(route_embs.values())).size(1)

        z_stack = torch.stack([route_embs[r] for r in ROUTES], dim=2)  
        z_list = [route_embs[r] for r in ROUTES]          
        Z = torch.stack(z_list, dim=2)                    
        Z = Z.permute(0, 2, 1).contiguous()              

        if l2norm_each:
            Z = F.normalize(Z, dim=2)

        Z_bc7d = Z.unsqueeze(1).expand(-1, self.n_tasks, -1, -1).contiguous()
        # Apply weights
        W_bc7d = route_w.unsqueeze(-1)                   
        Zw = W_bc7d * Z_bc7d                            

        weighted: Dict[str, torch.Tensor] = {}
        for j, r in enumerate(ROUTES):
            weighted[r] = Zw[:, :, j, :] 

        # Concatenate along routes → [B, C, 7*d]
        X_cat = Zw.reshape(B, self.n_tasks, 7 * d)
        return X_cat, weighted


class FinalConcatHeadPerTask(nn.Module):
    def __init__(self, d: int, n_tasks: int, hidden: Optional[Sequence[int]] = None, p_drop: float = 0.1):
        super().__init__()
        self.n_tasks = n_tasks
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
        B, C, _ = x_cat.shape
        x = x_cat.reshape(B * C, -1)
        y = self.mlp(x)                    
        return y.view(B, C)



def forward_emb_concat(
    z_unimodal: Dict[str, torch.Tensor],              
    fusion: Dict[str, nn.Module],                     
    route_heads: Dict[str, RouteHead],                
    final_head: FinalConcatHeadPerTask,               
    y_true: Optional[torch.Tensor] = None,            
    masks: Optional[Dict[str, torch.Tensor]] = None,  
    combiner: Optional[BCEWeightedEmbedCombiner] = None,
    pos_weight: Optional[torch.Tensor] = None,        
    l2norm_each: bool = False,
    update_ema: bool = True,
) -> Tuple[
    torch.Tensor,                 
    torch.Tensor,                 
    Dict[str, torch.Tensor],      
    Dict[str, torch.Tensor],      
    Dict[str, torch.Tensor],      
]:

    device = next(iter(z_unimodal.values())).device
    B = next(iter(z_unimodal.values())).size(0)

    route_embs = make_route_inputs(z_unimodal, fusion)  

    route_logits = compute_route_logits(route_embs, route_heads)  

    if combiner is None:
        combiner = BCEWeightedEmbedCombiner(n_tasks=route_logits[ROUTES[0]].size(1)).to(device)

    batch_route_bce = None
    if (y_true is not None) and update_ema:
        batch_route_bce = combiner.update_ema_from_batch(route_logits, y_true, pos_weight=pos_weight)

    route_w = combiner.compute_weights(B=B, masks=masks, device=device)

    x_cat, weighted = combiner.combine_embeddings(route_embs, route_w, l2norm_each=l2norm_each)  

    logits = final_head(x_cat)  

    aux = {
        "route_logits": route_logits,          
        "batch_route_bce": batch_route_bce,    
        "ema_loss": combiner.ema_loss.detach().clone(),  
    }
    return logits, route_w, route_embs, weighted, aux
