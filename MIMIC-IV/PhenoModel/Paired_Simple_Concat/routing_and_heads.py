
from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from env_config import ROUTES, DEVICE, CFG
import capsule_layers


def _dbg(msg: str) -> None:
    if getattr(CFG, "verbose", False):
        print(msg)

def _peek_tensor(name: str, x: torch.Tensor, k: int = 3) -> None:
    if not getattr(CFG, "verbose", False):
        return
    if not hasattr(_peek_tensor, "_printed"):
        _peek_tensor._printed = set()
    key = f"{name}_shape"
    if key in _peek_tensor._printed:
        return
    _peek_tensor._printed.add(key)
    try:
        with torch.no_grad():
            flat = x.reshape(-1)
            vals = flat[:k].detach().cpu().tolist()
        print(f"[peek] {name}: shape={tuple(x.shape)} sample={vals}")
    except Exception:
        print(f"[peek] {name}: shape={tuple(x.shape)} sample=<unavailable>")


@torch.no_grad()
def _nan_guard(tag: str, x: torch.Tensor) -> None:
    if torch.isnan(x).any() or torch.isinf(x).any():
        _dbg(
            f"[NaN/Inf WARNING] {tag}: "
            f"nan={torch.isnan(x).any().item()} "
            f"inf={torch.isinf(x).any().item()}"
        )


class _MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Optional[Sequence[int]] = None,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        _nan_guard("_MLP.out", y)
        return y


class PairwiseFusion(nn.Module):
    def __init__(
        self,
        d: int,
        hidden: Optional[Sequence[int]] = None,
        p_drop: float = 0.0,
        feature_mode: str = "concat",
    ):
        super().__init__()
        assert feature_mode == "concat", "Only 'concat' feature_mode is supported."
        self.feature_mode = feature_mode
        in_dim = 2 * d
        self.mlp = _MLP(in_dim, d, hidden=hidden, p_drop=p_drop)

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([za, zb], dim=-1)  # [B, 2d]
        z = self.mlp(x)                  # [B, d]
        _peek_tensor("fusion.pair_z", z)
        _nan_guard("fusion.pair_z", z)
        return z


class TrimodalFusion(nn.Module):
    def __init__(
       self,
        d: int,
        hidden: Optional[Sequence[int]] = None,
        p_drop: float = 0.0,
        feature_mode: str = "concat",
    ):
        super().__init__()
        assert feature_mode == "concat", "Only 'concat' feature_mode is supported."
        self.feature_mode = feature_mode
        in_dim = 3 * d
        self.mlp = _MLP(in_dim, d, hidden=hidden, p_drop=p_drop)

    def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
        x = torch.cat([zL, zN, zI], dim=-1)  # [B, 3d]
        z = self.mlp(x)                      # [B, d]
        _peek_tensor("fusion.tri_z", z)
        _nan_guard("fusion.tri_z", z)
        return z


def build_fusions(
    d: int,
    p_drop: float = 0.0,
    feature_mode: str = "concat",
) -> Dict[str, nn.Module]:
    dev = torch.device(DEVICE)
    LN = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    LI = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    NI = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    LNI = TrimodalFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    _dbg(f"[build_fusions] feature_mode={feature_mode} -> LN,LI,NI:Pairwise / LNI:Trimodal @ d={d}")
    return {"LN": LN, "LI": LI, "NI": NI, "LNI": LNI}


@torch.no_grad()
def _safe_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone() if x.requires_grad is False else x


def make_route_inputs(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
) -> Dict[str, torch.Tensor]:
    zL, zN, zI = z["L"], z["N"], z["I"]
    routes = {
        "L": _safe_clone(zL),
        "N": _safe_clone(zN),
        "I": _safe_clone(zI),
        "LN": fusion["LN"](zL, zN),
        "LI": fusion["LI"](zL, zI),
        "NI": fusion["NI"](zN, zI),
        "LNI": fusion["LNI"](zL, zN, zI),
    }
    if not hasattr(make_route_inputs, "_printed_once"):
        make_route_inputs._printed_once = True
        sizes = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in routes.items())
        _dbg(f"[make_route_inputs] routes -> {sizes}")
        for k, v in routes.items():
            _nan_guard(f"route.{k}", v)
            _peek_tensor(f"route.{k}", v)
    return routes


# Route primary projector
class RoutePrimaryProjector(nn.Module):
    def __init__(self, d_in: int, pc_dim: int):
        super().__init__()
        self.pc_dim = int(pc_dim)
        self.proj = nn.ModuleDict({
            r: nn.Linear(d_in, self.pc_dim + 1, bias=False)
            for r in ROUTES
        })

    def forward(self, route_embs):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        pcs = [self.proj[r](route_embs[r].to(device=device, dtype=dtype)) for r in ROUTES]
        pc_all = torch.stack(pcs, dim=1)            # [B, 7, pc_dim+1]
        poses = pc_all[:, :, :self.pc_dim]
        raw_logits = pc_all[:, :, self.pc_dim:]     # [B, 7, 1]

        acts = torch.sigmoid(raw_logits)           
        return poses, acts

class CapsuleMortalityHead(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        mc_caps_dim: int,
        num_routing: int,
        dp: float = 0.0,
        act_type: str = "ONES",
        layer_norm: bool = False,   
        dim_pose_to_vote: int = 0,  
        num_classes: int = 25,
    ):
        super().__init__()
        self.in_n_capsules = 7
        self.in_d_capsules = pc_dim
        self.out_n_capsules = num_classes
        self.out_d_capsules = mc_caps_dim
        self.num_routing = int(num_routing)
        self.capsule = capsule_layers.CapsuleFC(
            in_n_capsules=self.in_n_capsules,
            in_d_capsules=self.in_d_capsules,
            out_n_capsules=self.out_n_capsules,
            out_d_capsules=self.out_d_capsules,
            n_rank=0,
            dp=0.0,                        # NO dropout here
            dim_pose_to_vote=dim_pose_to_vote,
            uniform_routing_coefficient=False,
            act_type=act_type,             # "EM", "Hubert", "ONES"
            small_std=False,
        )

        self.embedding = nn.Parameter(
            torch.zeros(self.out_n_capsules, self.out_d_capsules)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_n_capsules))
        self.nonlinear_act = nn.Sequential()

    def forward(
        self,
        prim_pose: torch.Tensor,   # [B, 7, pc_dim]
        prim_act: torch.Tensor,    # [B, 7] or [B, 7, 1]
        uniform_routing: bool = False,
    ):
        # Ensure activations are [B, 7, 1] (CapsuleFC expects [B, N_in, 1])
        if prim_act.dim() == 2:
            prim_act = prim_act.unsqueeze(-1)
        elif prim_act.dim() == 3 and prim_act.size(-1) == 1:
            pass
        else:
            raise ValueError(f"prim_act must be [B,7] or [B,7,1], got {prim_act.shape}")

        decision_pose = None
        decision_act = None
        routing_coef = None

        for it in range(self.num_routing):
            decision_pose, decision_act, routing_coef = self.capsule(
                input=prim_pose,
                current_act=prim_act,
                num_iter=it,
                next_capsule_value=decision_pose,
                next_act=decision_act,
                uniform_routing=uniform_routing,
            )

        # decision_pose: [B, K, mc_caps_dim]
        logits = torch.einsum("bmd, md -> bm", decision_pose, self.embedding) + self.bias

        prim_act_out = prim_act.squeeze(-1)  # [B, 7]
        return logits, prim_act_out, routing_coef

def forward_capsule_from_routes(
    z_unimodal: Dict[str, torch.Tensor],          # {"L","N","I"} each [B,d]
    fusion: Dict[str, nn.Module],                 # {"LN","LI","NI","LNI"}
    projector: RoutePrimaryProjector,             # d -> (pc_dim+1) per route
    capsule_head: CapsuleMortalityHead,           # -> [B,K] logits
    *,
    acts_override: Optional[torch.Tensor] = None, # [B,7,1] (optional external priors)
    route_mask: Optional[torch.Tensor] = None,    # [B,7]  (1=keep, 0=mask)
    act_temperature: float = 1.0,                 # >1 = softer, <1 = sharper
    detach_priors: bool = False,
    return_routing: bool = True,                  # kept for API compatibility
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    if not hasattr(forward_capsule_from_routes, "_printed_once"):
        forward_capsule_from_routes._printed_once = True
        sizes = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in z_unimodal.items())
        _dbg(f"[caps-bridge] unimodal -> {sizes}")
        for k, v in z_unimodal.items():
            _peek_tensor(f"caps-bridge.uni.{k}", v)
            _nan_guard(f"caps-bridge.uni.{k}", v)

    # Build 7 route embeddings from unimodal
    route_embs = make_route_inputs(z_unimodal, fusion)
    dev = next(projector.parameters()).device
    dtype = next(projector.parameters()).dtype
    route_embs = {
        k: v.to(device=dev, dtype=dtype, non_blocking=True)
        for k, v in route_embs.items()
    }

    poses, acts = projector(route_embs)

    acts_prior = acts if acts_override is None else acts_override.to(
        device=acts.device, dtype=acts.dtype
    )

    if route_mask is not None:
        acts_prior = acts_prior * route_mask.unsqueeze(-1) + 1e-6  

    if act_temperature != 1.0:
        eps = 1e-6
        acts_prior = torch.clamp(acts_prior, eps, 1.0 - eps)
        logits_t = torch.log(acts_prior) - torch.log(1.0 - acts_prior)
        logits_t = logits_t / float(act_temperature)
        acts_prior = torch.sigmoid(logits_t)

    prior_floor = float(getattr(CFG, "route_prior_floor", 1e-3))
    prior_ceiling = float(getattr(CFG, "route_prior_ceiling", 0.999))
    lo = prior_floor if prior_floor > 0.0 else 0.0
    hi = prior_ceiling if prior_ceiling > 0.0 else 1.0
    acts_prior = torch.clamp(acts_prior, min=lo, max=hi)

    acts_for_caps = acts_prior.detach() if detach_priors else acts_prior
    logits, prim_act_out, routing_coef = capsule_head(
        prim_pose=poses,
        prim_act=acts_for_caps.squeeze(-1),  # [B,7]
        uniform_routing=False,
    )

    prim_acts = prim_act_out.detach()  # [B,7] for logging only
    _peek_tensor("caps-bridge.prim_acts", prim_acts)
    _nan_guard("caps-bridge.prim_acts", prim_acts)

    if not hasattr(forward_capsule_from_routes, "_printed_routes"):
        forward_capsule_from_routes._printed_routes = True
        sizes = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
        _dbg(f"[caps-bridge] routes -> {sizes}")

    return logits, prim_acts, route_embs, routing_coef


__all__ = [
    "PairwiseFusion",
    "TrimodalFusion",
    "build_fusions",
    "make_route_inputs",
    "RoutePrimaryProjector",
    "CapsuleMortalityHead",
    "forward_capsule_from_routes",
]
