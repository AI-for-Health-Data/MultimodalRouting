from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional

import torch
import torch.nn as nn

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
            f"nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()}"
        )


class _MLP(nn.Module):
    """
    Minimal projection used by fusion blocks:
    a single Linear(in_dim -> out_dim) with bias=False.
    No LayerNorm, activation, dropout, or residuals.
    """
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
    """
    Given two unimodal embeddings za, zb (both [B, d]), build a pairwise route embedding.
    This is a pure linear fusion: concat([za, zb]) -> Linear(2d -> d).
    """
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
    """
    Given three unimodal embeddings zL, zN, zI (all [B, d]), build the trimodal route embedding.
    This is a pure linear fusion: concat([zL, zN, zI]) -> Linear(3d -> d).
    """
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
    """
    Create the fusion blocks used to build the interaction routes from unimodal embeddings.
    Returns a dict with keys {"LN","LI","NI","LNI"}.
    """
    dev = torch.device(DEVICE)
    LN  = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    LI  = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
    NI  = PairwiseFusion(d, p_drop=p_drop, feature_mode=feature_mode).to(dev)
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
    """
    Build embeddings for all 7 routes from unimodal pooled embeddings.
    Input:
        z: {"L","N","I"} each [B, d]
        fusion: {"LN","LI","NI","LNI"}
    Output:
        {"L","N","I","LN","LI","NI","LNI"} each [B, d]
    """
    zL, zN, zI = z["L"], z["N"], z["I"]
    routes = {
        "L":   _safe_clone(zL),
        "N":   _safe_clone(zN),
        "I":   _safe_clone(zI),
        "LN":  fusion["LN"](zL, zN),
        "LI":  fusion["LI"](zL, zI),
        "NI":  fusion["NI"](zN, zI),
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


class RoutePrimaryProjector(nn.Module):
    """
    Per-route projection: d_in -> (pc_dim + 1)
    Split into:
      - pose: [B, pc_dim]
      - act:  sigmoid([B,1])
    Applied independently for each of the 7 routes in ROUTES.
    """
    def __init__(self, d_in: int, pc_dim: int):
        super().__init__()
        self.pc_dim = int(pc_dim)
        self.proj = nn.ModuleDict({r: nn.Linear(d_in, self.pc_dim + 1) for r in ROUTES})

        self.route_logit_bias = nn.Parameter(torch.zeros(len(ROUTES), 1))

        with torch.no_grad():
            self.route_logit_bias.zero_()

    def forward(self, route_embs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            route_embs: dict(route -> [B, d_in]) for all 7 routes
        Returns:
            poses: [B, 7, pc_dim]
            acts:  [B, 7, 1]   (sigmoid activations, with learned per-route logit bias)
        """
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        pcs = [self.proj[r](route_embs[r].to(device=device, dtype=dtype)) for r in ROUTES]
        pc_all = torch.stack(pcs, dim=1)            # [B, 7, pc_dim+1]
        poses = pc_all[:, :, :self.pc_dim]
        raw_logits = pc_all[:, :, self.pc_dim:]     # [B,7,1]
        acts = torch.sigmoid(raw_logits + self.route_logit_bias.unsqueeze(0))

        if not hasattr(self, "_printed_once"):
            self._printed_once = True
            _dbg(f"[projector] pc_all:{tuple(pc_all.shape)} poses:{tuple(poses.shape)} acts:{tuple(acts.shape)}")
            with torch.no_grad():
                a = acts.detach()
                _dbg(
                    "[projector] acts stats -> "
                    f"min:{a.min().item():.4f} max:{a.max().item():.4f} mean:{a.mean().item():.4f}"
                )
                _peek_tensor("projector.poses", poses)
                _peek_tensor("projector.acts", acts)

        _nan_guard("projector.poses", poses)
        _nan_guard("projector.acts", acts)
        return poses, acts


class CapsuleMortalityHead(nn.Module):
    """
    General capsule head for K phenotypes:
      - routes 7 primary capsules into K decision capsules
      - each decision capsule k produces a scalar logit for pheno k
      - output: [B, K] logits
    """
    def __init__(
        self,
        pc_dim: int,             
        mc_caps_dim: int,         
        num_routing: int = 3,
        dp: float = 0.0,
        act_type: str = "EM",
        layer_norm: bool = False,
        dim_pose_to_vote: int = 0,
        num_classes: int = 2,     
    ):
        super().__init__()
        self.pc_dim = int(pc_dim)
        self.mc_caps_dim = int(mc_caps_dim)
        self.num_routing = int(num_routing)
        self.mc_num_caps = int(num_classes)   

        self.mc = capsule_layers.CapsuleFC(
            in_n_capsules=7,
            in_d_capsules=self.pc_dim,
            out_n_capsules=self.mc_num_caps,
            out_d_capsules=self.mc_caps_dim,
            n_rank=None,
            dp=dp,
            dim_pose_to_vote=dim_pose_to_vote,
            uniform_routing_coefficient=False,
            act_type=act_type,
            small_std=not layer_norm,
        )

        # one scalar logit per decision capsule
        self.cls_weight = nn.Parameter(torch.empty(self.mc_num_caps, self.mc_caps_dim))
        self.cls_bias   = nn.Parameter(torch.zeros(self.mc_num_caps))

        nn.init.normal_(self.cls_weight, std=0.02)
        nn.init.zeros_(self.cls_bias)

    def forward(
        self,
        *,
        poses: torch.Tensor,      # [B,7,pc_dim]
        acts_prior: torch.Tensor, # [B,7,1]  (priors already composed)
        return_routing: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        poses = poses.to(device=device, dtype=dtype)
        acts  = acts_prior.to(device=device, dtype=dtype)

        if not hasattr(self, "_printed_once"):
            self._printed_once = True
            _dbg(
                f"[cap-head] input poses:{tuple(poses.shape)} acts:{tuple(acts.shape)} "
                f"(pc_dim={self.pc_dim}, mc_caps_dim={self.mc_caps_dim}, "
                f"out_caps={self.mc_num_caps}, iters={self.num_routing})"
            )
            _peek_tensor("cap-head.poses.in", poses)
            _peek_tensor("cap-head.acts.in", acts)

        decision_pose, decision_act, routing_coef = self.mc(poses, acts, 0)
        if decision_act is None:
            B = poses.size(0)
            decision_act = torch.full((B, self.mc_num_caps), 0.5, device=device, dtype=dtype)

        prev_act = decision_act
        for t in range(1, self.num_routing):
            decision_pose, maybe_act, routing_coef = self.mc(
                poses, acts, t, decision_pose, prev_act
            )
            if maybe_act is not None:
                prev_act = maybe_act

        # decision_pose: [B, K, D]
        logits = torch.einsum("bkd,kd->bk", decision_pose, self.cls_weight) \
                 + self.cls_bias.unsqueeze(0)   # [B, K]

        if not hasattr(self, "_printed_done"):
            self._printed_done = True
            _dbg(f"[cap-head] decision_pose:{tuple(decision_pose.shape)} -> logits:{tuple(logits.shape)}")
            with torch.no_grad():
                _dbg(
                    f"[cap-head] logit stats -> "
                    f"min:{logits.min().item():.4f} "
                    f"max:{logits.max().item():.4f} "
                    f"mean:{logits.mean().item():.4f}"
                )

        return (logits, routing_coef) if return_routing else (logits, None)


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
    return_routing: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Bridge from unimodal pooled embeddings → 7-route capsules → logits.

    Returns:
        logits:       [B, K]
        prim_acts:    [B, 7]       (priors actually used; detached for logging)
        route_embs:   dict of 7 -> [B, d]
        routing_coef: [B, 7, K] or None
    """
    if not hasattr(forward_capsule_from_routes, "_printed_once"):
        forward_capsule_from_routes._printed_once = True
        sizes = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in z_unimodal.items())
        _dbg(f"[caps-bridge] unimodal -> {sizes}")
        for k, v in z_unimodal.items():
            _peek_tensor(f"caps-bridge.uni.{k}", v)
            _nan_guard(f"caps-bridge.uni.{k}", v)

    route_embs = make_route_inputs(z_unimodal, fusion)
    dev   = next(projector.parameters()).device
    dtype = next(projector.parameters()).dtype
    route_embs = {k: v.to(device=dev, dtype=dtype, non_blocking=True) for k, v in route_embs.items()}

    poses, acts = projector(route_embs)  # acts: sigmoid probabilities [B,7,1]

    acts_prior = acts if acts_override is None else acts_override.to(
        device=acts.device, dtype=acts.dtype
    )

    if route_mask is not None:
        acts_prior = acts_prior * route_mask.unsqueeze(-1) + 1e-6  # keep grads

    if act_temperature != 1.0:
        eps = 1e-6
        acts_prior = torch.clamp(acts_prior, eps, 1.0 - eps)
        logits_t = torch.log(acts_prior) - torch.log(1.0 - acts_prior)
        logits_t = logits_t / float(act_temperature)
        acts_prior = torch.sigmoid(logits_t)

    prior_floor   = float(getattr(CFG, "route_prior_floor", 1e-3))
    prior_ceiling = float(getattr(CFG, "route_prior_ceiling", 0.999))
    lo = prior_floor if prior_floor > 0.0 else 0.0
    hi = prior_ceiling if prior_ceiling > 0.0 else 1.0
    acts_prior = torch.clamp(acts_prior, min=lo, max=hi)

    if detach_priors:
        acts_prior = acts_prior.detach()

    logits, routing_coef = capsule_head(
        poses=poses,
        acts_prior=acts_prior,
        return_routing=return_routing,
    )

    prim_acts = acts_prior.squeeze(-1).detach()  # [B,7] for logging only
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
