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
    """One-time compact peek at a tensor: shape + a few values."""
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
        _dbg(f"[NaN/Inf WARNING] {tag}: nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()}")


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
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([
                nn.LayerNorm(dims[i]),
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
                nn.Dropout(p_drop),
            ])
        layers.extend([nn.LayerNorm(dims[-2]), nn.Linear(dims[-2], dims[-1])])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        _nan_guard("_MLP.out", y)
        return y


class PairwiseFusion(nn.Module):
    """
    Given two unimodal embeddings za, zb (both [B, d]), build a pairwise route embedding.
    feature_mode='concat': [za, zb] -> MLP -> d (+ residual to 0.5*(za+zb))
    feature_mode='rich'  : [za, zb, za*zb, |za-zb|] -> MLP -> d (+ residual)
    """
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
        z = h + self.res_scale * base
        _peek_tensor("fusion.pair_z", z)
        _nan_guard("fusion.pair_z", z)
        return z


class TrimodalFusion(nn.Module):
    """
    Given three unimodal embeddings zL, zN, zI (all [B, d]), build the trimodal route embedding.
    feature_mode='concat': [zL, zN, zI] -> MLP -> d (+ residual to mean)
    feature_mode='rich'  : [zL, zN, zI, zL*zN, zL*zI, zN*zI, zL*zN*zI] -> MLP -> d (+ residual)
    """
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
        z = h + self.res_scale * base
        _peek_tensor("fusion.tri_z", z)
        _nan_guard("fusion.tri_z", z)
        return z


def build_fusions(
    d: int,
    p_drop: float = 0.1,
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
    # One-time sanity
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

    def forward(self, route_embs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            route_embs: dict(route -> [B, d_in]) for all 7 routes
        Returns:
            poses: [B, 7, pc_dim]
            acts:  [B, 7, 1]   (sigmoid activations)
        """
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        pcs = [ self.proj[r]( route_embs[r].to(device=device, dtype=dtype) ) for r in ROUTES ]
        pc_all = torch.stack(pcs, dim=1)            # [B, 7, pc_dim+1], already correct dtype/device
        poses = pc_all[:, :, :self.pc_dim]
        acts  = torch.sigmoid(pc_all[:, :, self.pc_dim:])
        
    
        # SANITY PRINT
        if not hasattr(self, "_printed_once"):
            self._printed_once = True
            _dbg(f"[projector] pc_all:{tuple(pc_all.shape)} poses:{tuple(poses.shape)} acts:{tuple(acts.shape)}")
            with torch.no_grad():
                a = acts.detach()
                _dbg(f"[projector] acts stats -> min:{a.min().item():.4f} max:{a.max().item():.4f} mean:{a.mean().item():.4f}")
                _peek_tensor("projector.poses", poses)
                _peek_tensor("projector.acts", acts)
        _nan_guard("projector.poses", poses)
        _nan_guard("projector.acts", acts)
        return poses, acts

class CapsuleMortalityHead(nn.Module):
    """
    Routes the 7 primary capsules (poses + acts) into TWO decision capsules
    (class 0=survive, class 1=expire) and returns 2-class logits.
    """
    def __init__(
        self,
        pc_dim: int,              # primary capsule pose dim
        mc_caps_dim: int,         # decision capsule pose dim
        num_routing: int = 3,
        dp: float = 0.1,
        act_type: str = "EM",
        layer_norm: bool = False,     # toggles CapsuleFC init (small_std)
        dim_pose_to_vote: int = 0,
    ):
        super().__init__()
        self.pc_dim = int(pc_dim)
        self.mc_caps_dim = int(mc_caps_dim)
        self.num_routing = int(num_routing)

        # TWO decision capsules (binary 2-class CE)
        self.mc_num_caps = 2

        self.mc = capsule_layers.CapsuleFC(
            in_n_capsules=7,
            in_d_capsules=self.pc_dim,
            out_n_capsules=self.mc_num_caps,
            out_d_capsules=self.mc_caps_dim,
            n_rank=None,
            dp=dp,
            act_type=act_type,
            small_std=not layer_norm,
            dim_pose_to_vote=dim_pose_to_vote,
        )

        # Map each decision capsule pose -> one scalar logit
        self.cls0 = nn.Linear(self.mc_caps_dim, 1)
        self.cls1 = nn.Linear(self.mc_caps_dim, 1)
        nn.init.normal_(self.cls0.weight, std=0.02); nn.init.zeros_(self.cls0.bias)
        nn.init.normal_(self.cls1.weight, std=0.02); nn.init.zeros_(self.cls1.bias)

    def forward(self, poses: torch.Tensor, acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            poses: [B, 7, pc_dim]
            acts:  [B, 7, 1]
        Returns:
            logits:        [B, 2]  (class-0, class-1)
            routing_coef:  [B, 7, 2]
        """
        device = torch.device(DEVICE)
        dtype  = next(self.cls0.parameters()).dtype  # or next(self.parameters()).dtype
        poses = poses.to(device=device, dtype=dtype)
        acts  = acts.to(device=device, dtype=dtype)

        # One-time debug
        if not hasattr(self, "_printed_once"):
            self._printed_once = True
            _dbg(
                f"[cap-head] input poses:{tuple(poses.shape)} acts:{tuple(acts.shape)} "
                f"(pc_dim={self.pc_dim}, mc_caps_dim={self.mc_caps_dim}, out_caps={self.mc_num_caps}, iters={self.num_routing})"
            )
            _peek_tensor("cap-head.poses.in", poses)
            _peek_tensor("cap-head.acts.in", acts)

        # Initial routing (iter 0) without priors
        decision_pose, decision_act, routing_coef = self.mc(poses, acts, 0)  # pose:[B,2,D], act:[B,2] or None

        # Ensure valid activations for next iterations
        B = poses.size(0)
        if decision_act is None:
            _dbg("[cap-head] warning: initial decision_act is None; initializing to 0.5.")
            decision_act = torch.full((B, self.mc_num_caps), 0.5, device=device)

        prev_act = decision_act
        for n in range(1, self.num_routing):
            decision_pose, maybe_act, routing_coef = self.mc(poses, acts, n, decision_pose, prev_act)
            if maybe_act is not None:
                prev_act = maybe_act  # keep updated
            else:
                _dbg(f"[cap-head] iter {n}: mc returned None next_act; reusing previous activations.")
        
        # --- FIX: match device/dtype of classifier weights before matmul ---
        wdev  = self.cls0.weight.device
        wdtyp = self.cls0.weight.dtype
        decision_pose = decision_pose.to(device=wdev, dtype=wdtyp)
        
        # Map poses from the TWO decision capsules to TWO logits
        log0 = self.cls0(decision_pose[:, 0, :]).squeeze(1)  # [B]
        log1 = self.cls1(decision_pose[:, 1, :]).squeeze(1)  # [B]
        logits = torch.stack([log0, log1], dim=1)            # [B,2]

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

        return logits, routing_coef


def forward_capsule_from_routes(
    z_unimodal: Dict[str, torch.Tensor],          # {"L","N","I"} each [B,d]
    fusion: Dict[str, nn.Module],                 # {"LN","LI","NI","LNI"}
    projector: RoutePrimaryProjector,             # d -> (pc_dim+1) per route
    capsule_head: CapsuleMortalityHead,           # CapsuleFC wrapper -> [B,2] logits
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    1) Build 7 route embeddings from unimodal pooled embeddings.
    2) Project each route -> primary capsule (pose + act).
    3) Run capsule routing -> 2-class logits.

    Returns:
        logits:       [B, 2]
        prim_acts:    [B, 7]       (primary capsule activations)
        route_embs:   dict of 7 -> [B, d] (for inspection/aux losses)
        routing_coef: [B, 7, 2]    (interaction weights per route -> per class)
    """
    if not hasattr(forward_capsule_from_routes, "_printed_once"):
        forward_capsule_from_routes._printed_once = True
        sizes = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in z_unimodal.items())
        _dbg(f"[caps-bridge] unimodal -> {sizes}")
        for k, v in z_unimodal.items():
            _peek_tensor(f"caps-bridge.uni.{k}", v)
            _nan_guard(f"caps-bridge.uni.{k}", v)

    # 1) Build 7 routes
    route_embs = make_route_inputs(z_unimodal, fusion)
    # --- ensure route_embs tensors are on same device as projector ---
    #dev = next(projector.parameters()).device
    #route_embs = {k: v.to(dev, non_blocking=True) for k, v in route_embs.items()}
    
    dev   = next(projector.parameters()).device
    dtype = next(projector.parameters()).dtype
    route_embs = {k: v.to(device=dev, dtype=dtype, non_blocking=True) for k, v in route_embs.items()}


    # 2) Project to primary capsules
    poses, acts = projector(route_embs)                        # [B,7,pc_dim], [B,7,1]

    # 3) Capsule routing -> logits + routing coefficients
    logits, routing_coef = capsule_head(poses, acts)           # [B,2], [B,7,2]

    # Primary activations for inspection
    prim_acts = acts.squeeze(-1).detach()                      # [B,7]
    _peek_tensor("caps-bridge.prim_acts", prim_acts)
    _nan_guard("caps-bridge.prim_acts", prim_acts)

    if not hasattr(forward_capsule_from_routes, "_printed_routes"):
        forward_capsule_from_routes._printed_routes = True
        sizes = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
        _dbg(f"[caps-bridge] routes -> {sizes}")

    return logits, prim_acts, route_embs, routing_coef



__all__ = [
    # Fusion builders
    "PairwiseFusion",
    "TrimodalFusion",
    "build_fusions",
    "make_route_inputs",
    # Capsule bridge
    "RoutePrimaryProjector",
    "CapsuleMortalityHead",
    "forward_capsule_from_routes",
]
