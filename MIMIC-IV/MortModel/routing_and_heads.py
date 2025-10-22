from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional

import torch
import torch.nn as nn

from env_config import ROUTES, DEVICE, CFG
import capsule_layers

def _dbg(msg: str) -> None:
    if getattr(CFG, "verbose", False):
        print(msg)

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
        return self.net(x)

class PairwiseFusion(nn.Module):
    """
    Given two unimodal embeddings za, zb (both [B, d]), build a pairwise route embedding.
    Mode 'concat': [za, zb] -> MLP -> d (+ residual to 0.5*(za+zb))
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
        return h + self.res_scale * base



class TrimodalFusion(nn.Module):
    """
    Given three unimodal embeddings zL,zN,zI (all [B, d]), build the trimodal route embedding.
    Mode 'concat': [zL, zN, zI] -> MLP -> d (+ residual to average)
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
        return h + self.res_scale * base


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
    Output:
        {"L","N","I","LN","LI","NI","LNI"} with each tensor [B, d]
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


# Primary capsule projection (pc heads)
class RoutePrimaryProjector(nn.Module):
    """
    Legacy-equivalent behavior:
      For each route r in {L,N,I,LN,LI,NI,LNI}, apply a dedicated Linear(d_in -> pc_dim+1),
      then split into:
        - pose: [B, 7, pc_dim]
        - act : sigmoid([B, 7, 1])
    """
    def __init__(self, d_in: int, pc_dim: int):
        super().__init__()
        self.pc_dim = int(pc_dim)
        self.proj = nn.ModuleDict({r: nn.Linear(d_in, self.pc_dim + 1) for r in ROUTES})

    def forward(self, route_embs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        pcs = [self.proj[r](route_embs[r]) for r in ROUTES]          # list of [B, pc_dim+1]
        pc_all = torch.stack(pcs, dim=1).to(device)                  # [B, 7, pc_dim+1]
        poses = pc_all[:, :, :self.pc_dim]                           # [B, 7, pc_dim]
        acts  = torch.sigmoid(pc_all[:, :, self.pc_dim:])            # [B, 7, 1]
        if not hasattr(self, "_printed_once"):
            self._printed_once = True
            _dbg(f"[projector] pc_all:{tuple(pc_all.shape)} poses:{tuple(poses.shape)} acts:{tuple(acts.shape)}")
            with torch.no_grad():
                a = acts.detach()
                _dbg(f"[projector] acts stats -> min:{a.min().item():.4f} max:{a.max().item():.4f} mean:{a.mean().item():.4f}")
        return poses, acts


# Capsule head wrapper (binary mortality -> one decision capsule)
class CapsuleMortalityHead(nn.Module):
    """
    Wraps CapsuleFC to route 7 primary capsules (poses + acts)
    and produce a single mortality logit per sample.
    Matches legacy routing behavior; the number of decision capsules here is 1 (binary).
    """
    def __init__(
        self,
        pc_dim: int,
        mc_caps_dim: int,
        num_routing: int = 3,
        dp: float = 0.1,
        act_type: str = "EM",
        layer_norm: bool = False,
        dim_pose_to_vote: int = 0,
    ):
        super().__init__()
        self.pc_dim = int(pc_dim)
        self.mc_caps_dim = int(mc_caps_dim)
        self.num_routing = int(num_routing)

        # Single decision capsule (binary task)
        self.mc_num_caps = 1

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

        self.embedding = nn.Parameter(torch.zeros(self.mc_num_caps, self.mc_caps_dim))

    def forward(self, poses: torch.Tensor, acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            poses: [B, 7, pc_dim]
            acts:  [B, 7, 1]
        Returns:
            final_logit:   [B, 1]
            routing_coef:  [B, 7, mc_num_caps]
        """
        device = torch.device(DEVICE)
        poses = poses.to(device)
        acts  = acts.to(device)

        if not hasattr(self, "_printed_once"):
            self._printed_once = True
            _dbg(f"[cap-head] input poses:{tuple(poses.shape)} acts:{tuple(acts.shape)} "
                 f"(pc_dim={self.pc_dim}, mc_caps_dim={self.mc_caps_dim}, out_caps={self.mc_num_caps}, iters={self.num_routing})")

        # First routing (no prior)
        decision_pose, decision_act, _ = self.mc(poses, acts, 0)

        routing_coef = None
        for n in range(self.num_routing):
            decision_pose, decision_act, routing_coef = self.mc(
                poses, acts, n, decision_pose, decision_act
            )

        decision_logit = torch.einsum('bcd,cd->bc', decision_pose, self.embedding)  # [B,1]

        if not hasattr(self, "_printed_done"):
            self._printed_done = True
            _dbg(f"[cap-head] decision_pose:{tuple(decision_pose.shape)} -> logit:{tuple(decision_logit.shape)}")
            with torch.no_grad():
                _dbg(f"[cap-head] logit stats -> "
                     f"min:{decision_logit.min().item():.4f} max:{decision_logit.max().item():.4f} mean:{decision_logit.mean().item():.4f}")

        return decision_logit, routing_coef


def forward_capsule_from_routes(
    z_unimodal: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
    projector: RoutePrimaryProjector,
    capsule_head: CapsuleMortalityHead,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        z_unimodal: {"L":[B,d], "N":[B,d], "I":[B,d]}
        fusion    : dict of fusion modules {"LN","LI","NI","LNI"}
        projector : per-route Linear -> (pc_dim+1) with sigmoid act (legacy-equivalent)
        capsule_head: CapsuleMortalityHead
    Returns:
        final_logit       : [B,1]
        primary_acts      : [B,7]         
        routing_coefficient: [B,7,1]      
    """
    # Build 7 route embeddings
    route_embs = make_route_inputs(z_unimodal, fusion)  # dict of 7 -> [B,d]
    if not hasattr(forward_capsule_from_routes, "_printed_routes"):
        forward_capsule_from_routes._printed_routes = True
        sizes = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
        _dbg(f"[caps-bridge] routes -> {sizes}")

    # Project to primary capsules (pose + act)
    poses, acts = projector(route_embs)                 # [B,7,pc_dim], [B,7,1]

    # Capsule routing → final logit (+ routing coef)
    final_logit, routing_coef = capsule_head(poses, acts)  # [B,1], [B,7,1]

    primary_acts = acts.squeeze(-1).detach()            # [B,7] 

    # Once-only peek at primary acts
    if not hasattr(forward_capsule_from_routes, "_printed_acts"):
        forward_capsule_from_routes._printed_acts = True
        with torch.no_grad():
            _dbg(f"[caps-bridge] prim_acts:{tuple(primary_acts.shape)} "
                 f"mean:{primary_acts.mean().item():.4f} min:{primary_acts.min().item():.4f} max:{primary_acts.max().item():.4f}")

    return final_logit, primary_acts, routing_coef


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
