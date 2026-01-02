from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from env_config import ROUTES, CFG
import capsule_layers
from mult_model import MULTModel


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

def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    # x: [B,T,D], m: [B,T] with 1=keep
    m = m.float()
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * m.unsqueeze(-1)).sum(dim=1) / denom


def build_route_mask_from_presence(
    hasL: torch.Tensor,
    hasN: torch.Tensor,
    hasI: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if device is None:
        device = hasL.device

    hasL = hasL.to(device=device, dtype=dtype)
    hasN = hasN.to(device=device, dtype=dtype)
    hasI = hasI.to(device=device, dtype=dtype)

    LN  = hasL * hasN
    LI  = hasL * hasI
    NI  = hasN * hasI
    LNI = hasL * hasN * hasI

    canonical = ["L","N","I","LN","NL","LI","IL","NI","IN","LNI"]
    if list(ROUTES) == canonical:
        return torch.stack([hasL, hasN, hasI, LN, LN, LI, LI, NI, NI, LNI], dim=1)

    # Fallback: robust for any custom ROUTES order/content
    B = hasL.shape[0]
    masks = []
    for r in ROUTES:
        m = torch.ones((B,), device=device, dtype=dtype)
        if "L" in r: m = m * hasL
        if "N" in r: m = m * hasN
        if "I" in r: m = m * hasI
        masks.append(m)
    return torch.stack(masks, dim=1)



class MulTRouteBuilder(nn.Module):
    """
    Builds ALL routes using MULTModel.
    Returns each route as [B,d] (mask-aware pooling if [B,T,d]).
    """
    def __init__(self, d: int):
        super().__init__()
        d = int(d)

        layers = int(getattr(CFG, "cross_attn_layers", 2))
        self_layers = int(getattr(CFG, "cross_attn_self_layers", 1))
        num_heads = int(getattr(CFG, "cross_attn_heads", 4))
        attn_dropout = float(getattr(CFG, "cross_attn_dropout", 0.0))
        relu_dropout = float(getattr(CFG, "mult_relu_dropout", 0.0))
        res_dropout  = float(getattr(CFG, "mult_res_dropout", 0.0))
        out_dropout  = float(getattr(CFG, "mult_out_dropout", 0.0))
        embed_dropout = float(getattr(CFG, "mult_embed_dropout", 0.0))
        attn_mask = bool(getattr(CFG, "mult_attn_mask", False))

        # IMPORTANT: MULTModel expects (L, N, I): structured, notes, images
        lonly = True
        nonly = True   # notes
        ionly = True   # images

        self.mult = MULTModel(
            orig_d_l=d, orig_d_n=d, orig_d_i=d,
            d_l=d, d_n=d, d_i=d,
            ionly=ionly, nonly=nonly, lonly=lonly,
            num_heads=num_heads,
            layers=layers,
            self_layers=self_layers,
            attn_dropout=attn_dropout,
            attn_dropout_n=attn_dropout,
            attn_dropout_i=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            out_dropout=out_dropout,
            embed_dropout=embed_dropout,
            attn_mask=attn_mask,
        )

    @staticmethod
    def _zero_pad(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: [B,T] (0/1 or bool)
        return seq * mask.float().unsqueeze(-1)

    @staticmethod
    def _pick_best_mask(xT: int, z: Dict[str, Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        """
        Pick a mask whose T matches x.size(1). Prefer L then N then I.
        """
        for key in ("L", "N", "I"):
            m = z[key]["mask"]
            if m.dim() == 2 and m.size(1) == xT:
                return m
        return None

    def forward(self, z: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        Ls, Lm = z["L"]["seq"], z["L"]["mask"]
        Ns, Nm = z["N"]["seq"], z["N"]["mask"]
        Is, Im = z["I"]["seq"], z["I"]["mask"]

        Ls0 = self._zero_pad(Ls, Lm)
        Ns0 = self._zero_pad(Ns, Nm)
        Is0 = self._zero_pad(Is, Im)

        out = self.mult(Ls0, Ns0, Is0)  # dict of all routes

        expected = set(ROUTES)
        got = set(out.keys())
        if expected != got:
            raise RuntimeError(f"MULTModel route keys mismatch. missing={expected-got}, extra={got-expected}")

        route_embs = {r: out[r] for r in ROUTES}

        if getattr(CFG, "debug_shapes", False) and (not hasattr(self, "_printed_raw_shapes")):
            self._printed_raw_shapes = True
            for r in ROUTES:
                print(f"[mult_raw] {r}: {tuple(route_embs[r].shape)}")

        # Ensure each route is [B,d] for the projector (mask-aware pooling).
        pooled: Dict[str, torch.Tensor] = {}
        for r in ROUTES:
            x = route_embs[r]
            if x.dim() == 3:
                m = self._pick_best_mask(x.size(1), z)
                if m is not None:
                    pooled[r] = masked_mean(x, m)
                else:
                    pooled[r] = x.mean(dim=1)
                    if getattr(CFG, "verbose", False):
                        print(f"[warn] route {r}: no matching mask for T={x.size(1)}; using mean over T")
            elif x.dim() == 2:
                pooled[r] = x
            else:
                raise ValueError(f"Route {r} must be [B,d] or [B,T,d], got {tuple(x.shape)}")

        if getattr(CFG, "verbose", False) and (not hasattr(self, "_printed_once")):
            self._printed_once = True
            for r in ROUTES:
                _peek_tensor(f"MulTRouteBuilder.{r}", pooled[r])
                _nan_guard(f"MulTRouteBuilder.{r}", pooled[r])

        return pooled


def build_fusions(d: int):
    """
    Keeps the old API name, but returns a single fusion module that produces all routes.
    """
    return MulTRouteBuilder(d)


class RoutePrimaryProjector(nn.Module):
    def __init__(self, d_in: int, pc_dim: int):
        super().__init__()
        self.pc_dim = int(pc_dim)
        self.proj = nn.ModuleDict({r: nn.Linear(d_in, self.pc_dim + 1, bias=False) for r in ROUTES})

    def forward(self, route_embs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # poses: [B,R,pc_dim]
        # acts : [B,R]  (sigmoid)
        pcs = [self.proj[r](route_embs[r]) for r in ROUTES]
        pc_all = torch.stack(pcs, dim=1)            # [B, R, pc_dim+1]
        poses = pc_all[:, :, :self.pc_dim]          # [B,R,pc_dim]
        raw_logits = pc_all[:, :, self.pc_dim]      # [B,R]
        acts = torch.sigmoid(raw_logits)            # [B,R]
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
        self.in_n_capsules = len(ROUTES)          # R
        self.in_d_capsules = pc_dim
        self.out_n_capsules = int(num_classes)    # K
        self.out_d_capsules = mc_caps_dim
        self.num_routing = int(num_routing)

        self.capsule = capsule_layers.CapsuleFC(
            in_n_capsules=self.in_n_capsules,
            in_d_capsules=self.in_d_capsules,
            out_n_capsules=self.out_n_capsules,
            out_d_capsules=self.out_d_capsules,
            n_rank=0,
            dp=float(dp) if dp is not None else 0.0,
            dim_pose_to_vote=dim_pose_to_vote,
            uniform_routing_coefficient=False,
            act_type=act_type,
            small_std=False,
        )

        self.embedding = nn.Parameter(torch.zeros(self.out_n_capsules, self.out_d_capsules))
        self.bias = nn.Parameter(torch.zeros(self.out_n_capsules))
        self.out_ln = nn.LayerNorm(self.out_d_capsules) if layer_norm else None


    def forward(
        self,
        prim_pose: torch.Tensor,   # [B,R,pc_dim]
        prim_act: torch.Tensor,    # [B,R]
        uniform_routing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if prim_act.dim() != 2:
            raise ValueError(f"prim_act must be [B,R], got {tuple(prim_act.shape)}")

        curr_act = prim_act.unsqueeze(-1)  # [B,R,1]

        decision_pose = None
        decision_act = None
        routing_coef = None

        for it in range(self.num_routing):
            decision_pose, decision_act, routing_coef = self.capsule(
                input=prim_pose,
                current_act=curr_act,
                num_iter=it,
                next_capsule_value=decision_pose,
                next_act=decision_act,
                uniform_routing=uniform_routing,
            )
        if self.out_ln is not None and decision_pose is not None:
            decision_pose = self.out_ln(decision_pose)

        logits = torch.einsum("bkd,kd->bk", decision_pose, self.embedding) + self.bias

        prim_act_out = prim_act
        return logits, prim_act_out, routing_coef


import torch.nn.functional as F

def orient_routing_coef_BRK(routing_coef, n_routes, n_classes, debug: bool = False, **kwargs):
    """
    Ensure routing_coef is shaped [B, R, K].
    Accepts:
      - [B, R, K]
      - [B, K, R] (transpose)
      - [B, R, K, 1] (squeeze trailing singleton)
    """
    if rc is None:
        return rc

    while rc.dim() > 3 and rc.size(-1) == 1:
        rc = rc.squeeze(-1)

    if rc.dim() != 3:
        raise ValueError(f"routing_coef must be 3D, got shape={tuple(rc.shape)}")

    B, A, C = rc.shape
    if A == n_routes and C == n_classes:
        return rc
    if A == n_classes and C == n_routes:
        return rc.transpose(1, 2)

    raise ValueError(
        f"routing_coef shape mismatch. got={tuple(rc.shape)} "
        f"expected [B,{n_routes},{n_classes}] or [B,{n_classes},{n_routes}]"
    )
    if debug:
        print(f"[orient] in={tuple(routing_coef.shape)} n_routes={n_routes} n_classes={n_classes}")

def routing_coef_to_p_class_given_route_for_report(rc: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Reporting-only: convert routing_coef [B,R,K] to p(class|route) by normalizing over K.

    - If CapsuleFC already returns probs over K, this is effectively a no-op (just re-normalizes safely).
    - If it returns logits/scores, this still produces a valid distribution without ever normalizing over routes.
    """
    if rc is None:
        return rc

    # If it looks like probabilities already (non-negative, sums approx 1), just renormalize safely.
    s = rc.sum(dim=2, keepdim=True)
    looks_prob = (rc.min() >= -1e-6) and ((s - 1.0).abs().mean() < 1e-3)

    if looks_prob:
        return rc / s.clamp_min(eps)

    # Otherwise treat as logits/scores: softmax over classes (K)
    return F.softmax(rc, dim=2)

def normalize_routing_over_classes(rc: torch.Tensor) -> torch.Tensor:
    """
    Paper-style: for each route i, softmax over classes j.
    Input rc is [B,R,K] (logits or scores); output is prob-like with sum_K = 1.
    """
    # softmax over K
    return F.softmax(rc, dim=2)



def forward_capsule_from_routes(
    z_unimodal: Dict[str, Dict[str, torch.Tensor]],          # {"L","N","I"} each has "seq","mask"
    fusion: nn.Module,                                      # MulTRouteBuilder
    projector: RoutePrimaryProjector,
    capsule_head: CapsuleMortalityHead,
    *,
    acts_override: Optional[torch.Tensor] = None,            # [B,R] or [B,R,1]
    route_mask: Optional[torch.Tensor] = None,               # [B,R]
    act_temperature: float = 1.0,
    detach_priors: bool = False,
    return_routing: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:

    # ---------- debug unimodal inputs (once) ---------
    if not hasattr(forward_capsule_from_routes, "_printed_once"):
        forward_capsule_from_routes._printed_once = True
        sizes = ", ".join(
            f"{k}: seq{tuple(v['seq'].shape)} mask{tuple(v['mask'].shape)}"
            for k, v in z_unimodal.items()
        )
        _dbg(f"[caps-bridge] unimodal inputs -> {sizes}")
        for k, v in z_unimodal.items():
            _peek_tensor(f"caps-bridge.uni.{k}.seq", v["seq"])
            _peek_tensor(f"caps-bridge.uni.{k}.mask", v["mask"].float())
            _nan_guard(f"caps-bridge.uni.{k}.seq", v["seq"])

    # ---------- Align devices: fuse + projector should agree ----------
    proj_dev = next(projector.parameters()).device
    proj_dtype = next(projector.parameters()).dtype

    # If fusion has params, use that device/dtype; else fall back to projector device
    try:
        fusion_dev = next(fusion.parameters()).device
        fusion_dtype = next(fusion.parameters()).dtype
    except StopIteration:
        fusion_dev = proj_dev
        fusion_dtype = proj_dtype

    # Move unimodal inputs to fusion device (prevents silent device mismatch)
    z_uni = {}
    for k in ("L", "N", "I"):
        seq = z_unimodal[k]["seq"].to(device=fusion_dev, dtype=fusion_dtype, non_blocking=True)
        mask = z_unimodal[k]["mask"].to(device=fusion_dev, non_blocking=True)
        z_uni[k] = {"seq": seq, "mask": mask}

    # ---------- Build route embeddings (MulT) ----------
    route_embs = fusion(z_uni)

    expected = set(ROUTES)
    got = set(route_embs.keys())
    if expected != got:
        missing = expected - got
        extra = got - expected
        raise RuntimeError(f"Route key mismatch. missing={missing}, extra={extra}")

    # ---------- Move route embs to projector device/dtype ----------
    route_embs = {k: v.to(device=proj_dev, dtype=proj_dtype, non_blocking=True) for k, v in route_embs.items()}

    # ---------- Project to primary capsules ----------
    poses, acts = projector(route_embs)  # poses [B,R,pc_dim], acts [B,R]
    _nan_guard("caps-bridge.poses", poses)
    _nan_guard("caps-bridge.acts", acts)

    # ---------- Priors override ----------
    if acts_override is None:
        acts_prior = acts
    else:
        ao = acts_override
        if ao.dim() == 3 and ao.size(-1) == 1:
            ao = ao.squeeze(-1)
        if ao.dim() != 2:
            raise ValueError(f"acts_override must be [B,R] or [B,R,1], got {tuple(acts_override.shape)}")
        acts_prior = ao.to(device=acts.device, dtype=acts.dtype)

    # ---------- Apply route mask (IMPORTANT: also mask poses) ----------
    # We apply mask AFTER clamps too to avoid “flooring” masked routes back up.
    eps_mask = float(getattr(CFG, "masked_route_eps", 1e-8))
    rm = None
    if route_mask is not None:
        rm = route_mask.to(device=acts_prior.device, dtype=acts_prior.dtype)
        if rm.dim() != 2:
            raise ValueError(f"route_mask must be [B,R], got {tuple(route_mask.shape)}")

        # Mask poses so masked routes contribute ~0 votes regardless of any CapsuleFC quirks.
        poses = poses * rm.unsqueeze(-1)

    # ---------- Temperature on priors ----------
    if act_temperature != 1.0:
        eps = 1e-6
        acts_prior = torch.clamp(acts_prior, eps, 1.0 - eps)
        logits_t = torch.log(acts_prior) - torch.log(1.0 - acts_prior)
        logits_t = logits_t / float(act_temperature)
        acts_prior = torch.sigmoid(logits_t)

    # ---------- Clamp priors (only meaningful for allowed routes) ----------
    prior_floor = float(getattr(CFG, "route_prior_floor", 1e-3))
    prior_ceiling = float(getattr(CFG, "route_prior_ceiling", 0.999))
    lo = prior_floor if prior_floor > 0.0 else 0.0
    hi = prior_ceiling if prior_ceiling > 0.0 else 1.0
    acts_prior = torch.clamp(acts_prior, min=lo, max=hi)

    # Re-apply mask AFTER clamp so masked routes do NOT get raised to floor
    if rm is not None:
        acts_prior = acts_prior * rm + eps_mask * (1.0 - rm)

    prim_act_in = acts_prior.detach() if detach_priors else acts_prior  # [B,R]
    _nan_guard("caps-bridge.prim_act_in", prim_act_in)

    # ---------- Capsule routing ----------
    logits, prim_act_out, routing_coef = capsule_head(
        prim_pose=poses,
        prim_act=prim_act_in,
        uniform_routing=False,
    )

    if prim_act_out is None:
        prim_act_out = prim_act_in

    prim_acts = prim_act_out.detach()
    _peek_tensor("caps-bridge.prim_acts", prim_acts)
    _nan_guard("caps-bridge.prim_acts", prim_acts)

    # ---------- Routing coef: keep paper semantics; DO NOT renormalize in train/eval ----------
    if not return_routing:
        routing_coef = None
    else:
        if routing_coef is not None:
            # Only make sure it's [B,R,K]. Do NOT softmax here; that's reporting-only.
            routing_coef = orient_routing_coef_BRK(
                routing_coef,
                n_routes=len(ROUTES),
                n_classes=logits.shape[-1],
            )

            # Optional: apply route mask WITHOUT any renormalization (no softmax over routes).
            # This preserves the meaning of "per-route distribution over classes" when you later normalize over K for reporting.
            if rm is not None:
                routing_coef = routing_coef * rm.unsqueeze(-1)


    # ---------- Return ----------
    return logits, prim_acts, route_embs, routing_coef



__all__ = [
    "MulTRouteBuilder",
    "build_fusions",
    "RoutePrimaryProjector",
    "CapsuleMortalityHead",
    "build_route_mask_from_presence",
    "forward_capsule_from_routes",
    "orient_routing_coef_BRK",
    "routing_coef_to_p_class_given_route_for_report",
]
