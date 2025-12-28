from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env_config import ROUTES, DEVICE, CFG
import capsule_layers

def build_route_mask_from_presence(
    hasL: torch.Tensor,
    hasN: torch.Tensor,
    hasI: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build [B, len(ROUTES)] route mask aligned EXACTLY with env_config.ROUTES.
    1 = route allowed, 0 = route disallowed.

    Rules:
      L  allowed if hasL
      N  allowed if hasN
      I  allowed if hasI
      LN/NL allowed if hasL & hasN
      LI/IL allowed if hasL & hasI
      NI/IN allowed if hasN & hasI
      LNI allowed if hasL & hasN & hasI
    """
    if device is None:
        device = hasL.device

    hasL = hasL.to(device=device).float()
    hasN = hasN.to(device=device).float()
    hasI = hasI.to(device=device).float()

    B = hasL.size(0)
    R = len(ROUTES)
    mask = torch.zeros(B, R, device=device, dtype=dtype)

    route2idx = {r: i for i, r in enumerate(ROUTES)}

    # unimodal
    mask[:, route2idx["L"]] = hasL
    mask[:, route2idx["N"]] = hasN
    mask[:, route2idx["I"]] = hasI

    # bimodal (both required)
    LN_ok = hasL * hasN
    LI_ok = hasL * hasI
    NI_ok = hasN * hasI

    mask[:, route2idx["LN"]] = LN_ok
    mask[:, route2idx["NL"]] = LN_ok
    mask[:, route2idx["LI"]] = LI_ok
    mask[:, route2idx["IL"]] = LI_ok
    mask[:, route2idx["NI"]] = NI_ok
    mask[:, route2idx["IN"]] = NI_ok

    # trimodal
    mask[:, route2idx["LNI"]] = hasL * hasN * hasI

    return mask

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

class CrossAttentionFusion(nn.Module):
    """
    Directional cross-attention:
      A attends to B:
        Q=A, K/V=B -> output aligned to A tokens -> pool over A -> [B,D]
    """
    def __init__(
        self,
        d: int,
        n_heads: int = 8,
        attn_dropout: float = 0.0,
        pool: str = "mean",        
        ff_mult: int = 4,
    ):
        super().__init__()
        self.pool = pool

        self.attn = nn.MultiheadAttention(
            d, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d),
            nn.ReLU(),
            nn.Linear(ff_mult * d, d),
        )
        self.ln2 = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(
        self,
        A: torch.Tensor, mA: torch.Tensor,
        B: torch.Tensor, mB: torch.Tensor
    ) -> torch.Tensor:
        # A: [B,TA,D], mA:[B,TA] (1=keep)
        # B: [B,TB,D], mB:[B,TB] (1=keep)
        mA = mA.float()
        mB = mB.float()
        # Safety: if B has no valid tokens for a sample, attention is undefined-ish.
        # We'll still run attention but will force output for those samples to zeros.
        validB = (mB > 0.5).any(dim=1)  # [B]

        key_pad = (mB < 0.5)  # True=PAD for MultiheadAttention
        A2B, _ = self.attn(query=A, key=B, value=B, key_padding_mask=key_pad, need_weights=False)

        X = self.ln1(A + A2B)
        X = self.ln2(X + self.ff(X))
        if self.pool == "first":
            # first valid token in A (fallback to 0 if all padded)
            has_any = (mA > 0.5).any(dim=1)
            idx = torch.where(
                has_any,
                (mA > 0.5).float().argmax(dim=1),
                torch.zeros_like(has_any, dtype=torch.long),
            )
            z = X[torch.arange(X.size(0), device=X.device), idx]  # [B,D]
        else:
            z = masked_mean(X, mA)
        if not validB.all():
            z = z * validB.float().unsqueeze(-1)
        return self.out(z)


class TriTokenAttentionFusion(nn.Module):
    """
    Learned query token attends to concat([L_seq, N_seq, I_seq]) -> [B,D]
    """
    def __init__(self, d: int, n_heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=attn_dropout, batch_first=True)
        self.ln_kv = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, L_seq, mL, N_seq, mN, I_seq, mI) -> torch.Tensor:
        B = L_seq.size(0)
        q = self.q.expand(B, 1, -1)  # [B,1,D]
        kv = torch.cat([L_seq, N_seq, I_seq], dim=1)  # [B, Tsum, D]
        kv = self.ln_kv(kv)
        m = torch.cat([mL, mN, mI], dim=1)            # [B, Tsum]
        validKV = (m > 0.5).any(dim=1)  # [B]

        kv_pad = (m < 0.5)
        out, _ = self.attn(query=q, key=kv, value=kv, key_padding_mask=kv_pad, need_weights=False)
        z = out[:, 0, :]  # [B,D]
        if not validKV.all():
            z = z * validKV.float().unsqueeze(-1)
        return self.out(z)


def build_fusions(d: int, feature_mode: str = "seq", p_drop: float = 0.0):
    dev = torch.device(DEVICE)
    h = int(getattr(CFG, "cross_attn_heads", 8))
    p = float(getattr(CFG, "cross_attn_dropout", p_drop))
    pool = str(getattr(CFG, "cross_attn_pool", "mean")).lower().strip()
    if pool not in {"mean", "first"}:
        if getattr(CFG, "verbose", False):
            print(f"[build_fusions] invalid cross_attn_pool={pool}; using 'mean'")
        pool = "mean"

    # Use p (from CFG) consistently, not p_drop
    LN  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=pool).to(dev)
    NL  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=pool).to(dev)
    LI  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=pool).to(dev)
    IL  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=pool).to(dev)
    NI  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=pool).to(dev)
    IN  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=pool).to(dev)
    LNI = TriTokenAttentionFusion(d, n_heads=h, attn_dropout=p).to(dev)

    return {"LN": LN, "NL": NL, "LI": LI, "IL": IL, "NI": NI, "IN": IN, "LNI": LNI}

@torch.no_grad()
def _safe_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone()

def make_route_inputs(z, fusion):
    Ls, Ns, Is = z["L"]["seq"], z["N"]["seq"], z["I"]["seq"]
    Lm, Nm, Im = z["L"]["mask"], z["N"]["mask"], z["I"]["mask"]

    routes = {
        "L":  z["L"]["pool"],
        "N":  z["N"]["pool"],
        "I":  z["I"]["pool"],
        "LN": fusion["LN"](Ls, Lm, Ns, Nm),
        "NL": fusion["NL"](Ns, Nm, Ls, Lm),
        "LI": fusion["LI"](Ls, Lm, Is, Im),
        "IL": fusion["IL"](Is, Im, Ls, Lm),
        "NI": fusion["NI"](Ns, Nm, Is, Im),
        "IN": fusion["IN"](Is, Im, Ns, Nm),
        "LNI": fusion["LNI"](Ls, Lm, Ns, Nm, Is, Im),
    }
    return routes

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
        pc_all = torch.stack(pcs, dim=1)            # [B, len(ROUTES), pc_dim+1]
        poses = pc_all[:, :, :self.pc_dim]
        raw_logits = pc_all[:, :, self.pc_dim:]     # [B, len(ROUTES), 1]

        acts = torch.sigmoid(raw_logits)            # no extra bias term
        return poses, acts

class CapsuleMortalityHead(nn.Module):
    """
    Wrapper around CapsuleFC (Multimodal Routing) to produce logits for K phenotypes.

    Inputs:
        prim_pose: [B, len(ROUTES), pc_dim]
        prim_act:  [B, len(ROUTES)] or [B, len(ROUTES), 1]

    Outputs:
        logits:       [B, K]
        prim_act_out: [B, len(ROUTES)]      (primary route activations, for analysis)
        routing_coef: [B, len(ROUTES), K]   (routing coefficients, query_key)
    """
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
        self.in_n_capsules = len(ROUTES)
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

        # Decision capsule embedding → logits
        self.embedding = nn.Parameter(
            torch.zeros(self.out_n_capsules, self.out_d_capsules)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_n_capsules))
        self.nonlinear_act = nn.Sequential()

    def forward(
        self,
        prim_pose: torch.Tensor,   # [B, len(ROUTES), pc_dim]
        prim_act: torch.Tensor,    # [B, len(ROUTES)] or [B, len(ROUTES), 1]
        uniform_routing: bool = False,
    ):
        # Ensure activations are [B, len(ROUTES), 1] (CapsuleFC expects [B, N_in, 1])
        if prim_act.dim() == 2:
            prim_act = prim_act.unsqueeze(-1)
        elif prim_act.dim() == 3 and prim_act.size(-1) == 1:
            pass
        else:
            raise ValueError(f"prim_act must be [B,len(ROUTES)] or [B,len(ROUTES),1], got {prim_act.shape}")

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

        prim_act_out = prim_act.squeeze(-1)  # [B, len(ROUTES)]
        return logits, prim_act_out, routing_coef


def forward_capsule_from_routes(
    z_unimodal: Dict[str, Dict[str, torch.Tensor]],          # {"L","N","I"} each [B,d]
    fusion: Dict[str, nn.Module],  # {"LN","NL","LI","IL","NI","IN","LNI"}
    projector: RoutePrimaryProjector,             # d -> (pc_dim+1) per route
    capsule_head: CapsuleMortalityHead,           # -> [B,K] logits
    *,
    acts_override: Optional[torch.Tensor] = None, # [B,len(ROUTES),1] (optional external priors)
    route_mask: Optional[torch.Tensor] = None,    # [B,len(ROUTES)]  (1=keep, 0=mask)
    act_temperature: float = 1.0,                 # >1 = softer, <1 = sharper
    detach_priors: bool = False, 
    return_routing: bool = True,                  
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Bridge from unimodal pooled embeddings → len(ROUTES)-route capsules → logits.

    Returns:
        logits:       [B, K]
        prim_acts:    [B, len(ROUTES)]       (activations actually used; detached for logging)
        route_embs:   dict of len(ROUTES) -> [B, d]
        routing_coef: [B, len(ROUTES), K] or None
    """
    if not hasattr(forward_capsule_from_routes, "_printed_once"):
        forward_capsule_from_routes._printed_once = True
        sizes = ", ".join(f"{k}:pool{tuple(v['pool'].shape)} seq{tuple(v['seq'].shape)}" for k,v in z_unimodal.items())
        _dbg(f"[caps-bridge] unimodal -> {sizes}")
        for k, v in z_unimodal.items():
            _peek_tensor(f"caps-bridge.uni.{k}.pool", v["pool"])
            _peek_tensor(f"caps-bridge.uni.{k}.seq", v["seq"])
            _nan_guard(f"caps-bridge.uni.{k}.pool", v["pool"])
            _nan_guard(f"caps-bridge.uni.{k}.seq", v["seq"])


    # Build 10 route embeddings from unimodal
    route_embs = make_route_inputs(z_unimodal, fusion)
    expected = set(ROUTES)
    got = set(route_embs.keys())
    if expected != got:
        missing = expected - got
        extra = got - expected
        raise RuntimeError(f"Route key mismatch. missing={missing}, extra={extra}")

    dev = next(projector.parameters()).device
    dtype = next(projector.parameters()).dtype
    route_embs = {
        k: v.to(device=dev, dtype=dtype, non_blocking=True)
        for k, v in route_embs.items()
    }

    # Project to primary capsules: poses [B,len(ROUTES),pc_dim], acts [B,len(ROUTES),1]
    poses, acts = projector(route_embs)

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

    # Clamp priors to avoid collapse / extremes
    prior_floor = float(getattr(CFG, "route_prior_floor", 1e-3))
    prior_ceiling = float(getattr(CFG, "route_prior_ceiling", 0.999))
    lo = prior_floor if prior_floor > 0.0 else 0.0
    hi = prior_ceiling if prior_ceiling > 0.0 else 1.0
    acts_prior = torch.clamp(acts_prior, min=lo, max=hi)

    acts_for_caps = acts_prior.detach() if detach_priors else acts_prior

    # Call capsule head in Multimodal Routing style
    logits, prim_act_out, routing_coef = capsule_head(
        prim_pose=poses,
        prim_act=acts_for_caps.squeeze(-1),  # [B,len(ROUTES)]
        uniform_routing=False,
    )

    prim_acts = prim_act_out.detach()  # [B,len(ROUTES)] for logging only
    _peek_tensor("caps-bridge.prim_acts", prim_acts)
    _nan_guard("caps-bridge.prim_acts", prim_acts)

    if not hasattr(forward_capsule_from_routes, "_printed_routes"):
        forward_capsule_from_routes._printed_routes = True
        sizes = ", ".join(f"{k}:pool{tuple(v['pool'].shape)} seq{tuple(v['seq'].shape)}" for k,v in z_unimodal.items())
        _dbg(f"[caps-bridge] routes -> {sizes}")

    return logits, prim_acts, route_embs, routing_coef


__all__ = [
    "CrossAttentionFusion",
    "TriTokenAttentionFusion",
    "build_fusions",
    "make_route_inputs",
    "RoutePrimaryProjector",
    "CapsuleMortalityHead",
    "forward_capsule_from_routes",
    "build_route_mask_from_presence",
]
