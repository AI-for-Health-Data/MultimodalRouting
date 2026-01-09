from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env_config import ROUTES, DEVICE, CFG
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

def _gate_norm_routes(
    rc: torch.Tensor,
    *,
    routes_dim: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    rc = rc.float().clamp_min(0.0)
    denom = rc.sum(dim=routes_dim, keepdim=True)   # [B,1,K]
    R = rc.size(routes_dim)
    rc_norm = rc / denom.clamp_min(eps)
    small = (denom < eps)
    if small.any():
        uniform = torch.full_like(rc_norm, 1.0 / float(R))
        rc_norm = torch.where(small.expand_as(rc_norm), uniform, rc_norm)
    rc_norm = rc_norm.clamp_min(eps)
    rc_norm = rc_norm / rc_norm.sum(dim=routes_dim, keepdim=True).clamp_min(eps)
    return rc_norm

def _mask_and_renorm_over_routes(
    rc: torch.Tensor,
    route_mask: Optional[torch.Tensor],
    *,
    routes_dim: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    if route_mask is None:
        return rc / rc.sum(dim=routes_dim, keepdim=True).clamp_min(eps)

    if route_mask.ndim == 1:
        view = [1] * rc.ndim
        view[routes_dim] = -1
        m = route_mask.to(rc.device).float().view(*view)
    elif route_mask.ndim == 2:
        if routes_dim != 1:
            raise ValueError("route_mask [B,R] currently supported only for routes_dim=1")
        m = route_mask.to(rc.device).float().unsqueeze(-1)  # [B,R,1]
    else:
        raise ValueError(f"route_mask must be [R] or [B,R], got {tuple(route_mask.shape)}")

    rc = rc * m
    denom = rc.sum(dim=routes_dim, keepdim=True)  # [B,1,K]
    active = m.sum(dim=routes_dim, keepdim=True).clamp_min(1.0)  # [B,1,1] or [1,1,1]
    uniform = m / active                                        
    rc = torch.where(denom > eps, rc / denom.clamp_min(eps), uniform.expand_as(rc))
    return rc


def route_given_pheno(
    rc_k_given_r: torch.Tensor,      # [B,R,K] == query_key
    prim_act: torch.Tensor,          # [B,R] or [B,R,1]
    route_mask: torch.Tensor | None = None,  # [R] bool or 0/1
    eps: float = 1e-10
) -> torch.Tensor:
    # make prim_act [B,R,1]
    if prim_act.ndim == 3:
        prim = prim_act
    else:
        prim = prim_act.unsqueeze(-1)
    resp = rc_k_given_r * prim  # [B,R,K]
    if route_mask is not None:
        m = route_mask.view(1, -1, 1).type_as(resp)
        resp = resp * m
    denom = resp.sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1,K]
    rc_r_given_k = resp / denom
    return rc_r_given_k  # [B,R,K], sums to 1 over R (dim=1)


def _normalize_routing_coef(
    rc: Optional[torch.Tensor],
    *,
    mode: Optional[str] = None,
    expect_routes: Optional[int] = None,
    eps: Optional[float] = None,
) -> Optional[torch.Tensor]:
    if rc is None:
        return None
    if rc.ndim != 3:
        raise ValueError(f"routing_coef must be 3D, got shape={tuple(rc.shape)}")

    mode = str(mode or getattr(CFG, "routing_coef_mode", "gate_norm")).lower().strip()
    eps = float(eps if eps is not None else getattr(CFG, "routing_coef_eps", 1e-6))

    B, d1, d2 = rc.shape
    if expect_routes is None:
        expect_routes = len(ROUTES)
    expect_routes = int(expect_routes)

    if d1 == expect_routes:
        rc_brk = rc
    elif d2 == expect_routes:
        rc_brk = rc.transpose(1, 2)  # [B,K,R] -> [B,R,K]
    else:
        rc_brk = rc

    if mode == "none":
        return rc_brk

    if mode == "gate_norm":
        return _gate_norm_routes(rc_brk, routes_dim=1, eps=eps)

    raise ValueError(f"Unknown routing_coef_mode={mode!r}")

def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    # x: [B,T,D], m: [B,T] with 1=keep
    m = m.float()
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * m.unsqueeze(-1)).sum(dim=1) / denom

class CrossAttentionFusion(nn.Module):
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

        return self.out(z)  # [B,D]

class TriTokenAttentionFusion(nn.Module):

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
        kv_pad = (m < 0.5)
        out, _ = self.attn(query=q, key=kv, value=kv, key_padding_mask=kv_pad, need_weights=False)
        z = out[:, 0, :]  # [B,D]
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

    LN  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=getattr(CFG, "cross_attn_pool", "mean")).to(dev)
    NL  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=getattr(CFG, "cross_attn_pool", "mean")).to(dev)
    LI  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=getattr(CFG, "cross_attn_pool", "mean")).to(dev)
    IL  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=getattr(CFG, "cross_attn_pool", "mean")).to(dev)
    NI  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=getattr(CFG, "cross_attn_pool", "mean")).to(dev)
    IN  = CrossAttentionFusion(d, n_heads=h, attn_dropout=p, pool=getattr(CFG, "cross_attn_pool", "mean")).to(dev)
    LNI = TriTokenAttentionFusion(d, n_heads=h, attn_dropout=p).to(dev)

    return {"LN": LN, "NL": NL, "LI": LI, "IL": IL, "NI": NI, "IN": IN, "LNI": LNI}

@torch.no_grad()
def _safe_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone()

def make_route_inputs(z, fusion) -> Dict[str, torch.Tensor]:
    Ls, Ns, Is = z["L"]["seq"], z["N"]["seq"], z["I"]["seq"]
    Lm, Nm, Im = z["L"]["mask"], z["N"]["mask"], z["I"]["mask"]

    routes = {
        # unimodal
        "L":  z["L"]["pool"],
        "N":  z["N"]["pool"],
        "I":  z["I"]["pool"],

        # bimodal directional (keep these exactly)
        "LN": fusion["LN"](Ls, Lm, Ns, Nm),  # L attends N
        "NL": fusion["NL"](Ns, Nm, Ls, Lm),  # N attends L
        "LI": fusion["LI"](Ls, Lm, Is, Im),  # L attends I
        "IL": fusion["IL"](Is, Im, Ls, Lm),  # I attends L
        "NI": fusion["NI"](Ns, Nm, Is, Im),  # N attends I
        "IN": fusion["IN"](Is, Im, Ns, Nm),  # I attends N

        # trimodal
        "LNI": fusion["LNI"](Ls, Lm, Ns, Nm, Is, Im),
    }
    expected = set(ROUTES)
    got = set(routes.keys())
    if expected != got:
        missing = expected - got
        extra = got - expected
        raise RuntimeError(f"[make_route_inputs] Route key mismatch. missing={missing}, extra={extra}")
    return routes



def make_route_inputs_mult(z, multmodel: MULTModel):
    Ls, Ns, Is = z["L"]["seq"], z["N"]["seq"], z["I"]["seq"]
    Lm, Nm, Im = z["L"]["mask"], z["N"]["mask"], z["I"]["mask"]
    Lp, Np, Ip = z["L"]["pool"], z["N"]["pool"], z["I"]["pool"]

    routes = multmodel.forward_from_encoders(
        L_seq=Ls, N_seq=Ns, I_seq=Is,
        mL=Lm, mN=Nm, mI=Im,
        L_pool=Lp, N_pool=Np, I_pool=Ip,
    )

    expected = set(ROUTES)
    got = set(routes.keys())
    if expected != got:
        missing = expected - got
        extra = got - expected
        raise RuntimeError(f"[make_route_inputs_mult] Route key mismatch. missing={missing}, extra={extra}")

    return routes


class RoutePrimaryProjector(nn.Module):
    def __init__(self, d_in: int, pc_dim: int):
        super().__init__()
        self.pc_dim = int(pc_dim)

        self.proj = nn.ModuleDict({
            r: nn.Linear(d_in, self.pc_dim + 1, bias=True)
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


class RouteDimAdapter(nn.Module):
    def __init__(self, d_in: int, d_l: int, d_n: int, d_i: int):
        super().__init__()
        d_in = int(d_in); d_l = int(d_l); d_n = int(d_n); d_i = int(d_i)

        def maybe_lin(src: int, dst: int) -> nn.Module:
            return nn.Identity() if src == dst else nn.Linear(src, dst, bias=False)

        self.adapt = nn.ModuleDict({
            # language-like routes (d_l)
            "L":   maybe_lin(d_l, d_in),
            "LN":  maybe_lin(d_l, d_in),
            "LI":  maybe_lin(d_l, d_in),
            "LNI": maybe_lin(d_l, d_in),

            # note-like routes (d_n)
            "N":   maybe_lin(d_n, d_in),
            "NL":  maybe_lin(d_n, d_in),
            "NI":  maybe_lin(d_n, d_in),

            # image-like routes (d_i)
            "I":   maybe_lin(d_i, d_in),
            "IL":  maybe_lin(d_i, d_in),
            "IN":  maybe_lin(d_i, d_in),
        })

    def forward(self, route_embs_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Keep keys identical; just map each tensor
        out: Dict[str, torch.Tensor] = {}
        for r in ROUTES:
            x = route_embs_in[r]
            out[r] = self.adapt[r](x)
        return out

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
            dp=dp,                          
            dim_pose_to_vote=dim_pose_to_vote,
            uniform_routing_coefficient=False,
            act_type=act_type,
            small_std=True,                  
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

            next_pose = decision_pose
            next_act  = decision_act

            if next_pose is not None:
                B = next_pose.size(0)
                M = next_pose.size(1)

                if next_act is None:
                    next_act = torch.ones(
                        (B, M),
                        device=next_pose.device,
                        dtype=prim_act.dtype,  
                    )
                else:
                    if next_act.dim() == 3 and next_act.size(-1) == 1:
                        next_act = next_act.squeeze(-1)
                    if next_act.dim() != 2:
                        raise ValueError(f"next_act must be [B,M], got {tuple(next_act.shape)}")

            decision_pose, decision_act, routing_coef = self.capsule(
                input=prim_pose,
                current_act=prim_act,
                num_iter=it,
                next_capsule_value=next_pose,
                next_act=next_act,                 
                uniform_routing=uniform_routing,
            )


        logits = torch.einsum("bmd, md -> bm", decision_pose, self.embedding) + self.bias

        prim_act_out = prim_act.squeeze(-1)  # [B, len(ROUTES)]

        routing_coef = _normalize_routing_coef(
            routing_coef,
            mode="none",                     
            expect_routes=self.in_n_capsules,
        )


        return logits, prim_act_out, routing_coef

def forward_capsule_from_route_dict(
    route_embs_in: Dict[str, torch.Tensor],            # {"L","N","I","LN","NL","LI","IL","NI","IN","LNI"} each [B,d]
    projector: RoutePrimaryProjector,                  # d -> (pc_dim+1) per route
    capsule_head: CapsuleMortalityHead,                # -> [B,K] logits
    *,
    acts_override: Optional[torch.Tensor] = None,      # [B,len(ROUTES),1] optional external priors
    route_mask: Optional[torch.Tensor] = None,         # [B,len(ROUTES)]  (1=keep, 0=mask)
    act_temperature: float = 1.0,                      # >1 softer, <1 sharper
    detach_priors: bool = False,
    return_routing: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    expected = set(ROUTES)
    got = set(route_embs_in.keys())
    if expected != got:
        missing = expected - got
        extra = got - expected
        raise RuntimeError(f"Route key mismatch. missing={missing}, extra={extra}")

    dev = next(projector.parameters()).device
    dtype = next(projector.parameters()).dtype

    route_embs: Dict[str, torch.Tensor] = {}
    for r in ROUTES:
        x = route_embs_in[r]
        if not torch.is_tensor(x):
            raise TypeError(f"route_embs_in['{r}'] must be a Tensor, got {type(x)}")

        # allow [B,d] OR [B,1,d] (squeeze the middle)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"route_embs_in['{r}'] must be [B,d] (or [B,1,d]), got {tuple(x.shape)}")

        route_embs[r] = x.to(device=dev, dtype=dtype, non_blocking=True)

    if getattr(CFG, "verbose", False) and not hasattr(forward_capsule_from_route_dict, "_printed_once"):
        forward_capsule_from_route_dict._printed_once = True
        _dbg("[caps-bridge] using precomputed route embeddings (e.g., MULTModel)")
        for r in ROUTES:
            _peek_tensor(f"caps-bridge.route.{r}", route_embs[r])
            _nan_guard(f"caps-bridge.route.{r}", route_embs[r])

    poses, acts = projector(route_embs)

    acts_prior = acts if acts_override is None else acts_override.to(device=acts.device, dtype=acts.dtype)

    if route_mask is not None:
        acts_prior = acts_prior * route_mask.unsqueeze(-1) + 1e-6  # keep grads

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
        prim_act=acts_for_caps.squeeze(-1),  # [B,R]
        uniform_routing=False,
    )

    if routing_coef is not None:
        routing_coef = _normalize_routing_coef(
            routing_coef,
            mode="none",                  
            expect_routes=len(ROUTES),
        )
        routing_coef = _mask_and_renorm_over_routes(
            routing_coef, route_mask, routes_dim=1, eps=1e-6
        )


    prim_acts = prim_act_out.detach()
    _peek_tensor("caps-bridge.prim_acts", prim_acts)
    _nan_guard("caps-bridge.prim_acts", prim_acts)

    if not return_routing:
        routing_coef = None

    return logits, prim_acts, route_embs, routing_coef

def forward_capsule_from_multmodel(
    multmodel: nn.Module,                                  # MULTModel
    x_l: torch.Tensor, x_n: torch.Tensor, x_i: torch.Tensor,# each [B,T,D]
    projector: RoutePrimaryProjector,
    capsule_head: CapsuleMortalityHead,
    *,
    route_adapter: Optional[RouteDimAdapter] = None,        # if dims differ, pass adapter
    acts_override: Optional[torch.Tensor] = None,           # [B,len(ROUTES),1]
    route_mask: Optional[torch.Tensor] = None,              # [B,len(ROUTES)]
    act_temperature: float = 1.0,
    detach_priors: bool = False,
    return_routing: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    route_embs_in = multmodel(x_l, x_n, x_i)  # keys must match ROUTES, tensors usually [B,d_*]

    expected = set(ROUTES)
    got = set(route_embs_in.keys())
    if expected != got:
        missing = expected - got
        extra = got - expected
        raise RuntimeError(f"[mult->caps] Route key mismatch. missing={missing}, extra={extra}")

    # Optional: adapt dims to projector's d_in
    if route_adapter is not None:
        route_embs_in = route_adapter(route_embs_in)

    # Now reuse the canonical bridge
    return forward_capsule_from_route_dict(
        route_embs_in=route_embs_in,
        projector=projector,
        capsule_head=capsule_head,
        acts_override=acts_override,
        route_mask=route_mask,
        act_temperature=act_temperature,
        detach_priors=detach_priors,
        return_routing=return_routing,
    )


__all__ = [
    "CrossAttentionFusion",
    "TriTokenAttentionFusion",
    "build_fusions",
    "make_route_inputs",
    "RoutePrimaryProjector",
    "RouteDimAdapter",
    "CapsuleMortalityHead",
    "forward_capsule_from_route_dict",
    "forward_capsule_from_multmodel",
    "route_given_pheno",
]
