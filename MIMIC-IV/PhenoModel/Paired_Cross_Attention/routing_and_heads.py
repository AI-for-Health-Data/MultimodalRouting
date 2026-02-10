from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
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

def route_given_pheno(q_brk: torch.Tensor, route_mask=None, eps=1e-10):
    resp = q_brk
    if route_mask is not None:
        m = route_mask
        if m.ndim == 1: m = m.view(1, -1, 1)
        elif m.ndim == 2: m = m.unsqueeze(-1)
        m = m.to(device=resp.device, dtype=resp.dtype)
        resp = resp * m
    denom = resp.sum(dim=1, keepdim=True).clamp_min(eps)
    return resp / denom  # sums to 1 over routes


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

    mode = str(mode or getattr(CFG, "routing_coef_mode", "none")).lower().strip()
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
    raise ValueError(f"Unknown routing_coef_mode={mode!r}")

def make_route_inputs_mult(z, multmodel: MULTModel):
    Ls, Ns, Is = z["L"]["seq"], z["N"]["seq"], z["I"]["seq"]
    Lm, Nm, Im = z["L"].get("mask", None), z["N"].get("mask", None), z["I"].get("mask", None)

    routes = multmodel(
        x_l=Ls, x_n=Ns, x_i=Is,
        mL=Lm, mN=Nm, mI=Im,
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
        self.d_in = int(d_in)          
        self.pc_dim = int(pc_dim)
        self.proj = nn.ModuleDict({
            r: nn.Linear(self.d_in, self.pc_dim + 1, bias=True)
            for r in ROUTES
        })

    def forward(self, route_embs):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        pcs = [self.proj[r](route_embs[r].to(device=device, dtype=dtype)) for r in ROUTES]
        pc_all = torch.stack(pcs, dim=1)            # [B, len(ROUTES), pc_dim+1]
        poses = pc_all[:, :, :self.pc_dim]
        raw_logits = pc_all[:, :, self.pc_dim:]     # [B, len(ROUTES), 1]

        acts = torch.sigmoid(raw_logits)           
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
        self.pose_to_mc = nn.Linear(self.in_d_capsules, self.out_d_capsules, bias=False)
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
        route_mask: Optional[torch.Tensor] = None,   # <--- ADD THIS

    ):
        if prim_act.dim() == 2:
            prim_act = prim_act.unsqueeze(-1)
        elif prim_act.dim() == 3 and prim_act.size(-1) == 1:
            pass
        else:
            raise ValueError(f"prim_act must be [B,len(ROUTES)] or [B,len(ROUTES),1], got {prim_act.shape}")

        if route_mask is not None:
            rm = route_mask
            if rm.ndim == 1:
                rm = rm.view(1, -1, 1).expand(prim_pose.size(0), -1, 1)   # [B,R,1]
            elif rm.ndim == 2:
                rm = rm.unsqueeze(-1)                                     # [B,R,1]
            else:
                raise ValueError(f"route_mask must be [R] or [B,R], got {tuple(rm.shape)}")

            rm = rm.to(device=prim_pose.device, dtype=prim_pose.dtype)
            prim_pose = prim_pose * rm
            prim_act  = prim_act  * rm

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
        # alpha: [B,R]
        alpha = prim_act.squeeze(-1)  # [B,R]

        q_brk = _normalize_routing_coef(
            routing_coef,
            mode="none",
            expect_routes=self.in_n_capsules,
        )  # [B,R,K]

        R_brk = route_given_pheno(q_brk, route_mask=route_mask)  # [B,R,K]
        alpha = prim_act.squeeze(-1)                              # [B,R]  (after mask! see next section)
        d_bkp = torch.einsum("brk, br, brp -> bkp", R_brk, alpha, prim_pose)

        # Score
        d_bkm = self.pose_to_mc(d_bkp)  # [B,K,mc_caps_dim]
        logits = torch.einsum("bkm, km -> bk", d_bkm, self.embedding) + self.bias

        prim_act_out = alpha  # [B,R]
        return logits, alpha, R_brk



def forward_capsule_from_route_dict(
    route_embs_in: Dict[str, torch.Tensor],            
    projector: RoutePrimaryProjector,                  
    capsule_head: CapsuleMortalityHead,                
    *,
    acts_override: Optional[torch.Tensor] = None,  
    route_mask: Optional[torch.Tensor] = None,         
    act_temperature: float = 1.0,                      
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
        rm = route_mask
        if rm.ndim == 1:
            rm = rm.view(1, -1).expand(acts_prior.size(0), -1)  # [B,R]
        elif rm.ndim != 2:
            raise ValueError(f"route_mask must be [R] or [B,R], got {tuple(rm.shape)}")

        rm = rm.to(device=acts_prior.device, dtype=acts_prior.dtype)
        keep = rm.unsqueeze(-1).bool()  # [B,R,1]

        # hard mask: masked routes stay exactly 0
        acts_prior = acts_prior * rm.unsqueeze(-1)
    else:
        keep = None


    if act_temperature != 1.0 and keep is not None:
        T = float(act_temperature)
        x32 = acts_prior[keep].to(torch.float32)
        x32 = torch.clamp(x32, 1e-6, 1.0 - 1e-6)
        logits32 = torch.log(x32) - torch.log1p(-x32)
        y32 = torch.sigmoid(logits32 / T)

        acts_prior = acts_prior.clone()
        acts_prior[keep] = y32.to(dtype=acts_prior.dtype)

    prior_floor = float(getattr(CFG, "route_prior_floor", 1e-3))
    prior_ceiling = float(getattr(CFG, "route_prior_ceiling", 0.999))
    lo = prior_floor if prior_floor > 0.0 else 0.0
    hi = prior_ceiling if prior_ceiling > 0.0 else 1.0

    if keep is None:
        acts_prior = torch.clamp(acts_prior, min=lo, max=hi)
    else:
        acts_prior = acts_prior.clone()
        acts_prior[keep] = torch.clamp(acts_prior[keep], min=lo, max=hi)

    acts_for_caps = acts_prior.detach() if detach_priors else acts_prior

    logits, prim_act_out, R_brk = capsule_head(
        prim_pose=poses,
        prim_act=acts_for_caps.squeeze(-1),
        uniform_routing=False,
        route_mask=route_mask,
    )
    q_brk = None  


    prim_acts = prim_act_out.detach()
    _peek_tensor("caps-bridge.prim_acts", prim_acts)
    _nan_guard("caps-bridge.prim_acts", prim_acts)

    if not return_routing:
        R_brk = None
    return logits, prim_acts, route_embs, R_brk


def forward_capsule_from_multmodel(
    multmodel: nn.Module,                                  # MULTModel
    x_l: torch.Tensor, x_n: torch.Tensor, x_i: torch.Tensor,
    projector: RoutePrimaryProjector,
    capsule_head: CapsuleMortalityHead,
    *,
    mL: Optional[torch.Tensor] = None,
    mN: Optional[torch.Tensor] = None,
    mI: Optional[torch.Tensor] = None,
    route_adapter: Optional[RouteDimAdapter] = None,        
    acts_override: Optional[torch.Tensor] = None,           # [B,len(ROUTES),1]
    route_mask: Optional[torch.Tensor] = None,              # [B,len(ROUTES)]
    act_temperature: float = 1.0,
    detach_priors: bool = False,
    return_routing: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    route_embs_in = multmodel(x_l, x_n, x_i, mL=mL, mN=mN, mI=mI)

    expected = set(ROUTES)
    got = set(route_embs_in.keys())
    if expected != got:
        missing = expected - got
        extra = got - expected
        raise RuntimeError(f"[mult->caps] Route key mismatch. missing={missing}, extra={extra}")

    if route_adapter is not None:
        route_embs_in = route_adapter(route_embs_in)

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
    "RoutePrimaryProjector",
    "RouteDimAdapter",
    "make_route_inputs_mult",
    "CapsuleMortalityHead",
    "forward_capsule_from_route_dict",
    "forward_capsule_from_multmodel",
    "route_given_pheno",
]
