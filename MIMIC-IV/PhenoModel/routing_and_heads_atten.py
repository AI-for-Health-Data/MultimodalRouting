from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env_config import ROUTES, DEVICE, CFG
import capsule_layers
from transformer import TransformerEncoder


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

def normalize_sigmoid_routing_for_scale(rc: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    """
    rc: routing gates in [0,1], e.g. [B,R,K]
    Returns rc normalized to sum=1 along `dim` ONLY for scale stability.
    This keeps sigmoid (non-competitive) routing, but prevents magnitude drift.
    """
    s = rc.sum(dim=dim, keepdim=True).clamp_min(eps)
    return rc / s

class PhenotypeRouteRouter(nn.Module):
    def __init__(self, d_model: int, n_routes: int, n_labels: int):
        super().__init__()
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.q = nn.Embedding(n_labels, d_model)
        self.scale = d_model ** -0.5

    def forward(self, route_embs, temp: float = 1.0):
        # route_embs: [B,R,D]
        k = self.key(route_embs)              # [B,R,D]
        q = self.q.weight                     # [K,D]

        # logits: [B,R,K]
        logits = torch.einsum("brd,kd->brk", k, q) * self.scale

        # --- SIGMOID ROUTING (independent gates), compute in fp32 to avoid bf16 quantization ---
        t = float(max(temp, 1e-6))
        with torch.cuda.amp.autocast(enabled=False):
            logits_fp32 = logits.float() / t
            probs = torch.sigmoid(logits_fp32)     # [B,R,K], NOT simplex, NOT sum-to-1

        return logits, probs

class MulTCrossAttentionFusion(nn.Module):
    """
    MulT-style directional cross-attention:
      A attends to B, then take LAST timestep from the returned A-aligned sequence.
    """
    def __init__(
        self,
        d: int,
        n_heads: int = 8,
        attn_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        layers: int = 1,
        attn_mask: bool = True,
        use_positional: bool = True,
        padding_idx: int = 0,
        left_pad: bool = False,
    ):
        super().__init__()
        self.trans = TransformerEncoder(
            embed_dim=d,
            num_heads=n_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            attn_mask=attn_mask,
            use_positional=use_positional,
            padding_idx=padding_idx,
            left_pad=left_pad,
        )

    def forward(self, A: torch.Tensor, mA: torch.Tensor, B: torch.Tensor, mB: torch.Tensor) -> torch.Tensor:
        # A: [B,TA,D], B: [B,TB,D]
        A_tbd = A.transpose(0, 1)  # [TA,B,D]
        B_tbd = B.transpose(0, 1)  # [TB,B,D]

        X = self.trans(A_tbd, B_tbd, B_tbd)   # [TA,B,D]
        X = X.transpose(0, 1)                 # [B,TA,D]

        # MulT uses last timestep [-1]. If you have masks, pick last valid.
        if mA is None:
            return X[:, -1, :]

        mA = mA.float()
        lengths = mA.sum(dim=1).long().clamp(min=1)   # [B]
        idx = (lengths - 1).clamp(min=0)              # [B]
        return X[torch.arange(X.size(0), device=X.device), idx]  # [B,D]


class MulTTriFusion(nn.Module):
    """
    MulT-style tri fusion, consistent with MULTModel.forward:
      - compute LN, LI, IN directional cross streams
      - use last timestep of each cross output
      - concat -> linear -> [B,D]
    """
    def __init__(
        self,
        d: int,
        n_heads: int = 8,
        attn_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        layers: int = 1,
        attn_mask: bool = False,
        use_positional: bool = True,
        padding_idx: int = 0,
        left_pad: bool = False,
    ):
        super().__init__()

        self.L_with_N = TransformerEncoder(
            embed_dim=d, num_heads=n_heads, layers=layers,
            attn_dropout=attn_dropout, relu_dropout=relu_dropout, res_dropout=res_dropout,
            embed_dropout=embed_dropout, attn_mask=attn_mask,
            use_positional=use_positional, padding_idx=padding_idx, left_pad=left_pad,
        )
        self.L_with_I = TransformerEncoder(
            embed_dim=d, num_heads=n_heads, layers=layers,
            attn_dropout=attn_dropout, relu_dropout=relu_dropout, res_dropout=res_dropout,
            embed_dropout=embed_dropout, attn_mask=attn_mask,
            use_positional=use_positional, padding_idx=padding_idx, left_pad=left_pad,
        )
        self.I_with_N = TransformerEncoder(
            embed_dim=d, num_heads=n_heads, layers=layers,
            attn_dropout=attn_dropout, relu_dropout=relu_dropout, res_dropout=res_dropout,
            embed_dropout=embed_dropout, attn_mask=attn_mask,
            use_positional=use_positional, padding_idx=padding_idx, left_pad=left_pad,
        )

        self.final = nn.Linear(3 * d, d)


    def forward(self, L_seq, mL, N_seq, mN, I_seq, mI) -> torch.Tensor:
        # [B,T,D] -> [T,B,D]
        L = L_seq.transpose(0, 1)
        N = N_seq.transpose(0, 1)
        I = I_seq.transpose(0, 1)

        # directional cross streams
        h_ln = self.L_with_N(L, N, N)   # [TL,B,D]
        h_li = self.L_with_I(L, I, I)   # [TL,B,D]
        h_in = self.I_with_N(I, N, N)   # [TI,B,D]

        # MulT-style "last timestep" pooling
        ln_last = h_ln[-1]  # [B,D]
        li_last = h_li[-1]  # [B,D]
        in_last = h_in[-1]  # [B,D]

        z = torch.cat([ln_last, in_last, li_last], dim=1)  # [B,3D]
        return self.final(z)  # [B,D]

def build_fusions(d: int, feature_mode: str = "seq", p_drop: float = 0.0):
    dev = torch.device(DEVICE)

    h = int(getattr(CFG, "cross_attn_heads", 8))
    p = float(getattr(CFG, "cross_attn_dropout", p_drop))

    layers = int(getattr(CFG, "cross_attn_layers", 1))
    relu_dp = float(getattr(CFG, "cross_relu_dropout", 0.0))
    res_dp  = float(getattr(CFG, "cross_res_dropout", 0.0))
    emb_dp  = float(getattr(CFG, "cross_embed_dropout", 0.0))
    attn_mask = bool(getattr(CFG, "cross_attn_mask", True))   # default True (MulT-like)
    use_positional = bool(getattr(CFG, "cross_use_positional", True))

    LN  = MulTCrossAttentionFusion(d, n_heads=h, attn_dropout=p, relu_dropout=relu_dp,
                                   res_dropout=res_dp, embed_dropout=emb_dp,
                                   layers=layers, attn_mask=attn_mask,
                                   use_positional=use_positional).to(dev)
    NL  = MulTCrossAttentionFusion(d, n_heads=h, attn_dropout=p, relu_dropout=relu_dp,
                                   res_dropout=res_dp, embed_dropout=emb_dp,
                                   layers=layers, attn_mask=attn_mask,
                                   use_positional=use_positional).to(dev)
    LI  = MulTCrossAttentionFusion(d, n_heads=h, attn_dropout=p, relu_dropout=relu_dp,
                                   res_dropout=res_dp, embed_dropout=emb_dp,
                                   layers=layers, attn_mask=attn_mask,
                                   use_positional=use_positional).to(dev)
    IL  = MulTCrossAttentionFusion(d, n_heads=h, attn_dropout=p, relu_dropout=relu_dp,
                                   res_dropout=res_dp, embed_dropout=emb_dp,
                                   layers=layers, attn_mask=attn_mask,
                                   use_positional=use_positional).to(dev)
    NI  = MulTCrossAttentionFusion(d, n_heads=h, attn_dropout=p, relu_dropout=relu_dp,
                                   res_dropout=res_dp, embed_dropout=emb_dp,
                                   layers=layers, attn_mask=attn_mask,
                                   use_positional=use_positional).to(dev)
    IN  = MulTCrossAttentionFusion(d, n_heads=h, attn_dropout=p, relu_dropout=relu_dp,
                                   res_dropout=res_dp, embed_dropout=emb_dp,
                                   layers=layers, attn_mask=attn_mask,
                                   use_positional=use_positional).to(dev)

    LNI = MulTTriFusion(d, n_heads=h, attn_dropout=p, relu_dropout=relu_dp,
                        res_dropout=res_dp, embed_dropout=emb_dp,
                        layers=layers, attn_mask=attn_mask,
                        use_positional=use_positional).to(dev)

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
            dp=dp,                        # NO dropout here
            dim_pose_to_vote=dim_pose_to_vote,
            uniform_routing_coefficient=False,
            act_type=act_type,             # "EM", "Hubert", "ONES"
            small_std=not layer_norm,
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


        # init routing (like their "first routing")
        decision_pose, decision_act, _ = self.capsule(
            input=prim_pose,
            current_act=prim_act,
            num_iter=0,
            next_capsule_value=None,
            next_act=None,
            uniform_routing=uniform_routing,
        )   

        # update routing (like their for-loop)
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

    # Match their behavior: primary activations are just sigmoid(linear) and go straight in
    acts_for_caps = acts if acts_override is None else acts_override.to(device=acts.device, dtype=acts.dtype)

    # optionally detach priors (warmup: stop capsule gradients flowing into projector priors)
    if detach_priors:
        acts_for_caps = acts_for_caps.detach()

    # apply route mask (0 = drop)
    if route_mask is not None:
        m = route_mask.float().to(device=acts_for_caps.device, dtype=acts_for_caps.dtype)
        if m.ndim == 2:
            m = m.unsqueeze(-1)  # [B,R,1]
        acts_for_caps = acts_for_caps * m

    # activation temperature on sigmoid-probs via logit temperature
    if act_temperature is not None and float(act_temperature) != 1.0:
        eps = 1e-6
        a = acts_for_caps.clamp(eps, 1.0 - eps)
        logits = torch.log(a) - torch.log1p(-a)
        logits = logits / float(act_temperature)
        acts_for_caps = torch.sigmoid(logits)


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
    "MulTCrossAttentionFusion",
    "MulTTriFusion",
    "build_fusions",
    "make_route_inputs",
    "RoutePrimaryProjector",
    "CapsuleMortalityHead",
    "forward_capsule_from_routes",
]
