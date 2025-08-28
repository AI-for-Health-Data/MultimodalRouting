import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from env_config import CFG, ROUTES, BLOCKS, DEVICE

# Fusion layers (produce d-dim embeddings for LN / LI / NI / LNI)
class PairwiseFusion(nn.Module):
    """
    Simple pairwise fusion that maps [z_a; z_b] in R^{2d} -> R^d.
    Uses LayerNorm + 2-layer MLP with GELU + dropout + residual gate.
    """
    def __init__(self, d: int, hidden: int = None, p_drop: float = 0.1):
        super().__init__()
        hidden = hidden or (2 * d)
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, d),
        )
        # Optional residual gate over the mean of the pair
        self.gate = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        # za, zb: [B, d]
        h = torch.cat([za, zb], dim=-1)  
        f = self.net(h)                  
        base = (za + zb) / 2.0           
        g = self.gate(base)              
        return g * f + (1.0 - g) * base  # gated residual


class TrimodalFusion(nn.Module):
    """
    Trimodal fusion that maps [z_L; z_N; z_I] in R^{3d} -> R^d.
    Two-layer MLP with GELU + dropout + learned residual gate.
    """
    def __init__(self, d: int, hidden: int = None, p_drop: float = 0.1):
        super().__init__()
        hidden = hidden or (3 * d)
        self.net = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, d),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )

    def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
        h = torch.cat([zL, zN, zI], dim=-1)  
        f = self.net(h)                      
        base = (zL + zN + zI) / 3.0          
        g = self.gate(base)                  
        return g * f + (1.0 - g) * base


def build_fusions(d: int, p_drop: float = 0.1) -> Dict[str, nn.Module]:
    """
    Returns fusion modules for LN, LI, NI, LNI (each output in R^d).
    """
    fusion = {
        "LN":  PairwiseFusion(d=d, p_drop=p_drop).to(DEVICE),
        "LI":  PairwiseFusion(d=d, p_drop=p_drop).to(DEVICE),
        "NI":  PairwiseFusion(d=d, p_drop=p_drop).to(DEVICE),
        "LNI": TrimodalFusion(d=d, p_drop=p_drop).to(DEVICE),
    }
    return fusion


@torch.no_grad()
def _safe_clone(x: torch.Tensor) -> torch.Tensor:
    # Tiny utility to avoid in-place surprises when composing inputs
    return x.clone() if x.requires_grad is False else x


def make_route_inputs(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
) -> Dict[str, torch.Tensor]:
    """
    Build d-dim inputs for all 7 routes using unimodal embeddings and fusion modules.

    Args:
      z: {"L":[B,d], "N":[B,d], "I":[B,d]}
      fusion: {"LN":PairwiseFusion, "LI":PairwiseFusion, "NI":PairwiseFusion, "LNI":TrimodalFusion}

    Returns:
      route_inputs: dict mapping each route to a [B, d] tensor.
    """
    zL, zN, zI = z["L"], z["N"], z["I"]
    route_inputs = {
        "L":   _safe_clone(zL),
        "N":   _safe_clone(zN),
        "I":   _safe_clone(zI),
        "LN":  fusion["LN"](zL, zN),
        "LI":  fusion["LI"](zL, zI),
        "NI":  fusion["NI"](zN, zI),
        "LNI": fusion["LNI"](zL, zN, zI),
    }
    return route_inputs


# Route Head (2-layer MLP) — now EVERY route head takes d-dim input
class RouteHead(nn.Module):
    """
    Small MLP that maps a route-specific embedding in R^d to task logits.
    Used for all 7 routes independently.
    """
    def __init__(self, d_in: int, n_tasks: int = 3, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 2 * d_in),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(2 * d_in, n_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Learned gate over (L, N, I) → per-task route/block weights
class LearnedGateRouter(nn.Module):
    """
    A gating network that takes unimodal embeddings (and optional present/absent masks),
    and outputs, for each task c:
      (1) a distribution over the 7 routes (L, N, I, LN, LI, NI, LNI),
      (2) a distribution over the 3 blocks (uni, bi, tri).

    It blends route-specific logits using those weights, builds block logits,
    and blends across blocks → final logits.

    Inputs:
      z_dict: {"L":[B,d], "N":[B,d], "I":[B,d]}
      route_logits_dict: {route:[B,C]} for all routes in ROUTES
      masks (optional): {"L":[B,1], "N":[B,1], "I":[B,1]} where 1=present, 0=missing

    Outputs:
      ylogits:     [B,C]   final logits (before sigmoid)
      route_w:     [B,C,7] per-task route weights (softmax over 7; unavailable routes get weight 0)
      block_w:     [B,C,3] per-task block weights (softmax over 3; unavailable blocks get weight 0)
      block_logits:[B,3,C] logits for (uni, bi, tri) before block weighting
    """
    def __init__(
        self,
        routes: List[str],
        blocks: Dict[str, List[str]],
        d: int,
        n_tasks: int = 3,
        hidden: int = 1024,
        p_drop: float = 0.1,
        use_masks: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert len(routes) == 7, "Expect routes = ['L','N','I','LN','LI','NI','LNI']"
        assert set(blocks.keys()) == {"uni", "bi", "tri"}
        self.routes = routes
        self.blocks = blocks
        self.n_tasks = n_tasks
        self.use_masks = use_masks
        self.temperature = temperature

        gate_in = 3 * d + (3 if use_masks else 0)  # concat z_L, z_N, z_I (+ 3 mask flags)
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_tasks * (7 + 3)),  # per-task: 7 route logits + 3 block logits
        )

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(dtype=logits.dtype)
        keep = (mask.sum(dim=dim, keepdim=True) > 0).to(logits.dtype)
        safe_mask = mask * keep + (1.0 - keep)           
        logits = logits + torch.log(safe_mask + 1e-12)   
        return torch.softmax(logits, dim=dim)

    def _stack_gate_input(
        self,
        z_dict: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        zL, zN, zI = z_dict["L"], z_dict["N"], z_dict["I"]  
        pieces = [zL, zN, zI]
        if self.use_masks:
            if masks is None:
                B = zL.size(0)
                device = zL.device
                m = torch.ones(B, 3, device=device)
            else:
                m = torch.cat([masks["L"], masks["N"], masks["I"]], dim=1)  
            pieces.append(m)
        return torch.cat(pieces, dim=1)  # [B, 3d(+3)]

    def _route_availability_mask(self, masks: Dict[str, torch.Tensor] | None, B: int, device) -> torch.Tensor:
        if not self.use_masks:
            return torch.ones(B, 7, device=device)
        if masks is None:
            mL = mN = mI = torch.ones(B, 1, device=device)
        else:
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
        return torch.cat([req[r] for r in self.routes], dim=1).clamp(0, 1) 

    def _block_availability_mask(self, route_mask: torch.Tensor) -> torch.Tensor:
        device = route_mask.device

        def block_any(names: List[str]) -> torch.Tensor:
            idx = torch.tensor([self.routes.index(r) for r in names], device=device)
            return (route_mask.index_select(1, idx).sum(dim=1, keepdim=True) > 0).float() 

        b_uni = block_any(self.blocks["uni"])
        b_bi  = block_any(self.blocks["bi"])
        b_tri = block_any(self.blocks["tri"])
        return torch.cat([b_uni, b_bi, b_tri], dim=1)  

    def forward(
        self,
        z_dict: Dict[str, torch.Tensor],
        route_logits_dict: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_dict: {"L":[B,d], "N":[B,d], "I":[B,d]}
        route_logits_dict: {route:[B,C]} for all routes in self.routes
        masks: optional modality-present masks {"L":[B,1], ...}
        """
        B = next(iter(route_logits_dict.values())).shape[0]
        C = self.n_tasks
        device = z_dict["L"].device

        # 1. Gate → per-task logits for routes & blocks
        g_in = self._stack_gate_input(z_dict, masks)              
        raw = self.gate(g_in).view(B, C, 7 + 3) / max(self.temperature, 1e-6)
        route_logits_gate = raw[:, :, :7]                          
        block_logits_gate = raw[:, :, 7:]                         

        # 2. Availability masks
        route_mask = self._route_availability_mask(masks, B, device)        
        block_mask = self._block_availability_mask(route_mask)               

        # 3. Masked softmax → weights
        route_w = self._masked_softmax(route_logits_gate, route_mask.unsqueeze(1).expand(-1, C, -1), dim=2)  
        block_w = self._masked_softmax(block_logits_gate, block_mask.unsqueeze(1).expand(-1, C, -1), dim=2)  

        # 4. Stack route logits → [B, C, 7]
        R = torch.cat([route_logits_dict[r].unsqueeze(-1) for r in self.routes], dim=2)

        # 5. Block-wise sums (per task)
        def sum_block(names: List[str]) -> torch.Tensor:
            idx = torch.tensor([self.routes.index(r) for r in names], device=R.device)
            R_blk = R.index_select(dim=2, index=idx)          
            w_blk = route_w.index_select(dim=2, index=idx)    
            return (w_blk * R_blk).sum(dim=2)                 

        y_uni = sum_block(self.blocks["uni"])  
        y_bi  = sum_block(self.blocks["bi"])   
        y_tri = sum_block(self.blocks["tri"])  

        block_logits = torch.stack([y_uni, y_bi, y_tri], dim=1) 
        block_logits = block_logits * block_mask.unsqueeze(-1)  

        # 6) Final blend across blocks per task
        y = (block_w.transpose(1, 2) * block_logits).sum(dim=1)  
        return y, route_w, block_w, block_logits


# Build fusion modules, route heads (all d-dim), and the router
fusion = build_fusions(d=CFG.d, p_drop=CFG.dropout)

# Every head now takes d-dim input 
route_heads = {
    r: RouteHead(d_in=CFG.d, n_tasks=3, p_drop=CFG.dropout).to(DEVICE)
    for r in ROUTES
}

router = LearnedGateRouter(
    routes=ROUTES,
    blocks=BLOCKS,
    d=CFG.d,
    n_tasks=3,
    hidden=1024,
    p_drop=CFG.dropout,
    use_masks=True,
    temperature=1.0,
).to(DEVICE)

print("Built fusion modules, route heads (d-dim), and learned-gate router.")


# Quick smoke test (random tensors)
B = 4
zL = torch.randn(B, CFG.d, device=DEVICE)
zN = torch.randn(B, CFG.d, device=DEVICE)
zI = torch.randn(B, CFG.d, device=DEVICE)
z  = {"L": zL, "N": zN, "I": zI}

masks = {k: torch.ones(B, 1, device=DEVICE) for k in ["L", "N", "I"]}

# Build route inputs via fusion and get per-route logits
route_inputs = make_route_inputs(z, fusion)               
route_logits = {r: route_heads[r](route_inputs[r]) for r in ROUTES} 

# Router combines them with learned per-task route/block weights
ylog, rW, bW, blk = router(z, route_logits, masks=masks)

print(
    "Final logits:", tuple(ylog.shape),
    "| route_w:", tuple(rW.shape),
    "| block_w:", tuple(bW.shape),
    "| block_logits:", tuple(blk.shape),
)
