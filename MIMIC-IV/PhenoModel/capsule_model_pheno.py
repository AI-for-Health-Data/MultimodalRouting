from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  

from .capsule_layers import CapsuleFC
from .routing_and_heads import build_capsule_inputs  


class Squash(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        scale = norm_sq / (1.0 + norm_sq).clamp_min(eps)
        return scale * x / (norm_sq.sqrt() + eps)


class PhenoCapsuleHead(nn.Module):
    """
    Phenotyping head with capsule routing over 7 route primaries (L, N, I, LN, LI, NI, LNI).
    Produces:
      - logits:          [B, K] via simple concat of 7 interaction vectors (no attention)
      - class_embed:     [B, K, 7*d_class] fixed-size per-phenotype embedding
      - poses:           [B, K, d_class] class capsule poses
      - acts:            [B, K]          class activations
      - q:               [B, 7, K]       routing coefficients (routes→class)
      - route_class_emb: [B, 7, K, d_class] per-route×class interaction embeddings
    """
    def __init__(
        self,
        d_in: int,
        n_primary: int,
        d_primary: int,
        n_classes: int,
        d_class: int = 16,
        routing_iters: int = 3,
        act_type: str = "EM",
        dp: float = 0.0,
        use_squash_primary: bool = False,
    ):
        super().__init__()
        self.n_primary = int(n_primary)         
        self.d_primary = int(d_primary)         
        self.n_classes = int(n_classes)         # = 25 phenotypes
        self.d_class = int(d_class)             # capsule class pose dim
        self.routing_iters = int(routing_iters)
        self.use_squash_primary = bool(use_squash_primary)

        self.to_primary = nn.Linear(d_in, self.n_primary * self.d_primary)
        self.primary_squash = Squash() if self.use_squash_primary else nn.Identity()
        self.primary_act = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, self.n_primary),
            nn.Sigmoid(),
        )

        # Primary -> Class capsule router
        self.caps = CapsuleFC(
            in_n_capsules=self.n_primary,
            in_d_capsules=self.d_primary,
            out_n_capsules=self.n_classes,
            out_d_capsules=self.d_class,
            dp=dp,
            act_type=act_type,
        )

        # Final prediction from concatenated interactions (simple concat, no attention)
        self.concat_dim = 7 * self.d_class  # 7 routes × d_class each
        self.class_concat_head = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Linear(self.concat_dim, 4 * self.concat_dim),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(4 * self.concat_dim, 1),  
        )

        # Kept for compatibility; not used in concat path, but harmless to keep
        self.to_logits = nn.Sequential(
            nn.LayerNorm(self.d_class),
            nn.Linear(self.d_class, 1),
        )

    def forward(
        self,
        z: Optional[torch.Tensor] = None,                        
        route_embs: Optional[Dict[str, torch.Tensor]] = None,    
        masks: Optional[Dict[str, torch.Tensor]] = None,        
    ) -> Dict[str, torch.Tensor]:

        # Primary capsules 
        if route_embs is not None:
            primary_pose, primary_act = build_capsule_inputs(route_embs, masks=masks)
        else:
            if z is None:
                raise ValueError("Either route_embs or z must be provided.")
            B = z.size(0)
            primary_pose = self.to_primary(z).view(B, self.n_primary, self.d_primary)
            primary_pose = self.primary_squash(primary_pose)
            primary_act = self.primary_act(z).clamp_(0.0, 1.0)  # [B, 7]

        # ---- Capsule routing (returns interactions too) ----
        # class_pose:      [B, K, d_class]
        # class_act:       [B, K]
        # q:               [B, 7, K]
        # route_class_emb: [B, 7, K, d_class]
        class_pose, class_act, q, route_class_emb = self.caps(
            input_pose=primary_pose,
            current_act=primary_act,
            num_iter=self.routing_iters,
            next_capsule_value=None,
            next_act=None,
            uniform_routing=False,
        )

        # Simple concat of 7 interactions per class (no attention) 
        B, R, K, D = route_class_emb.shape  # R is 7, D == d_class
        class_embed = (
            route_class_emb.permute(0, 2, 1, 3)       # [B, K, R, D]
                         .contiguous()
                         .view(B, K, R * D)           # [B, K, 7*d_class]
        )

        # Predict one logit per class from the concatenated interaction embedding
        logits = self.class_concat_head(class_embed).squeeze(-1)  # [B, K]

        return {
            "logits": logits,                    # [B, K] via concat head
            "class_embed": class_embed,          # [B, K, 7*d_class] fixed-size per-phenotype embedding
            "poses": class_pose,                 # [B, K, d_class]
            "acts": class_act,                   # [B, K]
            "q": q,                              # [B, 7, K] routing coefficients
            "primary_pose": primary_pose,        # [B, 7, d_primary]
            "primary_act": primary_act,          # [B, 7]
            "route_class_emb": route_class_emb,  # [B, 7, K, d_class]
        }
