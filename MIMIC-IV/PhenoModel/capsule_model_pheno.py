from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .capsule_layers import CapsuleFC


class Squash(nn.Module):
    """
    Classic capsule squash nonlinearity, applied along the last dimension.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        scale = norm_sq / (1.0 + norm_sq).clamp_min(eps)
        return scale * x / (norm_sq.sqrt() + eps)


class PhenoCapsuleHead(nn.Module):
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
        self.n_classes = int(n_classes)
        self.d_class = int(d_class)
        self.routing_iters = int(routing_iters)
        self.use_squash_primary = bool(use_squash_primary)

        self.to_primary = nn.Linear(d_in, self.n_primary * self.d_primary)
        self.primary_squash = Squash() if self.use_squash_primary else nn.Identity()

        self.primary_act = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, self.n_primary),
            nn.Sigmoid(),
        )

        # Primary -> Class capsules
        self.caps = CapsuleFC(
            in_n_capsules=self.n_primary,
            in_d_capsules=self.d_primary,
            out_n_capsules=self.n_classes,
            out_d_capsules=self.d_class,
            dp=dp,
            act_type=act_type,
        )

        # Parent pose -> logit
        self.to_logits = nn.Sequential(
            nn.LayerNorm(self.d_class),
            nn.Linear(self.d_class, 1),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z.size(0)

        primary_pose = self.to_primary(z).view(B, self.n_primary, self.d_primary)
        primary_pose = self.primary_squash(primary_pose)  
        primary_act = self.primary_act(z).clamp_(0.0, 1.0)  # [B, n_primary]

        class_pose, class_act, q = self.caps(
            input_pose=primary_pose,     # [B, n_primary, d_primary]
            current_act=primary_act,     # [B, n_primary]
            num_iter=self.routing_iters,
            next_capsule_value=None,
            next_act=None,
            uniform_routing=False,
        )  # class_pose: [B, n_classes, d_class], class_act: [B, n_classes], q: [B, n_primary, n_classes]

        logits = self.to_logits(class_pose).squeeze(-1)  

        return {
            "logits": logits,
            "acts": class_act,             
            "poses": class_pose,
            "q": q,                         
            "primary_pose": primary_pose,
            "primary_act": primary_act,
        }
