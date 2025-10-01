import torch
import torch.nn as nn
import torch.nn.functional as F

from .capsule_layers import CapsuleFC


class Squash(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        scale = norm_sq / (1.0 + norm_sq).clamp_min(1e-6)
        return scale * x / (norm_sq.sqrt() + 1e-6)


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
        self.n_primary = n_primary
        self.d_primary = d_primary
        self.n_classes = n_classes
        self.d_class = d_class
        self.routing_iters = routing_iters
        self.use_squash_primary = use_squash_primary

        self.to_primary = nn.Linear(d_in, n_primary * d_primary)
        self.primary_squash = Squash() if use_squash_primary else nn.Identity()

        self.primary_act = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, n_primary),
            nn.Sigmoid(),
        )

        self.caps = CapsuleFC(
            in_n_capsules=n_primary,
            in_d_capsules=d_primary,
            out_n_capsules=n_classes,
            out_d_capsules=d_class,
            dp=dp,
            act_type=act_type,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_class),
            nn.Linear(d_class, 1)
        )

    def forward(self, z: torch.Tensor) -> dict:
        B = z.size(0)

        # Primary capsules
        primary_pose = self.to_primary(z).view(B, self.n_primary, self.d_primary)
        primary_pose = self.primary_squash(primary_pose)
        primary_act = self.primary_act(z).clamp(0.0, 1.0)                
      
        # Route to class capsules
        class_pose, class_act, q = self.caps(
            input_pose=primary_pose,
            current_act=primary_act,
            num_iter=self.routing_iters,
            next_capsule_value=None,
            next_act=None,
            uniform_routing=False,
        )

        logits = self.to_logits(class_pose).squeeze(-1)                 

        acts = torch.linalg.vector_norm(class_pose, ord=2, dim=-1)      

        return {
            "logits": logits,
            "acts": acts,
            "poses": class_pose,
            "q": q,
            "primary": primary_pose,
        }
