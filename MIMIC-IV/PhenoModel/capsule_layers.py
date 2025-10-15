import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CapsuleFC(nn.Module):
    def __init__(
        self,
        in_n_capsules: int,
        in_d_capsules: int,
        out_n_capsules: int,
        out_d_capsules: int,
        n_rank: int | None = None,
        dp: float = 0.0,
        dim_pose_to_vote: int = 0,
        uniform_routing_coefficient: bool = False,
        act_type: str = "EM",
        small_std: bool = True,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.n_rank = n_rank
        self.eps = eps

        self.weight_init_const = float(
            np.sqrt(out_n_capsules / (in_d_capsules * max(1, in_n_capsules)))
        )
        self.w = nn.Parameter(
            self.weight_init_const
            * torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules)
        )

        self.dropout_rate = float(dp)
        if small_std:
            self.nonlinear_act = nn.Sequential()
        else:
            raise ValueError(
                "LayerNorm/nonlinear stage disabled to keep interpretability (small_std must be True)."
            )

        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1.0 / (out_d_capsules ** 0.5)

        self.act_type = act_type.upper()
        if self.act_type not in ("EM", "ONES"):
            raise ValueError(f"Unsupported act_type: {act_type}")

        self.uniform_routing_coefficient = uniform_routing_coefficient

    def extra_repr(self) -> str:
        return (
            f"in_n_capsules={self.in_n_capsules}, in_d_capsules={self.in_d_capsules}, "
            f"out_n_capsules={self.out_n_capsules}, out_d_capsules={self.out_d_capsules}, "
            f"n_rank={self.n_rank}, weight_init_const={self.weight_init_const:.4f}, "
            f"dropout_rate={self.dropout_rate}"
        )

    @torch.no_grad()
    def _init_uniform(self, B: int, device: torch.device) -> torch.Tensor:
        q = torch.zeros(B, self.in_n_capsules, self.out_n_capsules, device=device)
        return F.softmax(q, dim=2)

    def _ensure_act_shape(self, current_act: torch.Tensor) -> torch.Tensor:
        if current_act.dim() == 3 and current_act.size(-1) == 1:
            current_act = current_act.squeeze(-1)
        if current_act.dim() != 2:
            raise ValueError("current_act must be shape [B, N_in] or [B, N_in, 1].")
        return current_act

    def _pose_to_act(self, pose: torch.Tensor) -> torch.Tensor:
        act = torch.linalg.vector_norm(pose, ord=2, dim=-1)
        return act

    def forward(
        self,
        input_pose: torch.Tensor,        
        current_act: torch.Tensor,       
        num_iter: int,
        next_capsule_value: torch.Tensor | None = None, 
        next_act: torch.Tensor | None = None,            
        uniform_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Returns:
          next_capsule_value: [B, N_out, D_out]   (class poses)
          next_act:           [B, N_out]          (class activations)
          query_key:          [B, N_in, N_out]    (routing coeffs over routes per class)
          route_class_emb:    [B, N_in, N_out, D_out]  (per-route × per-class interaction embeddings)
        """
        B, Nin, Din = input_pose.shape
        assert Nin == self.in_n_capsules and Din == self.in_d_capsules, \
            f"Expected input_pose [B,{self.in_n_capsules},{self.in_d_capsules}], got {list(input_pose.shape)}"
        device = input_pose.device

        current_act = self._ensure_act_shape(current_act)  
        W = self.w  # [N_in, D_in, N_out, D_out]

        if next_capsule_value is None:
            q0 = torch.zeros(self.in_n_capsules, self.out_n_capsules, device=device)  
            q0 = F.softmax(q0, dim=1)
            next_capsule_value = torch.einsum('nm, bna, namd -> bmd', q0, input_pose, W)  
            query_key = q0.unsqueeze(0).expand(B, -1, -1)  
        else:
            if uniform_routing or self.uniform_routing_coefficient:
                query_key = self._init_uniform(B, device)  
            else:
                logits0 = torch.einsum('bna, namd, bmd -> bnm', input_pose, W, next_capsule_value)  
                logits0.mul_(self.scale)
                query_key = F.softmax(logits0, dim=2)  

        # Routing iterations
        for _ in range(max(1, num_iter)):
            if uniform_routing or self.uniform_routing_coefficient:
                query_key = self._init_uniform(B, device)
            else:
                logits = torch.einsum('bna, namd, bmd -> bnm', input_pose, W, next_capsule_value)  
                logits.mul_(self.scale)
                query_key = F.softmax(logits, dim=2)
                if next_act is not None:
                    query_key = query_key * next_act.unsqueeze(1) 
                    denom = query_key.sum(dim=2, keepdim=True).clamp_min(self.eps)
                    query_key = query_key / denom

            next_capsule_value = torch.einsum(
                'bnm, bna, namd, bn -> bmd',
                query_key, input_pose, W, current_act
            )  # [B, N_out, D_out]

        # Class activations
        if self.act_type == "ONES":
            next_act = torch.ones(next_capsule_value.shape[:2], device=device)  
        else:
            if next_act is None:
                next_act = self._pose_to_act(next_capsule_value)  

        next_capsule_value = self.drop(next_capsule_value)
        if next_capsule_value.shape[-1] != 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)

        # Per-route × class interaction embeddings 
        votes = torch.einsum('bna, namd -> bnkd', input_pose, W)
        weights = query_key * current_act.unsqueeze(-1)
        route_class_emb = weights.unsqueeze(-1) * votes  

        return next_capsule_value, next_act, query_key, route_class_emb
