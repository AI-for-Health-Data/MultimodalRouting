import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CapsuleFC(nn.Module):
    """
    Fully-connected capsule layer (routing over 7 primaries -> K class capsules).
    Matches the reference behavior:
      - no LayerNorm (small_std=True -> identity)
      - routing via softmax over out_n_capsules
      - supports uniform routing (ablation)
      - act_type: 'EM' params registered (unused) or 'ONES'
    Returns (next_capsule_value, next_act, query_key, route_class_emb)
    """
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

        # same init as reference
        self.weight_init_const = float(
            np.sqrt(out_n_capsules / (in_d_capsules * max(1, in_n_capsules)))
        )
        self.w = nn.Parameter(
            self.weight_init_const
            * torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules)
        )

        self.dropout_rate = float(dp)
        if small_std:
            # identity: no LayerNorm / nonlinearity to preserve interpretability
            self.nonlinear_act = nn.Sequential()
        else:
            # keep parity with reference (no LN allowed)
            raise ValueError("layer norm will destroy interpretability, thus not available")

        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1.0 / (out_d_capsules ** 0.5)

        self.act_type = act_type
        # register EM/Hubert params for parity (not used by our forward)
        if self.act_type == "EM":
            self.beta_u = nn.Parameter(torch.randn(out_n_capsules))
            self.beta_a = nn.Parameter(torch.randn(out_n_capsules))
        elif self.act_type == "Hubert":
            self.alpha = nn.Parameter(torch.ones(out_n_capsules))
            self.beta = nn.Parameter(
                np.sqrt(1.0 / (in_n_capsules * in_d_capsules))
                * torch.randn(in_n_capsules, in_d_capsules, out_n_capsules)
            )

        self.uniform_routing_coefficient = uniform_routing_coefficient

    def extra_repr(self) -> str:
        return (
            f"in_n_capsules={self.in_n_capsules}, in_d_capsules={self.in_d_capsules}, "
            f"out_n_capsules={self.out_n_capsules}, out_d_capsules={self.out_d_capsules}, "
            f"n_rank={self.n_rank}, weight_init_const={self.weight_init_const}, "
            f"dropout_rate={self.dropout_rate}"
        )

    @torch.no_grad()
    def _uniform_q(self, B: int, device) -> torch.Tensor:
        q = torch.zeros(B, self.in_n_capsules, self.out_n_capsules, device=device)
        return F.softmax(q, dim=2)

    def forward(
        self,
        input: torch.Tensor,               # [B, N_in, D_in]
        current_act: torch.Tensor,         # [B, N_in] or [B, N_in, 1]
        num_iter: int,
        next_capsule_value: torch.Tensor | None = None,  # [B, N_out, D_out]
        next_act: torch.Tensor | None = None,            # [B, N_out]
        uniform_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        # normalize current_act shape to [B, N_in]
        if current_act.dim() == 3 and current_act.size(-1) == 1:
            current_act = current_act.squeeze(-1)
        if current_act.dim() != 2:
            raise ValueError("current_act must be [B, N_in] or [B, N_in, 1].")

        B, Nin, Din = input.shape
        assert Nin == self.in_n_capsules and Din == self.in_d_capsules

        W = self.w  # [N_in, D_in, N_out, D_out]

        # init or update routing coefficients
        if next_capsule_value is None:
            q0 = torch.zeros(self.in_n_capsules, self.out_n_capsules, device=input.device)
            q0 = F.softmax(q0, dim=1)  # [N_in, N_out]
            next_capsule_value = torch.einsum("nm, bna, namd -> bmd", q0, input, W)
            query_key = q0.unsqueeze(0).expand(B, -1, -1)  # [B, N_in, N_out]
        else:
            if uniform_routing or self.uniform_routing_coefficient:
                query_key = self._uniform_q(B, input.device)  # [B, N_in, N_out]
            else:
                logits0 = torch.einsum("bna, namd, bmd -> bnm", input, W, next_capsule_value)
                logits0.mul_(self.scale)
                query_key = F.softmax(logits0, dim=2)        # softmax over N_out
                if next_act is not None:                     
                    query_key = query_key * next_act.unsqueeze(1)
                    query_key = query_key / (query_key.sum(dim=2, keepdim=True).clamp_min(1e-10))

            next_capsule_value = torch.einsum(
                "bnm, bna, namd, bn -> bmd", query_key, input, W, current_act
            )

        # routing refinement (num_iter >=1 to mimic reference loop)
        for _ in range(max(1, num_iter)):
            if uniform_routing or self.uniform_routing_coefficient:
                query_key = self._uniform_q(B, input.device)
            else:
                logits = torch.einsum("bna, namd, bmd -> bnm", input, W, next_capsule_value)
                logits.mul_(self.scale)
                query_key = F.softmax(logits, dim=2)
                if next_act is not None:
                    query_key = query_key * next_act.unsqueeze(1)
                    query_key = query_key / (query_key.sum(dim=2, keepdim=True).clamp_min(1e-10))

            next_capsule_value = torch.einsum(
                "bnm, bna, namd, bn -> bmd", query_key, input, W, current_act
            )

        # activations
        if self.act_type == "ONES":
            next_act = torch.ones(next_capsule_value.shape[:2], device=input.device)
        # EM path keeps next_act as passed/None (we don’t run EM steps here, matching ref)

        next_capsule_value = self.drop(next_capsule_value)
        if next_capsule_value.shape[-1] != 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)

        # Provide per-route × per-class “votes” so the head can concat interactions
        votes = torch.einsum("bna, namd -> bnkd", input, W)     # [B, N_in, N_out, D_out]
        weights = query_key * current_act.unsqueeze(-1)         # [B, N_in, N_out]
        route_class_emb = weights.unsqueeze(-1) * votes         # [B, N_in, N_out, D_out]

        return next_capsule_value, next_act, query_key, route_class_emb
