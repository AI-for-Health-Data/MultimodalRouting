import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CapsuleFC(nn.Module):
    r"""Capsule fully-connected layer (reference-style), with batched routing coefficients."""

    def __init__(
        self,
        in_n_capsules: int,
        in_d_capsules: int,
        out_n_capsules: int,
        out_d_capsules: int,
        n_rank: int,
        dp: float,
        dim_pose_to_vote: int,
        uniform_routing_coefficient: bool = False,
        act_type: str = "EM",
        small_std: bool = False,
    ):
        super().__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.n_rank = n_rank

        # same init as reference
        self.weight_init_const = np.sqrt(out_n_capsules / (in_d_capsules * in_n_capsules))
        self.w = nn.Parameter(
            self.weight_init_const
            * torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules)
        )

        self.dropout_rate = float(dp)

        # same nonlinear/dropout logic as reference
        if small_std:
            self.nonlinear_act = nn.Sequential()
        else:
            print("layer norm will destroy interpretability, thus not available")
            assert False

        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1.0 / (out_d_capsules ** 0.5)

        self.act_type = act_type
        if act_type == "EM":
            self.beta_u = nn.Parameter(torch.randn(out_n_capsules))
            self.beta_a = nn.Parameter(torch.randn(out_n_capsules))
        elif act_type == "Hubert":
            self.alpha = nn.Parameter(torch.ones(out_n_capsules))
            self.beta = nn.Parameter(
                np.sqrt(1.0 / (in_n_capsules * in_d_capsules))
                * torch.randn(in_n_capsules, in_d_capsules, out_n_capsules)
            )

        self.uniform_routing_coefficient = bool(uniform_routing_coefficient)

    def extra_repr(self) -> str:
        return (
            "in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, n_rank{}, "
            "weight_init_const={}, dropout_rate={}"
        ).format(
            self.in_n_capsules,
            self.in_d_capsules,
            self.out_n_capsules,
            self.out_d_capsules,
            self.n_rank,
            self.weight_init_const,
            self.dropout_rate,
        )

    def forward(
        self,
        input: torch.Tensor,                   # [B, N_in, A]
        current_act: torch.Tensor,              # [B, N_in, 1]
        num_iter: int,
        next_capsule_value: torch.Tensor = None,  # [B, N_out, D]
        next_act: torch.Tensor = None,            # [B, N_out]
        uniform_routing: bool = False,
    ):
        # reference flattens current_act exactly like this
        current_act = current_act.view(current_act.shape[0], -1)  # [B, N_in]

        B = input.shape[0]
        w = self.w  # [N_in, A, N_out, D]

        # ----------------------------
        # IMPORTANT FIX:
        # If next_capsule_value is None (first iteration), make query_key batched:
        #   query_key: [B, N_in, N_out]
        # so even when num_iter == 1, routing_coef is 3D (consistent everywhere).
        # Also: softmax should be over N_out -> dim=2 for [B,N_in,N_out].
        # ----------------------------
        if next_capsule_value is None:
            query_key = torch.zeros(B, self.in_n_capsules, self.out_n_capsules).type_as(input)  # [B,N_in,N_out]
            query_key = F.softmax(query_key, dim=2)  # over N_out
            next_capsule_value = torch.einsum("bnm, bna, namd->bmd", query_key, input, w)  # [B,N_out,D]

        else:
            if uniform_routing:
                query_key = torch.zeros(B, self.in_n_capsules, self.out_n_capsules).type_as(input)  # [B,N_in,N_out]
                _query_key = torch.zeros_like(query_key)
                # IMPORTANT FIX: uniform routing over N_out -> dim=2 (not dim=1)
                query_key = F.softmax(query_key, dim=2)
            else:
                # reference agreement: against next_capsule_value
                _query_key = torch.einsum("bna, namd, bmd->bnm", input, w, next_capsule_value)  # [B,N_in,N_out]
                _query_key.mul_(self.scale)

                # reference routing: softmax over N_out
                query_key = F.softmax(_query_key, dim=2)

                # gate by next_act then renormalize over N_out
                query_key = torch.einsum("bnm, bm->bnm", query_key, next_act)
                query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) + 1e-10)

            # reference aggregation uses W and outputs [B, N_out, D]
            next_capsule_value = torch.einsum("bnm, bna, namd, bn->bmd", query_key, input, w, current_act)

        if self.act_type == "ONES":
            next_act = torch.ones(next_capsule_value.shape[0:2]).type_as(next_capsule_value)

        next_capsule_value = self.drop(next_capsule_value)
        if next_capsule_value.shape[-1] != 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)

        # query_key is now ALWAYS [B, N_in, N_out]
        return next_capsule_value, next_act, query_key
