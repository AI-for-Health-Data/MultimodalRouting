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
        n_rank: int,
        dp: float = 0.0,
        dim_pose_to_vote: int = 0,
        uniform_routing_coefficient: bool = False,
        act_type: str = "EM",
        small_std: bool = False,
    ):
        super(CapsuleFC, self).__init__()

        self.in_n_capsules = in_n_capsules      # N_in (7 routes)
        self.in_d_capsules = in_d_capsules      # A (pc_dim)
        self.out_n_capsules = out_n_capsules    # N_out (#phenotypes)
        self.out_d_capsules = out_d_capsules    # D (mc_caps_dim)
        self.n_rank = n_rank

        # Shape: [N_in, A, N_out, D]
        self.weight_init_const = np.sqrt(
            out_n_capsules / (in_d_capsules * in_n_capsules)
        )
        self.w = nn.Parameter(
            self.weight_init_const
            * torch.randn(
                in_n_capsules,
                in_d_capsules,
                out_n_capsules,
                out_d_capsules,
            )
        )

        # Dropout (currently disabled; kept for API compatibility)
        self.dropout_rate = float(dp)
        self.drop = nn.Identity()

        # extra nonlinearity on poses (currently none)
        self.nonlinear_act = nn.Sequential()

        # Scale factor used before softmax to keep logits in a reasonable range
        self.scale = 1.0 / (out_d_capsules ** 0.5)

        # Routing / activation configuration
        self.act_type = act_type
        self.uniform_routing_coefficient = bool(uniform_routing_coefficient)

        # Parameters for different act_type modes (kept for future extension)
        if act_type == "EM":
            # Symmetric, unbiased initialization (all capsules equal at start)
            self.beta_u = nn.Parameter(torch.zeros(out_n_capsules))
            self.beta_a = nn.Parameter(torch.zeros(out_n_capsules))
        elif act_type == "Hubert":
            self.alpha = nn.Parameter(torch.ones(out_n_capsules))
            self.beta = nn.Parameter(
                torch.zeros(in_n_capsules, in_d_capsules, out_n_capsules)
            )
        else:
            # "ONES" or any other custom mode: no extra parameters required
            pass

    def extra_repr(self) -> str:
        return (
            "in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, "
            "out_d_capsules={}, n_rank={}, weight_init_const={}, dropout_rate={}"
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
        input: torch.Tensor,
        current_act: torch.Tensor,
        num_iter: int,
        next_capsule_value: torch.Tensor = None,
        next_act: torch.Tensor = None,
        uniform_routing: bool = False,
    ):
        B = input.shape[0]
        device = input.device
        dtype = input.dtype

        # Match W to current device/dtype
        W = self.w.to(device=device, dtype=dtype)

        # current_act: [B, N_in] (flatten last dimension if [B, N_in, 1])
        if current_act.dim() == 3 and current_act.size(-1) == 1:
            current_act = current_act.view(B, -1)
        elif current_act.dim() == 2:
            current_act = current_act
        else:
            raise ValueError(
                f"current_act must be [B, N_in] or [B, N_in, 1], got {tuple(current_act.shape)}"
            )

        if next_capsule_value is None:
            # query_key_unif: [N_in, N_out] -> softmax over N_out
            query_key_unif = torch.zeros(
                self.in_n_capsules,
                self.out_n_capsules,
                device=device,
                dtype=dtype,
            )
            query_key_unif = F.softmax(query_key_unif, dim=1)  # uniform probs

            next_capsule_value = torch.einsum(
                "nm, bna, namd->bmd", query_key_unif, input, W
            )  # [B, N_out, D]

        if next_act is None:
            # Seed from mean primary activations across routes.
            # current_act: [B, N_in] -> [B] -> [B, N_out]
            init_a = current_act.mean(dim=1)  # [B]
            next_act = init_a.unsqueeze(1).expand(B, self.out_n_capsules).contiguous()
        else:
            # Ensure provided next_act lives on the correct device/dtype.
            next_act = next_act.to(device=device, dtype=dtype)
          
        if uniform_routing or self.uniform_routing_coefficient:
            # Completely uniform routing over N_out for each (b, n).
            query_key = torch.zeros(
                B,
                self.in_n_capsules,
                self.out_n_capsules,
                device=device,
                dtype=dtype,
            )
            query_key = F.softmax(query_key, dim=2)  # [B, N_in, N_out]
        else:
            _query_key = torch.einsum("bna, namd, bmd->bnm", input, W, next_capsule_value)

            # Scale to control logit magnitude before softmax
            _query_key.mul_(self.scale)

            # Softmax over N_out (labels) for each (b,n):
            query_key = F.softmax(_query_key, dim=2)  # [B, N_in, N_out]

            # Weight by output activations next_act: [B, N_out]
            # Broadcast multiply: (b,n,m) * (b,m) -> (b,n,m)
            query_key = torch.einsum("bnm, bm->bnm", query_key, next_act)

            # Renormalize across N_out so each (b,n) remains a valid distribution
            denom = query_key.sum(dim=2, keepdim=True) + 1e-10
            query_key = query_key / denom

        next_capsule_value = torch.einsum(
            "bnm, bna, namd, bn->bmd", query_key, input, W, current_act
        )

        if self.act_type == "ONES":
            # Simple fixed activation of 1 for all output capsules
            next_act = torch.ones(
                next_capsule_value.shape[:2], device=device, dtype=dtype
            )
        else:
            pass

        next_capsule_value = self.drop(next_capsule_value)
        if next_capsule_value.shape[-1] != 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)

        # query_key: [B, N_in, N_out] (routing coefficients per route/phenotype)
        return next_capsule_value, next_act, query_key
