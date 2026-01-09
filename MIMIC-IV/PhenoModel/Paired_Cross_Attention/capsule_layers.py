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
        gate_temp: float = 1.0,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
        gate_eps: float = 1e-6,
    ):
        super().__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.n_rank = n_rank
        self.weight_init_const = np.sqrt(out_n_capsules / (in_d_capsules * in_n_capsules))
        self.w = nn.Parameter(
            self.weight_init_const
            * torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules)
        )
        self.dropout_rate = float(dp)
        self.drop = nn.Dropout(self.dropout_rate)
        if small_std:
            self.nonlinear_act = nn.Sequential()
        else:
            raise AssertionError("layer norm will destroy interpretability, thus not available")

        self.scale = 1.0 / (out_d_capsules ** 0.5)

        self.act_type = act_type
        self.uniform_routing_coefficient = bool(uniform_routing_coefficient)

        self.gate_temp = float(gate_temp)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.gate_eps = float(gate_eps)

        if act_type == "EM":
            self.beta_u = nn.Parameter(torch.randn(out_n_capsules))
            self.beta_a = nn.Parameter(torch.randn(out_n_capsules))
        elif act_type == "Hubert":
            self.alpha = nn.Parameter(torch.ones(out_n_capsules))
            self.beta = nn.Parameter(
                np.sqrt(1.0 / (in_n_capsules * in_d_capsules))
                * torch.randn(in_n_capsules, in_d_capsules, out_n_capsules)
            )

    def _apply_gate_temp_and_clamp(self, act: torch.Tensor) -> torch.Tensor:
        eps = self.gate_eps
        a = act.clamp(eps, 1.0 - eps)
        if self.gate_temp and self.gate_temp != 1.0:
            logits = torch.log(a) - torch.log1p(-a)
            logits = logits / self.gate_temp
            a = torch.sigmoid(logits)
        if self.gate_min > 0.0 or self.gate_max < 1.0:
            a = a.clamp(self.gate_min, self.gate_max)
        return a

    def forward(
        self,
        input: torch.Tensor,              # [B, N_in, A]
        current_act: torch.Tensor,         # [B, N_in, 1] or [B, N_in]
        num_iter: int,
        next_capsule_value: torch.Tensor = None,  # [B, N_out, D]
        next_act: torch.Tensor = None,            # [B, N_out]
        uniform_routing: bool = False,
    ):
        B = input.shape[0]
        if current_act.dim() == 3 and current_act.size(-1) == 1:
            current_act = current_act.view(B, -1)
        elif current_act.dim() == 2:
            pass
        else:
            raise ValueError(f"current_act must be [B,N_in] or [B,N_in,1], got {tuple(current_act.shape)}")
        current_act = self._apply_gate_temp_and_clamp(current_act)

        w = self.w  # reference uses self.w directly
        if next_capsule_value is None:
            # query_key: [N_in, N_out], softmax over phenotypes (dim=1)
            query_key = torch.zeros(self.in_n_capsules, self.out_n_capsules, device=input.device, dtype=input.dtype)
            query_key = F.softmax(query_key, dim=1)
            next_capsule_value = torch.einsum("nm, bna, namd->bmd", query_key, input, w)
        else:
            if uniform_routing or self.uniform_routing_coefficient:
                query_key = torch.zeros(B, self.in_n_capsules, self.out_n_capsules, device=input.device, dtype=input.dtype)
                _query_key = torch.zeros_like(query_key)  
                query_key = F.softmax(query_key, dim=1)    
            else:
                _query_key = torch.einsum("bna, namd, bmd->bnm", input, w, next_capsule_value)
                _query_key.mul_(self.scale)
                query_key = F.softmax(_query_key, dim=2)             
                if next_act is None:
                    raise ValueError("Reference behavior: next_act must be provided when routing (next_capsule_value is not None).")
                query_key = torch.einsum("bnm, bm->bnm", query_key, next_act)
                query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) + 1e-10)

            next_capsule_value = torch.einsum("bnm, bna, namd, bn->bmd", query_key, input, w, current_act)

        if self.act_type == "ONES":
            next_act = torch.ones(next_capsule_value.shape[0:2], device=next_capsule_value.device, dtype=next_capsule_value.dtype)

        next_capsule_value = self.drop(next_capsule_value)
        if next_capsule_value.shape[-1] != 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)

        query_key_out = locals().get("query_key", None)
        return next_capsule_value, next_act, query_key_out
