import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CapsuleFC(nn.Module):
    def __init__(
        self,
        in_n_capsules,
        in_d_capsules,
        out_n_capsules,
        out_d_capsules,
        n_rank=None,
        dp=0.0,
        dim_pose_to_vote=0,
        uniform_routing_coefficient=False,
        act_type='EM',
        small_std=True,
    ):
        super().__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.n_rank = n_rank

        # weight init follows the original idea (scaled normal)
        self.weight_init_const = np.sqrt(out_n_capsules / (in_d_capsules * in_n_capsules))
        self.w = nn.Parameter(
            self.weight_init_const
            * torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules)
        )

        self.dropout_rate = dp
        if small_std:
            # Keep identity (the original code disallows LN for interpretability)
            self.nonlinear_act = nn.Sequential()
        else:
            raise ValueError("Layer norm / non-empty nonlinearity disabled to keep interpretability.")

        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1.0 / (out_d_capsules ** 0.5)

        self.act_type = act_type
        if act_type == 'EM':
            self.beta_u = nn.Parameter(torch.randn(out_n_capsules))
            self.beta_a = nn.Parameter(torch.randn(out_n_capsules))
        elif act_type == 'ONES':
            pass
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

        self.uniform_routing_coefficient = uniform_routing_coefficient

    def extra_repr(self):
        return (
            f'in_n_capsules={self.in_n_capsules}, in_d_capsules={self.in_d_capsules}, '
            f'out_n_capsules={self.out_n_capsules}, out_d_capsules={self.out_d_capsules}, '
            f'n_rank={self.n_rank}, weight_init_const={self.weight_init_const:.4f}, '
            f'dropout_rate={self.dropout_rate}'
        )

    @torch.no_grad()
    def _init_uniform(self, B):
        q = torch.zeros(B, self.in_n_capsules, self.out_n_capsules)
        q = F.softmax(q, dim=2)
        return q

    def forward(
        self,
        input_pose,          
        current_act,          
        num_iter: int,        
        next_capsule_value=None,  
        next_act=None,            
        uniform_routing=False,
    ):

        B = input_pose.shape[0]
        current_act = current_act.view(B, -1)  

        w = self.w  # [N_in, in_dim, N_out, out_dim]

        if next_capsule_value is None:
            q = torch.zeros(self.in_n_capsules, self.out_n_capsules, device=input_pose.device)
            q = F.softmax(q, dim=1)  
            next_capsule_value = torch.einsum('nm, bna, namd -> bmd', q, input_pose, w)
            query_key = q.unsqueeze(0).expand(B, -1, -1)
        else:
            if uniform_routing:
                query_key = self._init_uniform(B).to(input_pose.device)
            else:
                logits = torch.einsum('bna, namd, bmd -> bnm', input_pose, w, next_capsule_value)
                logits.mul_(self.scale)
                query_key = F.softmax(logits, dim=2) 
                if next_act is not None:
                    query_key = torch.einsum('bnm, bm -> bnm', query_key, next_act)
                    query_key = query_key / (query_key.sum(dim=2, keepdim=True) + 1e-10)

            next_capsule_value = torch.einsum(
                'bnm, bna, namd, bn -> bmd',
                query_key, input_pose, w, current_act
            )

        if self.act_type == 'ONES':
            next_act = torch.ones(next_capsule_value.shape[:2], device=next_capsule_value.device)

        next_capsule_value = self.drop(next_capsule_value)
        if next_capsule_value.shape[-1] != 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)

        return next_capsule_value, next_act, query_key
