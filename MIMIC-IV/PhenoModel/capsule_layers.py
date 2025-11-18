import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CapsuleFC(nn.Module):
    r"""Fully-connected capsule layer with safe init for first routing."""
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules,
                 n_rank, dp, dim_pose_to_vote, uniform_routing_coefficient=False,
                 act_type='EM', small_std=False):
        super(CapsuleFC, self).__init__()
        self.in_n_capsules  = in_n_capsules   # N_in (e.g., 7 routes)
        self.in_d_capsules  = in_d_capsules   # A   (pc_dim)
        self.out_n_capsules = out_n_capsules  # N_out (e.g., 2 classes)
        self.out_d_capsules = out_d_capsules  # D   (mc_caps_dim)
        self.n_rank = n_rank

        self.weight_init_const = np.sqrt(out_n_capsules / (in_d_capsules * in_n_capsules))
        self.w = nn.Parameter(
            self.weight_init_const *
            torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules)
        )  # [N_in, A, N_out, D]

        self.dropout_rate = dp
        self.drop = nn.Dropout(self.dropout_rate)

        # Keep a no-op nonlinearity by default 
        self.nonlinear_act = nn.Sequential()
        self.scale = 1.0 / (out_d_capsules ** 0.5)

        self.act_type = act_type
        if act_type == 'EM':
            self.beta_u = nn.Parameter(torch.randn(out_n_capsules))
            self.beta_a = nn.Parameter(torch.randn(out_n_capsules))
        elif act_type == 'Hubert':
            self.alpha = nn.Parameter(torch.ones(out_n_capsules))
            self.beta  = nn.Parameter(
                np.sqrt(1.0 / (in_n_capsules * in_d_capsules)) *
                torch.randn(in_n_capsules, in_d_capsules, out_n_capsules)
            )

        self.uniform_routing_coefficient = uniform_routing_coefficient  

    def extra_repr(self):
        return ('in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, '
                'n_rank={}, weight_init_const={}, dropout_rate={}'
                ).format(self.in_n_capsules, self.in_d_capsules,
                         self.out_n_capsules, self.out_d_capsules,
                         self.n_rank, self.weight_init_const, self.dropout_rate)

    def forward(self, input, current_act, num_iter,
                next_capsule_value=None, next_act=None, uniform_routing=False):
        """
        input            : [B, N_in, A]       (primary poses)
        current_act      : [B, N_in, 1]       (primary activations)
        next_capsule_value: [B, N_out, D] or None  (decision poses from prev iter)
        next_act         : [B, N_out] or None      (decision activations from prev iter)
        returns:
          next_capsule_value: [B, N_out, D]
          next_act          : [B, N_out]
          query_key         : [B, N_in, N_out]   (routing coefficients)
        """
        B = input.shape[0]
        device = input.device
        dtype  = input.dtype
        
        W = self.w.to(device=device, dtype=dtype)
        
        # [B, N_in]
        current_act = current_act.view(B, -1)

        if next_capsule_value is None:
            # query_key_unif: [N_in, N_out] -> softmax over N_out axis
            query_key_unif = torch.zeros(self.in_n_capsules, self.out_n_capsules, device=device, dtype=dtype)
            query_key_unif = F.softmax(query_key_unif, dim=1)  # uniform probs

            # votes: [B, N_out, D]  via (N_in,N_out) ⊗ [B,N_in,A] ⊗ [N_in,A,N_out,D]
            next_capsule_value = torch.einsum('nm, bna, namd->bmd',
                                              query_key_unif, input, W)  # [B, M, D]

        # If next_act is None, seed it from current primary activations (mean over routes)
        if next_act is None:
            init_a = current_act.mean(dim=1)                         # [B]
            next_act = init_a.unsqueeze(1).expand(B, self.out_n_capsules).contiguous()  # [B, M]
        else:
            # ensure compatibility if caller provided it
            next_act = next_act.to(device=device, dtype=dtype)

        # ROUTING STEP 
        if uniform_routing or self.uniform_routing_coefficient:
            # batch-wise uniform over N_out
            query_key = torch.zeros(B, self.in_n_capsules, self.out_n_capsules, device=device, dtype=dtype)
            query_key = F.softmax(query_key, dim=2)  # [B, N_in, N_out]
        else:
            # agreement: [B, N_in, N_out] = input ⊗ W ⊗ next_capsule_value
            # input [B,N_in,A], W [N_in,A,N_out,D], next_capsule_value [B,N_out,D]
            _query_key = torch.einsum('bna, namd, bmd->bnm', input, W, next_capsule_value)  # [B,N_in,N_out]
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=2)  # softmax over N_out

            # weight by next_act [B, N_out] -> broadcast to [B,N_in,N_out]
            query_key = torch.einsum('bnm, bm->bnm', query_key, next_act)
            # normalize across N_out to keep it stochastic
            denom = query_key.sum(dim=2, keepdim=True) + 1e-10
            query_key = query_key / denom

        # aggregate votes to produce next decision poses: [B, M, D]
        next_capsule_value = torch.einsum('bnm, bna, namd, bn->bmd',
                                          query_key, input, W, current_act)

        if self.act_type == 'ONES':
            next_act = torch.ones(next_capsule_value.shape[:2], device=device, dtype=dtype)

        # regularization + optional nonlinearity
        next_capsule_value = self.drop(next_capsule_value)
        if next_capsule_value.shape[-1] != 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)

        return next_capsule_value, next_act, query_key
