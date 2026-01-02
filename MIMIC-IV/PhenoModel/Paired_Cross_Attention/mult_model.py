import torch
from torch import nn

import torch.nn.functional as F
from transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, orig_d_l, orig_d_n, orig_d_i, d_l, d_n, d_i,
                 ionly, nonly, lonly, num_heads, layers, self_layers,
                 attn_dropout, attn_dropout_n, attn_dropout_i,
                 relu_dropout, res_dropout, out_dropout, embed_dropout, attn_mask):
        super().__init__()
        self.orig_d_l, self.orig_d_n, self.orig_d_i = orig_d_l, orig_d_n, orig_d_i
        self.d_l, self.d_n, self.d_i = d_l, d_n, d_i
        self.ionly = ionly
        self.nonly = nonly
        self.lonly = lonly
        self.num_heads = num_heads
        self.layers = layers
        self.self_layers = self_layers
        self.attn_dropout = attn_dropout
        self.attn_dropout_n = attn_dropout_n
        self.attn_dropout_i = attn_dropout_i
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

        # 1. Temporal conv projections (MulT expects [B, T, D] inputs)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_n = nn.Conv1d(self.orig_d_n, self.d_n, kernel_size=1, padding=0, bias=False)
        self.proj_i = nn.Conv1d(self.orig_d_i, self.d_i, kernel_size=1, padding=0, bias=False)

        # Self-attn per modality
        self.trans_l = self.get_network(self_type='l_only', layers=self.self_layers)
        self.trans_n = self.get_network(self_type='n_only', layers=self.self_layers)
        self.trans_i = self.get_network(self_type='i_only', layers=self.self_layers)

        # Crossmodal attention blocks
        if self.lonly:
            self.trans_l_with_n = self.get_network(self_type='ln')
            self.trans_l_with_i = self.get_network(self_type='li')
        if self.nonly:
            self.trans_n_with_l = self.get_network(self_type='nl')
            self.trans_n_with_i = self.get_network(self_type='ni')
        if self.ionly:
            self.trans_i_with_l = self.get_network(self_type='il')
            self.trans_i_with_n = self.get_network(self_type='in')

        self.final_lni = nn.Linear(self.d_n + self.d_i + self.d_l, self.d_l)


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'nl', 'il']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['n', 'ln', 'in']:
            embed_dim, attn_dropout = self.d_n, self.attn_dropout_n
        elif self_type in ['i', 'li', 'ni']:
            embed_dim, attn_dropout = self.d_i, self.attn_dropout_i
        elif self_type == "l_only":
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == "n_only":
            embed_dim, attn_dropout = self.d_n, self.attn_dropout_n
        elif self_type == "i_only":
            embed_dim, attn_dropout = self.d_i, self.attn_dropout_i
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask
        )

    def forward(self, x_l, x_n, x_i):
        """
        Inputs MUST be [B, T, D]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)  # [B,D,T]
        x_n = x_n.transpose(1, 2)
        x_i = x_i.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_n = x_n if self.orig_d_n == self.d_n else self.proj_n(x_n)
        proj_x_i = x_i if self.orig_d_i == self.d_i else self.proj_i(x_i)

        # -> [T,B,D]
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_n = proj_x_n.permute(2, 0, 1)
        proj_x_i = proj_x_i.permute(2, 0, 1)

        # Self
        h_l_only = self.trans_l(proj_x_l)
        h_n_only = self.trans_n(proj_x_n)
        h_i_only = self.trans_i(proj_x_i)

        # Cross (directional)
        h_l_with_ns = self.trans_l_with_n(proj_x_l, proj_x_n, proj_x_n)
        h_l_with_is = self.trans_l_with_i(proj_x_l, proj_x_i, proj_x_i)

        h_n_with_ls = self.trans_n_with_l(proj_x_n, proj_x_l, proj_x_l)
        h_n_with_is = self.trans_n_with_i(proj_x_n, proj_x_i, proj_x_i)

        h_i_with_ls = self.trans_i_with_l(proj_x_i, proj_x_l, proj_x_l)
        h_i_with_ns = self.trans_i_with_n(proj_x_i, proj_x_n, proj_x_n)

        # last timestep pooled (MulT style)
        h_l_last = h_l_only[-1]      # [B, d]
        h_n_last = h_n_only[-1]
        h_i_last = h_i_only[-1]

        h_ln_last = h_l_with_ns[-1]  # [B, d_l]  (L with A)
        h_li_last = h_l_with_is[-1]  # [B, d_l]  (L with V)

        h_nl_last = h_n_with_ls[-1]  # [B, d_a]  (A with L)
        h_ni_last = h_n_with_is[-1]  # [B, d_a]  (A with V)

        h_il_last = h_i_with_ls[-1]  # [B, d_v]  (V with L)
        h_in_last = h_i_with_ns[-1]  # [B, d_v]  (V with A)

        # Tri (MulT style)
        h_lni_last = torch.cat([h_nl_last, h_in_last, h_li_last], dim=1)
        h_lni_last = self.final_lni(h_lni_last)
        
        return {
            "L":   h_l_last,
            "N":   h_n_last,
            "I":   h_i_last,

            "LN":  h_ln_last,
            "LI":  h_li_last,

            "NL":  h_nl_last,
            "NI":  h_ni_last,

            "IL":  h_il_last,
            "IN":  h_in_last,

            "LNI": h_lni_last,
        }
