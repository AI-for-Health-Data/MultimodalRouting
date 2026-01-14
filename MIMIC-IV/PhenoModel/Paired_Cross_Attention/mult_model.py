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

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_n = nn.Conv1d(self.orig_d_n, self.d_n, kernel_size=1, padding=0, bias=False)
        self.proj_i = nn.Conv1d(self.orig_d_i, self.d_i, kernel_size=1, padding=0, bias=False)

        self.trans_l = self.get_network(self_type='l_only', layers=self.self_layers)
        self.trans_n = self.get_network(self_type='n_only', layers=self.self_layers)
        self.trans_i = self.get_network(self_type='i_only', layers=self.self_layers)

        if self.lonly:
            self.trans_l_with_n = self.get_network(self_type='ln')
            self.trans_l_with_i = self.get_network(self_type='li')
        if self.nonly:
            self.trans_n_with_l = self.get_network(self_type='nl')
            self.trans_n_with_i = self.get_network(self_type='ni')
        if self.ionly:
            self.trans_i_with_l = self.get_network(self_type='il')
            self.trans_i_with_n = self.get_network(self_type='in')

        # Align N / I embeddings into d_l for pairing (no change to existing returned routes)
        self.proj_n_to_l = nn.Identity() if self.d_n == self.d_l else nn.Linear(self.d_n, self.d_l, bias=True)
        self.proj_i_to_l = nn.Identity() if self.d_i == self.d_l else nn.Linear(self.d_i, self.d_l, bias=True)

        # Pair projections (2*d_l -> d_l)
        self.proj_pair_ln = nn.Linear(2 * self.d_l, self.d_l, bias=True)
        self.proj_pair_li = nn.Linear(2 * self.d_l, self.d_l, bias=True)
        self.proj_pair_ni = nn.Linear(2 * self.d_l, self.d_l, bias=True)

        # Trimodal projection (3*d_l -> d_l)
        self.final_lni = nn.Linear(3 * self.d_l, self.d_l, bias=True)

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

    def _masked_mean_tbd(self, h_tbd, m_bt):
        """
        h_tbd: [T,B,D]
        m_bt:  [B,T] float (1=keep, 0=pad)
        returns [B,D]
        """
        if m_bt is None:
            return h_tbd.mean(dim=0)  # mean over T -> [B,D]
        m = m_bt.float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        h_btd = h_tbd.permute(1, 0, 2)                     # [B,T,D]
        return (h_btd * m.unsqueeze(-1)).sum(dim=1) / denom

    def _ensure_float_mask(self, m, B, T, device):
        if m is None:
            return None
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(B, -1)
        return m.to(device=device).float()

    def forward_from_encoders(
        self,
        L_seq, N_seq, I_seq,              # [B, TL, DL], [B, TN, DN], [B, TI, DI]
        mL=None, mN=None, mI=None,         # [B, TL], [B, TN], [B, TI] float (1=keep,0=pad)
        L_pool=None, N_pool=None, I_pool=None,  # [B, *] pooled from encoders (preferred)
    ):
        assert L_seq.dim() == 3 and N_seq.dim() == 3 and I_seq.dim() == 3
        B, TL, DL = L_seq.shape
        BN, TN, DN = N_seq.shape
        BI, TI, DI = I_seq.shape
        assert B == BN == BI

        device = L_seq.device
        mL = self._ensure_float_mask(mL, B, TL, device)
        mN = self._ensure_float_mask(mN, B, TN, device)
        mI = self._ensure_float_mask(mI, B, TI, device)

        # [B,T,D] -> [B,D,T]
        x_l = F.dropout(L_seq.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_n = N_seq.transpose(1, 2)
        x_i = I_seq.transpose(1, 2)

        # conv proj
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_n = x_n if self.orig_d_n == self.d_n else self.proj_n(x_n)
        proj_x_i = x_i if self.orig_d_i == self.d_i else self.proj_i(x_i)

        # [B,D,T] -> [T,B,D]
        proj_x_l = proj_x_l.permute(2, 0, 1)  # [TL,B,d_l]
        proj_x_n = proj_x_n.permute(2, 0, 1)  # [TN,B,d_n]
        proj_x_i = proj_x_i.permute(2, 0, 1)  # [TI,B,d_i]

        if not (self.lonly and self.nonly and self.ionly):
            raise RuntimeError(
                "forward_from_encoders requires lonly=nonly=ionly=True so cross modules exist "
                "(trans_l_with_n/trans_l_with_i/trans_n_with_l/trans_n_with_i/trans_i_with_l/trans_i_with_n)."
        )

        # Cross (directional) with masks
        h_l_with_ns = self.trans_l_with_n(proj_x_l, proj_x_n, proj_x_n, q_mask=mL, kv_mask=mN)
        h_l_with_is = self.trans_l_with_i(proj_x_l, proj_x_i, proj_x_i, q_mask=mL, kv_mask=mI)

        h_n_with_ls = self.trans_n_with_l(proj_x_n, proj_x_l, proj_x_l, q_mask=mN, kv_mask=mL)
        h_n_with_is = self.trans_n_with_i(proj_x_n, proj_x_i, proj_x_i, q_mask=mN, kv_mask=mI)

        h_i_with_ls = self.trans_i_with_l(proj_x_i, proj_x_l, proj_x_l, q_mask=mI, kv_mask=mL)
        h_i_with_ns = self.trans_i_with_n(proj_x_i, proj_x_n, proj_x_n, q_mask=mI, kv_mask=mN)

        # Pool cross outputs WITH QUERY mask (never use [-1] when padded)
        zLN = self._masked_mean_tbd(h_l_with_ns, mL)   # [B,d_l]  e_{L<-N}
        zLI = self._masked_mean_tbd(h_l_with_is, mL)   # [B,d_l]  e_{L<-I}

        zNL = self._masked_mean_tbd(h_n_with_ls, mN)   # [B,d_n]  e_{N<-L}
        zNI = self._masked_mean_tbd(h_n_with_is, mN)   # [B,d_n]  e_{N<-I}

        zIL = self._masked_mean_tbd(h_i_with_ls, mI)   # [B,d_i]  e_{I<-L}
        zIN = self._masked_mean_tbd(h_i_with_ns, mI)   # [B,d_i]  e_{I<-N}

        # UPDATED Tri: build from (LN, LI, NI) pair embeddings 
        zNL_l = self.proj_n_to_l(zNL)  # [B,d_l]
        zNI_l = self.proj_n_to_l(zNI)  # [B,d_l]
        zIL_l = self.proj_i_to_l(zIL)  # [B,d_l]
        zIN_l = self.proj_i_to_l(zIN)  # [B,d_l]

        eLN = self.proj_pair_ln(torch.cat([zLN, zNL_l], dim=1))  # [B,d_l]
        eLI = self.proj_pair_li(torch.cat([zLI, zIL_l], dim=1))  # [B,d_l]
        eNI = self.proj_pair_ni(torch.cat([zNI_l, zIN_l], dim=1))# [B,d_l]

        zLNI = torch.cat([eLN, eLI, eNI], dim=1)                 # [B,3*d_l]
        zLNI = self.final_lni(zLNI)                              # [B,d_l]

        zL = L_pool if L_pool is not None else self._masked_mean_tbd(proj_x_l, mL)  # proj_x_l is [T,B,D]
        zN = N_pool if N_pool is not None else self._masked_mean_tbd(proj_x_n, mN)
        zI = I_pool if I_pool is not None else self._masked_mean_tbd(proj_x_i, mI)

        return {
            "L": zL, "N": zN, "I": zI,
            "LN": zLN, "LI": zLI,
            "NL": zNL, "NI": zNI,
            "IL": zIL, "IN": zIN,
            "LNI": zLNI,
        }

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

        # last timestep pooled (as in your original forward)
        h_l_last = h_l_only[-1]      # [B, d_l]
        h_n_last = h_n_only[-1]      # [B, d_n]
        h_i_last = h_i_only[-1]      # [B, d_i]

        h_ln_last = h_l_with_ns[-1]  # [B, d_l]  e_{L<-N}
        h_li_last = h_l_with_is[-1]  # [B, d_l]  e_{L<-I}

        h_nl_last = h_n_with_ls[-1]  # [B, d_n]  e_{N<-L}
        h_ni_last = h_n_with_is[-1]  # [B, d_n]  e_{N<-I}

        h_il_last = h_i_with_ls[-1]  # [B, d_i]  e_{I<-L}
        h_in_last = h_i_with_ns[-1]  # [B, d_i]  e_{I<-N}

        h_nl_l = self.proj_n_to_l(h_nl_last)  # [B,d_l]
        h_ni_l = self.proj_n_to_l(h_ni_last)  # [B,d_l]
        h_il_l = self.proj_i_to_l(h_il_last)  # [B,d_l]
        h_in_l = self.proj_i_to_l(h_in_last)  # [B,d_l]

        eLN = self.proj_pair_ln(torch.cat([h_ln_last, h_nl_l], dim=1))  # [B,d_l]
        eLI = self.proj_pair_li(torch.cat([h_li_last, h_il_l], dim=1))  # [B,d_l]
        eNI = self.proj_pair_ni(torch.cat([h_ni_l,  h_in_l], dim=1))     # [B,d_l]

        h_lni_last = torch.cat([eLN, eLI, eNI], dim=1)                   # [B,3*d_l]
        h_lni_last = self.final_lni(h_lni_last)                          # [B,d_l]

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
