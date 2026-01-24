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

        self.trans_l_with_n = self.get_network(self_type='ln')
        self.trans_l_with_i = self.get_network(self_type='li')

        self.trans_n_with_l = self.get_network(self_type='nl')
        self.trans_n_with_i = self.get_network(self_type='ni')

        self.trans_i_with_l = self.get_network(self_type='il')
        self.trans_i_with_n = self.get_network(self_type='in')

        self.proj_n_to_l = nn.Identity() if self.d_n == self.d_l else nn.Linear(self.d_n, self.d_l, bias=True)
        self.proj_i_to_l = nn.Identity() if self.d_i == self.d_l else nn.Linear(self.d_i, self.d_l, bias=True)

        self.proj_pair_ln = nn.Linear(2 * self.d_l, self.d_l, bias=True)
        self.proj_pair_li = nn.Linear(2 * self.d_l, self.d_l, bias=True)
        self.proj_pair_ni = nn.Linear(2 * self.d_l, self.d_l, bias=True)

        self.final_lni = nn.Linear(3 * self.d_l, self.d_l, bias=True)

    def get_network(self, self_type: str = 'l', layers: int = -1):
        n_layers = self.layers if layers == -1 else layers

        q = self_type[0]
        if q == 'l':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif q == 'n':
            embed_dim, attn_dropout = self.d_n, self.attn_dropout_n
        elif q == 'i':
            embed_dim, attn_dropout = self.d_i, self.attn_dropout_i
        else:
            raise ValueError(f"Unknown network type: {self_type}")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=n_layers,             
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )


    def _masked_mean_tbd(self, h_tbd, m_bt):
        if m_bt is None:
            return h_tbd.mean(dim=0)  # mean over T -> [B,D]
        m = m_bt.float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        h_btd = h_tbd.permute(1, 0, 2)                     # [B,T,D]
        return (h_btd * m.unsqueeze(-1)).sum(dim=1) / denom

    def _masked_last_tbd(self, h_tbd, m_bt):
        if m_bt is None:
            return h_tbd[-1]  # [B,D]

        m = (m_bt > 0.5).long()          # [B,T]
        lengths = m.sum(dim=1)           # [B]
        idx = (lengths - 1).clamp_min(0) # [B]

        h_btd = h_tbd.permute(1, 0, 2)   # [B,T,D]
        out = h_btd[torch.arange(h_btd.size(0), device=h_btd.device), idx]  # [B,D]

        # if length==0 -> zero it out
        if (lengths == 0).any():
            out = out.clone()
            out[lengths == 0] = 0.0
        return out


    def _ensure_float_mask(self, m, B, T, device):
        if m is None:
            return None
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(B, -1)
        return m.to(device=device).float()

    def forward(self, x_l, x_n, x_i, mL=None, mN=None, mI=None):
        assert x_l.dim() == 3 and x_n.dim() == 3 and x_i.dim() == 3
        B, TL, _ = x_l.shape
        BN, TN, _ = x_n.shape
        BI, TI, _ = x_i.shape
        assert B == BN == BI

        device = x_l.device
        mL = self._ensure_float_mask(mL, B, TL, device)
        mN = self._ensure_float_mask(mN, B, TN, device)
        mI = self._ensure_float_mask(mI, B, TI, device)

        # [B,T,D] -> [B,D,T]
        xl = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        xn = x_n.transpose(1, 2)
        xi = x_i.transpose(1, 2)

        pl = xl if self.orig_d_l == self.d_l else self.proj_l(xl)
        pn = xn if self.orig_d_n == self.d_n else self.proj_n(xn)
        pi = xi if self.orig_d_i == self.d_i else self.proj_i(xi)

        # [B,d,T] -> [T,B,d]
        pl = pl.permute(2, 0, 1)
        pn = pn.permute(2, 0, 1)
        pi = pi.permute(2, 0, 1)

        hL = self.trans_l(pl, q_mask=mL, kv_mask=mL)
        hN = self.trans_n(pn, q_mask=mN, kv_mask=mN)
        hI = self.trans_i(pi, q_mask=mI, kv_mask=mI)

        zL = self._masked_last_tbd(hL, mL)   # [B,d_l]
        zN = self._masked_last_tbd(hN, mN)   # [B,d_n]
        zI = self._masked_last_tbd(hI, mI)   # [B,d_i]

        hLN = self.trans_l_with_n(pl, pn, pn, q_mask=mL, kv_mask=mN)  # L<-N  [TL,B,d_l]
        hLI = self.trans_l_with_i(pl, pi, pi, q_mask=mL, kv_mask=mI)  # L<-I

        hNL = self.trans_n_with_l(pn, pl, pl, q_mask=mN, kv_mask=mL)  # N<-L  [TN,B,d_n]
        hNI = self.trans_n_with_i(pn, pi, pi, q_mask=mN, kv_mask=mI)  # N<-I

        hIL = self.trans_i_with_l(pi, pl, pl, q_mask=mI, kv_mask=mL)  # I<-L  [TI,B,d_i]
        hIN = self.trans_i_with_n(pi, pn, pn, q_mask=mI, kv_mask=mN)  # I<-N

        zLN = self._masked_last_tbd(hLN, mL)  # [B,d_l]
        zLI = self._masked_last_tbd(hLI, mL)  # [B,d_l]

        zNL = self._masked_last_tbd(hNL, mN)  # [B,d_n]
        zNI = self._masked_last_tbd(hNI, mN)  # [B,d_n]

        zIL = self._masked_last_tbd(hIL, mI)  # [B,d_i]
        zIN = self._masked_last_tbd(hIN, mI)  # [B,d_i]

        zNL_l = self.proj_n_to_l(zNL)  # [B,d_l]
        zNI_l = self.proj_n_to_l(zNI)  # [B,d_l]
        zIL_l = self.proj_i_to_l(zIL)  # [B,d_l]
        zIN_l = self.proj_i_to_l(zIN)  # [B,d_l]

        eLN = self.proj_pair_ln(torch.cat([zLN,  zNL_l], dim=1))  # [B,d_l]
        eLI = self.proj_pair_li(torch.cat([zLI,  zIL_l], dim=1))  # [B,d_l]
        eNI = self.proj_pair_ni(torch.cat([zNI_l, zIN_l], dim=1))  # [B,d_l]

        zLNI = self.final_lni(torch.cat([eLN, eLI, eNI], dim=1))   # [B,d_l]

        out = {
            "L": zL, "N": zN, "I": zI,
            "LN": zLN, "LI": zLI,
            "NL": zNL, "NI": zNI,
            "IL": zIL, "IN": zIN,
            "LNI": zLNI,
        }

        target_dtype = next(self.parameters()).dtype
        for k, v in out.items():
            if torch.is_tensor(v) and v.dtype != target_dtype:
                out[k] = v.to(dtype=target_dtype)

        return out
