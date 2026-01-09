import math
import torch
from torch import nn
import torch.nn.functional as F

from position_embedding import SinusoidalPositionalEmbedding
from multihead_attention import MultiheadAttention


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        layers,
        attn_dropout=0.0,
        relu_dropout=0.0,
        res_dropout=0.0,
        embed_dropout=0.0,
        attn_mask=False,
        use_positional=True,
        padding_idx=0,
        left_pad=False,
    ):
        super().__init__()
        self.dropout = float(embed_dropout)
        self.attn_dropout = float(attn_dropout)
        self.embed_dim = int(embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.attn_mask = bool(attn_mask)

        self.embed_positions = (
            SinusoidalPositionalEmbedding(self.embed_dim, padding_idx=padding_idx, left_pad=left_pad)
            if use_positional else None
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask,
            )
            for _ in range(int(layers))
        ])

        self.normalize = True
        self.layer_norm = nn.LayerNorm(self.embed_dim) if self.normalize else None

    def forward(self, x_in, x_in_k=None, x_in_v=None, q_mask=None, kv_mask=None):
        """
        x_in:   [T, B, C]
        x_in_k: [T2, B, C] optional
        x_in_v: [T2, B, C] optional
        """
        x = self.embed_scale * x_in

        if self.embed_positions is not None:
            # Build dummy token ids [B,T] just to get positions
            B = x_in.size(1)
            T = x_in.size(0)
            dummy = torch.ones(B, T, device=x_in.device, dtype=torch.long)
            pos = self.embed_positions(dummy).transpose(0, 1)  # [T,B,C]
            pos = pos.to(dtype=x.dtype)
            x = x + pos


        x = F.dropout(x, p=self.dropout, training=self.training)

        if q_mask is not None:
            q_keep = q_mask.transpose(0, 1).to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            x = x * q_keep

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                pos_k = self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                pos_v = self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_k = x_k + pos_k
                x_v = x_v + pos_v
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        else:
            x_k = None
            x_v = None

        for layer in self.layers:
            if x_k is not None and x_v is not None:
                x = layer(x, x_k, x_v, q_mask=q_mask, kv_mask=kv_mask)
            else:
                x = layer(x, q_mask=q_mask, kv_mask=q_mask)

        if self.normalize:
            x = self.layer_norm(x)

        if q_mask is not None:
            q_keep = q_mask.transpose(0, 1).to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            x = x * q_keep

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=4,
        attn_dropout=0.1,
        relu_dropout=0.1,
        res_dropout=0.1,
        attn_mask=False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.attn_mask = bool(attn_mask)

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=float(attn_dropout),
        )

        self.relu_dropout = float(relu_dropout)
        self.res_dropout = float(res_dropout)
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)

        # exactly 2 layer norms (i=0 attn block, i=1 ffn block)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None, q_mask=None, kv_mask=None):
        """
        x:   [Tq,B,C]
        x_k: [Tk,B,C] optional
        x_v: [Tk,B,C] optional
        q_mask:  [B,Tq] float (1=keep, 0=pad)
        kv_mask: [B,Tk] float (1=keep, 0=pad) for cross-attn; for self-attn you can pass q_mask
        """
        # Build keep mask for query positions so PAD stays zero
        q_keep = None
        if q_mask is not None:
            q_keep = q_mask.transpose(0, 1).to(device=x.device, dtype=x.dtype).unsqueeze(-1)  # [Tq,B,1]

        # Choose key padding mask for attention (True means PAD)
        key_padding_mask = None
        if x_k is None and x_v is None:
            if q_mask is not None:
                key_padding_mask = (q_mask < 0.5).to(device=x.device)  # [B,Tq] bool-ish
        else:
            if kv_mask is not None:
                key_padding_mask = (kv_mask < 0.5).to(device=x.device)  # [B,Tk]


        residual = x
        x = self.maybe_layer_norm(0, x, before=True)

        if q_keep is not None:
            x = x * q_keep

        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(
                query=x, key=x, value=x,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
            )
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(
                query=x, key=x_k, value=x_v,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
            )

        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        if q_keep is not None:
            x = x * q_keep
        x = self.maybe_layer_norm(0, x, after=True)
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        if q_keep is not None:
            x = x * q_keep
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x

        if q_keep is not None:
            x = x * q_keep
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            ln = self.layer_norms[i]
            if ln.weight.device != x.device:
                self.layer_norms[i] = ln.to(x.device)
                ln = self.layer_norms[i]
            return ln(x)
        return x

def fill_with_neg_inf(t):
    return t.float().fill_(float("-inf")).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = tensor.size(0)
    dim2 = tensor2.size(0) if tensor2 is not None else dim1

    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    future_mask = future_mask.to(tensor.device)
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
