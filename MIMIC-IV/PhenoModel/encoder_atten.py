# encoders_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from env_config import CFG, DEVICE

# -------------------------
# Debug utils (optional)
# -------------------------
def _dbg(msg: str) -> None:
    if getattr(CFG, "verbose", False):
        print(msg)

def _peek_tensor(name: str, x: torch.Tensor, k: int = 3) -> None:
    if not getattr(CFG, "verbose", False):
        return
    if not hasattr(_peek_tensor, "_printed"):
        _peek_tensor._printed = set()
    key = f"{name}_shape"
    if key in _peek_tensor._printed:
        return
    _peek_tensor._printed.add(key)
    try:
        with torch.no_grad():
            flat = x.reshape(-1)
            vals = flat[:k].detach().cpu().tolist()
        print(f"[peek] {name}: shape={tuple(x.shape)} sample={vals}")
    except Exception:
        print(f"[peek] {name}: shape={tuple(x.shape)} sample=<unavailable>")


def _ensure_2d_mask(
    mask: Optional[torch.Tensor],
    B: int,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(B, T, device=device, dtype=torch.float32)
    if mask.dim() == 1:
        return mask.unsqueeze(0).expand(B, -1).contiguous().float()
    return mask.float().to(device)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    x: [B,T,D], mask: [B,T] (1=keep)
    """
    m = mask.float()
    denom = m.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * m.unsqueeze(-1)).sum(dim=dim) / denom


# ============================================================
# 1) BEHRT-style structured encoder (CLS output)
# ============================================================
class BEHRTLabEncoder(nn.Module):
    """
    Structured encoder over [B,T,F] (F=17).

    Returns:
      encode_seq_and_pool -> (seq [B,T,D], mask [B,T], cls [B,D])
      forward             -> cls [B,D]  (always CLS)
    """
    def __init__(
        self,
        n_feats: int,
        d: int,
        seq_len: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        super().__init__()

        self.out_dim = int(d)
        self.n_feats = int(n_feats)

        self.max_seq_len = int(seq_len)
        if self.max_seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")

        self.input_proj = nn.Linear(self.n_feats, d)

        # learnable absolute positional embeddings for time steps (not including CLS)
        self.pos = nn.Parameter(torch.randn(1, self.max_seq_len, d) * 0.02)

        # learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=0.0,
            activation=activation,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
        )

        self._printed_once = False

    def _pos(self, T: int) -> torch.Tensor:
        if T > self.pos.size(1):
            raise ValueError(
                f"Input length T={T} exceeds max_seq_len={self.pos.size(1)}. "
                "Increase structured_seq_len."
            )
        return self.pos[:, :T, :]

    # encoders.py  (inside class BioClinBERTEncoder)


    def encode_seq_and_pool(
        self,
        x: torch.Tensor,                  # [B,T,F]
        mask: Optional[torch.Tensor] = None,  # [B,T]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,T,F], got {tuple(x.shape)}")

        dev = next(self.parameters()).device
        x = x.to(dev)

        B, T, F = x.shape
        if F != self.n_feats:
            raise ValueError(f"Expected F={self.n_feats}, got F={F}")

        m = _ensure_2d_mask(mask, B, T, dev)  # [B,T]

        # project features + add time pos embedding
        h = self.input_proj(x) + self._pos(T)  # [B,T,D]

        # prepend CLS
        cls_tok = self.cls_token.expand(B, 1, -1)  # [B,1,D]
        h_in = torch.cat([cls_tok, h], dim=1)      # [B,T+1,D]

        # key padding mask: True = PAD
        pad_mask = torch.cat(
            [
                torch.zeros(B, 1, device=dev, dtype=torch.bool),
                (m < 0.5),
            ],
            dim=1,
        )  # [B,T+1]

        h_out = self.enc(h_in, src_key_padding_mask=pad_mask)  # [B,T+1,D]
        h_out = self.out(h_out)

        cls = h_out[:, 0, :]    # [B,D]  ✅ CLS embedding
        seq = h_out[:, 1:, :]   # [B,T,D]

        if not self._printed_once:
            self._printed_once = True
            _dbg(f"[BEHRTLabEncoder] seq={tuple(seq.shape)} cls={tuple(cls.shape)} mask={tuple(m.shape)}")
            _peek_tensor("behrt.cls", cls)

        return seq, m, cls



    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, _, cls = self.encode_seq_and_pool(x, mask=mask)
        return cls


# ============================================================
# 2) BioClinicalBERT notes encoder (CLS per chunk -> projected to D)
# ============================================================
class BioClinBERTEncoder(nn.Module):
    """
    For chunked, pre-tokenized notes.

    Preferred input (matches your parquet/collate):
      input_ids      : [B,S,L]
      attention_mask : [B,S,L]
      chunk_mask     : [B,S] (1=real chunk, 0=pad)

    Output:
      encode_seq_and_pool -> (chunk_cls [B,S,D], chunk_mask [B,S], pooled_cls [B,D])
      forward             -> pooled_cls [B,D]

    IMPORTANT:
      - We extract CLS from BERT: last_hidden_state[:,0,:]
      - Then we PROJECT to dimension D (so your routing expects consistent d).
      - pooled_cls is masked-mean across chunks (or max if CFG.note_agg == "max")
    """
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        d: int = 256,
        chunk_bs: int = 8,
        agg: Literal["mean", "max"] = "mean",
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name)
        hidden = int(getattr(self.bert.config, "hidden_size", 768))

        self.hidden = hidden
        self.out_dim = int(d)
        self.chunk_bs = max(1, int(chunk_bs))
        self.agg = str(agg).lower().strip()

        # project CLS(hidden) -> D  (keeps your pipeline consistent)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.out_dim, bias=False),
        )

        self._printed_once = False

    def encode_seq_and_pool(self, input_ids, attention_mask, chunk_mask=None):
        """
        Accepts:
          - input_ids: [B,L] or [B,S,L]
          - attention_mask: same shape
          - chunk_mask: [B,S] optional (1=valid chunk, 0=pad chunk)

        Returns:
          seq:  [B,S,D]   (each chunk -> CLS -> projected)
          mask: [B,S]     (valid chunks)
          pool: [B,D]     (mean/max over valid chunks)
        """
        device = input_ids.device
        if input_ids.ndim == 2:
            # treat as one chunk per sample
            input_ids = input_ids.unsqueeze(1)         # [B,1,L]
            attention_mask = attention_mask.unsqueeze(1)
            if chunk_mask is None:
                chunk_mask = torch.ones(input_ids.size(0), 1, device=device, dtype=torch.float32)

        assert input_ids.ndim == 3, f"expected [B,S,L], got {tuple(input_ids.shape)}"
        B, S, L = input_ids.shape

        if chunk_mask is None:
            # valid chunk if it has any attention > 0
            chunk_mask = (attention_mask.sum(dim=-1) > 0).float()  # [B,S]
        else:
            chunk_mask = chunk_mask.float()

        # flatten chunks: [B*S, L]
        flat_ids = input_ids.reshape(B * S, L)
        flat_attn = attention_mask.reshape(B * S, L)

        # run BERT in mini-batches to control VRAM
        bs = int(self.chunk_bs)

        cls_list = []
        for i in range(0, B * S, bs):
            out = self.bert(
                input_ids=flat_ids[i:i+bs],
                attention_mask=flat_attn[i:i+bs],
            )
            cls = out.last_hidden_state[:, 0, :]  # [bs, H]
            cls_list.append(cls)

        cls_all = torch.cat(cls_list, dim=0)      # [B*S, H]
        cls_all = self.proj(cls_all)              # [B*S, D]
        seq = cls_all.view(B, S, -1)              # [B,S,D]

        mask = chunk_mask                          # [B,S]
        # avoid divide-by-zero
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        if self.agg == "mean":

            # set invalid chunks to -inf then max
            seq_masked = seq.masked_fill(mask.unsqueeze(-1) < 0.5, float("-inf"))
            pool = torch.max(seq_masked, dim=1).values
            pool = torch.nan_to_num(pool, nan=0.0, neginf=0.0, posinf=0.0)
        else:
            # mean over valid chunks
            pool = (seq * mask.unsqueeze(-1)).sum(dim=1) / denom

        return seq, mask, pool

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, _, pooled = self.encode_seq_and_pool(input_ids, attention_mask, chunk_mask)
        return pooled


# ============================================================
# 3) Image encoder (MedFuse-style: ResNet backbone -> pooled + layer4 tokens)
# ============================================================
class ImageEncoder(nn.Module):
    """
    MedFuse-style image encoder using torchvision ResNet.

    Input:
      imgs: [B,3,H,W]

    Output:
      encode_seq_and_pool -> (I_seq [B,P,D], I_mask [B,P], I_pool [B,D])
      forward             -> I_pool [B,D]

    Notes:
      - I_pool is global pooled (resnet penultimate vector) projected to D.
      - I_seq are spatial tokens from layer4 feature map projected to D.
    """
    def __init__(
        self,
        d: int = 256,
        vision_backbone: str = "resnet34",
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        import torchvision

        dev = torch.device(DEVICE if device is None else device)

        # robust torchvision API (old/new)
        model_fn = getattr(torchvision.models, vision_backbone)
        try:
            if pretrained:
                weights_enum = getattr(torchvision.models, f"{vision_backbone.upper()}_Weights", None)
                weights = weights_enum.DEFAULT if weights_enum is not None else None
            else:
                weights = None
            m = model_fn(weights=weights)
        except TypeError:
            m = model_fn(pretrained=pretrained)

        if not hasattr(m, "layer4") or not hasattr(m, "fc"):
            raise ValueError("Only torchvision ResNet backbones supported (need layer4 + fc).")

        feats_dim = int(m.fc.in_features)
        m.fc = nn.Identity()
        self.backbone = m

        self.proj_pool = nn.Linear(feats_dim, d)
        self.token_in_dim = self._infer_layer4_channels()
        self.proj_tokens = nn.Linear(self.token_in_dim, d, bias=False)

        self.out_dim = int(d)
        self._printed_once = False

        self.to(dev)

    def _infer_layer4_channels(self) -> int:
        layer4 = self.backbone.layer4
        last_block = list(layer4.children())[-1]
        if hasattr(last_block, "conv2"):  # resnet18/34 basicblock
            return int(last_block.conv2.out_channels)
        if hasattr(last_block, "conv3"):  # resnet50+ bottleneck
            return int(last_block.conv3.out_channels)
        for mod in reversed(list(last_block.modules())):
            if isinstance(mod, nn.Conv2d):
                return int(mod.out_channels)
        raise ValueError("Could not infer layer4 output channels.")

    def encode_seq_and_pool(
        self,
        imgs: torch.Tensor,  # [B,3,H,W]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = next(self.parameters()).device
        x = imgs.to(dev)
        if x.dim() != 4:
            raise ValueError(f"imgs must be [B,3,H,W], got {tuple(x.shape)}")

        holder: Dict[str, torch.Tensor] = {}

        def _hook(_m, _inp, out):
            holder["fmap"] = out

        h = self.backbone.layer4.register_forward_hook(_hook)
        try:
            feats = self.backbone(x)  # [B, feats_dim]
        finally:
            h.remove()

        fmap = holder.get("fmap", None)
        if fmap is None:
            raise RuntimeError("Failed to capture layer4 fmap.")

        B, C, H4, W4 = fmap.shape
        if C != self.token_in_dim:
            raise ValueError(f"layer4 channels mismatch: got {C}, expected {self.token_in_dim}")

        I_pool = self.proj_pool(feats)  # [B,D]

        tokens = fmap.permute(0, 2, 3, 1).reshape(B, H4 * W4, C)  # [B,P,C]
        I_seq = self.proj_tokens(tokens)                          # [B,P,D]
        I_mask = torch.ones(B, H4 * W4, device=dev)

        if not self._printed_once:
            self._printed_once = True
            _dbg(f"[ImageEncoder] I_seq={tuple(I_seq.shape)} I_pool={tuple(I_pool.shape)}")
            _peek_tensor("img.I_pool", I_pool)

        return I_seq, I_mask, I_pool

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        _, _, I_pool = self.encode_seq_and_pool(imgs)
        return I_pool


# ============================================================
# Build + encode wrapper
# ============================================================
@dataclass
class EncoderConfig:
    d: int = 256

    # structured
    structured_seq_len: int = 256
    structured_n_feats: int = 17
    structured_layers: int = 2
    structured_heads: int = 8

    # notes
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    bert_chunk_bs: int = 8
    note_agg: Literal["mean", "max"] = "mean"

    # images
    vision_backbone: str = "resnet34"
    vision_pretrained: bool = True


def build_encoders(
    cfg: EncoderConfig,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder]:
    dev = torch.device(DEVICE if device is None else device)

    behrt = BEHRTLabEncoder(
        n_feats=cfg.structured_n_feats,
        d=cfg.d,
        seq_len=cfg.structured_seq_len,
        n_layers=cfg.structured_layers,
        n_heads=cfg.structured_heads,
        activation="gelu",
    ).to(dev)

    bbert = BioClinBERTEncoder(
        model_name=cfg.text_model_name,
        d=cfg.d,
        chunk_bs=cfg.bert_chunk_bs,
        agg=cfg.note_agg,
    ).to(dev)

    imgenc = ImageEncoder(
        d=cfg.d,
        vision_backbone=cfg.vision_backbone,
        pretrained=cfg.vision_pretrained,
        device=dev,
    ).to(dev)

    return behrt, bbert, imgenc


def encode_modalities_for_routing(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    x_struct: torch.Tensor,                 # [B,T,17]
    m_struct: Optional[torch.Tensor],       # [B,T]
    input_ids: torch.Tensor,                # [B,S,L]
    attn_mask: torch.Tensor,                # [B,S,L]
    chunk_mask: torch.Tensor,               # [B,S]
    imgs: torch.Tensor,                     # [B,3,H,W]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Returns exactly what routing/fusion code expects:
      z["L"]["seq"], z["L"]["mask"], z["L"]["pool"]  (pool = CLS)
      z["N"]["seq"], z["N"]["mask"], z["N"]["pool"]  (seq = chunk CLS tokens, pool = pooled CLS)
      z["I"]["seq"], z["I"]["mask"], z["I"]["pool"]
    """
    dev = next(behrt.parameters()).device

    L_seq, L_mask, L_cls = behrt.encode_seq_and_pool(x_struct.to(dev), mask=None if m_struct is None else m_struct.to(dev))

    N_seq, N_mask, N_cls = bbert.encode_seq_and_pool(
        input_ids=input_ids,
        attention_mask=attn_mask,
        chunk_mask=chunk_mask,
    )

    I_seq, I_mask, I_pool = imgenc.encode_seq_and_pool(imgs.to(dev))

    return {
        "L": {"seq": L_seq, "mask": L_mask, "pool": L_cls},   # ✅ CLS
        "N": {"seq": N_seq, "mask": N_mask, "pool": N_cls},   # ✅ CLS (pooled across chunks)
        "I": {"seq": I_seq, "mask": I_mask, "pool": I_pool},
    }


__all__ = [
    "BEHRTLabEncoder",
    "BioClinBERTEncoder",
    "ImageEncoder",
    "EncoderConfig",
    "build_encoders",
    "encode_modalities_for_routing",
]
