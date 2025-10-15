# MIMIC-IV/PhenoModel/encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Sequence, Union, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import DEVICE


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def _ensure_2d_mask(mask: Optional[torch.Tensor], B: int, T: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(B, T, device=device, dtype=torch.float32)
    if mask.dim() == 1:
        return mask.unsqueeze(0).expand(B, -1).contiguous().float().to(device)
    return mask.float().to(device)


class BEHRTLabEncoder(nn.Module):
    """
    Simple transformer encoder over structured time series (labs/vitals), with
    sinusoidal positional encodings and 'last'/'mean' pooling.
    """
    def __init__(
        self,
        n_feats: int,
        d: int,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        pool: Literal["last", "mean"] = "last",
        max_len_pos: int = 512,
        use_input_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.d = d
        self.n_feats = n_feats
        self.use_input_layernorm = bool(use_input_layernorm)

        self.in_norm = nn.LayerNorm(n_feats) if self.use_input_layernorm else nn.Identity()
        self.input_proj = nn.Linear(n_feats, d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        pos = self._build_sinusoidal_pos(max_len_pos, d)
        self.register_buffer("pos_table", pos, persistent=False)

    @staticmethod
    def _build_sinusoidal_pos(L: int, d: int) -> torch.Tensor:
        pe = torch.zeros(L, d)
        position = torch.arange(0, L, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [L, d]

    def _ensure_pos_len(self, T: int, device: torch.device) -> torch.Tensor:
        if self.pos_table is None or self.pos_table.size(0) < T or self.pos_table.device != device:
            L = max(T, 2 * T, 512)
            self.pos_table = self._build_sinusoidal_pos(L, self.d).to(device)
        return self.pos_table[:T]

    def _ensure_mask(self, mask: Optional[torch.Tensor], B: int, T: int, device: torch.device) -> torch.Tensor:
        if mask is None:
            return torch.ones(B, T, device=device, dtype=torch.float32)
        if mask.dim() == 1:
            return mask.unsqueeze(0).expand(B, -1).contiguous().float().to(device)
        return mask.float().to(device)

    def encode_seq(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F = x.shape
        dev = x.device
        M = self._ensure_mask(mask, B, T, dev)

        x = self.in_norm(x)
        h = self.input_proj(x)                                      # [B, T, d]
        pos = self._ensure_pos_len(T, dev).unsqueeze(0)             # [1, T, d]
        h = h + pos
        src_key_padding_mask = (M < 0.5)                            # True = pad
        h = self.enc(h, src_key_padding_mask=src_key_padding_mask)  # [B, T, d]
        h = self.out(h)                                             # [B, T, d]
        return h, M

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pooled representation:
          "last": last valid timestep per row
          "mean": mean over valid timesteps
        """
        H, M = self.encode_seq(x, mask=mask)                        # [B, T, d], [B, T]
        if self.pool == "last":
            valid_counts = M.sum(dim=1)                             # [B]
            idx = (valid_counts - 1).clamp_min(0).long()            # [B]
            batch = torch.arange(H.size(0), device=H.device)
            z = H[batch, idx]                                       # [B, d]
        else:
            denom = M.sum(dim=1, keepdim=True).clamp_min(1.0)       # [B,1]
            z = (H * M.unsqueeze(-1)).sum(dim=1) / denom            # [B, d]
        return z


class BioClinBERTEncoder(nn.Module):
    """
    Clinical note encoder with optional attention pooling across note/chunk CLS vectors.
    """
    def __init__(
        self,
        model_name: str,
        d: int,
        max_len: int = 512,
        dropout: float = 0.1,
        note_agg: Literal["mean", "attention"] = "mean",
        max_notes_concat: int = 8,
        attn_hidden: int = 256,
        device: Optional[torch.device] = None,
        project_to_d: bool = True,
        chunk_stride: int = 64,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.d = d
        self.max_len = max_len
        self.note_agg = note_agg
        self.max_notes_concat = max_notes_concat
        self.chunk_stride = max(0, int(chunk_stride))
        self.device_override = device

        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.bert = AutoModel.from_pretrained(model_name)
            self.bert.eval()
            self.hf_available = True
        except Exception:
            self.tokenizer = None
            self.bert = None
            self.hf_available = False

        if project_to_d and d != 768:
            self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d))
            self.out_dim = d
        else:
            self.proj = nn.Identity()
            self.out_dim = 768

        self.drop = nn.Dropout(dropout)

        self.attn = None
        if note_agg == "attention":
            self.attn = nn.Sequential(
                nn.LayerNorm(self.out_dim),
                nn.Linear(self.out_dim, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )

    def _device(self) -> torch.device:
        return self.device_override or next(self.parameters()).device

    def _pad_token_id(self) -> int:
        if self.tokenizer and self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer and self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        if self.tokenizer and self.tokenizer.sep_token_id is not None:
            return self.tokenizer.sep_token_id
        return 0

    def _chunk_note_to_token_windows(self, text: Optional[str]) -> List[Dict[str, torch.Tensor]]:
        device = self._device()
        if not self.hf_available:
            # deterministic stub (keeps graph shapes valid)
            return [{
                "input_ids": torch.tensor([101, 102], device=device, dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], device=device, dtype=torch.long),
            }]

        enc = self.tokenizer(
            text or "",
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_overflowing_tokens=True,
            stride=self.chunk_stride,
            return_tensors=None,
        )
        chunks: List[Dict[str, torch.Tensor]] = []
        for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
            chunks.append({
                "input_ids": torch.tensor(ids, device=device, dtype=torch.long),
                "attention_mask": torch.tensor(attn, device=device, dtype=torch.long),
            })
        if not chunks:
            # minimal [CLS][SEP] if empty
            cls_id = self.tokenizer.cls_token_id if (self.tokenizer and self.tokenizer.cls_token_id is not None) else 101
            sep_id = self.tokenizer.sep_token_id if (self.tokenizer and self.tokenizer.sep_token_id is not None) else 102
            chunks = [{
                "input_ids": torch.tensor([cls_id, sep_id], device=device, dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], device=device, dtype=torch.long),
            }]
        return chunks

    @torch.no_grad()
    def _encode_token_windows_to_cls(self, token_windows: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        device = self._device()
        if not self.hf_available or self.bert is None:
            return torch.randn(len(token_windows), 768, device=device)

        S = len(token_windows)
        maxL = max(tw["input_ids"].numel() for tw in token_windows)
        pad_id = self._pad_token_id()

        input_ids = torch.full((S, maxL), pad_id, device=device, dtype=torch.long)
        attention_mask = torch.zeros((S, maxL), device=device, dtype=torch.long)
        for i, tw in enumerate(token_windows):
            L = tw["input_ids"].numel()
            input_ids[i, :L] = tw["input_ids"]
            attention_mask[i, :L] = tw["attention_mask"]

        self.bert.eval()
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]  # CLS

    @staticmethod
    def _normalize_input(batch_notes: Union[Sequence[str], Sequence[Sequence[str]]]) -> List[List[str]]:
        if len(batch_notes) == 0:
            return []
        if isinstance(batch_notes[0], str):
            return [[t] for t in batch_notes]
        return [list(notes) for notes in batch_notes]

    def encode_seq(
        self,
        batch_notes: Union[List[str], List[List[str]]],
        max_total_chunks: int = 32,
        chunk_bs: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self._device()
        patients = self._normalize_input(batch_notes)

        seqs: List[torch.Tensor] = []
        lengths: List[int] = []

        for notes in patients:
            if self.max_notes_concat is not None and self.max_notes_concat > 0:
                notes = notes[: self.max_notes_concat]

            token_windows: List[Dict[str, torch.Tensor]] = []
            remaining = max_total_chunks
            for note_text in notes:
                if remaining <= 0:
                    break
                ws = self._chunk_note_to_token_windows(note_text)
                if len(ws) > remaining:
                    ws = ws[:remaining]
                token_windows.extend(ws)
                remaining -= len(ws)

            if len(token_windows) == 0:
                H = torch.zeros(1, self.out_dim, device=device)
            else:
                cls_list: List[torch.Tensor] = []
                for start in range(0, len(token_windows), chunk_bs):
                    end = min(start + chunk_bs, len(token_windows))
                    cls = self._encode_token_windows_to_cls(token_windows[start:end])
                    cls_list.append(cls)
                cls_all = torch.cat(cls_list, dim=0) if cls_list else torch.zeros(0, 768, device=device)
                H = self.drop(self.proj(cls_all)) if cls_all.numel() > 0 else torch.zeros(1, self.out_dim, device=device)

            seqs.append(H)
            lengths.append(H.size(0))

        if len(seqs) == 0:
            return torch.zeros(0, 1, self.out_dim, device=device), torch.zeros(0, 1, device=device)

        Smax = max(lengths)
        B = len(seqs)
        Hpad = torch.zeros(B, Smax, self.out_dim, device=device)
        M = torch.zeros(B, Smax, device=device)
        for i, H in enumerate(seqs):
            s = H.size(0)
            Hpad[i, :s] = H
            M[i, :s] = 1.0
        return Hpad, M

    def encode_chunks(
        self,
        batch_chunks: List[List[str]],
        max_total_chunks: Optional[int] = None,
        chunk_bs: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self._device()
        seqs: List[torch.Tensor] = []
        lengths: List[int] = []

        for chunks in batch_chunks:
            if chunks is None or len(chunks) == 0:
                H = torch.zeros(1, self.out_dim, device=device)
                seqs.append(H)
                lengths.append(1)
                continue

            if max_total_chunks is not None:
                chunks = chunks[:max_total_chunks]

            token_batches: List[Dict[str, torch.Tensor]] = []
            if not self.hf_available:
                token_batches = [{
                    "input_ids": torch.tensor([101, 102], device=device, dtype=torch.long),
                    "attention_mask": torch.tensor([1, 1], device=device, dtype=torch.long),
                }]
            else:
                for ch in chunks:
                    enc = self.tokenizer(
                        ch or "",
                        padding=False,
                        truncation=True,
                        max_length=self.max_len,
                        return_tensors=None,
                    )
                    ids = torch.tensor(enc["input_ids"], device=device, dtype=torch.long)
                    attn = torch.tensor(enc["attention_mask"], device=device, dtype=torch.long)
                    token_batches.append({"input_ids": ids, "attention_mask": attn})

            if len(token_batches) == 0:
                H = torch.zeros(1, self.out_dim, device=device)
            else:
                cls_list: List[torch.Tensor] = []
                for start in range(0, len(token_batches), chunk_bs):
                    end = min(start + chunk_bs, len(token_batches))
                    cls = self._encode_token_windows_to_cls(token_batches[start:end])
                    cls_list.append(cls)
                cls_all = torch.cat(cls_list, dim=0) if cls_list else torch.zeros(0, 768, device=device)
                H = self.drop(self.proj(cls_all)) if cls_all.numel() > 0 else torch.zeros(1, self.out_dim, device=device)

            seqs.append(H)
            lengths.append(H.size(0))

        if len(seqs) == 0:
            return torch.zeros(0, 1, self.out_dim, device=device), torch.zeros(0, 1, device=device)

        Smax = max(lengths)
        B = len(seqs)
        Hpad = torch.zeros(B, Smax, self.out_dim, device=device)
        M = torch.zeros(B, Smax, device=device)
        for i, H in enumerate(seqs):
            s = H.size(0)
            Hpad[i, :s] = H
            M[i, :s] = 1.0
        return Hpad, M

    def forward(self, notes_or_chunks: Union[List[str], List[List[str]]]) -> torch.Tensor:
        is_prechunked = (
            isinstance(notes_or_chunks, list)
            and len(notes_or_chunks) > 0
            and isinstance(notes_or_chunks[0], list)
        )
        if is_prechunked:
            H, M = self.encode_chunks(notes_or_chunks)
        else:
            H, M = self.encode_seq(notes_or_chunks)

        if self.attn is None or (M.sum(dim=1) == 0).any():
            return _masked_mean(H, M)

        scores = self.attn(H).squeeze(-1)                           # [B, S]
        scores = scores.masked_fill(M < 0.5, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, dim=1)                             # [B, S]
        return (w.unsqueeze(-1) * H).sum(dim=1)                      # [B, out_dim]


class ImageEncoder(nn.Module):
    """
    ResNet34 feature extractor with GAP + linear projection to dimension d.
    Accepts a single image tensor [3,H,W], a batch [B,3,H,W], a list of tensors,
    or a list-of-lists (sequence per sample). Default aggregation is 'last'.
    """
    def __init__(self, d: int, dropout: float = 0.0, img_agg: Literal["last", "mean", "attention"] = "last") -> None:
        super().__init__()
        import torchvision
        backbone = torchvision.models.resnet34(weights=None)
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = 512

        self.proj = nn.Linear(self.out_channels, d)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.img_agg = img_agg  # kept for API symmetry

    @torch.no_grad()
    def _encode_one_tensor(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device)
        h = self.backbone(x)
        h = self.gap(h)
        h = torch.flatten(h, 1)
        z = self.drop(self.proj(h))
        return z.squeeze(0)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                return self._encode_one_tensor(x)
            elif x.dim() == 4:
                x = x.to(device)
                h = self.backbone(x)
                h = self.gap(h).flatten(1)
                return self.drop(self.proj(h))
            else:
                raise ValueError("Tensor input must be [3,H,W] or [B,3,H,W].")

        if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], torch.Tensor)):
            if len(x) == 0:
                return torch.zeros(0, self.proj.out_features, device=device)
            xs = torch.stack(x, dim=0).to(device)
            h = self.backbone(xs)
            h = self.gap(h).flatten(1)
            return self.drop(self.proj(h))

        out: List[torch.Tensor] = []
        for imgs in x:
            if imgs is None or len(imgs) == 0:
                out.append(torch.zeros(self.proj.out_features, device=device))
                continue
            out.append(self._encode_one_tensor(imgs[-1]))
        return torch.stack(out, dim=0)

    def encode_seq(
        self,
        batch_images: Union[List[torch.Tensor], List[List[torch.Tensor]], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        Z = self.forward(batch_images)
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)
        B = Z.size(0)
        Hpad = Z.unsqueeze(1)
        M = torch.ones(B, 1, device=device)
        return Hpad, M

    def load_backbone_state(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
        try:
            self.backbone.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            remapped = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(remapped, strict=strict)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


@dataclass
class EncoderConfig:
    d: int = 256
    dropout: float = 0.1
    # structured
    structured_seq_len: int = 24          # kept for config parity; not required by BEHRT
    structured_n_feats: int = 128
    structured_layers: int = 2
    structured_heads: int = 8
    structured_pool: Literal["last", "mean"] = "last"
    # notes
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_max_len: int = 512
    note_agg: Literal["mean", "attention"] = "mean"
    max_notes_concat: int = 8
    # images
    img_agg: Literal["last", "mean", "attention"] = "last"


def build_encoders(
    cfg: EncoderConfig,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder]:
    dev = torch.device(DEVICE if device is None else device)

    behrt = BEHRTLabEncoder(
        n_feats=cfg.structured_n_feats,
        d=cfg.d,
        n_layers=cfg.structured_layers,
        n_heads=cfg.structured_heads,
        dropout=cfg.dropout,
        pool=cfg.structured_pool,
    ).to(dev)

    bbert = BioClinBERTEncoder(
        model_name=cfg.text_model_name,
        d=cfg.d,
        max_len=cfg.text_max_len,
        dropout=cfg.dropout,
        note_agg=cfg.note_agg,
        max_notes_concat=cfg.max_notes_concat,
        device=dev,
    ).to(dev)

    if getattr(bbert, "hf_available", False) and bbert.bert is not None:
        bbert.bert.to(dev)
        bbert.bert.eval()

    imgenc = ImageEncoder(
        d=cfg.d,
        dropout=cfg.dropout,
        img_agg=cfg.img_agg,
    ).to(dev)

    return behrt, bbert, imgenc


@torch.no_grad()
def encode_pooled_modalities(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    xL: torch.Tensor,                               
    notes_list: Union[List[str], List[List[str]]],  
    imgs: Union[List[torch.Tensor], List[List[torch.Tensor]], torch.Tensor],
    mL: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Produce pooled embeddings per modality (shape [B, d]) ready for route fusion:
      returns {"L": zL, "N": zN, "I": zI}
    """
    dev = next(behrt.parameters()).device
    zL = behrt(xL.to(dev), mask=mL.to(dev) if mL is not None else None)  # [B, d]
    zN = bbert(notes_list)                                               # [B, d]
    zI = imgenc(imgs)                                                    # [B, d]
    return {"L": zL, "N": zN, "I": zI}


__all__ = [
    # Encoders
    "BEHRTLabEncoder",
    "BioClinBERTEncoder",
    "ImageEncoder",
    # Config & builders
    "EncoderConfig",
    "build_encoders",
    # Helpers
    "encode_pooled_modalities",
]
