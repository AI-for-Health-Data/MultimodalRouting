from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Sequence, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import DEVICE, CFG

def _dbg(msg: str) -> None:
    if getattr(CFG, "verbose", False):
        print(msg)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def _ensure_2d_mask(mask: Optional[torch.Tensor], B: int, T: int, device) -> torch.Tensor:
    if mask is None:
        return torch.ones(B, T, device=device, dtype=torch.float32)
    if mask.dim() == 1:
        return mask.unsqueeze(0).expand(B, -1).contiguous().float()
    return mask.float()

class BEHRTLabEncoder(nn.Module):
    """
    Transformer encoder over structured sequences.
    Inputs: x [B, T, F] where F = number of variables (e.g., 17),
            mask [B, T] where 1=valid timestep.
    Pooling:
      - "mean": masked mean over time
      - "last": last valid timestep
      - "cls" : learnable CLS token (not included in returned sequence; used only for pooled output)
    """
    def __init__(
        self,
        n_feats: int,
        d: int,
        seq_len: int = 24,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        pool: Literal["last", "mean", "cls"] = "mean",
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.out_dim = d
        self.input_proj = nn.Linear(n_feats, d)
        # positional encoding for T steps (structured timesteps)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        # learnable CLS (used only if pool="cls")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
        )

        # init CLS a bit
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _pos(self, T: int, std: float = 0.02) -> torch.Tensor:
        """Expand positional embeddings if sequence longer than current cache."""
        if self.pos.size(1) < T:
            extra = torch.randn(
                1, T - self.pos.size(1), self.pos.size(-1),
                device=self.pos.device, dtype=self.pos.dtype
            ) * std
            self.pos = nn.Parameter(torch.cat([self.pos, extra], dim=1))
        return self.pos[:, :T, :]

    def _encode_with_optional_cls(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Run transformer. If pool='cls', prepend CLS and return:
          - seq_out_no_cls: [B,T,D] (timesteps only)
          - mask_out:       [B,T]   (unchanged mask)
          - cls_vec:        [B,D]   (pooled cls), else None
        If pool!='cls', cls_vec=None and output is just seq-out.
        """
        B, T, F = x.shape
        dev = x.device
        assert self.input_proj.in_features == F, \
            f"Expected F={self.input_proj.in_features}, got F={F}"

        H_in = self.input_proj(x) + self._pos(T)  # [B,T,D]

        if self.pool == "cls":
            cls_tok = self.cls_token.expand(B, 1, -1)  # [B,1,D]
            H_in = torch.cat([cls_tok, H_in], dim=1)   # [B,T+1,D]
            pad_mask = torch.cat([torch.zeros(B, 1, device=dev, dtype=torch.bool), (mask < 0.5)], dim=1)  # [B,T+1]
        else:
            pad_mask = (mask < 0.5)  # [B,T]

        H = self.enc(H_in, src_key_padding_mask=pad_mask)  # [B,T(+1),D]
        H = self.out(H)

        if self.pool == "cls":
            cls_vec = H[:, 0, :]                 # [B,D]
            seq_out = H[:, 1:, :]                # [B,T,D]
            return seq_out, mask, cls_vec
        else:
            return H, mask, None

    def encode_seq(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns per-timestep sequence representations (without CLS) and mask:
          - h: [B,T,D]
          - mask: [B,T] float
        """
        if x.dim() == 2:
            # Allow [B,T] only when n_feats == 1
            x = x.unsqueeze(-1)
        B, T, _ = x.shape
        dev = next(self.parameters()).device
        m = _ensure_2d_mask(mask, B, T, dev)

        h, m_out, _ = self._encode_with_optional_cls(x.to(dev), m.to(dev))
        return h, m_out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, T, _ = x.shape
        dev = next(self.parameters()).device
        m = _ensure_2d_mask(mask, B, T, dev)

        # If cls pooling, compute cls inside; otherwise use seq + masked pool
        seq_h, m_out, cls_vec = self._encode_with_optional_cls(x.to(dev), m.to(dev))
        if self.pool == "cls":
            return cls_vec  # [B,D]

        if self.pool == "last":
            if (m_out.sum(dim=1) != m_out.size(1)).any():
                idx = (m_out.sum(dim=1) - 1).clamp_min(0).long()
                z = seq_h[torch.arange(seq_h.size(0), device=seq_h.device), idx]
            else:
                z = seq_h[:, -1]
        else:  # mean
            z = _masked_mean(seq_h, m_out)
        return z


class BioClinBERTEncoder(nn.Module):
    """
    Aggregates CLS embeddings across chunked notes per patient.
    - encode_seq: accepts List[str] or List[List[str]] (per-patient)
    - forward   : masked mean over chunk-CLS (or attention if enabled)
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

        self.hf_available = False
        self.tokenizer = None
        self.bert = None
        hidden = 768

        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.bert = AutoModel.from_pretrained(model_name)
            hidden = int(getattr(self.bert.config, "hidden_size", 768))
            self.hf_available = True
        except Exception:
            self.hf_available = False

        self.hidden = hidden
        if not self.hf_available:
            project_to_d = False

        if project_to_d and d != hidden:
            self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))
            self.out_dim = d
        else:
            self.proj = nn.Identity()
            self.out_dim = hidden

        self.drop = nn.Dropout(dropout)

        self.attn = None
        if note_agg == "attention":
            self.attn = nn.Sequential(
                nn.LayerNorm(self.out_dim),
                nn.Linear(self.out_dim, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )

        if self.hf_available and self.bert is not None:
            self.bert.eval()
            self.bert.to(self._device())

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
            # minimal [CLS][SEP] fallback
            return [{
                "input_ids": torch.tensor([101, 102], device=device, dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], device=device, dtype=torch.long)
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
            cls_id = self.tokenizer.cls_token_id if (self.tokenizer and self.tokenizer.cls_token_id is not None) else 101
            sep_id = self.tokenizer.sep_token_id if (self.tokenizer and self.tokenizer.sep_token_id is not None) else 102
            chunks = [{
                "input_ids": torch.tensor([cls_id, sep_id], device=device, dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], device=device, dtype=torch.long)
            }]
        return chunks

    @torch.no_grad()
    def _encode_token_windows_to_cls(self, token_windows: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        device = self._device()

        if not self.hf_available or self.bert is None:
            return torch.randn(len(token_windows), self.hidden, device=device)

        if next(self.bert.parameters()).device != device:
            self.bert.to(device)

        maxL = max(tw["input_ids"].numel() for tw in token_windows)
        S = len(token_windows)
        pad_id = self._pad_token_id()

        input_ids = torch.full((S, maxL), pad_id, device=device, dtype=torch.long)
        attention_mask = torch.zeros((S, maxL), device=device, dtype=torch.long)
        for i, tw in enumerate(token_windows):
            L = tw["input_ids"].numel()
            input_ids[i, :L] = tw["input_ids"]
            attention_mask[i, :L] = tw["attention_mask"]

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            self.bert.eval()
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
        return cls

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
        """Return [B,S,D] chunk-CLS sequence and [B,S] mask."""
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
                    cls_hidden = self._encode_token_windows_to_cls(token_windows[start:end])
                    cls_proj = self.drop(self.proj(cls_hidden))
                    cls_list.append(cls_proj)

                cls_all = torch.cat(cls_list, dim=0) if cls_list else torch.zeros(0, self.out_dim, device=device)
                H = cls_all if cls_all.numel() > 0 else torch.zeros(1, self.out_dim, device=device)

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
        """Same as encode_seq, but when notes are already pre-chunked."""
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
                    "attention_mask": torch.tensor([1, 1], device=device, dtype=torch.long)
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
                    cls_hidden = self._encode_token_windows_to_cls(token_batches[start:end])
                    cls_proj = self.drop(self.proj(cls_hidden))
                    cls_list.append(cls_proj)

                cls_all = torch.cat(cls_list, dim=0) if cls_list else torch.zeros(0, self.out_dim, device=device)
                H = cls_all if cls_all.numel() > 0 else torch.zeros(1, self.out_dim, device=device)

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
        """Return pooled note embedding [B,D] (mean or attention over chunk CLS)."""
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

        scores = self.attn(H).squeeze(-1)
        scores = scores.masked_fill(M < 0.5, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, dim=1)
        return (w.unsqueeze(-1) * H).sum(dim=1)

class ImageEncoder(nn.Module):
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
            elif x.dim() == 4:  # [B,3,H,W]
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


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.1, hidden: Optional[Sequence[int]] = None):
        super().__init__()
        hidden = list(hidden) if hidden is not None else [4 * out_dim, 2 * out_dim]
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [
                nn.LayerNorm(dims[i]),
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
                nn.Dropout(p_drop),
            ]
        layers += [nn.LayerNorm(dims[-2]), nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairwiseConcatFusion(nn.Module):
    """(za, zb) -> concat/rich -> MLP -> d, with residual to average(za, zb)."""
    def __init__(self, d: int, p_drop: float = 0.1, feature_mode: str = "concat"):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        self.feature_mode = feature_mode
        in_dim = 2 * d if feature_mode == "concat" else 4 * d
        self.mlp = _MLP(in_dim, d, p_drop=p_drop)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, A: torch.Tensor, mA: torch.Tensor, B: torch.Tensor, mB: torch.Tensor) -> torch.Tensor:
        za = _masked_mean(A, mA)
        zb = _masked_mean(B, mB)
        if self.feature_mode == "concat":
            x = torch.cat([za, zb], dim=-1)
        else:
            had = za * zb
            diff = (za - zb).abs()
            x = torch.cat([za, zb, had, diff], dim=-1)
        h = self.mlp(x)
        base = 0.5 * (za + zb)
        return h + self.res_scale * base


class TrimodalConcatFusion(nn.Module):
    """(zL, zN, zI) -> concat/rich -> MLP -> d, with residual to their average."""
    def __init__(self, d: int, p_drop: float = 0.1, feature_mode: str = "concat"):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        in_dim = 3 * d if feature_mode == "concat" else 7 * d
        self.mlp = _MLP(in_dim, d, p_drop=p_drop)
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.feature_mode = feature_mode

    def forward(
        self,
        L: torch.Tensor, mL: torch.Tensor,
        N: torch.Tensor, mN: torch.Tensor,
        I: torch.Tensor, mI: torch.Tensor
    ) -> torch.Tensor:
        zL = _masked_mean(L, mL)
        zN = _masked_mean(N, mN)
        zI = _masked_mean(I, mI)
        if self.feature_mode == "concat":
            x = torch.cat([zL, zN, zI], dim=-1)
        else:
            zLN = zL * zN
            zLI = zL * zI
            zNI = zN * zI
            zLNI = zL * zN * zI
            x = torch.cat([zL, zN, zI, zLN, zLI, zNI, zLNI], dim=-1)
        h = self.mlp(x)
        base = (zL + zN + zI) / 3.0
        return h + self.res_scale * base


class RouteActivation(nn.Module):
    """Sigmoid activation per route (used when producing capsule-style acts)."""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


@dataclass
class MulTConfig:
    d: int = 256
    dropout: float = 0.1
    feature_mode: str = "concat"
    unimodal_pool: Literal["mean", "last"] = "mean"


class MultimodalFeatureExtractor(nn.Module):
    """
    Builds pairwise/trimodal interaction embeddings and (optionally) capsule-style
    per-route activations from pooled unimodal sequences.
    """
    def __init__(self, cfg: MulTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.pair_LN = PairwiseConcatFusion(d, p_drop=cfg.dropout, feature_mode=cfg.feature_mode)
        self.pair_LI = PairwiseConcatFusion(d, p_drop=cfg.dropout, feature_mode=cfg.feature_mode)
        self.pair_NI = PairwiseConcatFusion(d, p_drop=cfg.dropout, feature_mode=cfg.feature_mode)
        self.tri_LNI = TrimodalConcatFusion(d, p_drop=cfg.dropout, feature_mode=cfg.feature_mode)

        # Capsule-style route activations (sigmoid)
        self.act_L   = RouteActivation(d)
        self.act_N   = RouteActivation(d)
        self.act_I   = RouteActivation(d)
        self.act_LN  = RouteActivation(d)
        self.act_LI  = RouteActivation(d)
        self.act_NI  = RouteActivation(d)
        self.act_LNI = RouteActivation(d)

        self.unim_ln = nn.LayerNorm(d)

    def _pool_uni(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        if self.cfg.unimodal_pool == "last":
            last_idx = (M.sum(dim=1) - 1).clamp_min(0).long()
            return X[torch.arange(X.size(0), device=X.device), last_idx]
        return _masked_mean(X, M)

    def forward(
        self,
        L_seq: torch.Tensor, mL: torch.Tensor,
        N_seq: torch.Tensor, mN: torch.Tensor,
        I_seq: torch.Tensor, mI: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Unimodal pooled embeddings
        zL = self.unim_ln(self._pool_uni(L_seq, mL))
        zN = self.unim_ln(self._pool_uni(N_seq, mN))
        zI = self.unim_ln(self._pool_uni(I_seq, mI))

        # Pairwise concat → MLP
        zLN = self.pair_LN(L_seq, mL, N_seq, mN)
        zLI = self.pair_LI(L_seq, mL, I_seq, mI)
        zNI = self.pair_NI(N_seq, mN, I_seq, mI)

        # Trimodal concat → MLP
        zLNI = self.tri_LNI(L_seq, mL, N_seq, mN, I_seq, mI)

        route_embs: Dict[str, torch.Tensor] = {
            "L": zL, "N": zN, "I": zI, "LN": zLN, "LI": zLI, "NI": zNI, "LNI": zLNI
        }
        # Capsule-style per-route activations (sigmoid in [0,1])
        route_act: Dict[str, torch.Tensor] = {
            "L":  self.act_L(zL),
            "N":  self.act_N(zN),
            "I":  self.act_I(zI),
            "LN": self.act_LN(zLN),
            "LI": self.act_LI(zLI),
            "NI": self.act_NI(zNI),
            "LNI": self.act_LNI(zLNI),
        }
        return route_embs, route_act


@dataclass
class EncoderConfig:
    d: int = 256
    dropout: float = 0.1
    # structured
    structured_seq_len: int = 24            # 48h @ 2h bins
    structured_n_feats: int = 128           # set to 17 in CFG for your setup
    structured_layers: int = 2
    structured_heads: int = 8
    structured_pool: Literal["last", "mean", "cls"] = "mean"
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
        seq_len=cfg.structured_seq_len,
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


def build_multimodal_feature_extractor(
    d: int,
    dropout: float = 0.1,
    feature_mode: str = "concat",
    unimodal_pool: Literal["mean", "last"] = "mean",
) -> MultimodalFeatureExtractor:
    cfg = MulTConfig(
        d=d,
        dropout=dropout,
        feature_mode=feature_mode,
        unimodal_pool=unimodal_pool,
    )
    dev = torch.device(DEVICE)
    return MultimodalFeatureExtractor(cfg).to(dev)


@torch.no_grad()
def encode_all_routes_from_batch(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    extractor: MultimodalFeatureExtractor,
    xL: torch.Tensor,
    notes_list: Union[List[str], List[List[str]]],
    imgs: Union[List[torch.Tensor], List[List[torch.Tensor]]],
    mL: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Convenience for sequence-level encoding + feature extraction that returns:
      - route_embs: dict of 7 interaction embeddings (each [B,d])
      - route_act:  dict of 7 sigmoid activations (each [B,1])
    """
    dev = next(extractor.parameters()).device

    L_seq, mL_seq = behrt.encode_seq(xL.to(dev), mask=mL.to(dev) if mL is not None else None)

    if isinstance(notes_list, list) and len(notes_list) > 0 and isinstance(notes_list[0], list):
        N_seq, mN_seq = bbert.encode_chunks(notes_list)
    else:
        N_seq, mN_seq = bbert.encode_seq(notes_list)

    I_seq, mI_seq = imgenc.encode_seq(imgs)

    route_embs, route_act = extractor(L_seq, mL_seq, N_seq, mN_seq, I_seq, mI_seq)

    # One-time sanity print
    if not hasattr(extractor, "_printed_once"):
        extractor._printed_once = True
        keys = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_embs.items())
        acts = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in route_act.items())
        _dbg(f"[encoders] routes -> {keys}")
        _dbg(f"[encoders] route_acts (sigmoid) -> {acts}")

    return route_embs, route_act


@torch.no_grad()
def encode_unimodal_pooled(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    xL: torch.Tensor,
    notes_list: Union[List[str], List[List[str]]],
    imgs: Union[List[torch.Tensor], List[List[torch.Tensor]]],
    mL: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Helper that returns pooled unimodal embeddings ready for downstream fusion:
      {"L": [B,d], "N": [B,d], "I": [B,d]}
    """
    dev = next(behrt.parameters()).device
    zL = behrt(xL.to(dev), mask=mL.to(dev) if mL is not None else None)
    zN = bbert(notes_list)
    zI = imgenc(imgs.to(dev) if isinstance(imgs, torch.Tensor) else imgs)
    return {"L": zL, "N": zN, "I": zI}


__all__ = [
    # Encoders
    "BEHRTLabEncoder",
    "BioClinBERTEncoder",
    "ImageEncoder",
    # Multimodal features
    "MulTConfig",
    "MultimodalFeatureExtractor",
    "build_multimodal_feature_extractor",
    "encode_all_routes_from_batch",
    "encode_unimodal_pooled",
    # Config & builders
    "EncoderConfig",
    "build_encoders",
]
