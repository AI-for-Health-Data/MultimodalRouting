from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Sequence, Union, Dict

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from env_config import CFG, DEVICE  

def _dbg(msg: str) -> None:
    if getattr(CFG, "verbose", False):
        print(msg)


def _peek_tensor(name: str, x: torch.Tensor, k: int = 3) -> None:
    """One-time compact peek at a tensor: shape + a few values."""
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


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over time dimension (dim=1) for sequence tensors [B,T,D]."""
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def _ensure_2d_mask(mask: Optional[torch.Tensor], B: int, T: int, device) -> torch.Tensor:
    """Ensure mask shape [B,T] (float, 1=valid)."""
    if mask is None:
        return torch.ones(B, T, device=device, dtype=torch.float32)
    if mask.dim() == 1:
        return mask.unsqueeze(0).expand(B, -1).contiguous().float()
    return mask.float()


class BEHRTLabEncoder(nn.Module):
    """
    Transformer encoder over structured sequences.

    Inputs:
      x    : [B, T, F] where F = number of variables (e.g., 17). If [B,T], it's auto-expanded to [B,T,1].
      mask : [B, T] where 1=valid timestep.

    Pooling:
      - "mean": masked mean over time
      - "last": last valid timestep (by mask)
      - "cls" : learnable CLS token; pooled CLS

    Output: [B, D] embedding
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
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)  # positional embeddings up to seq_len
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))         # used only if pool="cls"

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

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self._printed_once = False
        self._warned_dead = False  # one-time "near-zero" warning

    def _pos(self, T: int, std: float = 0.02) -> torch.Tensor:
        """Expand positional embeddings if sequence longer than current cache."""
        if self.pos.size(1) < T:
            extra = torch.randn(
                1,
                T - self.pos.size(1),
                self.pos.size(-1),
                device=self.pos.device,
                dtype=self.pos.dtype,
            ) * std
            self.pos = nn.Parameter(torch.cat([self.pos, extra], dim=1))
        return self.pos[:, :T, :]

    def _encode_with_optional_cls(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Run transformer. If pool='cls', prepend CLS and return:
          - seq_out_no_cls: [B,T,D]
          - mask_out:       [B,T]
          - cls_vec:        [B,D] (else None)
        """
        B, T, F = x.shape
        dev = x.device
        assert self.input_proj.in_features == F, f"Expected F={self.input_proj.in_features}, got F={F}"

        H_in = self.input_proj(x) + self._pos(T)  # [B,T,D]

        if self.pool == "cls":
            cls_tok = self.cls_token.expand(B, 1, -1)  # [B,1,D]
            H_in = torch.cat([cls_tok, H_in], dim=1)   # [B,T+1,D]
            pad_mask = torch.cat(
                [torch.zeros(B, 1, device=dev, dtype=torch.bool), (mask < 0.5)], dim=1
            )
        else:
            pad_mask = mask < 0.5

        H = self.enc(H_in, src_key_padding_mask=pad_mask)  # [B,T(+1),D]
        H = self.out(H)

        if self.pool == "cls":
            cls_vec = H[:, 0, :]   # [B,D]
            seq_out = H[:, 1:, :]  # [B,T,D]
            return seq_out, mask, cls_vec
        else:
            return H, mask, None

    def encode_seq(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns per-timestep sequence representations (without CLS) and mask:
          - h: [B,T,D]
          - mask: [B,T] float
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B,T,1]
        B, T, _ = x.shape
        dev = next(self.parameters()).device
        m = _ensure_2d_mask(mask, B, T, dev)

        h, m_out, _ = self._encode_with_optional_cls(x.to(dev), m.to(dev))
        if not self._printed_once:
            self._printed_once = True
            _dbg(f"[BEHRTLabEncoder] encode_seq -> h:{tuple(h.shape)} mask:{tuple(m_out.shape)}")
            _peek_tensor("behrt.h", h)
        return h, m_out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return pooled embedding [B,D] using configured pooling."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, T, _ = x.shape
        dev = next(self.parameters()).device
        m = _ensure_2d_mask(mask, B, T, dev)

        seq_h, m_out, cls_vec = self._encode_with_optional_cls(x.to(dev), m.to(dev))
        if self.pool == "cls":
            z = cls_vec  # [B,D]
        elif self.pool == "last":
            if (m_out.sum(dim=1) != m_out.size(1)).any():
                idx = (m_out.sum(dim=1) - 1).clamp_min(0).long()
                z = seq_h[torch.arange(seq_h.size(0), device=seq_h.device), idx]
            else:
                z = seq_h[:, -1]
        else:  # mean
            z = _masked_mean(seq_h, m_out)

        # one-time guard to detect dead branch
        if not self._warned_dead:
            with torch.no_grad():
                if z.abs().mean().item() < 1e-6:
                    print("[warn:BEHRTLabEncoder] near-zero pooled embedding; check mask/inputs.")
            self._warned_dead = True

        return z  # [B,D]

class BioClinBERTEncoder(nn.Module):
    """
    Bio-ClinicalBERT encoder (pre-tokenized only).

    Expected per-patient input (pick one format and keep it consistent):
      1) Dict with stacked chunks:
         {"input_ids": LongTensor[S, L], "attention_mask": LongTensor[S, L]}
      2) List of (ids, attn) chunk pairs:
         [(LongTensor[L] or [1,L], LongTensor[L] or [1,L]), ...]
    Batch = list of those per-patient objects.

    Returns:
      forward(...)   -> [B, D]  (mean pooled over chunks)
      encode_seq(...) -> ([B, S_max, D], [B, S_max])  (per-chunk CLS with mask)
    """
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        d: Optional[int] = None,   
        dropout: float = 0.0,
        force_hf: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hf_available = False
        self.bert = None
        hidden = 768

        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(model_name)
            hidden = int(getattr(self.bert.config, "hidden_size", 768))
            self.hf_available = True
        except Exception as e:
            if force_hf:
                raise RuntimeError(
                    f"Failed to load '{model_name}'. Install transformers and cache the model. Error: {e}"
                )

        if d is not None and d != hidden:
            self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))
            self.out_dim = d
        else:
            self.proj = nn.Identity()
            self.out_dim = hidden

        self.drop = nn.Dropout(dropout)
        if self.hf_available and self.bert is not None:
            self.bert.eval()

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _is_pretok_item(item) -> bool:
        if isinstance(item, dict):
            return ("input_ids" in item) and ("attention_mask" in item)
        if isinstance(item, list):
            if len(item) == 0:
                return True  # allow patients with 0 note chunks
            a = item[0]
            return (len(a) == 2) and torch.is_tensor(a[0]) and torch.is_tensor(a[1])
        return False


    def _normalize_batch(self, notes_or_chunks):
        if len(notes_or_chunks) == 0:
            return []
        first = notes_or_chunks[0]
        if self._is_pretok_item(first):
            return notes_or_chunks  # already pre-tokenized per patient
        raise ValueError(
            "BioClinBERTEncoder now requires pre-tokenized inputs. "
            "Pass per-patient dict {'input_ids':[S,L], 'attention_mask':[S,L]} "
            "or list of (ids[L], attn[L]) chunks."
        )

    def _encode_chunks_to_cls(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """
        ids/attn: [S, L] (adds batch dim if [L])
        returns:  [S, D]
        """
        if ids.dim() == 1:  ids = ids.unsqueeze(0)
        if attn.dim() == 1: attn = attn.unsqueeze(0)

        if not self.hf_available or self.bert is None:
            return torch.zeros(ids.size(0), self.out_dim, device=self._device())

        # keep model on same device
        dev = self._device()
        if next(self.bert.parameters()).device != dev:
            self.bert.to(dev)

        self.bert.eval()
        out = self.bert(input_ids=ids.to(dev), attention_mask=attn.to(dev))
        cls = out.last_hidden_state[:, 0]        # [S, hidden]
        cls = self.drop(self.proj(cls))          # [S, out_dim]
        return cls

    def encode_seq(self, notes_or_chunks) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-patient sequence of CLS embeddings.
        Returns:
            Hpad: [B, S_max, D]
            M   : [B, S_max] (float mask)
        """
        dev = self._device()
        batch = self._normalize_batch(notes_or_chunks)

        seqs: List[torch.Tensor] = []
        lengths: List[int] = []

        for patient in batch:
            collected: List[torch.Tensor] = []

            if isinstance(patient, dict):
                ids = patient["input_ids"].to(dev)
                attn = patient["attention_mask"].to(dev)
                cls = self._encode_chunks_to_cls(ids, attn)         # [S, D]
                collected.append(cls)
            else:
                for (ids, attn) in patient:
                    cls = self._encode_chunks_to_cls(ids.to(dev), attn.to(dev))  # [1, D]
                    collected.append(cls)

            H = torch.zeros(1, self.out_dim, device=dev) if len(collected) == 0 else torch.cat(collected, dim=0)
            seqs.append(H)
            lengths.append(H.size(0))

        if len(seqs) == 0:
            return torch.zeros(0, 1, self.out_dim, device=dev), torch.zeros(0, 1, device=dev)

        Smax = max(lengths)
        B = len(seqs)
        Hpad = torch.zeros(B, Smax, self.out_dim, device=dev)
        M    = torch.zeros(B, Smax, device=dev)
        for i, H in enumerate(seqs):
            s = H.size(0)
            Hpad[i, :s] = H
            M[i, :s] = 1.0
        return Hpad, M

    def forward(self, notes_or_chunks) -> torch.Tensor:
        requires_grad = bool(getattr(CFG, "finetune_text", False))
        with torch.enable_grad() if requires_grad else torch.no_grad():
            H, M = self.encode_seq(notes_or_chunks)  # if encode_seq should also allow grads, wrap its internals similarly
            denom = M.sum(dim=1, keepdim=True).clamp_min(1.0)
            z = (H * M.unsqueeze(-1)).sum(dim=1) / denom
            return z

class MedFuseImageEncoder(nn.Module):
    """
    MedFuse-style image branch (CXRModels-equivalent):
      - build torchvision backbone with pretrained weights
      - remove final classifier ('fc' for ResNet, 'classifier' for DenseNet/EfficientNet)
      - expose pooled features as visual_feats
      - 1x Linear head -> sigmoid -> BCE loss
      - forward returns (preds, lossvalue_bce, visual_feats)
    """
    def __init__(
        self,
        vision_backbone: str = "resnet34",
        vision_num_classes: int = 14,
        pretrained: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        self.vision_backbone = getattr(torchvision.models, vision_backbone)(pretrained=pretrained)

        d_visual = None
        for classifier in ("classifier", "fc"):
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue

            # Try direct in_features first
            d_visual = getattr(cls_layer, "in_features", None)

            # If classifier is a Sequential (e.g., EfficientNet), find last Linear
            if d_visual is None:
                last_linear = None
                for m in reversed(list(cls_layer.modules())):
                    if isinstance(m, nn.Linear):
                        last_linear = m
                        break
                if last_linear is not None:
                    d_visual = last_linear.in_features

            if d_visual is None:
                raise ValueError(f"Cannot infer in_features from `{classifier}` of {type(self.vision_backbone).__name__}")

            # Replace with identity to expose pooled features
            setattr(self.vision_backbone, classifier, nn.Identity())
            break

        if d_visual is None:
            raise ValueError(f"Unsupported backbone `{vision_backbone}` (no `fc`/`classifier` head found).")

        self.bce_loss = nn.BCELoss(reduction="mean")

        self.classifier = nn.Sequential(nn.Linear(d_visual, vision_num_classes))

        self.feats_dim = d_visual
        self.vision_num_classes = vision_num_classes
        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        n_crops: int = 0,
        bs: int | None = None,
    ):
        device = next(self.parameters()).device
        x = x.to(device)

        # pooled features from backbone
        visual_feats = self.vision_backbone(x)          # [B, D]
        logits = self.classifier(visual_feats)          # [B, C]
        preds = torch.sigmoid(logits)                   # [B, C]

        # Multi-crop: average predictions across crops (exact MedFuse behavior)
        if n_crops and n_crops > 0:
            if bs is None:
                # default if not provided
                if preds.size(0) % n_crops != 0:
                    raise ValueError("When n_crops > 0, pass bs or ensure B % n_crops == 0.")
                bs = preds.size(0) // n_crops
            preds = preds.view(bs, n_crops, -1).mean(dim=1)  # [bs, C]

        # BCE loss (on post-sigmoid probabilities, matching MedFuse)
        if labels is not None:
            labels = labels.to(device)
            lossvalue_bce = self.bce_loss(preds, labels)
        else:
            lossvalue_bce = torch.zeros(1, device=device)

        return preds, lossvalue_bce, visual_feats

# Fusion-facing image encoder wrapper
class ImageEncoder(nn.Module):
    """
    Wrapper over MedFuseImageEncoder that:
      - keeps exact MedFuse behavior available via `medfuse_forward(...)`
      - adds a projection `visual_feats -> d` to integrate with fusion (returns embeddings)
      - provides encode_seq(...) returning ([B,1,d], [B,1]) for route builder

    Use:
      preds, loss, feats = imgenc.medfuse_forward(x, labels=..., n_crops=..., bs=...)
      z = imgenc(x)  # -> [B, d] embedding for fusion
    """

    def __init__(
        self,
        d: int,
        dropout: float = 0.0,
        img_agg: Literal["last", "mean", "attention"] = "last",
        vision_backbone: str = "resnet34",
        vision_num_classes: int = 14,
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        dev = torch.device(DEVICE if device is None else device)
        self.medfuse = MedFuseImageEncoder(
            vision_backbone=vision_backbone,
            vision_num_classes=vision_num_classes,
            pretrained=pretrained,
            device=str(dev),
        )
        self.img_agg = img_agg  # kept for API symmetry
        self.proj = nn.Linear(self.medfuse.feats_dim, d)  # type: ignore[arg-type]
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.to(dev)
    
    def _encode_batch_feats(self, x: torch.Tensor) -> torch.Tensor:
        _, _, feats = self.medfuse(x)        # [B, D_vis]
        z = self.drop(self.proj(feats))      # [B, d]
        _peek_tensor("imgenc.z", z)
        return z

    def medfuse_forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        n_crops: int = 0,
        bs: Optional[int] = None,
    ):
        """Exact MedFuse forward: returns (preds, loss, visual_feats)."""
        return self.medfuse(x, labels=labels, n_crops=n_crops, bs=bs)


    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Fusion-friendly forward:
          - If x is [B,3,H,W], returns [B, d]
          - If x is list[Tensors], stacks to [B,3,H,W] then returns [B, d]
          - If x is list[list[Tensors]], uses last image per sample -> [B, d]
        """
        device = next(self.parameters()).device

        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                x = x.unsqueeze(0)  # [1,3,H,W]
            if x.dim() != 4:
                raise ValueError("Tensor input must be [3,H,W] or [B,3,H,W].")
            return self._encode_batch_feats(x.to(device))

        if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], torch.Tensor)):
            if len(x) == 0:
                return torch.zeros(0, self.proj.out_features, device=device)
            xs = torch.stack(x, dim=0).to(device)  # [B,3,H,W]
            return self._encode_batch_feats(xs)

        # list of lists -> take last image per sequence
        out: List[torch.Tensor] = []
        for imgs in x:
            if imgs is None or len(imgs) == 0:
                out.append(torch.zeros(self.proj.out_features, device=device))
            else:
                img = imgs[-1]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                z = self._encode_batch_feats(img.to(device))  # [1, d]
                out.append(z.squeeze(0))
        Z = torch.stack(out, dim=0)  # [B, d]
        _peek_tensor("imgenc.seq_last_z", Z)
        return Z

    def encode_seq(
        self,
        batch_images: Union[List[torch.Tensor], List[List[torch.Tensor]], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return ([B,1,d], [B,1]) so it plugs into route extractor.
        """
        device = next(self.parameters()).device
        Z = self.forward(batch_images)  # [B, d]
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)
        B = Z.size(0)
        Hpad = Z.unsqueeze(1)  # [B,1,d]
        M = torch.ones(B, 1, device=device)
        _peek_tensor("imgenc.Hpad", Hpad)
        return Hpad, M

    def load_backbone_state(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
        # support loading only the backbone state_dict if needed
        try:
            self.medfuse.vision_backbone.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            remapped = {k.replace("vision_backbone.", ""): v for k, v in state_dict.items()}
            self.medfuse.vision_backbone.load_state_dict(remapped, strict=strict)

    def freeze_backbone(self) -> None:
        for p in self.medfuse.vision_backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.medfuse.vision_backbone.parameters():
            p.requires_grad = True

class SimpleHead(nn.Module):
    """LayerNorm + Linear (+ optional Dropout)."""
    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(p_drop) if p_drop and p_drop > 0 else nn.Identity(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PairwiseConcatFusion(nn.Module):
    """
    Pool -> concat -> Linear to d.
    Inputs are sequences [B,T,D] with masks [B,T].
    """
    def __init__(self, d: int, p_drop: float = 0.0):
        super().__init__()
        self.head = SimpleHead(in_dim=2 * d, out_dim=d, p_drop=p_drop)

    def forward(self, A: torch.Tensor, mA: torch.Tensor, B: torch.Tensor, mB: torch.Tensor) -> torch.Tensor:
        za = _masked_mean(A, mA)   # [B, D]
        zb = _masked_mean(B, mB)   # [B, D]
        x = torch.cat([za, zb], dim=-1)  # [B, 2D]
        z = self.head(x)                 # [B, D]
        _peek_tensor("fusion.pair_z", z)
        return z

class TrimodalConcatFusion(nn.Module):
    """
    Pool -> concat -> Linear to d.
    """
    def __init__(self, d: int, p_drop: float = 0.0):
        super().__init__()
        self.head = SimpleHead(in_dim=3 * d, out_dim=d, p_drop=p_drop)

    def forward(
        self,
        L: torch.Tensor, mL: torch.Tensor,
        N: torch.Tensor, mN: torch.Tensor,
        I: torch.Tensor, mI: torch.Tensor,
    ) -> torch.Tensor:
        zL = _masked_mean(L, mL)  # [B, D]
        zN = _masked_mean(N, mN)  # [B, D]
        zI = _masked_mean(I, mI)  # [B, D]
        x = torch.cat([zL, zN, zI], dim=-1)  # [B, 3D]
        z = self.head(x)                     # [B, D]
        _peek_tensor("fusion.tri_z", z)
        return z

class RouteActivation(nn.Module):
    """Sigmoid activation per route (useful for diagnostics; capsules don’t need it)."""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

@dataclass
class MulTConfig:
    d: int = 256
    dropout: float = 0.0             
    unimodal_pool: Literal["mean", "last"] = "mean"

class MultimodalFeatureExtractor(nn.Module):
    def __init__(self, cfg: MulTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.pair_LN = PairwiseConcatFusion(d, p_drop=cfg.dropout)
        self.pair_LI = PairwiseConcatFusion(d, p_drop=cfg.dropout)
        self.pair_NI = PairwiseConcatFusion(d, p_drop=cfg.dropout)
        self.tri_LNI = TrimodalConcatFusion(d, p_drop=cfg.dropout)

        self.act_L  = RouteActivation(d)
        self.act_N  = RouteActivation(d)
        self.act_I  = RouteActivation(d)
        self.act_LN = RouteActivation(d)
        self.act_LI = RouteActivation(d)
        self.act_NI = RouteActivation(d)
        self.act_LNI= RouteActivation(d)

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
        # Unimodal pooled (optionally LN)
        zL = self.unim_ln(self._pool_uni(L_seq, mL))
        zN = self.unim_ln(self._pool_uni(N_seq, mN))
        zI = self.unim_ln(self._pool_uni(I_seq, mI))

        # Pairwise + Trimodal (simple concat + linear)
        zLN = self.pair_LN(L_seq, mL, N_seq, mN)
        zLI = self.pair_LI(L_seq, mL, I_seq, mI)
        zNI = self.pair_NI(N_seq, mN, I_seq, mI)
        zLNI = self.tri_LNI(L_seq, mL, N_seq, mN, I_seq, mI)

        route_embs = {"L": zL, "N": zN, "I": zI, "LN": zLN, "LI": zLI, "NI": zNI, "LNI": zLNI}
        route_act  = {
            "L": self.act_L(zL), "N": self.act_N(zN), "I": self.act_I(zI),
            "LN": self.act_LN(zLN), "LI": self.act_LI(zLI), "NI": self.act_NI(zNI), "LNI": self.act_LNI(zLNI),
        }
        return route_embs, route_act

@dataclass
class EncoderConfig:
    d: int = 256
    dropout: float = 0.1
    # structured
    structured_seq_len: int = 24
    structured_n_feats: int = 17
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
    vision_backbone: str = "resnet34"
    vision_num_classes: int = 14
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
        dropout=cfg.dropout,
        pool=cfg.structured_pool,
    ).to(dev)

    bbert = BioClinBERTEncoder(
        model_name=cfg.text_model_name,
        d=cfg.d,
        dropout=cfg.dropout,
        force_hf=True,
    ).to(dev)

    if getattr(bbert, "hf_available", False) and bbert.bert is not None:
        bbert.bert.to(dev)
        if not getattr(CFG, "finetune_text", False):
            bbert.bert.eval()

    imgenc = ImageEncoder(
        d=cfg.d,
        dropout=cfg.dropout,
        img_agg=cfg.img_agg,
        vision_backbone=cfg.vision_backbone,
        vision_num_classes=cfg.vision_num_classes,
        pretrained=cfg.vision_pretrained,
        device=dev,
    ).to(dev)

    return behrt, bbert, imgenc


def build_multimodal_feature_extractor(
    d: int,
    dropout: float = 0.1,
    unimodal_pool: Literal["mean", "last"] = "mean",
) -> MultimodalFeatureExtractor:
    cfg = MulTConfig(
        d=d,
        dropout=dropout,
        unimodal_pool=unimodal_pool,
    )
    dev = torch.device(DEVICE)
    return MultimodalFeatureExtractor(cfg).to(dev)

NoteItem = Union[
    Dict[str, torch.Tensor],                
    List[Tuple[torch.Tensor, torch.Tensor]] 
]
BatchNotes = List[NoteItem]

def encode_all_routes_from_batch(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    extractor: MultimodalFeatureExtractor,
    xL: torch.Tensor,
    notes_list: BatchNotes,
    imgs: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
    mL: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Convenience for sequence-level encoding + feature extraction that returns:
      - route_embs: dict of 7 interaction embeddings (each [B,d])
      - route_act:  dict of 7 sigmoid activations (each [B,1])
    """
    dev = next(extractor.parameters()).device

    L_seq, mL_seq = behrt.encode_seq(xL.to(dev), mask=mL.to(dev) if mL is not None else None)

    # BioClinBERTEncoder.encode_seq expects pre-tokenized notes (BatchNotes).
    N_seq, mN_seq = bbert.encode_seq(notes_list)  
    
    I_seq, mI_seq = imgenc.encode_seq(imgs)

    route_embs, route_act = extractor(L_seq, mL_seq, N_seq, mN_seq, I_seq, mI_seq)

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
    notes_list: BatchNotes,
    imgs: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
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
    "MedFuseImageEncoder",
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
