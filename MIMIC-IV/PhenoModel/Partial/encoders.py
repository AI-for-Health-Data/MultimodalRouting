from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Literal,
    Tuple,
    Sequence,
    Union,
    Dict,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import CFG, DEVICE

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


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked mean over time dimension:
        x:    [B, T, D]
        mask: [B, T] (1 = keep, 0 = ignore)
    Returns:
        [B, D]
    """
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def _ensure_2d_mask(
    mask: Optional[torch.Tensor],
    B: int,
    T: int,
    device,
) -> torch.Tensor:
    """
    Ensure mask is [B, T] float tensor.
    If None: all ones.
    If [T]: broadcast to [B,T].
    """
    if mask is None:
        return torch.ones(B, T, device=device, dtype=torch.float32)
    if mask.dim() == 1:
        return mask.unsqueeze(0).expand(B, -1).contiguous().float()
    return mask.float()

class BEHRTLabEncoder(nn.Module):
    def __init__(
        self,
        n_feats: int,
        d: int,
        seq_len: int = 24,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,  # ignored; kept for API compatibility
        pool: Literal["last", "mean", "cls"] = "cls",
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()

        self.pool = pool
        self.out_dim = d

        self.input_proj = nn.Linear(n_feats, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

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
            nn.ReLU() if activation == "relu" else nn.GELU(),
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self._printed_once = False
        self._warned_dead = False

    def _pos(self, T: int, std: float = 0.02) -> torch.Tensor:
        if self.pos.size(1) < T:
            extra = (
                torch.randn(
                    1,
                    T - self.pos.size(1),
                    self.pos.size(-1),
                    device=self.pos.device,
                    dtype=self.pos.dtype,
                )
                * std
            )
            self.pos = nn.Parameter(torch.cat([self.pos, extra], dim=1))
        return self.pos[:, :T, :]

    def _encode_with_optional_cls(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T, F = x.shape
        dev = x.device

        # Input projection
        assert (
            self.input_proj.in_features == F
        ), f"Expected F={self.input_proj.in_features}, got F={F}"
        H_in = self.input_proj(x) + self._pos(T)  # [B,T,D]

        if self.pool == "cls":
            cls_tok = self.cls_token.expand(B, 1, -1)  # [B,1,D]
            H_in = torch.cat([cls_tok, H_in], dim=1)   # [B,T+1,D]
            pad_mask = torch.cat(
                [
                    torch.zeros(B, 1, device=dev, dtype=torch.bool),
                    (mask < 0.5),
                ],
                dim=1,
            )  # [B,T+1]
        else:
            pad_mask = mask < 0.5  # [B,T]

        H = self.enc(H_in, src_key_padding_mask=pad_mask)  # [B,T(+1),D]
        H = self.out(H)

        if self.pool == "cls":
            cls_vec = H[:, 0, :]   # [B,D]
            seq_out = H[:, 1:, :]  # [B,T,D]
            return seq_out, mask, cls_vec
        else:
            return H, mask, None

    def encode_seq(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B,T,1]

        B, T, _ = x.shape
        dev = next(self.parameters()).device
        m = _ensure_2d_mask(mask, B, T, dev)

        h, m_out, _ = self._encode_with_optional_cls(x.to(dev), m.to(dev))

        if not self._printed_once:
            self._printed_once = True
            _dbg(
                f"[BEHRTLabEncoder] encode_seq -> "
                f"h:{tuple(h.shape)} mask:{tuple(m_out.shape)}"
            )
            _peek_tensor("behrt.h", h)

        return h, m_out

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B,T,1]

        B, T, _ = x.shape
        dev = next(self.parameters()).device
        m = _ensure_2d_mask(mask, B, T, dev)

        seq_h, m_out, cls_vec = self._encode_with_optional_cls(x.to(dev), m.to(dev))

        valid = (m_out.sum(dim=1) > 0)  # [B] bool

        if self.pool == "cls":
            z = cls_vec  # [B,D]
        elif self.pool == "last":
            idx = (m_out.sum(dim=1) - 1).clamp_min(0).long() 
            z = seq_h[torch.arange(B, device=seq_h.device), idx]  # [B,D]
        else:
            z = _masked_mean(seq_h, m_out)  # [B,D]

        # force exact zeros when no structured timesteps exist
        if (~valid).any():
            z = z.clone()
            z[~valid] = 0.0

        if not self._warned_dead:
            with torch.no_grad():
                if z.abs().mean().item() < 1e-6:
                    print("[warn:BEHRTLabEncoder] near-zero pooled embedding; check mask/inputs.")
                    self._warned_dead = True

        return z  # [B,D]

class BioClinBERTEncoder(nn.Module):
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
        self.bert: Optional[nn.Module] = None

        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(model_name)
            hidden = int(getattr(self.bert.config, "hidden_size", 768))
            self.hf_available = True
        except Exception as e:
            if force_hf:
                raise RuntimeError(
                    f"Failed to load '{model_name}'. "
                    f"Install transformers and cache the model. Error: {e}"
                )
            self.bert = None
            hidden = 768
            self.hf_available = False

        self.hidden = hidden

        if d is not None and d != hidden:
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, d, bias=False),
            )
            self.out_dim = int(d)
        else:
            self.proj = nn.Identity()
            self.out_dim = int(hidden)

        self.drop = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

        # keep HF model in eval by default; finetune_text will override in training loop
        if self.hf_available and self.bert is not None:
            self.bert.eval()

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _is_valid_patient_dict(item) -> bool:
        if not isinstance(item, dict):
            return False
        if ("input_ids" not in item) or ("attention_mask" not in item):
            return False
        # allow empty [0,L]
        return True

    def _normalize_batch(self, notes_batch):
        if notes_batch is None:
            return []
        if not isinstance(notes_batch, (list, tuple)):
            raise TypeError(
                "BioClinBERTEncoder expects a batch that is a list of per-patient dicts. "
                "Pass the output of prepare_notes_batch(notes_batch)."
            )
        if len(notes_batch) == 0:
            return []

        for i, it in enumerate(notes_batch):
            if not self._is_valid_patient_dict(it):
                raise ValueError(
                    f"Invalid notes item at index {i}. Expected dict with "
                    f"'input_ids' and 'attention_mask'. Got type={type(it)} keys={getattr(it,'keys',lambda:[])()}."
                )
        return list(notes_batch)

    def _encode_chunks_to_cls(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)

        # handle empty chunk-batch safely
        if ids.numel() == 0 or ids.size(0) == 0:
            return torch.zeros(0, self.out_dim, device=self._device())

        # fallback if HF not available
        if (not self.hf_available) or (self.bert is None):
            return torch.zeros(ids.size(0), self.out_dim, device=self._device())

        dev = self._device()
        if next(self.bert.parameters()).device != dev:
            self.bert.to(dev)

        out = self.bert(
            input_ids=ids.to(dev, non_blocking=True),
            attention_mask=attn.to(dev, non_blocking=True),
        )
        cls = out.last_hidden_state[:, 0]   # [S, hidden]
        cls = self.proj(cls)                # [S, D]
        cls = self.drop(cls)
        return cls

    def encode_seq(self, notes_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self._device()
        batch = self._normalize_batch(notes_batch)
        B = len(batch)

        if B == 0:
            return (
                torch.zeros(0, 1, self.out_dim, device=dev),
                torch.zeros(0, 1, device=dev),
            )

        seqs: List[torch.Tensor] = []
        lengths: List[int] = []

        for patient in batch:
            ids = patient["input_ids"]
            attn = patient["attention_mask"]

            if torch.is_tensor(ids):
                ids = ids.to(dev, non_blocking=True)
            else:
                ids = torch.tensor(ids, dtype=torch.long, device=dev)

            if torch.is_tensor(attn):
                attn = attn.to(dev, non_blocking=True)
            else:
                attn = torch.tensor(attn, dtype=torch.long, device=dev)

            cls = self._encode_chunks_to_cls(ids, attn)  # [S, D] (S may be 0)

            seqs.append(cls)
            lengths.append(int(cls.size(0)))

        Smax = max(lengths) if max(lengths) > 0 else 1  # keep tensors non-empty for downstream
        Hpad = torch.zeros(B, Smax, self.out_dim, device=dev)
        M = torch.zeros(B, Smax, device=dev)

        for i, H in enumerate(seqs):
            s = int(H.size(0))
            if s > 0:
                Hpad[i, :s] = H
                M[i, :s] = 1.0

        return Hpad, M

    def forward(self, notes_batch) -> torch.Tensor:
        requires_grad = bool(getattr(CFG, "finetune_text", False))
        ctx = torch.enable_grad if requires_grad else torch.no_grad

        with ctx():
            H, M = self.encode_seq(notes_batch)  # [B, Smax, D], [B, Smax]
            denom = M.sum(dim=1, keepdim=True).clamp_min(1.0)
            z = (H * M.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
        return z

class MedFuseImageEncoder(nn.Module):
    def __init__(
        self,
        vision_backbone: str = "resnet34",
        vision_num_classes: int = 14,
        pretrained: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        import torchvision  

        self.device = torch.device(device)

        # Build backbone
        self.vision_backbone = getattr(torchvision.models, vision_backbone)(
            pretrained=pretrained
        )

        # Try to deduce feature dimension from classifier/fc
        d_visual = None
        for classifier in ("classifier", "fc"):
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue

            d_visual = getattr(cls_layer, "in_features", None)
            if d_visual is None:
                last_linear = None
                for m in reversed(list(cls_layer.modules())):
                    if isinstance(m, nn.Linear):
                        last_linear = m
                        break
                if last_linear is not None:
                    d_visual = last_linear.in_features

            if d_visual is None:
                raise ValueError(
                    f"Cannot infer in_features from {classifier} of "
                    f"{type(self.vision_backbone).__name__}"
                )

            # Replace with identity to expose pooled features
            setattr(self.vision_backbone, classifier, nn.Identity())
            break

        if d_visual is None:
            raise ValueError(
                f"Unsupported backbone {vision_backbone} "
                f"(no fc/classifier head found)."
            )

        # BCE
        self.bce_loss = nn.BCELoss(reduction="mean")

        # Single Linear head
        self.classifier = nn.Sequential(
            nn.Linear(d_visual, vision_num_classes),
        )

        self.feats_dim = d_visual
        self.vision_num_classes = vision_num_classes

        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        n_crops: int = 0,
        bs: Optional[int] = None,
    ):
        device = next(self.parameters()).device
        x = x.to(device)

        # pooled features from backbone
        visual_feats = self.vision_backbone(x)  # [B, D_vis]

        logits = self.classifier(visual_feats)  # [B, C]
        preds = torch.sigmoid(logits)           # [B, C]

        # Multi-crop: average predictions across crops
        if n_crops and n_crops > 0:
            if bs is None:
                # default if not provided: require B % n_crops == 0
                if preds.size(0) % n_crops != 0:
                    raise ValueError(
                        "When n_crops > 0, pass bs or ensure B % n_crops == 0."
                    )
                bs = preds.size(0) // n_crops
            preds = preds.view(bs, n_crops, -1).mean(dim=1)  # [bs, C]

        # BCE loss
        if labels is not None:
            labels = labels.to(device)
            lossvalue_bce = self.bce_loss(preds, labels)
        else:
            lossvalue_bce = torch.zeros(1, device=device)

        return preds, lossvalue_bce, visual_feats


class ImageEncoder(nn.Module):
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

        self.img_agg = img_agg
        self.proj = nn.Linear(self.medfuse.feats_dim, d)
        self.drop = nn.Identity()

        self.to(dev)

    def _encode_batch_feats(self, x: torch.Tensor) -> torch.Tensor:
        _, _, feats = self.medfuse(x)  # [B, D_vis]
        z = self.drop(self.proj(feats))  # [B, d]
        _peek_tensor("imgenc.z", z)
        return z

    def medfuse_forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        n_crops: int = 0,
        bs: Optional[int] = None,
    ):
        return self.medfuse(x, labels=labels, n_crops=n_crops, bs=bs)

    def forward(
        self,
        x: Union[
            torch.Tensor,
            List[torch.Tensor],
            List[List[torch.Tensor]],
        ],
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        d_out = int(self.proj.out_features)

        def _zero_embed(B: int) -> torch.Tensor:
            return torch.zeros(B, d_out, device=device)

        # Case 1: tensor input
        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                x = x.unsqueeze(0)  # [1,3,H,W]
            if x.dim() != 4:
               raise ValueError("Tensor input must be [3,H,W] or [B,3,H,W].")
            x = x.to(device, non_blocking=True)

            # detect missing images: exactly all zeros per sample
            flat = x.view(x.size(0), -1)
            missing = (flat.abs().sum(dim=1) == 0)  # [B] bool

            # encode only non-missing
            if missing.all():
                return _zero_embed(x.size(0))

            z = _zero_embed(x.size(0))
            keep_idx = (~missing).nonzero(as_tuple=False).squeeze(1)
            z_keep = self._encode_batch_feats(x[keep_idx])  # [B_keep, d]
            z[keep_idx] = z_keep
            _peek_tensor("imgenc.z(masked)", z)
            return z

        # Case 2: list of tensors [B]
        if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], torch.Tensor)):
            if len(x) == 0:
                return torch.zeros(0, d_out, device=device)

            xs = torch.stack(x, dim=0).to(device, non_blocking=True)  # [B,3,H,W]
            flat = xs.view(xs.size(0), -1)
            missing = (flat.abs().sum(dim=1) == 0)

            if missing.all():
                return _zero_embed(xs.size(0))

            z = _zero_embed(xs.size(0))
            keep_idx = (~missing).nonzero(as_tuple=False).squeeze(1)
            z_keep = self._encode_batch_feats(xs[keep_idx])
            z[keep_idx] = z_keep
            _peek_tensor("imgenc.z(masked)", z)
            return z

        # Case 3: list of list-of-images -> use last image per sample
        out: List[torch.Tensor] = []
        for imgs in x:
            if imgs is None or len(imgs) == 0:
                out.append(torch.zeros(d_out, device=device))
                continue

            img = imgs[-1]
            if not isinstance(img, torch.Tensor):
                out.append(torch.zeros(d_out, device=device))
                continue

            # if last image is all zeros, treat as missing
            if img.abs().sum().item() == 0:
                out.append(torch.zeros(d_out, device=device))
                continue

            if img.dim() == 3:
                img = img.unsqueeze(0)  # [1,3,H,W]
            z1 = self._encode_batch_feats(img.to(device, non_blocking=True))  # [1,d]
            out.append(z1.squeeze(0))

        Z = torch.stack(out, dim=0)
        _peek_tensor("imgenc.seq_last_z(masked)", Z)
        return Z


    def encode_seq(
        self,
        batch_images: Union[
            List[torch.Tensor],
            List[List[torch.Tensor]],
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        Z = self.forward(batch_images)  # [B, d]
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)
        B = Z.size(0)

        # If your forward guarantees missing images -> exact zero embedding,
        # we can define presence as "any nonzero in embedding.
        present = (Z.abs().sum(dim=1) > 0).float()  # [B]
        M = present.unsqueeze(1)                    # [B,1]

        Hpad = Z.unsqueeze(1)                       # [B,1,d]
        _peek_tensor("imgenc.Hpad", Hpad)
        _peek_tensor("imgenc.M", M)
        return Hpad, M


    def load_backbone_state(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ) -> None:
        try:
            self.medfuse.vision_backbone.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            remapped = {
                k.replace("vision_backbone.", ""): v for k, v in state_dict.items()
            }
            self.medfuse.vision_backbone.load_state_dict(remapped, strict=strict)

    def freeze_backbone(self) -> None:
        for p in self.medfuse.vision_backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.medfuse.vision_backbone.parameters():
            p.requires_grad = True

class SimpleHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairwiseConcatFusion(nn.Module):
    def __init__(
        self,
        d: int,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.head = SimpleHead(in_dim=2 * d, out_dim=d, p_drop=0.0)

    def forward(
        self,
        A: torch.Tensor,
        mA: torch.Tensor,
        B: torch.Tensor,
        mB: torch.Tensor,
    ) -> torch.Tensor:
        za = _masked_mean(A, mA)  # [B, D]
        zb = _masked_mean(B, mB)  # [B, D]
        x = torch.cat([za, zb], dim=-1)  # [B, 2D]
        z = self.head(x)                 # [B, D]
        _peek_tensor("fusion.pair_z", z)
        return z


class TrimodalConcatFusion(nn.Module):
    def __init__(
        self,
        d: int,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.head = SimpleHead(in_dim=3 * d, out_dim=d, p_drop=0.0)

    def forward(
        self,
        L: torch.Tensor,
        mL: torch.Tensor,
        N: torch.Tensor,
        mN: torch.Tensor,
        I: torch.Tensor,
        mI: torch.Tensor,
    ) -> torch.Tensor:
        zL = _masked_mean(L, mL)  # [B, D]
        zN = _masked_mean(N, mN)  # [B, D]
        zI = _masked_mean(I, mI)  # [B, D]
        x = torch.cat([zL, zN, zI], dim=-1)  # [B, 3D]
        z = self.head(x)                     # [B, D]
        _peek_tensor("fusion.tri_z", z)
        return z


class RouteActivation(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 1),
        )

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

        # Per-route activations (sigmoid)
        self.act_L = RouteActivation(d)
        self.act_N = RouteActivation(d)
        self.act_I = RouteActivation(d)
        self.act_LN = RouteActivation(d)
        self.act_LI = RouteActivation(d)
        self.act_NI = RouteActivation(d)
        self.act_LNI = RouteActivation(d)

        self.unim_ln = nn.LayerNorm(d)

    def _pool_uni(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        if self.cfg.unimodal_pool == "last":
            last_idx = (M.sum(dim=1) - 1).clamp_min(0).long()
            return X[torch.arange(X.size(0), device=X.device), last_idx]
        return _masked_mean(X, M)

    def forward(
        self,
        L_seq: torch.Tensor,
        mL: torch.Tensor,
        N_seq: torch.Tensor,
        mN: torch.Tensor,
        I_seq: torch.Tensor,
        mI: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Unimodal pooled
        zL = self.unim_ln(self._pool_uni(L_seq, mL))
        zN = self.unim_ln(self._pool_uni(N_seq, mN))
        zI = self.unim_ln(self._pool_uni(I_seq, mI))

        # Pairwise + Trimodal (simple concat + linear)
        zLN = self.pair_LN(L_seq, mL, N_seq, mN)
        zLI = self.pair_LI(L_seq, mL, I_seq, mI)
        zNI = self.pair_NI(N_seq, mN, I_seq, mI)
        zLNI = self.tri_LNI(L_seq, mL, N_seq, mN, I_seq, mI)

        route_embs: Dict[str, torch.Tensor] = {
            "L": zL,
            "N": zN,
            "I": zI,
            "LN": zLN,
            "LI": zLI,
            "NI": zNI,
            "LNI": zLNI,
        }

        route_act: Dict[str, torch.Tensor] = {
            "L": self.act_L(zL),
            "N": self.act_N(zN),
            "I": self.act_I(zI),
            "LN": self.act_LN(zLN),
            "LI": self.act_LI(zLI),
            "NI": self.act_NI(zNI),
            "LNI": self.act_LNI(zLNI),
        }

        return route_embs, route_act

@dataclass
class EncoderConfig:
    d: int = 256
    dropout: float = 0.0

    # structured
    structured_seq_len: int = 24
    structured_n_feats: int = 17
    structured_layers: int = 2
    structured_heads: int = 8
    structured_pool: Literal["last", "mean", "cls"] = "cls"

    # notes
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_max_len: int = 512
    note_agg: Literal["mean", "attention"] = "attention"

    # images
    img_agg: Literal["last", "mean", "attention"] = "last"
    vision_backbone: str = "resnet34"
    vision_num_classes: int = 14
    vision_pretrained: bool = True


def build_encoders(
    cfg: EncoderConfig,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder]:
    """
    Construct structured/text/image encoders given EncoderConfig.
    """
    dev = torch.device(DEVICE if device is None else device)

    behrt = BEHRTLabEncoder(
        n_feats=cfg.structured_n_feats,
        d=cfg.d,
        seq_len=cfg.structured_seq_len,
        n_layers=cfg.structured_layers,
        n_heads=cfg.structured_heads,
        dropout=0.0,
        pool=cfg.structured_pool,
    ).to(dev)

    bbert = BioClinBERTEncoder(
        model_name=cfg.text_model_name,
        d=cfg.d,
        dropout=cfg.dropout,
    ).to(dev)

    if getattr(bbert, "hf_available", False) and bbert.bert is not None:
        bbert.bert.to(dev)
        if not getattr(CFG, "finetune_text", False):
            bbert.bert.eval()

    imgenc = ImageEncoder(
        d=cfg.d,
        dropout=0.0,
        img_agg=cfg.img_agg,
        vision_backbone=cfg.vision_backbone,
        vision_num_classes=cfg.vision_num_classes,
        pretrained=cfg.vision_pretrained,
        device=dev,
    ).to(dev)

    return behrt, bbert, imgenc


def build_multimodal_feature_extractor(
    d: int,
    dropout: float = 0.0,
    unimodal_pool: Literal["mean", "last"] = "mean",
) -> MultimodalFeatureExtractor:
    cfg = MulTConfig(
        d=d,
        dropout=dropout,
        unimodal_pool=unimodal_pool,
    )
    dev = torch.device(DEVICE)
    return MultimodalFeatureExtractor(cfg).to(dev)


NoteItem = Dict[str, torch.Tensor]   # {"input_ids":[S,L], "attention_mask":[S,L]}
BatchNotes = List[NoteItem]


def encode_all_routes_from_batch(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    extractor: MultimodalFeatureExtractor,
    xL: torch.Tensor,
    notes_list: BatchNotes,
    imgs: Union[
        torch.Tensor,
        List[torch.Tensor],
        List[List[torch.Tensor]],
    ],
    mL: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Convenience for sequence-level encoding + feature extraction that returns:
      - route_embs: dict of 7 interaction embeddings (each [B,d])
      - route_act: dict of 7 sigmoid activations (each [B,1])
    """
    dev = next(extractor.parameters()).device

    # Structured (Lab) encoder
    L_seq, mL_seq = behrt.encode_seq(
        xL.to(dev),
        mask=mL.to(dev) if mL is not None else None,
    )

    # BioClinBERTEncoder.encode_seq expects pre-tokenized notes (BatchNotes).
    N_seq, mN_seq = bbert.encode_seq(notes_list)  # type: ignore[arg-type]

    # Image encoder
    I_seq, mI_seq = imgenc.encode_seq(imgs)

    route_embs, route_act = extractor(
        L_seq,
        mL_seq,
        N_seq,
        mN_seq,
        I_seq,
        mI_seq,
    )

    # One-time sanity print
    if not hasattr(extractor, "_printed_once"):
        extractor._printed_once = True  # type: ignore[attr-defined]
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
    imgs: Union[
        torch.Tensor,
        List[torch.Tensor],
        List[List[torch.Tensor]],
    ],
    mL: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Return unimodal pooled embeddings:
        {"L": zL, "N": zN, "I": zI}
    """
    dev = next(behrt.parameters()).device

    zL = behrt(
        xL.to(dev),
        mask=mL.to(dev) if mL is not None else None,
    )
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
