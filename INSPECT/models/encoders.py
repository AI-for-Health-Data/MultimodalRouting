from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv
from transformers import AutoConfig, AutoModel


# 1) Structured Lab Sequence Encoder
class _LearnablePositionalEncoding(nn.Module):
    """
    Adds a learnable positional embedding to each position in a sequence.
    Intended for sequences where the order matters (e.g., time-ordered labs).
    """
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        # A parameter of shape (seq_len, d_model) that will be learned.
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # simply add the learned positional vector to each token
        return x + self.pos


class BEHRTLabEncoder(nn.Module):
    """
    Encodes a fixed-length sequence of scalar lab values into a single embedding.
    Implements:
      1) Linear tokenization of each scalar to a vector
      2) Addition of learnable positional encodings
      3) Transformer encoder layers
      4) Projection + dropout
    """
    def __init__(
        self,
        *,
        seq_len: int,
        out_dim: int = 768,
        nhead: int = 8,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1) Turn each scalar lab value into an out_dim-dim vector
        self.token = nn.Linear(1, out_dim)
        # 2) Learnable positional embeddings
        self.pos = _LearnablePositionalEncoding(seq_len, out_dim)

        # 3) Stacked Transformer encoder
        block = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)

        # 4) Final projection + nonlinearity + dropout
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        """
        x: (batch, seq_len) of raw lab values
        Returns: (batch, out_dim) combined embedding
        """
        # [B, L] -> [B, L, 1] -> token embeddings [B, L, out_dim]
        h = self.token(x.unsqueeze(-1))
        # add positional info and run through transformer
        h = self.transformer(self.pos(h))
        # mean-pool over the sequence dimension -> [B, out_dim]
        h = h.mean(dim=1)
        return self.proj(h)

# 2) Clinical Text Encoder
class BioClinicalBERTEncoder(nn.Module):
    """
    Wraps a pretrained Bio_ClinicalBERT to produce a single CLS-vector per input.
    Any long‐note chunking and patient‐level aggregation should be handled outside this module.
    """
    def __init__(
        self,
        *,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        # Load from local cache_dir or HF hub
        bert_kwargs = {}
        if cache_dir is not None:
            bert_kwargs["cache_dir"] = str(cache_dir)
        self.bert = AutoModel.from_pretrained(model_name, **bert_kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len) token IDs, already truncated to <= max_length
            attention_mask: (batch, seq_len)
        Returns:
            cls_emb:        (batch, hidden_size) embedding of [CLS] token
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]

# 3) Image Encoder (updated with INSPECT CT support)
class ImageCXREncoder(nn.Module):
    """
    Encodes either:
      • 2D CXR images via torchvision backbones (DenseNet121, ResNet50, etc.)
      • 3D CT volumes (stack of slices) via the ResNetV2-CT Lightning model
        from SMU-ZW/inspect_image_code (through radfusion3).
    """
    def __init__(
        self,
        *,
        modality: str = "CXR",               # "CXR" or "CT"
        backbone: str = "densenet121",       
        pretrained: bool = True,             
        ct_config: Optional[Path] = None,     
        ct_ckpt: Optional[str] = None,       # path to ResNetV2-CT .ckpt
        out_dim: int = 256,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.modality = modality.upper()

        if self.modality == "CXR":
            import torchvision.models as tv
            # -- pick a 2D backbone and strip off its classifier --
            if backbone == "densenet121":
                model = tv.densenet121(pretrained=pretrained)
                feat_dim = model.classifier.in_features
                model.classifier = nn.Identity()
            elif backbone == "resnet50":
                model = tv.resnet50(pretrained=pretrained)
                feat_dim = model.fc.in_features
                model.fc = nn.Identity()
            else:
                raise ValueError(f"Unsupported CXR backbone: {backbone}")

            if freeze_backbone:
                for p in model.parameters():
                    p.requires_grad = False

            self.backbone = model
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )

        elif self.modality == "CT":
            # load the slice encoder from the INSPECT repo via radfusion3 
            if ct_config is None or ct_ckpt is None:
                raise ValueError("For CT modality, both ct_config and ct_ckpt must be provided")

            import radfusion3
            # build_lightning_model will return a LightningModule wrapping ResNetV2-CT
            self.ct_model = radfusion3.builder.build_lightning_model(ct_config, ckpt=ct_ckpt)
            # drop the classification head so we only get features
            if hasattr(self.ct_model, "head"):
                self.ct_model.head = nn.Identity()
            if freeze_backbone:
                for p in self.ct_model.parameters():
                    p.requires_grad = False

            # figure out the feature dimension
            feat_dim = getattr(self.ct_model, "hidden_size", None) \
                     or self.ct_model.lit_model.hparams.hidden_size

            # optionally project to out_dim
            self.proj = nn.Identity() if feat_dim == out_dim else nn.Linear(feat_dim, out_dim)

        else:
            raise ValueError(f"Unknown modality: {modality!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is either:
          • [B, 3, H, W] for CXR
          • [B, S, 1, H, W] for CT volumes (S = # of slices)
        Returns:
          • [B, out_dim] feature vectors
        """
        if self.modality == "CXR":
            feats = self.backbone(x)        
        else:
            B, S, C, H, W = x.shape
            x_flat = x.view(B * S, C, H, W)
            feat_flat = self.ct_model(x_flat)  
            feat_seq = feat_flat.view(B, S, -1)
            feats = feat_seq.mean(dim=1)     

        return self.proj(feats)
