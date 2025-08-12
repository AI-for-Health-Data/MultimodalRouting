from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as tv
from transformers import AutoConfig, AutoModel

try:
    import medfuse  
except Exception:
    medfuse = None


# 1. Structured sequence
class _LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


class BEHRTLabEncoder(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int = 24,          
        out_dim: int = 768,
        nhead: int = 8,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.token = nn.Linear(1, out_dim)
        self.pos   = _LearnablePositionalEncoding(seq_len, out_dim)

        block = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.tr = nn.TransformerEncoder(block, layers)

        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, *_unused) -> torch.Tensor:
        h = self.token(x.unsqueeze(-1))    
        h = self.tr(self.pos(h)).mean(1)   
        return self.proj(h)


# 2. Notes 
class BioClinBERTEncoder(nn.Module):
    def __init__(
        self,
        *,
        out_dim: int = 768,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        ckpt_dir: Optional[Path] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        cfg   = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        model = ckpt_dir or model_name
        self.bert = AutoModel.from_pretrained(model, config=cfg)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.proj    = (
            nn.Linear(cfg.hidden_size, out_dim)
            if cfg.hidden_size != out_dim else nn.Identity()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        B, N, L = input_ids.shape
        out = self.bert(
            input_ids      = input_ids.view(B * N, L),
            attention_mask = attention_mask.view(B * N, L)
        ).last_hidden_state[:, 0]  

        out = out.view(B, N, -1).mean(1)   
        return self.dropout(self.proj(out)) 


# 3. Images (MedFuse) 
class ImageCXREncoder(nn.Module):
    def __init__(
        self,
        *,
        out_dim: int = 768,
        model_name: str = "medfuse_base",
        ckpt_path: Optional[Path] = None,
        freeze: bool = False,
        dropout_p: float = 0.10,
    ) -> None:
        super().__init__()

        self.backbone, feat_dim = self._build_backbone(model_name, ckpt_path)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

    @staticmethod
    def _remove_resnet_fc(m: nn.Module) -> Tuple[nn.Module, int]:
        """Return a resnet backbone that outputs pooled features."""
        if isinstance(m, tv.ResNet):
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()  
        else:
            feat_dim = getattr(m, "feat_dim", 2048)
        return m, feat_dim

    def _build_backbone(self, model_name: str, ckpt_path: Optional[Path]) -> Tuple[nn.Module, int]:
        if medfuse is not None:
            try:
                backbone = medfuse.create_model(model_name, pretrained=True)
                feat_dim = getattr(backbone, "feat_dim", None)
                if feat_dim is None:
                    feat_dim = getattr(backbone, "num_features", None)
                if feat_dim is None:
                    raise AttributeError("Could not infer feature dimension from MedFuse backbone.")

                if ckpt_path is not None:
                    state = torch.load(str(ckpt_path), map_location="cpu")
                    backbone.load_state_dict(state, strict=False)
                return backbone, int(feat_dim)
            except Exception:
                pass

        m = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT)
        m, feat_dim = self._remove_resnet_fc(m)
        return m, feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))
        return self.proj(feats)
