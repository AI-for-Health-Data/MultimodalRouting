from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Sequence, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import DEVICE  # keep a single source of truth for device



class BEHRTLabEncoder(nn.Module):
    def __init__(
        self,
        n_feats: int,
        d: int,
        seq_len: int = 24,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        pool: Literal["last", "mean"] = "last",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.input_proj = nn.Linear(n_feats, d)
        # Learnable positional encoding over seq_len hours
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        h = self.input_proj(x) + self.pos[:, :T, :]   
        h = self.enc(h)                                
        h = h[:, -1] if self.pool == "last" else h.mean(dim=1)
        z = self.out(h)                               
        return z



class BioClinBERTEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        d: int,
        max_len: int = 512,
        dropout: float = 0.1,
        note_agg: Literal["mean", "attention", "concat"] = "mean",
        max_notes_concat: int = 8,
        attn_hidden: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.d = d
        self.max_len = max_len
        self.note_agg = note_agg
        self.max_notes_concat = max_notes_concat

        try:
            from transformers import AutoTokenizer, AutoModel  
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            self.bert.eval()  # deterministic (disable dropout)
            self.hf_available = True
        except Exception:
            self.tokenizer = None
            self.bert = None
            self.hf_available = False

        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d))
        self.drop = nn.Dropout(dropout)

        if note_agg == "attention":
            self.attn = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )
        else:
            self.attn = None

        self.device_override = device

    def _encode_chunks(self, texts: List[str]) -> torch.Tensor:
        device = self.device_override or next(self.parameters()).device
        if not self.hf_available:
            return torch.randn(len(texts), 768, device=device)

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = self.bert(**batch).last_hidden_state[:, 0]  
        return out  

    def _chunk_note(self, text: str) -> List[str]:
        if text is None:
            return [""]
        toks = text.split()
        chunks = []
        for i in range(0, len(toks), self.max_len):
            piece = " ".join(toks[i:i + self.max_len])
            if piece:
                chunks.append(piece)
        if not chunks:
            chunks = [""]
        return chunks

    def _pool_notes(self, note_vecs: List[torch.Tensor]) -> torch.Tensor:
        device = self.device_override or next(self.parameters()).device
        if len(note_vecs) == 0:
            return torch.zeros(self.d, device=device)

        H = torch.stack(note_vecs, dim=0)  

        if self.note_agg == "mean":
            return H.mean(dim=0)

        elif self.note_agg == "attention":
            scores = self.attn(H)  
            w = torch.softmax(scores.squeeze(-1), dim=0)
            return (w.unsqueeze(0) @ H).squeeze(0)

        else:  
            K = min(self.max_notes_concat, H.size(0))
            Hk = H[-K:]                 
            concat = Hk.reshape(-1)     
            proj_in = concat.numel()
            key = f"concat_proj_{proj_in}"
            if not hasattr(self, key):
                setattr(self, key, nn.Linear(proj_in, self.d).to(device))
            layer: nn.Linear = getattr(self, key)
            return layer(concat)

    @staticmethod
    def _normalize_input(
        batch_notes: Union[Sequence[str], Sequence[Sequence[str]]]
    ) -> List[List[str]]:
        if len(batch_notes) == 0:
            return []
        if isinstance(batch_notes[0], str):
            return [[t] for t in batch_notes]
        return [list(notes) for notes in batch_notes]

    def forward(self, batch_notes: Union[List[str], List[List[str]]]) -> torch.Tensor:
        device = self.device_override or next(self.parameters()).device
        patients = self._normalize_input(batch_notes)
        Z: List[torch.Tensor] = []

        for notes in patients:
            note_embs: List[torch.Tensor] = []
            for note_text in notes:
                chunks = self._chunk_note(note_text)
                cls_vecs = self._encode_chunks(chunks)              
                cls_mean = cls_vecs.mean(dim=0, keepdim=True)       
                z_note = self.drop(self.proj(cls_mean)).squeeze(0)  
                note_embs.append(z_note)

            z_patient = self._pool_notes(note_embs)
            Z.append(z_patient.to(device))

        if len(Z) == 0:
            return torch.zeros(0, self.d, device=device)

        return torch.stack(Z, dim=0) 



class ImageEncoder(nn.Module):
    def __init__(
        self,
        d: int,
        dropout: float = 0.1,
        img_agg: Literal["last", "mean", "attention"] = "last",
        attn_hidden: int = 256,
    ) -> None:
        super().__init__()
        import torchvision
        try:
            backbone = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.DEFAULT
            )
        except Exception:
            backbone = torchvision.models.resnet34(weights=None)  
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = 512

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.out_channels),
            nn.Linear(self.out_channels, d),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.img_agg = img_agg
        self.attn = None
        if img_agg == "attention":
            self.attn = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )

    def _encode_one(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self.backbone(x)    
        h = self.gap(h)         
        z = self.proj(h)        
        return z

    def _pool_images(self, embs: List[torch.Tensor]) -> torch.Tensor:
        H = torch.stack(embs, dim=0)  
        if self.img_agg == "last":
            return H[-1]
        elif self.img_agg == "mean":
            return H.mean(dim=0)
        else:
            scores = self.attn(H)                     
            w = torch.softmax(scores.squeeze(-1), dim=0)
            return (w.unsqueeze(0) @ H).squeeze(0)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]) -> torch.Tensor:
        device = next(self.parameters()).device

        if isinstance(x, torch.Tensor):
            z = self._encode_one(x.to(device))   
            return z

        if len(x) == 0:
            d_out = self.proj[2].out_features
            return torch.zeros(0, d_out, device=device)

        if isinstance(x[0], torch.Tensor):
            embs = [self._encode_one(img.to(device)).squeeze(0) for img in x]
            return torch.stack(embs, dim=0)

        out: List[torch.Tensor] = []
        for imgs in x:
            if len(imgs) == 0:
                d_out = self.proj[2].out_features
                out.append(torch.zeros(d_out, device=device))
                continue
            embs = [self._encode_one(img.to(device)).squeeze(0) for img in imgs]
            out.append(self._pool_images(embs))
        return torch.stack(out, dim=0)



class PairwiseFusion(nn.Module):
    def __init__(self, d: int,
                 hidden: Optional[Sequence[int]] = None,
                 dropout: float = 0.1,
                 feature_mode: str = "rich"):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        self.d = d
        self.feature_mode = feature_mode
        in_dim = 2 * d if feature_mode == "concat" else 4 * d
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden or (4 * d)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden or (4 * d), d),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, zA: torch.Tensor, zB: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "concat":
            x = torch.cat([zA, zB], dim=-1)
        else:
            had = zA * zB
            diff = (zA - zB).abs()
            x = torch.cat([zA, zB, had, diff], dim=-1)
        h = self.net(x)
        base = 0.5 * (zA + zB)
        return h + self.res_scale * base


class TrimodalFusion(nn.Module):
    def __init__(self, d: int,
                 hidden: Optional[Sequence[int]] = None,
                 dropout: float = 0.1,
                 feature_mode: str = "rich"):
        super().__init__()
        assert feature_mode in {"concat", "rich"}
        self.d = d
        self.feature_mode = feature_mode
        in_dim = 3 * d if feature_mode == "concat" else 7 * d
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden or (4 * d)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden or (4 * d), d),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, zL: torch.Tensor, zN: torch.Tensor, zI: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "concat":
            x = torch.cat([zL, zN, zI], dim=-1)
        else:
            zLN  = zL * zN
            zLI  = zL * zI
            zNI  = zN * zI
            zLNI = zL * zN * zI
            x = torch.cat([zL, zN, zI, zLN, zLI, zNI, zLNI], dim=1)
        h = self.net(x)
        base = (zL + zN + zI) / 3.0
        return h + self.res_scale * base


@dataclass
class EncoderConfig:
    d: int = 256
    dropout: float = 0.1
    # structured
    structured_seq_len: int = 24
    structured_n_feats: int = 128
    structured_layers: int = 2
    structured_heads: int = 8
    structured_pool: Literal["last", "mean"] = "last"
    # notes
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_max_len: int = 512
    note_agg: Literal["mean", "attention", "concat"] = "mean"
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


@dataclass
class FusionConfig:
    d: int = 256
    dropout: float = 0.1
    hidden: Optional[Sequence[int]] = None
    feature_mode: str = "rich"  


def build_fusions_enc(
    d: int,
    p_drop: float,
    hidden: Optional[Sequence[int]] = None,
    feature_mode: str = "rich",
) -> Dict[str, nn.Module]:
    dev = torch.device(DEVICE)
    LN  = PairwiseFusion(d=d, hidden=hidden, dropout=p_drop, feature_mode=feature_mode).to(dev)
    LI  = PairwiseFusion(d=d, hidden=hidden, dropout=p_drop, feature_mode=feature_mode).to(dev)
    NI  = PairwiseFusion(d=d, hidden=hidden, dropout=p_drop, feature_mode=feature_mode).to(dev)
    LNI = TrimodalFusion(d=d, hidden=hidden, dropout=p_drop, feature_mode=feature_mode).to(dev)
    return {"LN": LN, "LI": LI, "NI": NI, "LNI": LNI}


def build_fusions_from_cfg(cfg: FusionConfig) -> Dict[str, nn.Module]:
    return build_fusions_enc(
        d=cfg.d,
        p_drop=cfg.dropout,
        hidden=cfg.hidden,
        feature_mode=cfg.feature_mode,
    )


def make_route_inputs(
    z: Dict[str, torch.Tensor],
    fusion: Dict[str, nn.Module],
) -> Dict[str, torch.Tensor]:
    assert {"L", "N", "I"}.issubset(z), "z must contain L, N, I"
    assert {"LN", "LI", "NI", "LNI"}.issubset(fusion), "Missing fusion blocks"

    return {
        "L":   z["L"],
        "N":   z["N"],
        "I":   z["I"],
        "LN":  fusion["LN"](z["L"], z["N"]),
        "LI":  fusion["LI"](z["L"], z["I"]),
        "NI":  fusion["NI"](z["N"], z["I"]),
        "LNI": fusion["LNI"](z["L"], z["N"], z["I"]),
    }


__all__ = [
    # Encoders
    "BEHRTLabEncoder",
    "BioClinBERTEncoder",
    "ImageEncoder",
    # Fusion blocks (embedding-level)
    "PairwiseFusion",
    "TrimodalFusion",
    # Configs & builders
    "EncoderConfig",
    "FusionConfig",
    "build_encoders",
    "build_fusions_enc",
    "build_fusions_from_cfg",
    # Utility
    "make_route_inputs",
]
