from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Sequence, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import DEVICE  



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

    def encode_seq(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        dev = next(self.parameters()).device
        mask = _ensure_2d_mask(mask, B, T, dev)
        h = self.input_proj(x) + self.pos[:, :T, :]
        pad_mask = (mask < 0.5)  
        h = self.enc(h, src_key_padding_mask=pad_mask)
        h = self.out(h)  
        return h, mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, m = self.encode_seq(x, mask=mask)
        if self.pool == "last":
            if (m.sum(dim=1) != m.size(1)).any():
                idx = (m.sum(dim=1) - 1).clamp_min(0).long()  
                z = h[torch.arange(h.size(0), device=h.device), idx]  
            else:
                z = h[:, -1]
        else:
            z = _masked_mean(h, m)
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
            self.bert.eval()
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

    def _encode_chunk_cls(self, texts: List[str]) -> torch.Tensor:
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
            hidden = self.bert(**batch).last_hidden_state
            cls_vecs = hidden[:, 0]  
        return cls_vecs

    def _chunk_note(self, text: str) -> List[str]:
        if text is None:
            return [""]
        toks = text.split()
        chunks = []
        step = self.max_len  
        for i in range(0, len(toks), step):
            piece = " ".join(toks[i: i + step])
            if piece:
                chunks.append(piece)
        if not chunks:
            chunks = [""]
        return chunks

    @staticmethod
    def _normalize_input(
        batch_notes: Union[Sequence[str], Sequence[Sequence[str]]]
    ) -> List[List[str]]:
        if len(batch_notes) == 0:
            return []
        if isinstance(batch_notes[0], str):
            return [[t] for t in batch_notes]
        return [list(notes) for notes in batch_notes]

    def encode_seq(
        self,
        batch_notes: Union[List[str], List[List[str]]],
        max_total_chunks: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = self.device_override or next(self.parameters()).device
        patients = self._normalize_input(batch_notes)

        seqs: List[torch.Tensor] = []
        lengths: List[int] = []

        for notes in patients:
            chunk_cls: List[torch.Tensor] = []
            remaining = max_total_chunks
            for note_text in notes:
                if remaining <= 0:
                    break
                chunks = self._chunk_note(note_text)
                if len(chunks) > remaining:
                    chunks = chunks[:remaining]
                if len(chunks) == 0:
                    continue
                cls_vecs = self._encode_chunk_cls(chunks)  
                z = self.drop(self.proj(cls_vecs))        
                chunk_cls.append(z)
                remaining -= z.size(0)

            if len(chunk_cls) == 0:
                chunk_cls.append(torch.zeros(1, self.d, device=device))

            H = torch.cat(chunk_cls, dim=0)  
            seqs.append(H)
            lengths.append(H.size(0))

        if len(seqs) == 0:
            return torch.zeros(0, 1, self.d, device=device), torch.zeros(0, 1, device=device)

        Smax = max(lengths)
        B = len(seqs)
        Hpad = torch.zeros(B, Smax, self.d, device=device)
        M = torch.zeros(B, Smax, device=device)

        for i, H in enumerate(seqs):
            s = H.size(0)
            Hpad[i, :s] = H
            M[i, :s] = 1.0

        return Hpad, M

    def forward(self, batch_notes: Union[List[str], List[List[str]]]) -> torch.Tensor:

        device = self.device_override or next(self.parameters()).device
        H, M = self.encode_seq(batch_notes)
        if self.attn is None or (M.sum(dim=1) == 0).any():
            return _masked_mean(H, M)
        scores = self.attn(H)  
        scores = scores.squeeze(-1)
        scores = scores + (M < 0.5) * (-1e9)
        w = torch.softmax(scores, dim=1)  
        z = (w.unsqueeze(-1) * H).sum(dim=1)
        return z




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

    def encode_seq(
        self,
        batch_images: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        min_token_if_empty: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = next(self.parameters()).device

        if isinstance(batch_images, list) and len(batch_images) > 0 and isinstance(batch_images[0], torch.Tensor):
            batch_images = [[img] for img in batch_images] 

        seqs: List[torch.Tensor] = []
        lengths: List[int] = []

        for imgs in batch_images: 
            embs: List[torch.Tensor] = []
            for img in imgs:
                z = self._encode_one(img.to(device)).squeeze(0) 
                embs.append(z)
            if len(embs) == 0 and min_token_if_empty:
                embs.append(torch.zeros(self.proj[2].out_features, device=device))
            H = torch.stack(embs, dim=0) if len(embs) > 0 else torch.zeros(0, self.proj[2].out_features, device=device)
            seqs.append(H)
            lengths.append(max(1, H.size(0)))

        if len(seqs) == 0:
            D = self.proj[2].out_features
            return torch.zeros(0, 1, D, device=device), torch.zeros(0, 1, device=device)

        T_max = max(lengths)
        B = len(seqs)
        D = seqs[0].size(-1) if seqs[0].numel() > 0 else self.proj[2].out_features

        Hpad = torch.zeros(B, T_max, D, device=device)
        M = torch.zeros(B, T_max, device=device)

        for i, H in enumerate(seqs):
            t = H.size(0)
            if t > 0:
                Hpad[i, :t] = H
                M[i, :t] = 1.0
            else:
                M[i, 0] = 1.0  

        return Hpad, M

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        # Single tensor -> one image
        if isinstance(x, torch.Tensor):
            z = self._encode_one(x.to(device))
            return z

        # Batch of images (one per patient)
        if len(x) == 0:
            d_out = self.proj[2].out_features
            return torch.zeros(0, d_out, device=device)

        if isinstance(x[0], torch.Tensor):
            embs = [self._encode_one(img.to(device)).squeeze(0) for img in x]  
            return torch.stack(embs, dim=0)

        # Variable number per patient
        out: List[torch.Tensor] = []
        for imgs in x:  
            if len(imgs) == 0:
                d_out = self.proj[2].out_features
                out.append(torch.zeros(d_out, device=device))
                continue
            embs = [self._encode_one(img.to(device)).squeeze(0) for img in imgs]
            H = torch.stack(embs, dim=0)  # [T, D]
            if self.img_agg == "last":
                out.append(H[-1])
            elif self.img_agg == "mean":
                out.append(H.mean(dim=0))
            else:
                scores = self.attn(H)  # [T, 1]
                w = torch.softmax(scores.squeeze(-1), dim=0)
                out.append((w.unsqueeze(0) @ H.unsqueeze(0)).squeeze(0).squeeze(0))
        return torch.stack(out, dim=0)



class CrossAttentionBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(
        self,
        X: torch.Tensor,              
        Y: torch.Tensor,              
        key_mask: Optional[torch.Tensor] = None,   
        query_mask: Optional[torch.Tensor] = None, 
    ) -> torch.Tensor:
        # MultiheadAttention expects True for pads
        kpm = (key_mask < 0.5) if key_mask is not None else None  
        # attn output
        H, _ = self.attn(query=X, key=Y, value=Y, key_padding_mask=kpm, need_weights=False)
        X = self.norm1(X + H)
        X2 = self.ff(X)
        X = self.norm2(X + X2)
        if query_mask is not None:
            X = X * query_mask.unsqueeze(-1)
        return X


class PairwiseMulTHead(nn.Module):
    def __init__(self, d: int, n_heads: int = 8, n_layers: int = 1, dropout: float = 0.1, feature_mode: str = "rich"):
        super().__init__()
        self.feature_mode = feature_mode
        self.layers_ab = nn.ModuleList([CrossAttentionBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.layers_ba = nn.ModuleList([CrossAttentionBlock(d, n_heads, dropout) for _ in range(n_layers)])
        in_dim = 2 * d if feature_mode == "concat" else 4 * d
        self.fuse = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        A: torch.Tensor, mA: torch.Tensor,   
        B: torch.Tensor, mB: torch.Tensor,   
    ) -> torch.Tensor:
        Ha = A
        Hb = B
        for blk in self.layers_ab:
            Ha = blk(Ha, Hb, key_mask=mB, query_mask=mA)
        for blk in self.layers_ba:
            Hb = blk(Hb, Ha, key_mask=mA, query_mask=mB)

        za = _masked_mean(Ha, mA)  
        zb = _masked_mean(Hb, mB)  

        if self.feature_mode == "concat":
            x = torch.cat([za, zb], dim=-1)
        else:
            had = za * zb
            diff = (za - zb).abs()
            x = torch.cat([za, zb, had, diff], dim=-1)

        h = self.fuse(x)
        base = 0.5 * (za + zb)
        return h + self.res_scale * base  # [B, D]


class TrimodalMulTHead(nn.Module):
    def __init__(self, d: int, n_heads: int = 8, n_layers: int = 1, dropout: float = 0.1, feature_mode: str = "rich"):
        super().__init__()
        self.feature_mode = feature_mode
        self.l_from_ni = nn.ModuleList([CrossAttentionBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.n_from_li = nn.ModuleList([CrossAttentionBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.i_from_ln = nn.ModuleList([CrossAttentionBlock(d, n_heads, dropout) for _ in range(n_layers)])

        in_dim = 3 * d if feature_mode == "concat" else 7 * d
        self.fuse = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        L: torch.Tensor, mL: torch.Tensor,  
        N: torch.Tensor, mN: torch.Tensor,  
        I: torch.Tensor, mI: torch.Tensor,  
    ) -> torch.Tensor:
        # concat helper
        def cat2(A, mA, B, mB):
            return torch.cat([A, B], dim=1), torch.cat([mA, mB], dim=1)

        NI, mNI = cat2(N, mN, I, mI)
        LI, mLI = cat2(L, mL, I, mI)
        LN, mLN = cat2(L, mL, N, mN)

        HL = L
        for blk in self.l_from_ni:
            HL = blk(HL, NI, key_mask=mNI, query_mask=mL)
        HN = N
        for blk in self.n_from_li:
            HN = blk(HN, LI, key_mask=mLI, query_mask=mN)
        HI = I
        for blk in self.i_from_ln:
            HI = blk(HI, LN, key_mask=mLN, query_mask=mI)

        zL = _masked_mean(HL, mL)
        zN = _masked_mean(HN, mN)
        zI = _masked_mean(HI, mI)

        if self.feature_mode == "concat":
            x = torch.cat([zL, zN, zI], dim=-1)
        else:
            zLN = zL * zN
            zLI = zL * zI
            zNI = zN * zI
            zLNI = zL * zN * zI
            x = torch.cat([zL, zN, zI, zLN, zLI, zNI, zLNI], dim=-1)

        h = self.fuse(x)
        base = (zL + zN + zI) / 3.0
        return h + self.res_scale * base  


class RouteActivation(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))  



@dataclass
class MulTConfig:
    d: int = 256
    dropout: float = 0.1
    n_heads: int = 8
    n_layers_pair: int = 1
    n_layers_tri: int = 1
    feature_mode: str = "rich"  # "rich" | "concat"
    unimodal_pool: Literal["mean", "last"] = "mean"


class MultimodalFeatureExtractor(nn.Module):
    def __init__(self, cfg: MulTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.pair_LN = PairwiseMulTHead(d, cfg.n_heads, cfg.n_layers_pair, cfg.dropout, cfg.feature_mode)
        self.pair_LI = PairwiseMulTHead(d, cfg.n_heads, cfg.n_layers_pair, cfg.dropout, cfg.feature_mode)
        self.pair_NI = PairwiseMulTHead(d, cfg.n_heads, cfg.n_layers_pair, cfg.dropout, cfg.feature_mode)
        self.tri_LNI = TrimodalMulTHead(d, cfg.n_heads, cfg.n_layers_tri, cfg.dropout, cfg.feature_mode)

        # route activations p_i
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
        # Unimodal
        zL = self.unim_ln(self._pool_uni(L_seq, mL))
        zN = self.unim_ln(self._pool_uni(N_seq, mN))
        zI = self.unim_ln(self._pool_uni(I_seq, mI))

        # Bimodal
        zLN = self.pair_LN(L_seq, mL, N_seq, mN)
        zLI = self.pair_LI(L_seq, mL, I_seq, mI)
        zNI = self.pair_NI(N_seq, mN, I_seq, mI)

        # Trimodal
        zLNI = self.tri_LNI(L_seq, mL, N_seq, mN, I_seq, mI)

        route_embs: Dict[str, torch.Tensor] = {
            "L": zL, "N": zN, "I": zI, "LN": zLN, "LI": zLI, "NI": zNI, "LNI": zLNI
        }
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


def build_multimodal_feature_extractor(
    d: int,
    dropout: float = 0.1,
    n_heads: int = 8,
    n_layers_pair: int = 1,
    n_layers_tri: int = 1,
    feature_mode: str = "rich",
    unimodal_pool: Literal["mean", "last"] = "mean",
) -> MultimodalFeatureExtractor:
    cfg = MulTConfig(
        d=d,
        dropout=dropout,
        n_heads=n_heads,
        n_layers_pair=n_layers_pair,
        n_layers_tri=n_layers_tri,
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
    xL: torch.Tensor,               # [B, TL, F]
    notes_list: Union[List[str], List[List[str]]],
    imgs: Union[List[torch.Tensor], List[List[torch.Tensor]]],
    mL: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Convenience: from raw batch inputs → route embeddings + p_i.
    - Builds sequences + masks for each modality
    - Runs MulT-style extractor
    """
    dev = next(extractor.parameters()).device

    L_seq, mL_seq = behrt.encode_seq(xL.to(dev), mask=mL.to(dev) if mL is not None else None)  
    N_seq, mN_seq = bbert.encode_seq(notes_list)  
    I_seq, mI_seq = imgenc.encode_seq(imgs)       

    return extractor(L_seq, mL_seq, N_seq, mN_seq, I_seq, mI_seq)


__all__ = [
    # Encoders
    "BEHRTLabEncoder",
    "BioClinBERTEncoder",
    "ImageEncoder",
    # MulT-style heads
    "MulTConfig",
    "MultimodalFeatureExtractor",
    "build_multimodal_feature_extractor",
    "encode_all_routes_from_batch",
    # Config & builders
    "EncoderConfig",
    "build_encoders",
]
