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
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

def _ensure_2d_mask(
    mask: Optional[torch.Tensor],
    B: int,
    T: int,
    device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(B, T, device=device, dtype=torch.float32)
    if mask.dim() == 1:
        return mask.unsqueeze(0).expand(B, -1).contiguous().float()
    return mask.float()

# Structured encoder (BEHRT-style)
class BEHRTLabEncoder(nn.Module):
    def __init__(
        self,
        n_feats: int,
        d: int,
        seq_len: int = 48,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,  
        pool: Literal["last", "mean", "cls"] = "cls",
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.out_dim = d
        self.input_proj = nn.Linear(n_feats, d)
        self.max_seq_len = int(seq_len)
        if self.max_seq_len <= 0:
            raise ValueError(f"BEHRTLabEncoder seq_len must be > 0, got {seq_len}.")
        self.pos = nn.Parameter(torch.randn(1, self.max_seq_len, d) * 0.02)
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

    def _pos(self, T: int) -> torch.Tensor:
        if T > self.pos.size(1):
            raise ValueError(
                f"Input length T={T} exceeds max_seq_len={self.pos.size(1)}. "
                "Increase structured_seq_len (max_seq_len) in config/build_encoders."
            )
        return self.pos[:, :T, :]


    def _encode_with_optional_cls(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T, F = x.shape
        dev = x.device
        assert self.input_proj.in_features == F, (
            f"Expected F={self.input_proj.in_features}, got F={F}"
        )
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
            _dbg(f"[BEHRTLabEncoder] encode_seq -> h:{tuple(h.shape)} mask:{tuple(m_out.shape)}")
            _peek_tensor("behrt.h", h)
        return h, m_out

    def encode_seq_and_pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, T, _ = x.shape
        dev = next(self.parameters()).device
        m = _ensure_2d_mask(mask, B, T, dev)
        seq_h, m_out, cls_vec = self._encode_with_optional_cls(x.to(dev), m.to(dev))
        if self.pool == "cls":
            z = cls_vec
        elif self.pool == "last":
            if (m_out.sum(dim=1) != m_out.size(1)).any():
                idx = (m_out.sum(dim=1) - 1).clamp_min(0).long()
                z = seq_h[torch.arange(seq_h.size(0), device=seq_h.device), idx]
            else:
                z = seq_h[:, -1]
        else:
            z = _masked_mean(seq_h, m_out)

        if not self._warned_dead:
            with torch.no_grad():
                if z.abs().mean().item() < 1e-6:
                    print("[warn:BEHRTLabEncoder] near-zero pooled embedding; check mask/inputs.")
                    self._warned_dead = True
        return seq_h, m_out, z

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, z = self.encode_seq_and_pool(x, mask=mask)
        return z


# Bio-ClinicalBERT encoder (text)
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
        self.chunk_bs = int(getattr(CFG, "bert_chunk_bs", 8))
        self.chunk_bs = max(1, self.chunk_bs)
        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(model_name)
            hidden = int(getattr(self.bert.config, "hidden_size", 768))
            self.hf_available = True
        except Exception as e:
            if force_hf:
                raise RuntimeError(
                    f"Failed to load '{model_name}'. Install transformers & cache model. Error: {e}"
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
        self.drop = nn.Identity()
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
                return True
            a = item[0]
            return (
                isinstance(a, (tuple, list))
                and len(a) == 2
                and torch.is_tensor(a[0])
                and torch.is_tensor(a[1])
            )
        return False

    def _normalize_batch(self, notes_or_chunks):
        if len(notes_or_chunks) == 0:
            return []
        first = notes_or_chunks[0]
        if self._is_pretok_item(first):
            return notes_or_chunks
        raise ValueError(
            "BioClinBERTEncoder requires pre-tokenized inputs: "
            "dict {'input_ids':[S,L],'attention_mask':[S,L]} "
            "or list of (ids[L], attn[L]) chunks."
        )

    def _encode_chunks_to_cls(
        self,
        ids: torch.Tensor,
        attn: torch.Tensor,
    ) -> torch.Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)

        if not self.hf_available or self.bert is None:
            return torch.zeros(ids.size(0), self.out_dim, device=self._device())

        dev = self._device()
        if next(self.bert.parameters()).device != dev:
            self.bert.to(dev)

        ids = ids.to(device=dev, dtype=torch.long)
        attn = attn.to(device=dev, dtype=torch.long)

        S = ids.size(0)

        chunk_bs = int(getattr(self, "chunk_bs", 8))   
        chunk_bs = max(1, chunk_bs)

        outs = []
        for s0 in range(0, S, chunk_bs):
            s1 = min(S, s0 + chunk_bs)

            out = self.bert(
                input_ids=ids[s0:s1],
                attention_mask=attn[s0:s1],
            )
            cls = out.last_hidden_state[:, 0]  # [bs, hidden]
            cls = self.proj(cls)               # [bs, out_dim]
            cls = self.drop(cls)
            outs.append(cls)
        return torch.cat(outs, dim=0)  # [S, out_dim]

    def encode_seq(self, notes_or_chunks):
        if isinstance(notes_or_chunks, dict) and "input_ids" in notes_or_chunks and "attention_mask" in notes_or_chunks:
            ids  = notes_or_chunks["input_ids"]         # [B,S,L]
            attn = notes_or_chunks["attention_mask"]    # [B,S,L]
            cmask = notes_or_chunks.get("chunk_mask", None)  # [B,S] optional

            ids  = ids.to(next(self.bert.parameters()).device)
            attn = attn.to(next(self.bert.parameters()).device)
            B, S, L = ids.shape

            # flatten chunks -> [B*S, L]
            ids2  = ids.reshape(B * S, L)
            attn2 = attn.reshape(B * S, L)

            # Encode CLS for each chunk
            cls2 = self._encode_chunks_to_cls(ids2, attn2)     # [B*S, D]
            D = cls2.size(-1)

            # reshape back -> [B, S, D]
            H = cls2.view(B, S, D)

            # Chunk-level mask [B,S]
            if cmask is not None:
                M = cmask.to(H.device).float()
            else:
                # valid chunk if it has any attended tokens
                M = (attn.sum(dim=-1) > 0).to(H.device).float()

            return H, M


    def encode_seq_and_pool(self, notes_or_chunks):
        H, M = self.encode_seq(notes_or_chunks)
        z = self.pool_from_seq(H, M)
        return H, M, z

    @staticmethod
    def pool_from_seq(H: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        denom = M.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (H * M.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, notes_or_chunks) -> torch.Tensor:
        requires_grad = bool(getattr(CFG, "finetune_text", False))
        with torch.set_grad_enabled(requires_grad):
            H, M = self.encode_seq(notes_or_chunks)
            z = self.pool_from_seq(H, M)
        return z

# MedFuse-style image encoder 
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
        self.vision_backbone = getattr(torchvision.models, vision_backbone)(pretrained=pretrained)
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
                    f"Cannot infer in_features from {classifier} of {type(self.vision_backbone).__name__}"
                )
            setattr(self.vision_backbone, classifier, nn.Identity())
            break

        if d_visual is None:
            raise ValueError(f"Unsupported backbone {vision_backbone} (no fc/classifier head found).")

        self.bce_loss = nn.BCELoss(reduction="mean")
        self.classifier = nn.Sequential(nn.Linear(d_visual, vision_num_classes))
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
        visual_feats = self.vision_backbone(x)   # [B, D_vis]
        logits = self.classifier(visual_feats)   # [B, C]
        preds = torch.sigmoid(logits)            # [B, C]
        if n_crops and n_crops > 0:
            if bs is None:
                if preds.size(0) % n_crops != 0:
                    raise ValueError("When n_crops>0, pass bs or ensure B % n_crops == 0.")
                bs = preds.size(0) // n_crops
            preds = preds.view(bs, n_crops, -1).mean(dim=1)

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
        self.proj = nn.Linear(self.medfuse.feats_dim, d)
        self.out_dim = int(d)
        self.drop = nn.Identity()
        self.token_in_dim = self._infer_resnet_layer4_channels()
        self.token_proj = nn.Linear(self.token_in_dim, d, bias=False)
        self.to(dev)

    def _infer_resnet_layer4_channels(self) -> int:
        m = self.medfuse.vision_backbone
        if not hasattr(m, "layer4"):
            raise ValueError(
                "encode_seq_and_pool currently supports torchvision ResNet backbones "
                "because it uses .layer1..layer4. If you want DenseNet/EfficientNet, "
                "we need a different token extraction path."
            )

        layer4 = m.layer4
        last_block = list(layer4.children())[-1]
        if hasattr(last_block, "conv3"):   
            return int(last_block.conv3.out_channels)
        if hasattr(last_block, "conv2"):   
            return int(last_block.conv2.out_channels)

        last_conv = None
        for mod in reversed(list(last_block.modules())):
            if isinstance(mod, nn.Conv2d):
                last_conv = mod
                break
        if last_conv is None:
            raise ValueError("Could not infer layer4 output channels.")
        return int(last_conv.out_channels)

    def _encode_batch_feats(self, x: torch.Tensor) -> torch.Tensor:
        I_pool, _fmap = self._encode_pool_and_layer4_once(x)
        _peek_tensor("imgenc.z", I_pool)
        return I_pool

    def _encode_pool_and_layer4_once(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.medfuse.vision_backbone
        if not hasattr(m, "layer4"):
            raise ValueError("Hook path requires torchvision ResNet with .layer4")
        holder: Dict[str, torch.Tensor] = {}
        def _hook(_module, _inp, out):
            holder["fmap"] = out
        h = m.layer4.register_forward_hook(_hook)
        try:
            _, _, feats = self.medfuse(x)  
        finally:
            h.remove()

        if "fmap" not in holder:
            raise RuntimeError("Failed to capture layer4 fmap via hook.")

        fmap = holder["fmap"]  # [B,C,H,W]
        I_pool = self.drop(self.proj(feats))  # [B,d]
        return I_pool, fmap

    def medfuse_forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        n_crops: int = 0,
        bs: Optional[int] = None,
    ):
        return self.medfuse(x, labels=labels, n_crops=n_crops, bs=bs)


    def _as_batch_tensor(
        self,
        batch_images: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(batch_images, torch.Tensor):
            x = batch_images
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.dim() != 4:
                raise ValueError("Tensor input must be [3,H,W] or [B,3,H,W].")
            return x.to(device)

        if isinstance(batch_images, list) and (len(batch_images) == 0 or isinstance(batch_images[0], torch.Tensor)):
            if len(batch_images) == 0:
                return torch.zeros(0, 3, 224, 224, device=device)
            return torch.stack(batch_images, dim=0).to(device)
        xs: List[torch.Tensor] = []
        for imgs in batch_images:
            if imgs is None or len(imgs) == 0:
                xs.append(torch.zeros(3, 224, 224))
            else:
                xs.append(imgs[-1])
        return torch.stack(xs, dim=0).to(device)

    def encode_seq_and_pool(
        self,
        batch_images: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        x = self._as_batch_tensor(batch_images, device=device)

        if x.numel() == 0:
            I_seq = torch.zeros(0, 1, self.proj.out_features, device=device)
            I_mask = torch.zeros(0, 1, device=device)
            I_pool = torch.zeros(0, self.proj.out_features, device=device)
            return I_seq, I_mask, I_pool

        I_pool, fmap = self._encode_pool_and_layer4_once(x)  # [B,d], [B,C,H,W]
        B, C, H, W = fmap.shape
        if C != self.token_in_dim:
            raise ValueError(
                f"Layer4 channels mismatch: got C={C}, expected {self.token_in_dim}. "
                f"Backbone={type(self.medfuse.vision_backbone).__name__}."
            )
        tokens = fmap.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,P,C]
        I_seq = self.token_proj(tokens)                          # [B,P,d]
        I_mask = torch.ones(B, H * W, device=device)             # [B,P]

        _peek_tensor("imgenc.I_seq", I_seq)
        _peek_tensor("imgenc.I_pool", I_pool)
        return I_seq, I_mask, I_pool


    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        xb = self._as_batch_tensor(x, device=device)
        if xb.numel() == 0:
            return torch.zeros(0, self.proj.out_features, device=device)
        return self._encode_batch_feats(xb)

    def encode_seq(
        self,
        batch_images: Union[List[torch.Tensor], List[List[torch.Tensor]], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        Z = self.forward(batch_images)
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)
        B = Z.size(0)
        Hpad = Z.unsqueeze(1)  # [B,1,d]
        M = torch.ones(B, 1, device=device)
        _peek_tensor("imgenc.Hpad", Hpad)
        return Hpad, M

    def load_backbone_state(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
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
        temp = float(getattr(CFG, "route_gate_temp", 1.0))
        gmin = float(getattr(CFG, "route_gate_min", 0.0))
        gmax = float(getattr(CFG, "route_gate_max", 1.0))
        self.act_L = RouteActivation(d, temp=temp, gate_min=gmin, gate_max=gmax)
        self.act_N = RouteActivation(d, temp=temp, gate_min=gmin, gate_max=gmax)
        self.act_I = RouteActivation(d, temp=temp, gate_min=gmin, gate_max=gmax)
        self.act_LN = RouteActivation(d, temp=temp, gate_min=gmin, gate_max=gmax)
        self.act_LI = RouteActivation(d, temp=temp, gate_min=gmin, gate_max=gmax)
        self.act_NI = RouteActivation(d, temp=temp, gate_min=gmin, gate_max=gmax)
        self.act_LNI = RouteActivation(d, temp=temp, gate_min=gmin, gate_max=gmax)
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

        zL = self.unim_ln(self._pool_uni(L_seq, mL))
        zN = self.unim_ln(self._pool_uni(N_seq, mN))
        zI = self.unim_ln(self._pool_uni(I_seq, mI))

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
    structured_seq_len: int = 256
    structured_n_feats: int = 61
    structured_layers: int = 2
    structured_heads: int = 8
    structured_pool: Literal["last", "mean", "cls"] = "cls"

    # notes
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_max_len: int = 512
    note_agg: Literal["mean", "attention"] = "attention"
    bert_chunk_bs: int = 8

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
    seq_len = cfg.structured_seq_len
    if seq_len <= 0:
        seq_len = int(getattr(CFG, "structured_pos_max_len", 2048))

    behrt = BEHRTLabEncoder(
        n_feats=cfg.structured_n_feats,
        d=cfg.d,
        seq_len=seq_len,
        n_layers=cfg.structured_layers,
        n_heads=cfg.structured_heads,
        dropout=0.0,
        pool=cfg.structured_pool,
    ).to(dev)
    CFG.bert_chunk_bs = int(getattr(cfg, "bert_chunk_bs", getattr(CFG, "bert_chunk_bs", 8)))

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

NoteItem = Union[
    Dict[str, torch.Tensor],              # {"input_ids":[S,L], "attention_mask":[S,L]}
    List[Tuple[torch.Tensor, torch.Tensor]]  # [(ids[L], attn[L]), ...]
]
BatchNotes = List[NoteItem]

def encode_modalities_for_routing(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: "ImageEncoder",
    xL: torch.Tensor,
    notes_list,
    imgs,
    mL: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    dev = next(behrt.parameters()).device

    # L (single pass: seq + mask + pool)
    mL_dev = (mL.to(dev) if mL is not None else None)
    L_seq, L_mask, L_pool = behrt.encode_seq_and_pool(xL.to(dev), mask=mL_dev)

    # N (single pass through BERT)
    N_seq, N_mask = bbert.encode_seq(notes_list)
    N_pool = bbert.pool_from_seq(N_seq, N_mask)

    # I
    imgs_dev = imgs.to(dev) if isinstance(imgs, torch.Tensor) else imgs
    I_seq, I_mask, I_pool = imgenc.encode_seq_and_pool(imgs_dev)

    return {
        "L": {"seq": L_seq, "pool": L_pool, "mask": L_mask},
        "N": {"seq": N_seq, "pool": N_pool, "mask": N_mask},
        "I": {"seq": I_seq, "pool": I_pool, "mask": I_mask},
    }


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

    dev = next(behrt.parameters()).device

    mL_dev = (mL.to(dev) if mL is not None else None)
    _, _, zL = behrt.encode_seq_and_pool(xL.to(dev), mask=mL_dev)

    zN = bbert(notes_list)
    zI = imgenc(imgs.to(dev) if isinstance(imgs, torch.Tensor) else imgs)

    return {"L": zL, "N": zN, "I": zI}


__all__ = [
    "BEHRTLabEncoder",
    "BioClinBERTEncoder",
    "MedFuseImageEncoder",
    "ImageEncoder",
    "MulTConfig",
    "MultimodalFeatureExtractor",
    "build_multimodal_feature_extractor",
    "encode_modalities_for_routing",
    "encode_unimodal_pooled",
    "EncoderConfig",
    "build_encoders",
]
