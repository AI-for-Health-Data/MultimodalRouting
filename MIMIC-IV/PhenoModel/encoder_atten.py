# encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Literal,
    Tuple,
    Union,
    Dict,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from env_config import CFG, DEVICE


# =========================
# Debug helpers
# =========================
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


# =========================
# Mask utils
# =========================
def _make_pad_mask(mask_bt: torch.Tensor) -> torch.Tensor:
    """
    mask_bt: [B,T] float/bool with 1=valid timestep, 0=pad
    returns: src_key_padding_mask [B,T] bool with True=PAD (ignore)
    """
    if mask_bt.dtype != torch.bool:
        mask_bt = mask_bt > 0.5
    return ~mask_bt  # True where PAD


def _infer_mask_from_x(x: torch.Tensor) -> torch.Tensor:
    """
    Robust default for structured sequences:
    timestep valid if ANY feature is non-zero.
    x: [B,T,F] -> mask: [B,T] float(0/1)
    """
    return (x.abs().sum(dim=-1) > 0).float()

def _ensure_bt_mask(
    mask: Optional[torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Ensure mask is [B,T] float with 1=valid.
    Accepts:
      - None -> infer from x (recommended)
      - [B,T]
      - [T]  -> broadcast across batch
      - [B]  -> NOT valid for timestep mask (raise)
    """
    B, T = x.shape[0], x.shape[1]
    dev = x.device

    if mask is None:
        return _infer_mask_from_x(x).to(dev)

    mask = mask.to(dev)
    if mask.dim() == 2:
        if mask.shape[0] != B or mask.shape[1] != T:
            raise ValueError(f"mask must be [B,T]={B,T}, got {tuple(mask.shape)}")
        return mask.float()

    if mask.dim() == 1:
        if mask.shape[0] == T:
            return mask.unsqueeze(0).expand(B, -1).contiguous().float()
        if mask.shape[0] == B:
            raise ValueError(
                f"mask has shape [B]={B} but must be [B,T] timestep mask; "
                "pass lengths -> build [B,T] mask in collate."
            )
        raise ValueError(f"mask dim=1 must be [T] or [B], got {tuple(mask.shape)}")

    raise ValueError(f"mask must be None, [B,T], or [T]; got dim={mask.dim()}")

# ============================================================
# L encoder (Structured) -> returns seq [B,T,D], mask [B,T], pool [B,D]
# ============================================================
class BEHRTLabEncoder(nn.Module):
    def __init__(
        self,
        n_feats_in: int,
        d: int,
        seq_len: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        pool: Literal["last", "mean", "cls"] = "cls",
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        super().__init__()

        self.pool = pool
        self.out_dim = int(d)

        self.n_feats_raw = int(n_feats_in)

        # If you *intend* n_feats_in to be raw features, we build 2F model input (values + obs_mask).
        # If someone passes already-prepared Z, we also allow that at runtime.
        self.add_obs = True
        self.n_feats_model = self.n_feats_raw * 2 if self.add_obs else self.n_feats_raw

        self.input_proj = nn.Linear(self.n_feats_model, d)

        pos_max = max(16, int(getattr(CFG, "structured_pos_max_len", seq_len)))
        self.time = nn.Embedding(pos_max, d)
        nn.init.normal_(self.time.weight, mean=0.0, std=0.02)

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
    def _add_time(self, H: torch.Tensor, hour_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Add timestep embedding.
        If hour_idx is None -> use positions 0..T-1.
        hour_idx can be [B,T] or [T].
        """
        B, T, D = H.shape
        dev = H.device

        if hour_idx is None:
            t = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
        else:
            t = hour_idx.to(dev)
            if t.dim() == 1 and t.numel() == T:
                t = t.unsqueeze(0).expand(B, -1)
            if t.dim() != 2 or t.shape[0] != B or t.shape[1] != T:
                raise ValueError(f"hour_idx must be [B,T] or [T]; got {tuple(t.shape)}")

            t = t.long()

        t = t.clamp(min=0, max=self.time.num_embeddings - 1)
        return H + self.time(t)

    def _pool(self, seq_h: torch.Tensor, mask_bt: torch.Tensor, cls_vec: Optional[torch.Tensor]) -> torch.Tensor:
        """
        seq_h: [B,T,D]
        mask_bt: [B,T] with 1=valid, 0=pad
        """
        if self.pool == "cls":
            if cls_vec is None:
                raise ValueError("pool='cls' requires cls_vec")
            return cls_vec

        mask_bt = _ensure_bt_mask(mask_bt, seq_h)  # float [B,T]
        B, T, D = seq_h.shape

        if self.pool == "mean":
            denom = mask_bt.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (seq_h * mask_bt.unsqueeze(-1)).sum(dim=1) / denom

        if self.pool == "last":
            lengths = mask_bt.sum(dim=1).clamp_min(1.0).long()  # [B]
            idx = (lengths - 1).clamp_min(0)                    # [B]
            return seq_h[torch.arange(B, device=seq_h.device), idx]

        raise ValueError(f"Unknown pool='{self.pool}'")

    def _prepare_Z(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - raw continuous with NaNs: [B,T,Fraw]  -> returns [B,T,2*Fraw]
          - already-prepared:        [B,T,Fin] where Fin == input_proj.in_features
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        B, T, Fin = x.shape
        expected = int(self.input_proj.in_features)

        # Case A: already matches model input
        if Fin == expected:
            return x

        # Case B: raw -> expand if model expects doubled input
        if self.add_obs and expected == 2 * Fin:
            M = (~torch.isnan(x)).float()           # [B,T,Fin]
            x_filled = torch.nan_to_num(x, nan=0.0) # [B,T,Fin]
            return torch.cat([x_filled, M], dim=-1) # [B,T,2*Fin]

        raise ValueError(
            f"Structured input feature mismatch: got Fin={Fin}, "
            f"but model expects {expected}. "
            f"Pass raw [B,T,F] (with NaNs if available) OR prepared [B,T,{expected}]."
        )


    def forward(
        self,
        x: torch.Tensor,                   # [B,T,17] raw w/ NaN OR [B,T,34]/[B,T,51]
        mask_bt: torch.Tensor,             # [B,T] 1=valid timestep, 0=pad  (REQUIRED for correctness)
        hour_idx: Optional[torch.Tensor] = None,  # [B,T] optional
        *,
        return_seq: bool = False,
    ):
        # --- prepare inputs ---
        dev = next(self.parameters()).device
        x = x.to(dev)
        mask_bt = _ensure_bt_mask(mask_bt, x).to(dev)  # <- force [B,T] float

        if (mask_bt.sum(dim=1) == 0).any():
            mask_bt = mask_bt.clone()
            mask_bt[:, -1] = 1.0

        Z = self._prepare_Z(x)  # [B,T,34] or [B,T,51]
        B, T, Fin = Z.shape

        # Z is guaranteed to match input_proj.in_features by _prepare_Z

 
        # --- embed ---
        H = self.input_proj(Z)            # [B,T,D]
        H = self._add_time(H, hour_idx)   # add hour/bin embedding

        # --- add CLS + pad mask ---
        pad_mask = _make_pad_mask(mask_bt)  # [B,T] True=PAD

        if self.pool == "cls":
            cls_tok = self.cls_token.expand(B, 1, -1)
            H_in = torch.cat([cls_tok, H], dim=1)  # [B,T+1,D]
            pad_mask = torch.cat(
                [torch.zeros(B, 1, device=dev, dtype=torch.bool), pad_mask],
                dim=1
            )  # [B,T+1]
        else:
            H_in = H

        # --- encode ---
        out = self.enc(H_in, src_key_padding_mask=pad_mask)
        out = self.out(out)

        if self.pool == "cls":
            cls_vec = out[:, 0, :]      # [B,D]
            seq_h = out[:, 1:, :]       # [B,T,D]
        else:
            cls_vec = None
            seq_h = out                # [B,T,D]

        z = self._pool(seq_h, mask_bt, cls_vec)

        if return_seq:
            return seq_h, mask_bt, z
        return z


# ============================================================
# N encoder (Notes) -> returns seq [B,S,D], mask [B,S], pool [B,D]
# ============================================================
NoteItem = Union[
    Dict[str, torch.Tensor],                  # {"input_ids":[S,L], "attention_mask":[S,L]}
    List[Tuple[torch.Tensor, torch.Tensor]],  # [(ids[L], attn[L]), ...]
]
BatchNotes = List[NoteItem]


class BioClinBERTEncoder(nn.Module):
    """
    Bio-ClinicalBERT encoder (pre-tokenized only).

    encode_seq(...) -> (seq [B,Smax,D], mask [B,Smax])
    forward(...)    -> pooled [B,D] (masked mean over S)
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

        self.hidden = int(hidden)

        if d is not None and int(d) != self.hidden:
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden),
                nn.Linear(self.hidden, int(d), bias=False),
            )
            self.out_dim = int(d)
        else:
            self.proj = nn.Identity()
            self.out_dim = int(self.hidden)

        # keep for compatibility (you currently set drop=Identity)
        self.drop = nn.Identity()

        if self.hf_available and self.bert is not None:
            # Freeze or finetune controlled by CFG.finetune_text
            finetune = bool(getattr(CFG, "finetune_text", False))
            for p in self.bert.parameters():
                p.requires_grad = finetune
            self.bert.train(finetune)


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

    def _encode_chunks_to_cls(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
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

        # Decide once (not inside loop): is BERT trainable?
        bert_trainable = any(p.requires_grad for p in self.bert.parameters())
        ctx = torch.enable_grad if bert_trainable else torch.no_grad

        S = ids.size(0)
        chunk_bs = max(1, int(getattr(self, "chunk_bs", 8)))

        outs = []
        for s0 in range(0, S, chunk_bs):
            s1 = min(S, s0 + chunk_bs)

            # Only the BERT forward is gated
            with ctx():
                out = self.bert(input_ids=ids[s0:s1], attention_mask=attn[s0:s1])

            cls = out.last_hidden_state[:, 0]  # [bs, hidden]
            cls = self.proj(cls)               # proj can still train
            cls = self.drop(cls)
            outs.append(cls)

        return torch.cat(outs, dim=0)  # [S, out_dim]


    def _coerce_ids_attn(self, ids, attn, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accept ids/attn as torch tensors OR python lists.
        Supports:
          - ids: Tensor [S,L] or [L]
          - ids: list[int] or list[list[int]]
          - attn: same shapes/types, or None (will be created)
        Returns:
          ids: LongTensor [S,L]
          attn: LongTensor [S,L]
        """
        pad_id = 0
        tok = getattr(self, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token_id", None) is not None:
            pad_id = int(tok.pad_token_id)

        # ---- ids -> tensor [S,L] ----
        if torch.is_tensor(ids):
            ids_t = ids
            if ids_t.dim() == 1:
                ids_t = ids_t.unsqueeze(0)
            ids_t = ids_t.to(device=dev, dtype=torch.long)
        elif isinstance(ids, list):
            if len(ids) == 0:
                ids_t = torch.full((1, 1), pad_id, dtype=torch.long, device=dev)
            else:
                # list[int] -> [1,L]
                if isinstance(ids[0], int):
                    ids = [ids]

                # ragged? pad it
                is_ragged = any(len(x) != len(ids[0]) for x in ids if isinstance(x, list))
                if is_ragged:
                    from torch.nn.utils.rnn import pad_sequence
                    seqs = [torch.tensor(x, dtype=torch.long) for x in ids]
                    ids_t = pad_sequence(seqs, batch_first=True, padding_value=pad_id).to(dev)
                else:
                    ids_t = torch.tensor(ids, dtype=torch.long, device=dev)
        else:
            raise TypeError(f"Unsupported input_ids type: {type(ids)}")

        # ---- attn -> tensor [S,L] ----
        if attn is None:
            attn_t = (ids_t != pad_id).long()
        elif torch.is_tensor(attn):
            attn_t = attn
            if attn_t.dim() == 1:
                attn_t = attn_t.unsqueeze(0)
            attn_t = attn_t.to(device=dev, dtype=torch.long)
        elif isinstance(attn, list):
            if len(attn) == 0:
                attn_t = torch.zeros_like(ids_t, dtype=torch.long, device=dev)
            else:
                if isinstance(attn[0], int):
                    attn = [attn]

                is_ragged = any(len(x) != len(attn[0]) for x in attn if isinstance(x, list))
                if is_ragged:
                    from torch.nn.utils.rnn import pad_sequence
                    seqs = [torch.tensor(x, dtype=torch.long) for x in attn]
                    attn_t = pad_sequence(seqs, batch_first=True, padding_value=0).to(dev)
                else:
                    attn_t = torch.tensor(attn, dtype=torch.long, device=dev)
        else:
            raise TypeError(f"Unsupported attention_mask type: {type(attn)}")

        # ensure same shape
        if attn_t.shape != ids_t.shape:
            raise ValueError(f"attention_mask shape {attn_t.shape} != input_ids shape {ids_t.shape}")

        return ids_t, attn_t


    def encode_seq(self, notes_or_chunks) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self._device()
        batch = self._normalize_batch(notes_or_chunks)

        seqs: List[torch.Tensor] = []
        lengths: List[int] = []

        for patient in batch:
            collected: List[torch.Tensor] = []

            if isinstance(patient, dict):
                ids, attn = self._coerce_ids_attn(
                    patient["input_ids"],
                    patient.get("attention_mask", None),
                    dev,
                )
                cls = self._encode_chunks_to_cls(ids, attn)

                collected.append(cls)
            else:
                for (ids, attn) in patient:
                    cls = self._encode_chunks_to_cls(ids, attn)

                    collected.append(cls)

            if len(collected) == 0:
                H = torch.zeros(1, self.out_dim, device=dev)
            else:
                H = torch.cat(collected, dim=0)  # [S, D]

            seqs.append(H)
            lengths.append(H.size(0))

        if len(seqs) == 0:
            return (
                torch.zeros(0, 1, self.out_dim, device=dev),
                torch.zeros(0, 1, device=dev),
            )

        Smax = max(lengths)
        B = len(seqs)
        Hpad = torch.zeros(B, Smax, self.out_dim, device=dev)
        M = torch.zeros(B, Smax, device=dev)

        for i, H in enumerate(seqs):
            s = H.size(0)
            Hpad[i, :s] = H
            M[i, :s] = 1.0

        _peek_tensor("bert.N_seq", Hpad)
        _peek_tensor("bert.N_mask", M)
        return Hpad, M

    @staticmethod
    def pool_from_seq(H: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        denom = M.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (H * M.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, notes_or_chunks) -> torch.Tensor:
        H, M = self.encode_seq(notes_or_chunks)
        return self.pool_from_seq(H, M)


# ============================================================
# I encoder (Images) -> returns seq [B,P,D], mask [B,P], pool [B,D]
# ============================================================
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
        fn = getattr(torchvision.models, vision_backbone)
        try:
            weights = "DEFAULT" if pretrained else None
            self.vision_backbone = fn(weights=weights)
        except TypeError:
            self.vision_backbone = fn(pretrained=pretrained)

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

        self.feats_dim = int(d_visual)
        self.vision_num_classes = int(vision_num_classes)
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
    """
    For routing, we need:
      - I_seq: tokens [B,P,D] from ResNet layer4 feature map
      - I_mask: [B,P]
      - I_pool: pooled image vector [B,D]
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
        self.proj = nn.Linear(self.medfuse.feats_dim, int(d))
        self.drop = nn.Identity()

        self.token_in_dim = self._infer_resnet_layer4_channels()
        self.token_proj = nn.Linear(self.token_in_dim, int(d), bias=False)

        self.to(dev)

    def _infer_resnet_layer4_channels(self) -> int:
        m = self.medfuse.vision_backbone
        if not hasattr(m, "layer4"):
            raise ValueError(
                "encode_seq_and_pool currently supports torchvision ResNet backbones "
                "because it uses .layer4 hook."
            )

        layer4 = m.layer4
        last_block = list(layer4.children())[-1]
        if hasattr(last_block, "conv3"):  # Bottleneck (resnet50+)
            return int(last_block.conv3.out_channels)
        if hasattr(last_block, "conv2"):  # BasicBlock (resnet18/34)
            return int(last_block.conv2.out_channels)

        last_conv = None
        for mod in reversed(list(last_block.modules())):
            if isinstance(mod, nn.Conv2d):
                last_conv = mod
                break
        if last_conv is None:
            raise ValueError("Could not infer layer4 output channels.")
        return int(last_conv.out_channels)

    def _encode_pool_and_layer4_once(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.medfuse.vision_backbone
        if not hasattr(m, "layer4"):
            raise ValueError("Hook path requires torchvision ResNet with .layer4")

        holder: Dict[str, torch.Tensor] = {}

        def _hook(_module, _inp, out):
            holder["fmap"] = out

        h = m.layer4.register_forward_hook(_hook)
        try:
            _, _, feats = self.medfuse(x)  # feats: [B, feats_dim]
        finally:
            h.remove()

        if "fmap" not in holder:
            raise RuntimeError("Failed to capture layer4 fmap via hook.")

        fmap = holder["fmap"]                 # [B,C,H,W]
        I_pool = self.drop(self.proj(feats))  # [B,D]
        return I_pool, fmap

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

        I_pool, fmap = self._encode_pool_and_layer4_once(x)  # [B,D], [B,C,H,W]
        B, C, H, W = fmap.shape

        if C != self.token_in_dim:
            raise ValueError(
                f"Layer4 channels mismatch: got C={C}, expected {self.token_in_dim}. "
                f"Backbone={type(self.medfuse.vision_backbone).__name__}."
            )

        tokens = fmap.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,P,C]
        I_seq = self.token_proj(tokens)                          # [B,P,D]
        I_mask = torch.ones(B, H * W, device=device)            # [B,P]

        _peek_tensor("img.I_seq", I_seq)
        _peek_tensor("img.I_pool", I_pool)
        return I_seq, I_mask, I_pool

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]) -> torch.Tensor:
        device = next(self.parameters()).device
        xb = self._as_batch_tensor(x, device=device)
        if xb.numel() == 0:
            return torch.zeros(0, self.proj.out_features, device=device)
        I_pool, _ = self._encode_pool_and_layer4_once(xb)
        return I_pool


# ============================================================
# Minimal builder configs
# ============================================================
@dataclass
class EncoderConfig:
    d: int = 256
    dropout: float = 0.0

    # structured
    structured_seq_len: int = 48
    structured_n_feats: int = 17
    structured_layers: int = 2
    structured_heads: int = 8
    structured_pool: Literal["last", "mean", "cls"] = "cls"

    # notes
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"

    # images
    vision_backbone: str = "resnet34"
    vision_num_classes: int = 14
    vision_pretrained: bool = True


def build_encoders(
    cfg: EncoderConfig,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder]:
    dev = torch.device(DEVICE if device is None else device)

    seq_len = int(cfg.structured_seq_len)
    if seq_len <= 0:
        seq_len = int(getattr(CFG, "structured_pos_max_len", 2048))

    behrt = BEHRTLabEncoder(
        n_feats_in=int(cfg.structured_n_feats),
        d=int(cfg.d),
        seq_len=seq_len,
        n_layers=int(cfg.structured_layers),
        n_heads=int(cfg.structured_heads),
        pool=cfg.structured_pool,
    ).to(dev)

    bbert = BioClinBERTEncoder(
        model_name=str(cfg.text_model_name),
        d=int(cfg.d),
        dropout=float(cfg.dropout),
    ).to(dev)

    if getattr(bbert, "hf_available", False) and bbert.bert is not None:
        bbert.bert.to(dev)
        if not getattr(CFG, "finetune_text", False):
            bbert.bert.eval()

    imgenc = ImageEncoder(
        d=int(cfg.d),
        dropout=0.0,
        img_agg="last",
        vision_backbone=str(cfg.vision_backbone),
        vision_num_classes=int(cfg.vision_num_classes),
        pretrained=bool(cfg.vision_pretrained),
        device=dev,
    ).to(dev)

    return behrt, bbert, imgenc


# ============================================================
# ONLY thing routing needs: unimodal seq/mask/pool dict
# ============================================================
def encode_modalities_for_routing(
    behrt: BEHRTLabEncoder,
    bbert: BioClinBERTEncoder,
    imgenc: ImageEncoder,
    xL: torch.Tensor,
    notes_list: BatchNotes,
    imgs: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
    mL: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Returns dict used by routing_and_heads.forward_capsule_from_routes / MulTRouteBuilder:

      {
        "L": {"seq":[B,TL,D], "mask":[B,TL], "pool":[B,D]},
        "N": {"seq":[B,TN,D], "mask":[B,TN], "pool":[B,D]},
        "I": {"seq":[B,TI,D], "mask":[B,TI], "pool":[B,D]},
      }

    IMPORTANT:
      - seq dims are all D == cfg.d
      - masks are float with 1=valid, 0=pad
    """
    dev = next(behrt.parameters()).device

    # L (structured)
    if mL is None:
        raise ValueError("mL (structured timestep mask [B,T]) is required for BEHRTLabEncoder.")
    L_seq, L_mask, L_pool = behrt(
        xL.to(dev),
        mask_bt=mL.to(dev),
        hour_idx=None,          # or pass your hour/bin indices if you have them
        return_seq=True,
    )
    L_mask = L_mask.float()


    # N (notes)
    N_seq, N_mask = bbert.encode_seq(notes_list)  # [B,S,D], [B,S]
    N_pool = bbert.pool_from_seq(N_seq, N_mask)
    N_mask = N_mask.float()

    # I (images)
    imgs_dev = imgs.to(dev) if isinstance(imgs, torch.Tensor) else imgs
    I_seq, I_mask, I_pool = imgenc.encode_seq_and_pool(imgs_dev)
    I_mask = I_mask.float()

    # Optional sanity checks
    if getattr(CFG, "verbose", False):
        _peek_tensor("uni.L_seq", L_seq); _peek_tensor("uni.L_mask", L_mask); _peek_tensor("uni.L_pool", L_pool)
        _peek_tensor("uni.N_seq", N_seq); _peek_tensor("uni.N_mask", N_mask); _peek_tensor("uni.N_pool", N_pool)
        _peek_tensor("uni.I_seq", I_seq); _peek_tensor("uni.I_mask", I_mask); _peek_tensor("uni.I_pool", I_pool)

    return {
        "L": {"seq": L_seq, "mask": L_mask, "pool": L_pool},
        "N": {"seq": N_seq, "mask": N_mask, "pool": N_pool},
        "I": {"seq": I_seq, "mask": I_mask, "pool": I_pool},
    }


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
    Convenience if you ever need pooled-only:
      {"L": [B,D], "N": [B,D], "I": [B,D]}
    """
    dev = next(behrt.parameters()).device
    if mL is None:
        raise ValueError("mL is required.")
    _, _, zL = behrt(
        xL.to(dev),
        mask_bt=mL.to(dev),
        hour_idx=None,
        return_seq=True,
    )
    zN = bbert(notes_list)
    zI = imgenc(imgs.to(dev) if isinstance(imgs, torch.Tensor) else imgs)
    return {"L": zL, "N": zN, "I": zI}


__all__ = [
    "BEHRTLabEncoder",
    "BioClinBERTEncoder",
    "MedFuseImageEncoder",
    "ImageEncoder",
    "EncoderConfig",
    "build_encoders",
    "NoteItem",
    "BatchNotes",
    "encode_modalities_for_routing",
    "encode_unimodal_pooled",
]
