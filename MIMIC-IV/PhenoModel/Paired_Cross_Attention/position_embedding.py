import math
from typing import Optional, Dict

import torch
import torch.nn as nn

def _device_key(t: torch.Tensor) -> str:
    if t.device.type == "cuda":
        return f"cuda:{t.device.index}"
    return "cpu"

def make_positions(tensor: torch.Tensor, padding_idx: int, left_pad: bool) -> torch.Tensor:
    """
    Build a [bsz, seqlen] LongTensor of position ids.
    If tensor is not integer token ids (e.g., float features), we treat everything as non-padding.
    """
    bsz, seqlen = tensor.size()
    device = tensor.device

    # If tensor is integer-like, use padding mask; otherwise assume no padding.
    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        mask = tensor.ne(padding_idx)
    else:
        mask = torch.ones((bsz, seqlen), device=device, dtype=torch.bool)

    # positions: [1..seqlen] shifted by padding_idx
    positions = torch.arange(
        padding_idx + 1,
        padding_idx + 1 + seqlen,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).expand(bsz, seqlen)

    if left_pad:
        nonpad = mask.long().sum(dim=1, keepdim=True)
        positions = positions - seqlen + nonpad

    # output is ALWAYS long
    out = torch.full((bsz, seqlen), padding_idx, device=device, dtype=torch.long)
    out[mask] = positions[mask]
    return out

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Produces sinusoidal positional embeddings of any length.

    Input:  [bsz, seqlen] token indices (with padding_idx as padding)
    Output: [bsz, seqlen, embedding_dim]
    """

    def __init__(self, embedding_dim: int, padding_idx: int = 0, left_pad: bool = False, init_size: int = 128):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = int(padding_idx)
        self.left_pad = bool(left_pad)

        # Cache per device to avoid recomputing for every forward
        self.weights: Dict[str, torch.Tensor] = {}

        # A tiny buffer used only for dtype/device casting
        self.register_buffer("_float_tensor", torch.FloatTensor(1), persistent=False)

        # (Optional) warm cache for CPU to init_size
        if init_size is not None and init_size > 0:
            w = self.get_embedding(init_size + self.padding_idx + 1, self.embedding_dim, self.padding_idx)
            self.weights["cpu"] = w

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None) -> torch.Tensor:
        """
        Build sinusoidal embeddings. Shape: [num_embeddings, embedding_dim]
        """
        half_dim = embedding_dim // 2
        if half_dim == 0:
            raise ValueError(f"embedding_dim must be >= 2 for sinusoidal embeddings, got {embedding_dim}")

        # Avoid division by zero when half_dim == 1
        if half_dim == 1:
            emb = torch.ones(1, dtype=torch.float32)
        else:
            emb = math.log(10000.0) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-emb))

        pos = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1)  # [num_embeddings,1]
        emb = pos * emb.unsqueeze(0)  # [num_embeddings, half_dim]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [num_embeddings, 2*half_dim]
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1, dtype=torch.float32)], dim=1)

        if padding_idx is not None:
            emb[padding_idx, :] = 0.0
        return emb

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"Expected input of shape [bsz, seqlen], got {tuple(input.shape)}")

        bsz, seqlen = input.size()
        max_pos = self.padding_idx + 1 + seqlen
        key = _device_key(input)

        w = self.weights.get(key, None)
        if w is None or w.size(0) < max_pos:
            w = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
            self.weights[key] = w  # still CPU float32 here

        if w.device != input.device:
            w = w.to(device=input.device)

        w = w.to(dtype=input.dtype)

        self.weights[key] = w

        positions = make_positions(input, self.padding_idx, self.left_pad) 
        out = w.index_select(0, positions.reshape(-1)).view(bsz, seqlen, -1)
        return out.detach()



    def max_positions(self) -> int:
        return int(1e5)
