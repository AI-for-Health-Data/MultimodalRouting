from __future__ import annotations
import torch.nn as nn
__all__ = ["RouteMLP"]


class RouteMLP(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        width_mul: float = 2.0,
        dropout_p: float = 0.10,
    ) -> None:
        super().__init__()

        hidden = int(in_dim * width_mul)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, x):
        return self.net(x)

