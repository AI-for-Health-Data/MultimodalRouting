from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["RouteMLP"]


class RouteMLP(nn.Module):
    """
    A simple multi-layer perceptron for routing or transforming feature vectors.

    Architecture:
      1) LayerNorm over the last dimension
      2) Linear(in_dim → hidden_dim)
      3) GELU activation
      4) Dropout
      5) Linear(hidden_dim → out_dim)

    Args:
        in_dim (int):   Size of each input feature vector.
        out_dim (int):  Size of each output feature vector.
        width_mul (float): Hidden layer size will be in_dim * width_mul.
        dropout_p (float): Dropout probability after the activation.
    """
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        width_mul: float = 2.0,
        dropout_p: float = 0.10,
    ) -> None:
        super().__init__()
        hidden_dim = int(in_dim * width_mul)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
