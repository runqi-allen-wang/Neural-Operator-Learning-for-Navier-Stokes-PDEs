from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class CNNBaseline(nn.Module):
    def __init__(self, t_in: int, t_out: int, width: int = 64, depth: int = 6, dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Conv2d(t_in, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.GELU(), nn.Conv2d(width, t_out, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
