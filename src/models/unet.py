from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2d(nn.Module):
    def __init__(self, t_in: int, t_out: int, width: int = 32, depth: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        # depth controls the number of resolution levels; cap for small images.
        levels = max(2, min(depth, 5))
        channels = [width * (2 ** i) for i in range(levels)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = t_in
        for ch in channels:
            self.downs.append(DoubleConv(in_ch, ch, dropout))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch
        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2, dropout)
        self.up_trans = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        cur = channels[-1] * 2
        for ch in reversed(channels):
            self.up_trans.append(nn.ConvTranspose2d(cur, ch, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(ch * 2, ch, dropout))
            cur = ch
        self.head = nn.Conv2d(channels[0], t_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x
        for down, pool in zip(self.downs, self.pools):
            out = down(out)
            skips.append(out)
            out = pool(out)
        out = self.bottleneck(out)
        for up, conv, skip in zip(self.up_trans, self.up_convs, reversed(skips)):
            out = up(out)
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = conv(out)
        return self.head(out)
