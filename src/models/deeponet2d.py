from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def make_mlp(in_dim: int, hidden: int, out_dim: int, depth: int, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    cur = in_dim
    for _ in range(max(depth - 1, 1)):
        layers += [nn.Linear(cur, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        cur = hidden
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class CNNBranch(nn.Module):
    def __init__(self, in_channels: int, latent: int, t_out: int, width: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.GELU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = make_mlp(2 * width * 4 * 4, 2 * width, t_out * latent, depth=2, dropout=dropout)
        self.t_out = t_out
        self.latent = latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        z = self.net(x)
        z = self.head(z)
        return z.view(b, self.t_out, self.latent)


class MLPBranch(nn.Module):
    def __init__(self, in_channels: int, latent: int, t_out: int, hidden: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.t_out = t_out
        self.latent = latent
        # LazyLinear allows different resolutions without changing config.
        layers = [nn.Flatten(), nn.LazyLinear(hidden), nn.GELU()]
        for _ in range(max(depth - 2, 0)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, t_out * latent))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        z = self.net(x)
        return z.view(b, self.t_out, self.latent)


class DeepONet2d(nn.Module):
    """DeepONet-style branch/trunk operator network.

    The implementation follows the DeepXDE DeepONet idea: a branch net encodes
    the input function, a trunk net encodes query coordinates, and their latent
    inner product gives the output function value. For Navier--Stokes, we predict
    T_out channels on every spatial coordinate.
    """

    def __init__(
        self,
        t_in: int,
        t_out: int,
        latent: int = 128,
        hidden: int = 128,
        depth: int = 4,
        branch: str = "cnn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if branch == "mlp":
            self.branch = MLPBranch(t_in, latent, t_out, hidden, depth, dropout)
        else:
            self.branch = CNNBranch(t_in, latent, t_out, width=max(hidden // 2, 8), dropout=dropout)
        self.trunk = make_mlp(2, hidden, latent, depth=depth, dropout=dropout)
        self.bias = nn.Parameter(torch.zeros(t_out))
        self.latent = latent
        self.t_out = t_out
        self._coord_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def _coords(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device)
        if key not in self._coord_cache:
            xs = torch.linspace(0, 1, height, device=device)
            ys = torch.linspace(0, 1, width, device=device)
            try:
                gx, gy = torch.meshgrid(xs, ys, indexing="ij")
            except TypeError:  # older torch
                gx, gy = torch.meshgrid(xs, ys)
            coords = torch.stack([gx, gy], dim=-1).view(-1, 2)
            self._coord_cache[key] = coords
        return self._coord_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T_in,H,W]
        b, _, h, w = x.shape
        branch_feat = self.branch(x)                 # [B,T_out,P]
        trunk_feat = self.trunk(self._coords(h, w, x.device))  # [H*W,P]
        trunk_feat = trunk_feat / math.sqrt(self.latent)
        out = torch.einsum("btp,np->btn", branch_feat, trunk_feat)
        out = out + self.bias.view(1, self.t_out, 1)
        return out.view(b, self.t_out, h, w)
