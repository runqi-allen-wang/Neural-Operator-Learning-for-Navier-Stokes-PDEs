from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier layer used by FNO.

    The FFT path is explicitly computed in float32/complex64. This avoids common
    AMP + cuFFT failures on Kaggle when padding creates non-power-of-two sizes.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(input_fft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_fft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        x_float = x.float()
        x_ft = torch.fft.rfft2(x_float, norm="ortho")

        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        x = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x)
        y = self.norm(y)
        return F.gelu(y)


class FNO2d(nn.Module):
    """Fourier Neural Operator for mapping T_in fields to T_out fields."""

    def __init__(
        self,
        t_in: int,
        t_out: int,
        width: int = 32,
        depth: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        padding: int = 8,
    ) -> None:
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.width = width
        self.padding = padding
        self.fc0 = nn.Linear(t_in + 2, width)
        self.blocks = nn.ModuleList([FNOBlock2d(width, modes1, modes2) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, t_out)

    @staticmethod
    def get_grid(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        gridx = torch.linspace(0, 1, height, device=device).view(1, height, 1, 1).repeat(batch_size, 1, width, 1)
        gridy = torch.linspace(0, 1, width, device=device).view(1, 1, width, 1).repeat(batch_size, height, 1, 1)
        return torch.cat([gridx, gridy], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T_in,H,W]
        bsz, _, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,T_in]
        grid = self.get_grid(bsz, height, width, x.device)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2).contiguous()  # [B,width,H,W]

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for block in self.blocks:
            x = block(x)
        if self.padding > 0:
            x = x[..., :height, :width]

        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2).contiguous()
