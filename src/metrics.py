from __future__ import annotations

from typing import Dict

import torch


def batch_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Metrics for tensors shaped [B, T, H, W]."""
    diff = pred - target
    mse = torch.mean(diff ** 2)
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(mse + eps)
    rel_l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / (
        torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1) + eps
    )
    return {
        "mse": mse.detach(),
        "mae": mae.detach(),
        "rmse": rmse.detach(),
        "rel_l2": rel_l2.mean().detach(),
    }


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
