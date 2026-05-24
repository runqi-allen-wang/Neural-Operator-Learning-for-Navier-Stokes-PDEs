from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Normalizer:
    mean: float
    std: float

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def _load_array(path: str | Path, key: str = "u") -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Prepare the local .npz first with scripts/prepare_official_data.py "
            "or override data.path=/absolute/or/relative/path/to/file.npz"
        )
    obj = np.load(path)
    if key in obj:
        arr = obj[key]
    else:
        # Fallback: use the first 4D array in the npz.
        arr = None
        for k in obj.files:
            if obj[k].ndim == 4:
                arr = obj[k]
                break
        if arr is None:
            raise KeyError(f"Could not find key={key!r} or any 4D array in {path}")
    if arr.ndim != 4:
        raise ValueError(f"Expected [N,H,W,T], got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _make_indices(n_available: int, n_train: int, n_val: int, n_test: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = n_train + n_val + n_test
    if total > n_available:
        raise ValueError(f"Need {total} samples, but data only has {n_available}.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_available)[:total]
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _compute_stats(arr: np.ndarray, train_idx: np.ndarray, t_total: int, normalize: bool = True) -> Normalizer:
    if not normalize:
        return Normalizer(mean=0.0, std=1.0)
    train_block = arr[train_idx, :, :, :t_total]
    mean = float(train_block.mean())
    std = float(train_block.std() + 1e-6)
    return Normalizer(mean=mean, std=std)


class NavierStokesDataset(Dataset):
    """Dataset for tensor-to-tensor Navier--Stokes operator learning.

    Raw data shape: [N, H, W, T].
    Input: first t_in frames as [t_in, H, W].
    Target: next t_out frames as [t_out, H, W].
    """

    def __init__(
        self,
        arr: np.ndarray,
        indices: np.ndarray,
        t_in: int,
        t_out: int,
        resolution: int,
        normalizer: Normalizer,
    ) -> None:
        self.arr = arr
        self.indices = indices.astype(np.int64)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.resolution = int(resolution)
        self.normalizer = normalizer
        if self.t_in + self.t_out > arr.shape[-1]:
            raise ValueError(f"Need t_in+t_out={self.t_in+self.t_out}, but data has T={arr.shape[-1]}")

    def __len__(self) -> int:
        return len(self.indices)

    def _resize(self, u: torch.Tensor) -> torch.Tensor:
        # u: [T,H,W].  Interpolate over spatial dimensions.
        if u.shape[-1] == self.resolution and u.shape[-2] == self.resolution:
            return u
        u = u.unsqueeze(0)  # [1,T,H,W]
        u = F.interpolate(u, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return u.squeeze(0)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.arr[self.indices[item], :, :, : self.t_in + self.t_out]
        u = torch.from_numpy(raw).permute(2, 0, 1).contiguous()  # [T,H,W]
        u = self._resize(u)
        u = self.normalizer.encode(u)
        x = u[: self.t_in]
        y = u[self.t_in : self.t_in + self.t_out]
        return x.float(), y.float()


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Normalizer]:
    data_cfg = cfg["data"]
    arr = _load_array(data_cfg["path"], data_cfg.get("key", "u"))
    n_train, n_val, n_test = int(data_cfg["n_train"]), int(data_cfg["n_val"]), int(data_cfg["n_test"])
    seed = int(cfg["project"].get("seed", 2026))
    train_idx, val_idx, test_idx = _make_indices(arr.shape[0], n_train, n_val, n_test, seed)
    t_in, t_out = int(data_cfg["t_in"]), int(data_cfg["t_out"])
    normalizer = _compute_stats(arr, train_idx, t_in + t_out, bool(data_cfg.get("normalize", True)))

    train_ds = NavierStokesDataset(arr, train_idx, t_in, t_out, int(data_cfg["train_resolution"]), normalizer)
    val_ds = NavierStokesDataset(arr, val_idx, t_in, t_out, int(data_cfg["eval_resolution"]), normalizer)
    test_ds = NavierStokesDataset(arr, test_idx, t_in, t_out, int(data_cfg["eval_resolution"]), normalizer)

    common = dict(
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=bool(data_cfg.get("shuffle_train", True)), drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader, normalizer
