#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_toy_ns(n: int, size: int, t: int, seed: int) -> np.ndarray:
    """Create a tiny smooth advection-like dataset for debugging only.

    This is not a replacement for the official benchmark; it is used by smoke tests
    so that the code can be verified without downloading the large .mat file.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 2 * np.pi, size, endpoint=False)
    ys = np.linspace(0, 2 * np.pi, size, endpoint=False)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    data = np.zeros((n, size, size, t), dtype=np.float32)
    for i in range(n):
        k1, k2 = rng.integers(1, 5, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.5)
        vx, vy = rng.uniform(-0.15, 0.15, size=2)
        for j in range(t):
            field = amp * np.sin(k1 * (gx - vx * j) + k2 * (gy - vy * j) + phase)
            field += 0.3 * np.cos((k2 + 1) * (gx + vx * j) - k1 * (gy + vy * j))
            data[i, :, :, j] = field
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/processed/toy_ns.npz")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--t", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = make_toy_ns(args.n, args.size, args.t, args.seed)
    np.savez_compressed(out, u=data)
    print(f"Saved toy data to {out}, shape={data.shape}")


if __name__ == "__main__":
    main()
