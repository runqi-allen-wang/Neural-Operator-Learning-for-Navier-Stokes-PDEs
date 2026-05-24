#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


def load_mat_4d(path: Path) -> np.ndarray:
    """Load the first 4D array from a MATLAB .mat file.

    Supports both classic MAT files readable by scipy.io.loadmat and v7.3 HDF5 MAT files.
    The target FNO Navier--Stokes file is usually shaped [N, 64, 64, T].
    """
    try:
        mat = sio.loadmat(path)
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.ndim == 4:
                print(f"Found key {key!r} with shape {value.shape} using scipy.io.loadmat")
                return value.astype(np.float32)
    except NotImplementedError:
        pass
    except Exception as exc:
        print(f"scipy.io.loadmat failed with {exc!r}; trying h5py...")

    with h5py.File(path, "r") as f:
        for key in f.keys():
            arr = np.array(f[key])
            if arr.ndim == 4:
                # h5py may return transposed MATLAB ordering; use a simple heuristic.
                print(f"Found key {key!r} with raw HDF5 shape {arr.shape}")
                arr = np.asarray(arr)
                if arr.shape[0] not in {5000, 10000} and arr.shape[-1] in {5000, 10000}:
                    arr = np.transpose(arr, (3, 2, 1, 0))
                return arr.astype(np.float32)
    raise RuntimeError(f"No 4D array found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official FNO Navier-Stokes .mat into .npz")
    parser.add_argument("--input", type=str, required=True, help="Path to NavierStokes_V1e-3_N5000_T50.mat")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--n", type=int, default=1400, help="Number of trajectories to keep")
    parser.add_argument("--t", type=int, default=20, help="Number of time steps to keep")
    parser.add_argument("--key", type=str, default="u", help="Key name inside output npz")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    print(f"Loading official subset from {inp} ...")
    arr = load_mat_4d(inp)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D data [N,H,W,T], got {arr.shape}")
    arr = arr[: args.n, :, :, : args.t].astype(np.float32, copy=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{args.key: arr})
    size_mb = out.stat().st_size / 1024**2
    print(f"Loaded subset shape [N,H,W,T] = {arr.shape}")
    print(f"Saved {out} ({size_mb:.1f} MB).")
    print("This file can now be used by configs/default.yaml or configs/*_fast.yaml.")


if __name__ == "__main__":
    main()
