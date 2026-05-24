#!/usr/bin/env python
from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print("=" * 80)
    print("Local GPU check")
    print("=" * 80)
    print(f"Python        : {sys.version.split()[0]}")
    print(f"Platform      : {platform.platform()}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version  : {torch.version.cuda}")
    print(f"cuDNN version : {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA is not available. Training will run on CPU.")
        print("Install a CUDA-enabled PyTorch build, then rerun this script.")
        return

    n = torch.cuda.device_count()
    print(f"GPU count     : {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        print(f"GPU {i}        : {props.name} | {total_gb:.2f} GB | capability {props.major}.{props.minor}")

    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    print(f"\nMatmul test   : OK, result mean={z.mean().item():.6f}")
    print("Device        : cuda")


if __name__ == "__main__":
    main()
