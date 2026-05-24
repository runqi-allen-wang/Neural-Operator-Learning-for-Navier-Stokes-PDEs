#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local GPU experiment pipeline")
    parser.add_argument("--data", type=str, default="data/processed/NavierStokes_V1e-3_N1400_T20.npz")
    parser.add_argument("--suite", choices=["model_comparison", "data_scaling", "depth_scaling", "resolution_scaling", "all"], default="all")
    parser.add_argument("--profile", choices=["smoke", "fast", "report"], default="fast")
    parser.add_argument("--num_workers", type=int, default=0, help="Use 0 on Windows; 2-4 is OK on Linux.")
    parser.add_argument("--batch_size", type=int, default=8, help="Increase if your GPU memory allows.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision. Recommended mainly for CNN/U-Net, not FNO.")
    parser.add_argument("--skip_gpu_check", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find {data_path}. Prepare the data first, e.g.\n"
            "  python scripts/prepare_official_data.py --input data/raw/NavierStokes_V1e-3_N5000_T50.mat "
            "--output data/processed/NavierStokes_V1e-3_N1400_T20.npz --n 1400 --t 20"
        )

    if not args.skip_gpu_check:
        run([sys.executable, "scripts/check_gpu.py"])

    extra = [
        f"data.path={data_path.as_posix()}",
        f"data.num_workers={args.num_workers}",
        f"data.batch_size={args.batch_size}",
        f"training.amp={str(args.amp).lower()}",
        "system.device=auto",
        "system.tf32=true",
    ]

    run([
        sys.executable,
        "scripts/run_experiments.py",
        "--suite",
        args.suite,
        "--profile",
        args.profile,
        "--extra",
        *extra,
    ])

    if not args.skip_summary:
        run([sys.executable, "scripts/summarize_results.py", "--runs", "results/runs", "--out", "results/tables/summary.csv"])
        run([sys.executable, "scripts/plot_results.py", "--summary", "results/tables/summary.csv", "--out_dir", "results/figures"])
        print("\nDone. Main outputs:")
        print("  results/tables/summary.csv")
        print("  results/figures/")
        print("  results/runs/")


if __name__ == "__main__":
    main()
