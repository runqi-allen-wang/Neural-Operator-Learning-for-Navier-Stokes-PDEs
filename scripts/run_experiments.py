#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_profile(profile: Dict) -> List[str]:
    return [f"{k}={v}" for k, v in profile.items()]


def run_train(extra: List[str], config: str | None = None) -> None:
    cmd = [sys.executable, "scripts/train.py"]
    if config:
        cmd += ["--config", config]
    if extra:
        cmd += ["--extra"] + extra
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment suites")
    parser.add_argument("--suite", choices=["model_comparison", "data_scaling", "depth_scaling", "resolution_scaling", "all"], default="all")
    parser.add_argument("--profile", choices=["smoke", "fast", "report"], default="fast")
    parser.add_argument("--experiments", type=str, default="configs/experiments.yaml")
    parser.add_argument("--extra", nargs="*", default=[], help="Global overrides, e.g. data.path=/kaggle/input/.../file.npz")
    args = parser.parse_args()

    exp_cfg = load_yaml(args.experiments)
    profile_extra = flatten_profile(exp_cfg.get("profiles", {}).get(args.profile, {}))
    global_extra = profile_extra + args.extra
    suites = [args.suite] if args.suite != "all" else ["model_comparison", "data_scaling", "depth_scaling", "resolution_scaling"]

    if "model_comparison" in suites:
        for model in exp_cfg["model_comparison"]["models"]:
            run_train(
                global_extra
                + [
                    f"model.name={model}",
                    f"project.experiment_name=model_comparison_{model}",
                ]
            )

    if "data_scaling" in suites:
        model = exp_cfg["data_scaling"].get("model", "fno")
        for n_train in exp_cfg["data_scaling"]["n_train"]:
            run_train(
                global_extra
                + [
                    f"model.name={model}",
                    f"data.n_train={n_train}",
                    f"project.experiment_name=data_scaling_{model}_n{n_train}",
                ]
            )

    if "depth_scaling" in suites:
        model = exp_cfg["depth_scaling"].get("model", "fno")
        for depth in exp_cfg["depth_scaling"]["depth"]:
            run_train(
                global_extra
                + [
                    f"model.name={model}",
                    f"model.depth={depth}",
                    f"project.experiment_name=depth_scaling_{model}_d{depth}",
                ]
            )

    if "resolution_scaling" in suites:
        model = exp_cfg["resolution_scaling"].get("model", "fno")
        for res in exp_cfg["resolution_scaling"]["resolution"]:
            # Train and evaluate at the same resolution for a clean controlled comparison.
            # For FNO zero-shot super-resolution, manually set train_resolution=32 eval_resolution=64.
            run_train(
                global_extra
                + [
                    f"model.name={model}",
                    f"data.train_resolution={res}",
                    f"data.eval_resolution={res}",
                    f"project.experiment_name=resolution_scaling_{model}_r{res}",
                ]
            )

    print("\nAll requested experiments finished. Now run:")
    print("  python scripts/summarize_results.py --runs results/runs --out results/tables/summary.csv")
    print("  python scripts/plot_results.py --summary results/tables/summary.csv --out_dir results/figures")


if __name__ == "__main__":
    main()
