#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    toy = Path("data/processed/toy_ns.npz")
    if not toy.exists():
        subprocess.run([sys.executable, "scripts/make_toy_data.py", "--output", str(toy), "--n", "48", "--size", "16", "--t", "20"], check=True)
    for model in ["fno", "deeponet", "cnn", "unet"]:
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--extra",
            f"model.name={model}",
            f"project.experiment_name=smoke_{model}",
            "data.path=data/processed/toy_ns.npz",
            "data.n_train=4",
            "data.n_val=2",
            "data.n_test=2",
            "data.batch_size=2",
            "data.train_resolution=8",
            "data.eval_resolution=8",
            "model.width=4",
            "model.hidden=16",
            "model.depth=1",
            "model.modes1=4",
            "model.modes2=4",
            "model.padding=0",
            "model.deeponet_latent=16",
            "training.epochs=1",
            "training.amp=false",
        ]
        subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "scripts/summarize_results.py", "--runs", "results/runs", "--out", "results/tables/smoke_summary.csv"], check=True)
    subprocess.run([sys.executable, "scripts/plot_results.py", "--summary", "results/tables/smoke_summary.csv", "--out_dir", "results/figures/smoke"], check=True)
    print("Smoke test finished.")


if __name__ == "__main__":
    main()
