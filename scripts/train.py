#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data import build_dataloaders
from src.models import build_model
from src.trainer import train_model
from src.utils import configure_torch, count_parameters, ensure_dir, load_config, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural operator/baseline on Navier-Stokes data")
    parser.add_argument("--config", type=str, default=None, help="Optional model config yaml")
    parser.add_argument("--default_config", type=str, default="configs/default.yaml")
    parser.add_argument("--extra", nargs="*", default=[], help="Overrides such as model.name=fno training.epochs=30")
    args = parser.parse_args()

    cfg = load_config(args.default_config, args.config, args.extra)
    configure_torch(cfg)
    set_seed(int(cfg["project"].get("seed", 2026)))

    model_name = str(cfg["model"]["name"]).lower()
    exp_name = str(cfg["project"].get("experiment_name", model_name))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(cfg["project"].get("output_root", "results/runs")) / f"{exp_name}_{timestamp}")

    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Run dir   : {run_dir}")
    print(f"Model     : {model_name}")
    print(f"Data path : {cfg['data']['path']}")
    print("=" * 80)

    train_loader, val_loader, test_loader, normalizer = build_dataloaders(cfg)
    model = build_model(cfg)
    n_params = count_parameters(model)
    print(f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
    print(f"Normalizer: mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")
    print(f"Trainable parameters: {n_params:,}")

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    save_json({"n_parameters": n_params}, run_dir / "model_info.json")

    metrics, _ = train_model(model, train_loader, val_loader, test_loader, cfg, run_dir)
    metrics["n_parameters"] = n_params
    save_json(metrics, run_dir / "metrics.json")
    print("Final metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
