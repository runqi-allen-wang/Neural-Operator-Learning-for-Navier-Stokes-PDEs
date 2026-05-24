#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils import flatten_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default="results/runs")
    parser.add_argument("--out", type=str, default="results/tables/summary.csv")
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(Path(args.runs).glob("*/metrics.json")):
        run_dir = metrics_path.parent
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        config_path = run_dir / "config.yaml"
        cfg = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        row = {"run_dir": str(run_dir), "run_name": run_dir.name}
        row.update(flatten_dict(cfg))
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics.json found under {args.runs}")
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved summary to {out}")
    cols = [c for c in ["project.experiment_name", "model.name", "data.n_train", "model.depth", "data.train_resolution", "test_rel_l2", "test_rmse", "n_parameters"] if c in df]
    print(df[cols].sort_values("test_rel_l2").to_string(index=False))


if __name__ == "__main__":
    main()
