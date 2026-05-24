#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, out: Path) -> None:
    if df.empty or x not in df or y not in df:
        return
    d = df.sort_values(y)
    plt.figure(figsize=(7, 5))
    plt.bar(d[x].astype(str), d[y])
    plt.ylabel(y)
    plt.xlabel(x)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_line(df: pd.DataFrame, x: str, y: str, title: str, out: Path) -> None:
    if df.empty or x not in df or y not in df:
        return
    d = df.sort_values(x)
    plt.figure(figsize=(7, 5))
    plt.plot(d[x], d[y], marker="o")
    plt.ylabel(y)
    plt.xlabel(x)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/tables/summary.csv")
    parser.add_argument("--out_dir", type=str, default="results/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "project.experiment_name" not in df:
        raise KeyError("summary.csv must contain project.experiment_name")

    name = df["project.experiment_name"].astype(str)
    save_bar(
        df[name.str.startswith("model_comparison")],
        "model.name",
        "test_rel_l2",
        "Model comparison on Navier-Stokes",
        out_dir / "model_comparison_rel_l2.png",
    )
    save_line(
        df[name.str.startswith("data_scaling")],
        "data.n_train",
        "test_rel_l2",
        "Effect of training data size",
        out_dir / "data_scaling_rel_l2.png",
    )
    save_line(
        df[name.str.startswith("depth_scaling")],
        "model.depth",
        "test_rel_l2",
        "Effect of network depth",
        out_dir / "depth_scaling_rel_l2.png",
    )
    save_line(
        df[name.str.startswith("resolution_scaling")],
        "data.train_resolution",
        "test_rel_l2",
        "Effect of spatial resolution",
        out_dir / "resolution_scaling_rel_l2.png",
    )
    # Overall scatter: parameters vs error.
    if "n_parameters" in df and "test_rel_l2" in df:
        plt.figure(figsize=(7, 5))
        plt.scatter(df["n_parameters"], df["test_rel_l2"])
        for _, r in df.iterrows():
            plt.annotate(str(r.get("model.name", "")), (r["n_parameters"], r["test_rel_l2"]), fontsize=8)
        plt.xscale("log")
        plt.xlabel("Trainable parameters (log scale)")
        plt.ylabel("Test relative L2")
        plt.title("Parameter count vs prediction error")
        plt.tight_layout()
        plt.savefig(out_dir / "params_vs_error.png", dpi=200)
        plt.close()
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
