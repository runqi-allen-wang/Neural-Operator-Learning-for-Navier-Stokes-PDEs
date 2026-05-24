from pathlib import Path
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_NAME = {
    "fno": "FNO",
    "deeponet": "DeepONet",
    "cnn": "CNN",
    "unet": "U-Net",
}

MODEL_COLOR = {
    "FNO": "#1f77b4",
    "DeepONet": "#ff7f0e",
    "CNN": "#2ca02c",
    "U-Net": "#d62728",
}


def canonical_model_name(x):
    x = str(x).lower()
    return MODEL_NAME.get(x, str(x))


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig(out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def get_suite(df, suite):
    if "suite" in df.columns:
        return df[df["suite"] == suite].copy()
    return df[df["project.experiment_name"].astype(str).str.startswith(suite)].copy()


def infer_depth(row):
    if "model.depth" in row and pd.notna(row["model.depth"]):
        return int(row["model.depth"])
    name = str(row.get("project.experiment_name", ""))
    m = re.search(r"d(?:epth)?(\d+)", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else np.nan


def infer_resolution(row):
    if "data.train_resolution" in row and pd.notna(row["data.train_resolution"]):
        return int(row["data.train_resolution"])
    name = str(row.get("project.experiment_name", ""))
    m = re.search(r"r(\d+)", name)
    if m:
        return int(m.group(1))
    return np.nan


def plot_model_comparison(df, out_dir):
    sub = get_suite(df, "model_comparison").copy()
    sub["model_label"] = sub["model.name"].map(canonical_model_name)

    order = ["FNO", "U-Net", "CNN", "DeepONet"]
    sub["model_label"] = pd.Categorical(sub["model_label"], categories=order, ordered=True)
    sub = sub.sort_values("model_label")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    colors = [MODEL_COLOR[str(m)] for m in sub["model_label"]]
    ax.bar(sub["model_label"].astype(str), sub["test_rel_l2"], color=colors)

    ax.set_xlabel("Model architecture")
    ax.set_ylabel(r"Test relative $L_2$ error")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=20)

    savefig(out_dir, "model_comparison_relative_l2")


def plot_data_scaling(df, out_dir):
    sub = get_suite(df, "data_scaling").copy()

    # Optionally add the large-data FNO point from model_comparison.
    # This is valid if model_comparison_fno uses the same data, resolution, and training protocol.
    model_sub = get_suite(df, "model_comparison").copy()
    if len(model_sub) > 0 and "model.name" in model_sub.columns:
        fno_4000 = model_sub[model_sub["model.name"].astype(str).str.lower() == "fno"].copy()
        if len(fno_4000) > 0:
            sub = pd.concat([sub, fno_4000], ignore_index=True)

    sub = sub[sub["model.name"].astype(str).str.lower() == "fno"].copy()
    sub = sub.dropna(subset=["data.n_train", "test_rel_l2"])
    sub["data.n_train"] = sub["data.n_train"].astype(int)
    sub = sub.sort_values("data.n_train")
    sub = sub.drop_duplicates("data.n_train", keep="last")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(
        sub["data.n_train"],
        sub["test_rel_l2"],
        marker="o",
        color=MODEL_COLOR["FNO"],
        label="FNO",
    )

    ax.set_xlabel("Number of training trajectories")
    ax.set_ylabel(r"Test relative $L_2$ error")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(out_dir, "data_scaling_relative_l2")


def plot_resolution_scaling(df, out_dir):
    sub = get_suite(df, "resolution_scaling").copy()
    sub = sub[sub["model.name"].astype(str).str.lower() == "fno"].copy()
    sub["resolution"] = sub.apply(infer_resolution, axis=1)
    sub = sub.dropna(subset=["resolution", "test_rel_l2"])
    sub["resolution"] = sub["resolution"].astype(int)
    sub = sub.sort_values("resolution")
    sub = sub.drop_duplicates("resolution", keep="last")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(
        sub["resolution"],
        sub["test_rel_l2"],
        marker="o",
        color=MODEL_COLOR["FNO"],
        label="FNO",
    )

    ax.set_xlabel("Spatial grid resolution")
    ax.set_ylabel(r"Test relative $L_2$ error")
    ax.set_xticks(sorted(sub["resolution"].unique()))
    ax.set_xticklabels([f"{r}$\\times${r}" for r in sorted(sub["resolution"].unique())])
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(out_dir, "resolution_scaling_relative_l2")


def plot_depth_scaling(df, out_dir):
    sub = get_suite(df, "depth_scaling").copy()
    sub = sub[sub["model.name"].astype(str).str.lower() == "fno"].copy()
    sub["depth"] = sub.apply(infer_depth, axis=1)
    sub = sub.dropna(subset=["depth", "test_rel_l2"])
    sub["depth"] = sub["depth"].astype(int)
    sub = sub.sort_values("depth")
    sub = sub.drop_duplicates("depth", keep="last")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(
        sub["depth"],
        sub["test_rel_l2"],
        marker="o",
        color=MODEL_COLOR["FNO"],
        label="FNO",
    )

    ax.set_xlabel("Number of Fourier layers")
    ax.set_ylabel(r"Test relative $L_2$ error")
    ax.set_xticks(sorted(sub["depth"].unique()))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(out_dir, "depth_scaling_relative_l2")


def plot_parameter_efficiency(df, out_dir):
    # Use only model_comparison to avoid mixing data-size and resolution effects.
    sub = get_suite(df, "model_comparison").copy()
    sub["model_label"] = sub["model.name"].map(canonical_model_name)
    sub = sub.dropna(subset=["n_parameters", "test_rel_l2"])

    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    for _, row in sub.iterrows():
        label = row["model_label"]
        ax.scatter(
            row["n_parameters"],
            row["test_rel_l2"],
            s=60,
            color=MODEL_COLOR[label],
            label=label,
            zorder=3,
        )
        ax.annotate(
            label,
            (row["n_parameters"], row["test_rel_l2"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=11,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)

    ax.set_xscale("log")
    ax.set_xlabel("Number of trainable parameters")
    ax.set_ylabel(r"Test relative $L_2$ error")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(out_dir, "parameter_efficiency")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=str,
        default="results/tables/final/summary_all_clean.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/figures/final_neurips",
    )
    args = parser.parse_args()

    setup_matplotlib()

    df = pd.read_csv(args.summary)

    required = [
        "project.experiment_name",
        "model.name",
        "test_rel_l2",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    plot_model_comparison(df, args.out_dir)
    plot_data_scaling(df, args.out_dir)
    plot_resolution_scaling(df, args.out_dir)
    plot_depth_scaling(df, args.out_dir)
    plot_parameter_efficiency(df, args.out_dir)

    print(f"Saved final figures to: {args.out_dir}")


if __name__ == "__main__":
    main()