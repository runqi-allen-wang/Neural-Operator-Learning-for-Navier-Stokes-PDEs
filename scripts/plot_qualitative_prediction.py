#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders
from src.models import build_model
from src.utils import configure_torch, get_device, set_seed


def find_latest_run(prefix: str, runs_root: str = "results/runs") -> Path:
    runs_root = Path(runs_root)
    candidates = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not candidates:
        raise FileNotFoundError(f"No run directory found with prefix={prefix!r} under {runs_root}")
    return candidates[-1]


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, run_dir: Path, device: torch.device) -> torch.nn.Module:
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    # Compatible with possible torch.compile prefixes.
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    return model


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to a trained run directory. If omitted, use latest model_comparison_fno run.",
    )
    parser.add_argument(
        "--run_prefix",
        type=str,
        default="model_comparison_fno",
        help="Used only when --run_dir is omitted.",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--time_index", type=int, default=-1)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/figures/final_neurips",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="qualitative_prediction_fno",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(args.run_prefix)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using run directory: {run_dir}")

    cfg = load_run_config(run_dir)
    configure_torch(cfg)
    set_seed(int(cfg["project"].get("seed", 2026)))

    device = get_device(cfg)
    _, _, test_loader, normalizer = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    model = load_checkpoint(model, run_dir, device)
    model.eval()

    test_ds = test_loader.dataset
    if args.sample_index < 0 or args.sample_index >= len(test_ds):
        raise ValueError(f"sample_index must be in [0, {len(test_ds)-1}], got {args.sample_index}")

    x, y = test_ds[args.sample_index]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)

    # Choose forecast time index.
    t_out = y.shape[1]
    time_index = args.time_index
    if time_index < 0:
        time_index = t_out + time_index
    if time_index < 0 or time_index >= t_out:
        raise ValueError(f"time_index must be in [0, {t_out-1}] or negative indexing, got {args.time_index}")

    # Decode from normalized fields to physical scale.
    y_dec = normalizer.decode(y)
    pred_dec = normalizer.decode(pred)
    err = (pred_dec - y_dec).abs()

    truth_img = to_numpy(y_dec[0, time_index])
    pred_img = to_numpy(pred_dec[0, time_index])
    err_img = to_numpy(err[0, time_index])

    rel_l2 = torch.linalg.norm(pred_dec[0] - y_dec[0]) / (torch.linalg.norm(y_dec[0]) + 1e-12)
    frame_rel_l2 = torch.linalg.norm(pred_dec[0, time_index] - y_dec[0, time_index]) / (
        torch.linalg.norm(y_dec[0, time_index]) + 1e-12
    )

    vmin = min(float(truth_img.min()), float(pred_img.min()))
    vmax = max(float(truth_img.max()), float(pred_img.max()))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), constrained_layout=True)

    panels = [
        ("Ground truth", truth_img, vmin, vmax, "viridis"),
        ("Prediction", pred_img, vmin, vmax, "viridis"),
        ("Absolute error", err_img, 0.0, float(err_img.max()), "magma"),
    ]

    for ax, (label, img, lo, hi, cmap) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(label, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # Small caption-like annotation, not a title.
    fig.text(
        0.5,
        -0.03,
        rf"FNO prediction at forecast step {time_index + 1}; "
        rf"trajectory relative $L_2$ = {float(rel_l2):.4f}, "
        rf"frame relative $L_2$ = {float(frame_rel_l2):.4f}.",
        ha="center",
        va="top",
        fontsize=10,
    )

    png_path = out_dir / f"{args.filename}.png"
    pdf_path = out_dir / f"{args.filename}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


if __name__ == "__main__":
    main()