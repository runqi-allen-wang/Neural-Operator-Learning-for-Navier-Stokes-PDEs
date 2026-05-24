from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .metrics import AverageMeter, batch_metrics
from .utils import ensure_dir, get_device, maybe_compile_model, print_device_info, save_json


def make_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict):
    train_cfg = cfg["training"]
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("epochs", 30)),
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )
    if name in {"none", "null"}:
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def run_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp: bool,
    grad_clip: float | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    meters = {k: AverageMeter() for k in ["loss", "mse", "mae", "rmse", "rel_l2"]}
    criterion = nn.MSELoss()

    pbar = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bsz = x.shape[0]

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        met = batch_metrics(pred.detach(), y.detach())
        meters["loss"].update(float(loss.detach().cpu()), bsz)
        for k, v in met.items():
            meters[k].update(float(v.cpu()), bsz)
        pbar.set_postfix(loss=meters["loss"].avg, rel_l2=meters["rel_l2"].avg)

    return {k: m.avg for k, m in meters.items()}


def plot_history(history: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_rel_l2"], label="train rel_l2")
    plt.plot(history["epoch"], history["val_rel_l2"], label="val rel_l2")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_examples(model: nn.Module, loader, out_path: Path, device: torch.device, time_index: int = 0, num_examples: int = 3) -> None:
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = model(x)
    x = x.cpu()
    y = y.cpu()
    pred = pred.cpu()
    n = min(num_examples, x.shape[0])
    time_index = min(time_index, y.shape[1] - 1)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        imgs = [x[i, -1], y[i, time_index], pred[i, time_index], (pred[i, time_index] - y[i, time_index]).abs()]
        titles = ["input last", "target", "prediction", "absolute error"]
        for j, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[i, j]
            im = ax.imshow(img.numpy())
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    cfg: Dict,
    run_dir: str | Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    run_dir = ensure_dir(run_dir)
    device = get_device(cfg)
    print_device_info(device)
    model.to(device)
    model = maybe_compile_model(model, cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    amp = bool(cfg["training"].get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    grad_clip = float(cfg["training"].get("grad_clip", 0.0))
    patience = int(cfg["training"].get("early_stop_patience", 20))
    save_best = bool(cfg["training"].get("save_best", True))

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    rows = []
    start = time.time()

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, amp, grad_clip, scaler)
        val_metrics = run_one_epoch(model, val_loader, None, device, amp=False)
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        rows.append(row)
        print(
            f"Epoch {epoch:03d} | train rel_l2={train_metrics['rel_l2']:.5f} | "
            f"val rel_l2={val_metrics['rel_l2']:.5f} | lr={row['lr']:.2e}"
        )

        if val_metrics["rel_l2"] < best_val:
            best_val = val_metrics["rel_l2"]
            best_epoch = epoch
            bad_epochs = 0
            if save_best:
                torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    history = pd.DataFrame(rows)
    history.to_csv(run_dir / "history.csv", index=False)
    plot_history(history, run_dir / "training_curve.png")

    if save_best and (run_dir / "best.pt").exists():
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    test_metrics = run_one_epoch(model, test_loader, None, device, amp=False)
    elapsed = time.time() - start
    metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    metrics.update({"best_val_rel_l2": best_val, "best_epoch": best_epoch, "train_seconds": elapsed})
    save_json(metrics, run_dir / "metrics.json")
    plot_examples(
        model,
        test_loader,
        run_dir / "prediction_examples.png",
        device,
        time_index=int(cfg.get("visualization", {}).get("time_index", 0)),
        num_examples=int(cfg.get("visualization", {}).get("num_examples", 3)),
    )
    return metrics, history
