from __future__ import annotations

from typing import Dict

from torch import nn

from .cnn import CNNBaseline
from .deeponet2d import DeepONet2d
from .fno2d import FNO2d
from .unet import UNet2d


def build_model(cfg: Dict) -> nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    t_in = int(data_cfg["t_in"])
    t_out = int(data_cfg["t_out"])
    name = str(model_cfg["name"]).lower()
    if name == "fno":
        return FNO2d(
            t_in=t_in,
            t_out=t_out,
            width=int(model_cfg.get("width", 32)),
            depth=int(model_cfg.get("depth", 4)),
            modes1=int(model_cfg.get("modes1", 12)),
            modes2=int(model_cfg.get("modes2", 12)),
            padding=int(model_cfg.get("padding", 8)),
        )
    if name == "deeponet":
        return DeepONet2d(
            t_in=t_in,
            t_out=t_out,
            latent=int(model_cfg.get("deeponet_latent", 128)),
            hidden=int(model_cfg.get("hidden", 128)),
            depth=int(model_cfg.get("depth", 4)),
            branch=str(model_cfg.get("deeponet_branch", "cnn")),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if name == "cnn":
        return CNNBaseline(
            t_in=t_in,
            t_out=t_out,
            width=int(model_cfg.get("width", 64)),
            depth=int(model_cfg.get("depth", 6)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if name == "unet":
        return UNet2d(
            t_in=t_in,
            t_out=t_out,
            width=int(model_cfg.get("width", 32)),
            depth=int(model_cfg.get("depth", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    raise ValueError(f"Unknown model.name={name!r}")
