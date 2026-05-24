from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def configure_torch(cfg: Dict[str, Any]) -> None:
    """Apply local hardware-related PyTorch settings."""
    system_cfg = cfg.get("system", {}) or {}
    num_threads = system_cfg.get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    tf32 = bool(system_cfg.get("tf32", True))
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = tf32
    try:
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    except Exception:
        pass


def get_device(cfg: Dict[str, Any]) -> torch.device:
    requested = str((cfg.get("system", {}) or {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device={requested}, but torch.cuda.is_available() is False. "
            "Run `python scripts/check_gpu.py` and install a CUDA-enabled PyTorch build."
        )
    return torch.device(requested)


def print_device_info(device: torch.device) -> None:
    print(f"Device    : {device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU       : {props.name}")
        print(f"GPU memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"CUDA      : {torch.version.cuda}")


def maybe_compile_model(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.nn.Module:
    system_cfg = cfg.get("system", {}) or {}
    if bool(system_cfg.get("compile", False)):
        if not hasattr(torch, "compile"):
            print("torch.compile is not available in this PyTorch version; using the original model.")
            return model
        print("Compiling model with torch.compile ...")
        return torch.compile(model)
    return model


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_value(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        if any(c in value for c in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        cur = cfg
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = parse_value(value)
    return cfg


def load_config(default_path: str | Path, config_path: str | Path | None = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    cfg = load_yaml(default_path)
    if config_path is not None:
        cfg = deep_update(cfg, load_yaml(config_path))
    cfg = apply_overrides(cfg, overrides)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, name))
        else:
            out[name] = v
    return out
