"""Shared utilities: seeding, logging, IO, paths."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"

for d in (DATA_DIR, OUTPUTS_DIR, MODELS_DIR, CONFIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_logger(name: str = "tube_pose", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(h)
    logger.propagate = False
    return logger


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def device_str() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def list_images(folder: str | Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> list[Path]:
    folder = Path(folder)
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
