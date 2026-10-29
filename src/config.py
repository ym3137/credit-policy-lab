"""Configuration loading. One source of truth: config/config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and resolve every path relative to the repo root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["paths"] = {k: str(REPO_ROOT / v) for k, v in cfg["paths"].items()}
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg
