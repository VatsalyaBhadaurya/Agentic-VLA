"""Config loading. Thin YAML loader with a merged view of env/models/run configs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Config:
    """Merged configuration view. Access raw sub-dicts via .env / .models / .run."""

    env: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        return cls(
            env=_load_yaml("env.yaml"),
            models=_load_yaml("models.yaml"),
            run=_load_yaml("run.yaml"),
        )

    # convenience accessors used across the codebase ----------------------------------
    @property
    def suite(self) -> str:
        return self.env.get("suite", "libero_object")

    @property
    def checkpoint(self) -> str:
        return self.models["policy"]["checkpoint"]

    @property
    def device(self) -> str:
        return self.models["policy"].get("device", "cuda")

    @property
    def retry_budget(self) -> int:
        return self.run.get("recovery", {}).get("retry_budget", 3)


def load_config() -> Config:
    return Config.load()
