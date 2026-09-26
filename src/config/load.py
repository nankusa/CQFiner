from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from .schema import validate_config
from .runtime import _semantic_to_train_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LoadedConfig:
    train: dict[str, Any]
    resolved: dict[str, Any]
    schema: str
    source_path: Path


def save_resolved_config(path: str | Path, cfg: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return cfg


def _load_semantic_config(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Recursive config inheritance detected: {chain}")

    cfg = _load_yaml_file(path)
    inherits = cfg.pop("inherits", [])
    if isinstance(inherits, (str, Path)):
        inherits = [inherits]
    if not isinstance(inherits, list):
        raise TypeError(f"'inherits' must be a string or list in {path}")

    merged: dict[str, Any] = {}
    for parent in inherits:
        parent_path = _resolve_parent_path(path, parent)
        parent_cfg = _load_semantic_config(parent_path, (*stack, path))
        merged = _deep_merge(merged, parent_cfg)

    return _deep_merge(merged, cfg)


def _resolve_parent_path(path: Path, parent: str | Path) -> Path:
    parent_path = Path(parent).expanduser()
    if parent_path.is_absolute():
        return parent_path.resolve()
    return (path.parent / parent_path).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_training_config(path: str | Path) -> LoadedConfig:
    source = Path(path).expanduser().resolve()
    raw = _load_yaml_file(source)
    if "train" in raw:
        if not isinstance(raw["train"], dict) or not isinstance(raw["resolved"], dict):
            raise TypeError("Saved run must contain train and resolved mappings.")
        return LoadedConfig(
            deepcopy(raw["train"]), deepcopy(raw["resolved"]), raw["schema"], source
        )
    resolved = _load_semantic_config(source)
    validate_config(resolved)
    return LoadedConfig(
        _semantic_to_train_config(resolved), resolved, "surfqnet.v1", source
    )
