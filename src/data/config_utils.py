from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_default_config(default_path: str | Path | None = None) -> dict:
    resolved_default_path = (
        Path(default_path).expanduser().resolve()
        if default_path is not None
        else DEFAULT_CONFIG_PATH
    )
    if not resolved_default_path.exists():
        return {}
    with resolved_default_path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def _with_default_config(
    cfg: dict | None, default_path: str | Path | None = None
) -> dict:
    default_cfg = normalize_cutoff_config(_load_default_config(default_path))
    override_cfg = normalize_cutoff_config(cfg or {})
    return normalize_cutoff_config(_deep_merge_dicts(default_cfg, override_cfg))


def normalize_cutoff_config(cfg: dict | None) -> dict:
    """Normalize legacy cutoff keys into the current three-cutoff layout."""
    normalized = dict(cfg or {})
    feature_build_cfg = normalized.get("feature_build")
    if isinstance(feature_build_cfg, dict):
        feature_build_cfg = dict(feature_build_cfg)
        feature_build_cfg.pop("ligand_feature_modal", None)
        feature_build_cfg.pop("ligand_batch_size", None)
        feature_build_cfg.pop("molformer_model_name", None)
        normalized["feature_build"] = feature_build_cfg
    if "data" not in normalized and "model" not in normalized:
        return normalized

    has_model = "model" in normalized
    model_cfg = dict(normalized.get("model") or {})
    if "data" not in normalized:
        model_cfg.pop("host_cutoff", None)
        model_cfg.pop("host_query_cutoff", None)
        if has_model:
            normalized["model"] = model_cfg
        return normalized

    data_cfg = dict(normalized.get("data") or {})
    data_cfg.pop("ligand_feature_modal", None)
    flow_cfg = dict(normalized.get("flow") or {}) if "flow" in normalized else None
    loss_weights = (
        dict(normalized.get("loss_weights") or {})
        if "loss_weights" in normalized
        else None
    )

    if "host_host_cutoff" not in data_cfg:
        if "graph_cutoff" in data_cfg:
            data_cfg["host_host_cutoff"] = data_cfg["graph_cutoff"]
        else:
            data_cfg["host_host_cutoff"] = model_cfg.get("cutoff", 12.0)
    data_cfg.pop("graph_cutoff", None)

    if "host_query_cutoff" not in data_cfg:
        data_cfg["host_query_cutoff"] = model_cfg.get(
            "host_query_cutoff", data_cfg["host_host_cutoff"]
        )

    test_datasets = data_cfg.get("test_datasets")
    if isinstance(test_datasets, list):
        normalized_tests = []
        for test_cfg in test_datasets:
            if not isinstance(test_cfg, dict):
                normalized_tests.append(test_cfg)
                continue
            normalized_test = dict(test_cfg)
            normalized_test.pop("ligand_feature_modal", None)
            if (
                "host_host_cutoff" not in normalized_test
                and "graph_cutoff" in normalized_test
            ):
                normalized_test["host_host_cutoff"] = normalized_test["graph_cutoff"]
            normalized_test.pop("graph_cutoff", None)
            if "host_query_cutoff" not in normalized_test:
                normalized_test["host_query_cutoff"] = data_cfg["host_query_cutoff"]
            normalized_tests.append(normalized_test)
        data_cfg["test_datasets"] = normalized_tests

    model_cfg.pop("host_cutoff", None)
    model_cfg.pop("host_query_cutoff", None)
    normalized["data"] = data_cfg
    if has_model:
        normalized["model"] = model_cfg
    if flow_cfg is not None:
        flow_cfg.pop("query_field_sigma", None)
        normalized["flow"] = flow_cfg
    if loss_weights is not None:
        loss_weights.pop("query_field", None)
        normalized["loss_weights"] = loss_weights
    return normalized


def load_yaml_config(
    path: str | Path | None,
    *,
    include_defaults: bool = True,
    default_path: str | Path | None = None,
) -> dict:
    if path is None:
        return {}
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r") as handle:
        cfg = normalize_cutoff_config(yaml.safe_load(handle) or {})

    if not include_defaults:
        return cfg

    resolved_default_path = (
        Path(default_path).expanduser().resolve()
        if default_path is not None
        else DEFAULT_CONFIG_PATH
    )
    if cfg_path == resolved_default_path:
        return cfg

    default_cfg = normalize_cutoff_config(_load_default_config(default_path))
    return normalize_cutoff_config(_deep_merge_dicts(default_cfg, cfg))


def dataset_root_from_config(cfg: dict) -> Path | None:
    data_cfg = _with_default_config(cfg).get("data", {})
    root = data_cfg.get("root")
    dataset_name = data_cfg.get("dataset_name")
    if root is None or dataset_name is None:
        return None
    return Path(root).expanduser() / dataset_name


def resolve_feature_build_config(cfg: dict) -> dict:
    resolved_cfg = _with_default_config(cfg)
    data_cfg = resolved_cfg.get("data", {})
    feat_cfg = resolved_cfg.get("feature_build", {})

    dataset_root = feat_cfg.get("dataset_root")
    if dataset_root is None:
        inferred_root = dataset_root_from_config(resolved_cfg)
        dataset_root = inferred_root if inferred_root is not None else None

    embedding_modalities = data_cfg.get("embedding_modalities") or ["esm"]
    return {
        "dataset_root": Path(dataset_root).expanduser()
        if dataset_root is not None
        else None,
        "input_dir": feat_cfg.get("input_dir", data_cfg.get("input_dir")),
        "embedding_modal": feat_cfg.get("embedding_modal", embedding_modalities[0]),
        "target_modal": feat_cfg.get("target_modal", data_cfg.get("target_modal")),
        "use_residue_depths": feat_cfg.get("use_residue_depths"),
        "device": feat_cfg.get("device"),
        "n_jobs": feat_cfg.get("n_jobs"),
        "threshold": feat_cfg.get("threshold"),
        "surface_sasa_threshold": feat_cfg.get("surface_sasa_threshold"),
        "batch_size": feat_cfg.get("batch_size"),
        "max_num_tokens": feat_cfg.get("max_num_tokens"),
        "model_path": Path(feat_cfg["model_path"]).expanduser()
        if feat_cfg.get("model_path")
        else None,
        "overwrite": feat_cfg.get("overwrite"),
    }
