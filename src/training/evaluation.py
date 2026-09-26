import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import lightning as L
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_training_config
from src.config.load import save_resolved_config
from src.metrics import (
    ComplexityRecorder,
    EpochTimeRecorder,
    NearFarRecorder,
    OversmoothingRecorder,
    PostprocessRecorder,
    SiteEvalRecorder,
    SiteSizeRecorder,
    StageTimingRecorder,
)
from src.training.runtime import build_pocket_runtime


EVAL_TOP_LEVEL_KEYS = {
    "run_config",
    "checkpoint",
    "seed",
    "output",
    "trainer",
    "data",
    "model",
    "flow",
    "minimal_prediction",
    "complexity",
    "oversmoothing",
    "nearfar",
    "postprocess",
    "site_eval",
    "site_size",
    "test_datasets",
}

SITE_EVAL_CONFIG_KEYS = {"enabled"}
OVERSMOOTHING_CONFIG_KEYS = {"targets", "feature_transforms"}
POSTPROCESS_CONFIG_KEYS = {"enabled"}
SITE_SIZE_CONFIG_KEYS = {"enabled", "bins"}
NEARFAR_CONFIG_KEYS = {
    "split_cutoff_A",
    "split_reference",
    "success_cutoff_A",
    "store_per_query",
}
COMPLEXITY_CONFIG_KEYS = {
    "enabled",
    "train",
    "test",
    "train_batch_size",
    "test_batch_size",
    "train_limit_batches",
    "test_limit_batches",
    "timing",
    "measure_memory",
}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Config file does not exist: {source}")
    with source.open("r") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {source}")
    return payload


def _load_train_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = _load_yaml(source)
    if "train" in payload:
        train_cfg = payload["train"]
        if not isinstance(train_cfg, dict):
            raise TypeError(
                f"Saved resolved config has non-mapping train section: {source}"
            )
        return deepcopy(train_cfg), payload

    loaded = load_training_config(source)
    return deepcopy(loaded.train), {
        "schema": loaded.schema,
        "source_path": str(loaded.source_path),
        "resolved": loaded.resolved,
        "train": loaded.train,
    }


def _load_eval_config(path: str | Path) -> dict[str, Any]:
    cfg = _load_yaml(path)
    unknown = set(cfg) - EVAL_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown eval config keys: {sorted(unknown)}")
    required = {"run_config", "checkpoint", "output", "trainer", "test_datasets"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing required eval config keys: {sorted(missing)}")
    if not isinstance(cfg["test_datasets"], list) or not cfg["test_datasets"]:
        raise ValueError("eval config requires a non-empty test_datasets list.")
    for idx, dataset in enumerate(cfg["test_datasets"]):
        if not isinstance(dataset, dict):
            raise TypeError(f"test_datasets[{idx}] must be a mapping.")
        for key in ("name", "dataset_name", "split_tag"):
            if key not in dataset:
                raise ValueError(f"test_datasets[{idx}] is missing required key: {key}")
    return cfg


def _build_oversmoothing_recorder(cfg: Any) -> OversmoothingRecorder | None:
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError("oversmoothing config must be a mapping.")
    unknown = set(cfg) - OVERSMOOTHING_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown oversmoothing config keys: {sorted(unknown)}")
    if "targets" not in cfg:
        raise ValueError("oversmoothing config requires targets.")
    targets = cfg["targets"]
    if not isinstance(targets, list):
        raise TypeError("oversmoothing.targets must be a list.")
    feature_transforms = cfg.get("feature_transforms")
    if feature_transforms is not None and not isinstance(feature_transforms, list):
        raise TypeError("oversmoothing.feature_transforms must be a list.")
    return OversmoothingRecorder(
        targets=[str(target) for target in targets],
        feature_transforms=None
        if feature_transforms is None
        else [str(feature_transform) for feature_transform in feature_transforms],
    )


def _build_site_eval_recorder(cfg: Any) -> SiteEvalRecorder | None:
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError("site_eval config must be a mapping.")
    unknown = set(cfg) - SITE_EVAL_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown site_eval config keys: {sorted(unknown)}")
    if not bool(cfg.get("enabled", True)):
        return None
    return SiteEvalRecorder()


def _build_postprocess_recorder(cfg: Any) -> PostprocessRecorder | None:
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError("postprocess config must be a mapping.")
    unknown = set(cfg) - POSTPROCESS_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown postprocess config keys: {sorted(unknown)}")
    if not bool(cfg.get("enabled", True)):
        return None
    return PostprocessRecorder()


def _build_site_size_recorder(cfg: Any) -> SiteSizeRecorder | None:
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError("site_size config must be a mapping.")
    unknown = set(cfg) - SITE_SIZE_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown site_size config keys: {sorted(unknown)}")
    if not bool(cfg.get("enabled", True)):
        return None
    bins = cfg.get("bins")
    if bins is not None and not isinstance(bins, list):
        raise TypeError("site_size.bins must be a list.")
    return SiteSizeRecorder(bins=bins)


def _build_nearfar_recorder(
    cfg: Any,
    *,
    checkpoint_epoch: int,
    checkpoint_global_step: int,
) -> NearFarRecorder | None:
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError("nearfar config must be a mapping.")
    unknown = set(cfg) - NEARFAR_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown nearfar config keys: {sorted(unknown)}")
    return NearFarRecorder(
        split_cutoff_A=float(cfg.get("split_cutoff_A", 10.0)),
        split_reference=str(cfg.get("split_reference", "site_center")),
        success_cutoff_A=float(
            cfg.get("success_cutoff_A", cfg.get("split_cutoff_A", 10.0))
        ),
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_global_step=checkpoint_global_step,
        store_per_query=bool(cfg.get("store_per_query", True)),
    )


def _parse_batch_limit(value: Any, name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or float, got bool.")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
        return value
    if isinstance(value, float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Float {name} must be in [0, 1], got {value}.")
        return value
    raise TypeError(f"{name} must be an int or float, got {type(value).__name__}.")


def _positive_int_or_none(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, got bool.")
    if not isinstance(value, int):
        raise TypeError(
            f"{name} must be a positive integer, got {type(value).__name__}."
        )
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _build_complexity_config(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {"enabled": False, "train": False, "test": False}
    if not isinstance(cfg, dict):
        raise TypeError("complexity config must be a mapping.")
    unknown = set(cfg) - COMPLEXITY_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown complexity config keys: {sorted(unknown)}")
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return {"enabled": False, "train": False, "test": False}
    train = bool(cfg.get("train", False))
    test = bool(cfg.get("test", True))
    if not train and not test:
        raise ValueError(
            "complexity config enabled=true requires at least one of train/test."
        )
    timing = str(cfg.get("timing", "step")).lower()
    if timing not in {"step", "total", "stage"}:
        raise ValueError(
            f"complexity.timing must be 'step', 'total', or 'stage', got {timing!r}."
        )
    measure_memory = bool(cfg.get("measure_memory", True))
    if timing == "stage" and train:
        raise ValueError(
            "complexity.timing=stage is only supported for test inference timing."
        )
    if timing in {"total", "stage"} and measure_memory:
        raise ValueError(
            f"complexity.timing={timing} requires complexity.measure_memory=false."
        )
    if timing == "step" and not measure_memory:
        raise ValueError(
            "complexity.measure_memory=false requires complexity.timing=total or stage."
        )
    return {
        "enabled": True,
        "train": train,
        "test": test,
        "train_batch_size": _positive_int_or_none(
            cfg.get("train_batch_size"), "complexity.train_batch_size"
        ),
        "test_batch_size": _positive_int_or_none(
            cfg.get("test_batch_size"), "complexity.test_batch_size"
        ),
        "train_limit_batches": _parse_batch_limit(
            cfg.get("train_limit_batches"), "complexity.train_limit_batches"
        ),
        "test_limit_batches": _parse_batch_limit(
            cfg.get("test_limit_batches"), "complexity.test_limit_batches"
        ),
        "timing": timing,
        "measure_memory": measure_memory,
    }


def _load_lightning_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a Lightning state_dict: {path}")
    return checkpoint


def _log_metrics_to_logger(logger: Any, metrics: dict[str, float], step: int) -> None:
    if not hasattr(logger, "log_metrics"):
        raise TypeError(
            f"Expected a Lightning logger with log_metrics, got {type(logger).__name__}."
        )
    logger.log_metrics(metrics, step=step)
    experiment = logger.experiment
    if not hasattr(experiment, "flush"):
        raise TypeError(
            f"Expected TensorBoard experiment to expose flush, got {type(experiment).__name__}."
        )
    experiment.flush()


def _write_complexity_summary(log_dir: Path, recorders: list[Any]) -> None:
    rows: list[dict[str, Any]] = []
    for recorder in recorders:
        rows.extend(recorder.summary_rows())
    if not rows:
        raise RuntimeError("No complexity rows were produced.")
    with (log_dir / "complexity_summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    fieldnames = list(rows[0].keys())
    for row in rows:
        extra = set(row) - set(fieldnames)
        if extra:
            raise ValueError(f"Complexity row has unexpected keys: {sorted(extra)}.")
    with (log_dir / "complexity_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics(log_dir: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        raise RuntimeError("Trainer.test returned no metric dictionaries.")
    empty_indices = [index for index, result in enumerate(results) if not result]
    if empty_indices:
        raise RuntimeError(
            "Trainer.test returned empty metric dictionaries at indices "
            f"{empty_indices}; enable the required evaluation recorder(s)."
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    for result in results:
        serializable.append(
            {key: float(value) for key, value in sorted(result.items())}
        )
    with (log_dir / "test_metrics.json").open("w") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)

    metric_keys = sorted({key for result in serializable for key in result})
    with (log_dir / "test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in metric_keys:
            values = [result[key] for result in serializable if key in result]
            if len(values) == 0:
                raise ValueError(f"Expected at least one value for metric {key}, got 0")
            reference = values[0]
            if any(
                not math.isclose(value, reference, rel_tol=1e-7, abs_tol=1e-8)
                for value in values[1:]
            ):
                raise ValueError(
                    f"Metric {key} has inconsistent duplicate values: {values}"
                )
            writer.writerow({"metric": key, "value": reference})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a SiteFlow checkpoint on explicit test datasets."
    )
    parser.add_argument(
        "--config",
        "--eval-config",
        dest="eval_config",
        required=True,
        help="Path to the eval-only yaml config.",
    )
    parser.add_argument(
        "--only-test-dataset",
        action="append",
        default=None,
        help="Evaluate only the named test dataset. May be passed multiple times.",
    )
    parser.add_argument(
        "--logger-version",
        default=None,
        help="Override output.version in the eval config. Required for parallel per-dataset eval jobs.",
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=None,
        help="Override flow.metric_threshold for this eval run.",
    )
    parser.add_argument(
        "--checkpoint", help="Override the checkpoint in the evaluation config."
    )
    parser.add_argument("--data-root", type=Path, help="Override the dataset root.")
    args = parser.parse_args()

    eval_cfg = _load_eval_config(args.eval_config)
    if args.only_test_dataset is not None:
        requested = [str(name) for name in args.only_test_dataset]
        requested_set = set(requested)
        if len(requested_set) != len(requested):
            raise ValueError(
                f"Duplicate --only-test-dataset values are not allowed: {requested}"
            )
        filtered = [
            cfg
            for cfg in eval_cfg["test_datasets"]
            if str(cfg["name"]) in requested_set
        ]
        found = {str(cfg["name"]) for cfg in filtered}
        missing = requested_set - found
        if missing:
            available = [str(cfg["name"]) for cfg in eval_cfg["test_datasets"]]
            raise ValueError(
                f"Requested test dataset(s) not found: {sorted(missing)}; available={available}"
            )
        eval_cfg["test_datasets"] = filtered
    if args.logger_version is not None:
        eval_cfg["output"] = deepcopy(eval_cfg["output"])
        eval_cfg["output"]["version"] = str(args.logger_version)
    if args.metric_threshold is not None:
        eval_cfg["flow"] = deepcopy(eval_cfg.get("flow", {}) or {})
        eval_cfg["flow"]["metric_threshold"] = float(args.metric_threshold)
    complexity_cfg = _build_complexity_config(eval_cfg.get("complexity"))
    run_cfg_path = Path(eval_cfg["run_config"]).expanduser().resolve()
    checkpoint_value = args.checkpoint or eval_cfg["checkpoint"]
    if not checkpoint_value:
        raise ValueError(
            "Pass --checkpoint or set checkpoint in the evaluation configuration."
        )
    eval_cfg["checkpoint"] = str(checkpoint_value)
    ckpt_path = Path(checkpoint_value).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    seed = eval_cfg.get("seed")
    if seed is not None:
        L.seed_everything(int(seed), workers=True)

    flow_overrides = deepcopy(eval_cfg.get("flow", {}) or {})
    data_overrides = deepcopy(eval_cfg.get("data", {}) or {})
    if complexity_cfg["test"] and complexity_cfg.get("test_batch_size") is not None:
        data_overrides["batch_size"] = int(complexity_cfg["test_batch_size"])
    data_overrides["test_datasets"] = deepcopy(eval_cfg["test_datasets"])
    if args.data_root is not None:
        data_overrides["root"] = str(args.data_root.resolve())

    trainer_overrides = deepcopy(eval_cfg["trainer"])
    output_cfg = deepcopy(eval_cfg["output"])
    for key in ("save_dir", "name", "version", "default_hp_metric"):
        if key not in output_cfg:
            raise ValueError(f"output config missing required key: {key}")
    trainer_overrides["logger"] = {
        "save_dir": str(Path(output_cfg["save_dir"]).expanduser()),
        "name": str(output_cfg["name"]),
        "version": output_cfg["version"],
        "default_hp_metric": bool(output_cfg["default_hp_metric"]),
    }
    trainer_overrides["enable_checkpointing"] = False
    trainer_overrides.setdefault("enable_model_summary", False)
    if complexity_cfg["test"] and complexity_cfg.get("test_limit_batches") is not None:
        trainer_overrides["limit_test_batches"] = complexity_cfg["test_limit_batches"]

    test_complexity_recorder = None
    if complexity_cfg["test"]:
        if complexity_cfg["timing"] == "total":
            test_complexity_recorder = EpochTimeRecorder(stages={"test"})
        elif complexity_cfg["timing"] == "stage":
            test_complexity_recorder = StageTimingRecorder()
        else:
            test_complexity_recorder = ComplexityRecorder(stages={"test"})
    extra_callbacks = (
        [test_complexity_recorder] if test_complexity_recorder is not None else None
    )

    runtime = build_pocket_runtime(
        run_cfg_path,
        data_overrides=data_overrides,
        model_overrides=deepcopy(eval_cfg.get("model", {}) or {}),
        flow_overrides=flow_overrides,
        trainer_overrides=trainer_overrides,
        extra_callbacks=extra_callbacks,
        include_checkpoint_callbacks=False,
        include_periodic_test_callback=False,
        include_memory_monitor=False,
    )

    checkpoint = _load_lightning_checkpoint(ckpt_path)
    runtime.module.load_state_dict(checkpoint["state_dict"], strict=True)
    runtime.module.eval_pocket_minimal = bool(eval_cfg.get("minimal_prediction", True))
    runtime.module.eval_collect_epoch_metrics = False

    logger = runtime.trainer.logger
    log_dir = Path(logger.log_dir)
    train_complexity_recorder = None
    if complexity_cfg["train"]:
        train_data_overrides = deepcopy(eval_cfg.get("data", {}) or {})
        train_data_overrides.pop("batch_size", None)
        if complexity_cfg.get("train_batch_size") is not None:
            train_data_overrides["batch_size"] = int(complexity_cfg["train_batch_size"])
        train_trainer_overrides = deepcopy(eval_cfg["trainer"])
        train_trainer_overrides["logger"] = False
        train_trainer_overrides["enable_checkpointing"] = False
        train_trainer_overrides["enable_model_summary"] = False
        train_trainer_overrides["enable_progress_bar"] = False
        train_trainer_overrides["max_epochs"] = 1
        train_trainer_overrides["num_sanity_val_steps"] = 0
        train_trainer_overrides["limit_val_batches"] = 1
        train_trainer_overrides["test_every_n_epoch"] = 0
        train_trainer_overrides["monitor_gpu_memory"] = False
        if complexity_cfg.get("train_limit_batches") is not None:
            train_trainer_overrides["limit_train_batches"] = complexity_cfg[
                "train_limit_batches"
            ]
        if complexity_cfg["timing"] == "total":
            train_complexity_recorder = EpochTimeRecorder(stages={"train"})
        else:
            train_complexity_recorder = ComplexityRecorder(stages={"train"})
        train_runtime = build_pocket_runtime(
            run_cfg_path,
            data_overrides=train_data_overrides,
            flow_overrides=flow_overrides,
            trainer_overrides=train_trainer_overrides,
            extra_callbacks=[train_complexity_recorder],
            include_checkpoint_callbacks=False,
            include_periodic_test_callback=False,
            include_memory_monitor=False,
        )
        train_runtime.module.load_state_dict(checkpoint["state_dict"], strict=True)
        train_runtime.trainer.fit(
            train_runtime.module, datamodule=train_runtime.data, ckpt_path=None
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    checkpoint_global_step = int(checkpoint.get("global_step", -1))
    site_eval_recorder = _build_site_eval_recorder(eval_cfg.get("site_eval"))
    oversmoothing_recorder = _build_oversmoothing_recorder(
        eval_cfg.get("oversmoothing")
    )
    postprocess_recorder = _build_postprocess_recorder(eval_cfg.get("postprocess"))
    site_size_recorder = _build_site_size_recorder(eval_cfg.get("site_size"))
    nearfar_recorder = _build_nearfar_recorder(
        eval_cfg.get("nearfar"),
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_global_step=checkpoint_global_step,
    )
    feature_recorders = [
        recorder
        for recorder in (
            site_eval_recorder,
            oversmoothing_recorder,
            postprocess_recorder,
            site_size_recorder,
            nearfar_recorder,
        )
        if recorder is not None
    ]
    if feature_recorders:
        num_devices = int(getattr(runtime.trainer, "num_devices", 1))
        if num_devices != 1:
            raise NotImplementedError(
                "eval feature recording currently requires a single eval device."
            )
        runtime.module.eval_feature_recorders = feature_recorders
    runtime.module.eval_collect_epoch_metrics = site_eval_recorder is not None

    results = runtime.trainer.test(
        runtime.module, datamodule=runtime.data, ckpt_path=None
    )
    complexity_recorders = [
        recorder
        for recorder in (train_complexity_recorder, test_complexity_recorder)
        if recorder is not None
    ]
    if complexity_recorders:
        if not results:
            raise RuntimeError("Trainer.test returned no result dictionaries.")
        complexity_metrics: dict[str, float] = {}
        for recorder in complexity_recorders:
            complexity_metrics.update(recorder.metric_dict())
        results[0].update(complexity_metrics)

    if runtime.trainer.is_global_zero:
        if complexity_recorders:
            _log_metrics_to_logger(
                runtime.trainer.logger,
                complexity_metrics,
                int(checkpoint.get("global_step", 0) or 0),
            )
        save_resolved_config(
            log_dir / "resolved_eval_config.yaml",
            {
                "eval_config_path": str(Path(args.eval_config).expanduser().resolve()),
                "run_config_path": str(run_cfg_path),
                "checkpoint": str(ckpt_path),
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_global_step": checkpoint_global_step,
                "strict_state_dict_load": True,
                "seed": seed,
                "source": {
                    "schema": runtime.loaded_cfg.schema,
                    "source_path": str(runtime.loaded_cfg.source_path),
                    "resolved": runtime.loaded_cfg.resolved,
                    "train": runtime.source_cfg,
                },
                "eval": eval_cfg,
                "effective_train": runtime.effective_cfg,
            },
        )
        _write_metrics(log_dir, results)
        if complexity_recorders:
            _write_complexity_summary(log_dir, complexity_recorders)
        if oversmoothing_recorder is not None:
            oversmoothing_recorder.write(log_dir)
        if postprocess_recorder is not None:
            postprocess_recorder.write(log_dir)
        if site_size_recorder is not None:
            site_size_recorder.write(log_dir)
        if nearfar_recorder is not None:
            nearfar_recorder.write(log_dir)


if __name__ == "__main__":
    main()
