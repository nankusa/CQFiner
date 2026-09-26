import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import UnifiedStructureDataModule
from src.config import load_training_config
from src.config.load import save_resolved_config
from src.model import build_site_model, resolve_model_config, summarize_site_model
from src.training import (
    PeriodicTestCallback,
    SiteDisplacementModule,
    build_best_val_metric_checkpoints,
    build_gpu_memory_monitor,
    build_last_checkpoint,
    build_periodic_checkpoints,
)


def _parse_logger_version(value):
    if value is None:
        return None
    text = str(value)
    if text.isdigit():
        return int(text)
    return text


@dataclass
class PocketRuntime:
    loaded_cfg: Any
    effective_cfg: dict[str, Any]
    source_cfg: dict[str, Any]
    data: UnifiedStructureDataModule
    module: SiteDisplacementModule
    trainer: L.Trainer
    ckpt_path: Path | None


def _resolve_ckpt_path(ckpt_path: str | Path | None) -> Path | None:
    if ckpt_path is None:
        return None
    resolved = Path(ckpt_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resolved}")
    return resolved


def build_pocket_runtime(
    config_path: str | Path,
    *,
    ckpt_path: str | Path | None = None,
    logger_version: str | int | None = None,
    data_overrides: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    flow_overrides: Mapping[str, Any] | None = None,
    trainer_overrides: Mapping[str, Any] | None = None,
    extra_callbacks: list[Any] | None = None,
    include_checkpoint_callbacks: bool = True,
    include_periodic_test_callback: bool = True,
    include_memory_monitor: bool = True,
) -> PocketRuntime:
    resolved_ckpt_path = _resolve_ckpt_path(ckpt_path)
    loaded_cfg = load_training_config(config_path)
    cfg = deepcopy(loaded_cfg.train)
    if flow_overrides:
        cfg["flow"] = deepcopy(cfg["flow"])
        cfg["flow"].update(deepcopy(dict(flow_overrides)))
    if model_overrides:
        cfg["model"] = deepcopy(cfg["model"])
        cfg["model"].update(deepcopy(dict(model_overrides)))

    if cfg.get("seed") is not None:
        L.seed_everything(cfg["seed"], workers=True)

    flow_cfg = cfg["flow"]
    query_loss_balance_mode = str(
        flow_cfg.get("query_loss_balance_mode", "manual")
    ).lower()
    query_ranking_mode = str(
        flow_cfg.get("query_ranking_mode", "distance_distribution")
    ).lower()
    if query_ranking_mode in {
        "distance",
        "distance_distribution",
        "distribution",
        "dfl",
    }:
        query_ranking_mode = "distance_distribution"
    elif query_ranking_mode in {"confidence", "vnegnn", "scalar_confidence"}:
        query_ranking_mode = "confidence"
    else:
        raise ValueError(f"Unsupported flow.query_ranking_mode: {query_ranking_mode}")
    train_forward_passes = int(flow_cfg.get("train_forward_passes", 1))
    eval_forward_passes = int(flow_cfg.get("eval_forward_passes", 2))
    eval_use_stage0_offset_without_second_forward = bool(
        flow_cfg.get("eval_use_stage0_offset_without_second_forward", False)
    )
    if eval_use_stage0_offset_without_second_forward and eval_forward_passes != 1:
        raise ValueError(
            "flow.eval_use_stage0_offset_without_second_forward=true requires "
            "flow.eval_forward_passes=1."
        )
    eval_query_position_cache = bool(flow_cfg.get("eval_query_position_cache", False))
    eval_query_sampling_mode = str(
        flow_cfg.get("eval_query_sampling_mode", "fps")
    ).lower()
    if eval_query_sampling_mode not in {"fps", "random", "curvature", "atom_volume"}:
        raise ValueError(
            f"Unsupported flow.eval_query_sampling_mode: {eval_query_sampling_mode!r}"
        )
    if eval_query_sampling_mode != "fps" and eval_query_position_cache:
        raise ValueError(
            "flow.eval_query_position_cache=true is only supported with flow.eval_query_sampling_mode='fps'."
        )
    eval_num_query_nodes = flow_cfg.get("eval_num_query_nodes")
    if eval_num_query_nodes is None:
        eval_num_query_nodes = int(cfg["data"]["num_query_nodes"]) * 2
    else:
        eval_num_query_nodes = int(eval_num_query_nodes)
    optimizer_cfg = deepcopy(cfg["optimizer"])

    model_cfg = resolve_model_config(deepcopy(cfg["model"]))
    # Graph cutoffs are authoritative for both topology and backbone radial support.
    graph_cutoff = max(
        float(cfg["graph"]["host_host_cutoff"]),
        float(cfg["graph"]["host_query_cutoff"]),
    )
    if (
        model_overrides
        and "cutoff" in model_overrides
        and float(model_overrides["cutoff"]) != graph_cutoff
    ):
        raise ValueError(
            "Set cutoff through graph.host_host_cutoff/host_query_cutoff, not model overrides."
        )
    model_cfg["cutoff"] = graph_cutoff
    model_cfg["use_time_conditioning"] = False
    model_cfg["query_ranking_head"] = query_ranking_mode
    data_cfg = deepcopy(cfg["data"])
    if data_overrides:
        data_cfg.update(deepcopy(dict(data_overrides)))
    host_feature_mode = str(data_cfg["host_feature_mode"]).lower()
    if host_feature_mode in {"zeros", "grasp"}:
        if host_feature_mode == "grasp":
            if data_cfg.get("host_feature_dim") is None:
                data_cfg["host_feature_dim"] = 60
        else:
            if data_cfg.get("host_feature_dim") is None:
                data_cfg["host_feature_dim"] = int(model_cfg["input_dim"])
        if int(data_cfg["host_feature_dim"]) != int(model_cfg["input_dim"]):
            raise ValueError(
                f"When data.host_feature_mode='{host_feature_mode}', data.host_feature_dim must match model.input_dim "
                f"({data_cfg['host_feature_dim']} != {model_cfg['input_dim']})."
            )
    data = UnifiedStructureDataModule(**data_cfg)
    model = build_site_model(model_cfg)
    model_runtime = summarize_site_model(model)

    trainer_cfg = dict(cfg["trainer"])
    if trainer_overrides:
        trainer_cfg.update(deepcopy(dict(trainer_overrides)))
    test_every_n_epoch = int(trainer_cfg.pop("test_every_n_epoch", 0) or 0)
    checkpoint_cfg = dict(trainer_cfg.pop("checkpoint", {}) or {})
    memory_monitor_cfg = {
        "monitor_gpu_memory": trainer_cfg.pop("monitor_gpu_memory", True)
    }
    if "gpu_memory_monitor" in trainer_cfg:
        memory_monitor_cfg["gpu_memory_monitor"] = trainer_cfg.pop("gpu_memory_monitor")
    logger_cfg = trainer_cfg.get("logger", None)
    resolved_logger_cfg = logger_cfg
    parsed_logger_version = _parse_logger_version(logger_version)
    if logger_cfg is None or logger_cfg is True:
        default_logger_cfg = cfg["trainer"]["logger"]
        resolved_logger_cfg = {
            "save_dir": str(Path(default_logger_cfg["save_dir"]).expanduser()),
            "name": default_logger_cfg["name"],
            "version": parsed_logger_version
            if parsed_logger_version is not None
            else default_logger_cfg.get("version"),
            "default_hp_metric": bool(default_logger_cfg["default_hp_metric"]),
        }
        trainer_cfg["logger"] = TensorBoardLogger(**resolved_logger_cfg)
    elif isinstance(logger_cfg, dict):
        resolved_logger_cfg = {
            "save_dir": str(Path(logger_cfg["save_dir"]).expanduser()),
            "name": logger_cfg["name"],
            "version": parsed_logger_version
            if parsed_logger_version is not None
            else logger_cfg.get("version"),
            "default_hp_metric": bool(logger_cfg["default_hp_metric"]),
        }
        trainer_cfg["logger"] = TensorBoardLogger(**resolved_logger_cfg)

    callbacks = trainer_cfg.get("callbacks") or []
    if not isinstance(callbacks, list):
        callbacks = [callbacks]
    if include_checkpoint_callbacks:
        callbacks.extend(build_last_checkpoint(checkpoint_cfg))
        callbacks.extend(build_periodic_checkpoints(checkpoint_cfg))
        callbacks.extend(
            build_best_val_metric_checkpoints(
                checkpoint_cfg,
                enabled=bool(flow_cfg.get("rank_based_eval", False)),
            )
        )
    if (
        include_periodic_test_callback
        and test_every_n_epoch > 0
        and getattr(data, "test_datasets", None)
    ):
        callbacks.append(PeriodicTestCallback(test_every_n_epoch))
    if include_memory_monitor:
        memory_monitor = build_gpu_memory_monitor(memory_monitor_cfg)
        if memory_monitor is not None:
            callbacks.append(memory_monitor)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    if callbacks:
        trainer_cfg["callbacks"] = callbacks

    effective_cfg = deepcopy(cfg)
    effective_cfg["data"] = deepcopy(data_cfg)
    effective_cfg["model"] = deepcopy(model_cfg)
    effective_cfg["model_runtime"] = deepcopy(model_runtime)
    effective_cfg["optimizer"] = deepcopy(optimizer_cfg)
    effective_cfg["flow"] = {
        **deepcopy(flow_cfg),
        "query_loss_balance_mode": query_loss_balance_mode,
        "query_ranking_mode": query_ranking_mode,
        "train_forward_passes": train_forward_passes,
        "eval_forward_passes": eval_forward_passes,
        "eval_use_stage0_offset_without_second_forward": eval_use_stage0_offset_without_second_forward,
        "eval_query_position_cache": eval_query_position_cache,
        "eval_query_sampling_mode": eval_query_sampling_mode,
        "eval_num_query_nodes": eval_num_query_nodes,
    }
    effective_cfg["trainer"] = {
        **deepcopy(cfg["trainer"]),
        **deepcopy(dict(trainer_overrides or {})),
        "test_every_n_epoch": test_every_n_epoch,
        "logger": resolved_logger_cfg,
    }
    if resolved_ckpt_path is not None:
        effective_cfg["resume"] = {"ckpt_path": str(resolved_ckpt_path)}

    module = SiteDisplacementModule(
        model=model,
        optimizer_cfg=optimizer_cfg,
        run_config=effective_cfg,
        num_query_nodes=cfg["data"]["num_query_nodes"],
        eval_num_query_nodes=eval_num_query_nodes,
        host_host_cutoff=cfg["graph"]["host_host_cutoff"],
        host_query_cutoff=cfg["graph"]["host_query_cutoff"],
        graph_max_neighbors=cfg["graph"]["max_neighbors"],
        rank_based_eval=flow_cfg["rank_based_eval"],
        loss_weights=cfg["loss_weights"],
        query_loss_balance_mode=query_loss_balance_mode,
        query_ranking_mode=query_ranking_mode,
        confidence_gamma=flow_cfg.get("confidence_gamma", 4.0),
        confidence_c0=flow_cfg.get("confidence_c0", 0.001),
        confidence_positive_weight=flow_cfg.get("confidence_positive_weight", 1.0),
        confidence_negative_weight=flow_cfg.get("confidence_negative_weight", 1.0),
        metric_threshold=flow_cfg["metric_threshold"],
        query_distance_bin_size=flow_cfg["query_distance_bin_size"],
        query_contrastive_temperature=flow_cfg.get(
            "query_contrastive_temperature", 0.1
        ),
        query_surface_min_offset=flow_cfg["query_surface_min_offset"],
        query_surface_max_offset=flow_cfg["query_surface_max_offset"],
        query_surface_tangent_jitter=flow_cfg["query_surface_tangent_jitter"],
        query_surface_normal_neighbors=flow_cfg["query_surface_normal_neighbors"],
        query_sampling=flow_cfg.get("query_sampling", "surface"),
        query_volume_radius=flow_cfg.get("query_volume_radius", 10.0),
        query_sampling_seed=flow_cfg.get("query_sampling_seed", 42),
        query_nms_score_aggregation=flow_cfg.get("query_nms_score_aggregation"),
        query_eval_nms_radius=flow_cfg["query_eval_nms_radius"],
        site_mask_nms_radius=flow_cfg.get("site_mask_nms_radius"),
        site_mask_query_score_threshold=flow_cfg.get(
            "site_mask_query_score_threshold", 0.5
        ),
        site_mask_score_aggregation=flow_cfg.get("site_mask_score_aggregation", "mean"),
        site_mask_threshold=flow_cfg.get("site_mask_threshold", 0.5),
        site_mask_grouping=flow_cfg.get("site_mask_grouping", "nms"),
        site_mask_cluster_threshold=flow_cfg.get("site_mask_cluster_threshold"),
        site_mask_affinity_threshold=flow_cfg.get("site_mask_affinity_threshold", 0.5),
        site_mask_affinity_loss_weight=flow_cfg.get(
            "site_mask_affinity_loss_weight", 0.0
        ),
        site_mask_affinity_max_pairs=flow_cfg.get("site_mask_affinity_max_pairs", 4096),
        query_disp_supervision_cutoff=flow_cfg["query_disp_supervision_cutoff"],
        interaction_host_to_query=flow_cfg["interaction_host_to_query"],
        interaction_query_to_host=flow_cfg["interaction_query_to_host"],
        interaction_query_to_query=flow_cfg["interaction_query_to_query"],
        interaction_graph_mode=flow_cfg.get("interaction_graph_mode", "cutoff"),
        train_forward_passes=train_forward_passes,
        eval_forward_passes=eval_forward_passes,
        eval_use_stage0_offset_without_second_forward=eval_use_stage0_offset_without_second_forward,
        eval_query_position_cache=eval_query_position_cache,
        eval_query_sampling_mode=eval_query_sampling_mode,
        runtime_host_host_filter=flow_cfg.get("runtime_host_host_filter"),
    )

    trainer = L.Trainer(**trainer_cfg)
    return PocketRuntime(
        loaded_cfg=loaded_cfg,
        effective_cfg=effective_cfg,
        source_cfg=cfg,
        data=data,
        module=module,
        trainer=trainer,
        ckpt_path=resolved_ckpt_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train SiteFlow pocket model with direct query displacement"
    )
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Lightning checkpoint path for exact training resume.",
    )
    parser.add_argument(
        "--logger-version",
        default=None,
        help="TensorBoard logger version override. Numeric values keep Lightning's version_N directory format.",
    )
    parser.add_argument("--data-root", type=Path, help="Override the dataset root.")
    args = parser.parse_args()

    runtime = build_pocket_runtime(
        args.config,
        ckpt_path=args.ckpt_path,
        logger_version=args.logger_version,
        data_overrides={"root": str(args.data_root.resolve())} if args.data_root is not None else None,
    )
    trainer = runtime.trainer
    logger = getattr(runtime.trainer, "logger", None)
    log_dir = getattr(logger, "log_dir", None)
    if log_dir is not None and trainer.is_global_zero:
        save_resolved_config(
            Path(log_dir) / "resolved_config.yaml",
            {
                "schema": runtime.loaded_cfg.schema,
                "source_path": str(runtime.loaded_cfg.source_path),
                "resolved": runtime.loaded_cfg.resolved,
                "train": runtime.effective_cfg,
            },
        )
    trainer.fit(
        runtime.module,
        datamodule=runtime.data,
        ckpt_path=str(runtime.ckpt_path) if runtime.ckpt_path is not None else None,
    )


if __name__ == "__main__":
    main()
