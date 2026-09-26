from __future__ import annotations

from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint


BEST_VAL_METRIC_MONITORS: tuple[str, ...] = (
    "val/ap_iou_0.3",
    "val/ap_iou_0.5",
    "val/site_ap_iou_0.3",
    "val/site_ap_iou_0.5",
    "val/query_dcc_topn",
    "val/query_dca_topn",
    "val/query_dcc_topn_plus_2",
    "val/query_dca_topn_plus_2",
    "val/site_dcc_topn",
    "val/site_dca_topn",
    "val/site_dcc_topn_plus_2",
    "val/site_dca_topn_plus_2",
)


def _safe_monitor_name(monitor: str) -> str:
    return monitor.replace("/", "_").replace(".", "p")


def build_best_val_metric_checkpoints(
    checkpoint_cfg: dict[str, Any] | None = None,
    *,
    enabled: bool = True,
) -> list[ModelCheckpoint]:
    """Build one best-checkpoint callback for each standard validation metric."""
    if not enabled:
        return []
    cfg = dict(checkpoint_cfg or {})
    if cfg.get("save_best_val_metric_checkpoints", True) is False:
        return []

    save_top_k = int(cfg.get("best_val_metric_save_top_k", 1))
    if save_top_k == 0:
        return []

    dirpath = cfg.get("best_val_metric_dirpath", cfg.get("dirpath"))
    monitors = cfg.get("best_val_metric_monitors", BEST_VAL_METRIC_MONITORS)
    if isinstance(monitors, str):
        monitors = [monitors]
    callbacks: list[ModelCheckpoint] = []
    for monitor in monitors:
        safe_name = _safe_monitor_name(monitor)
        kwargs: dict[str, Any] = {}
        if dirpath is not None:
            kwargs["dirpath"] = dirpath
        callbacks.append(
            ModelCheckpoint(
                monitor=monitor,
                mode="max",
                save_top_k=save_top_k,
                save_last=False,
                filename=str(
                    cfg.get(
                        "best_val_metric_filename", f"best-{safe_name}-{{epoch:03d}}"
                    )
                ),
                auto_insert_metric_name=False,
                **kwargs,
            )
        )
    return callbacks


def build_last_checkpoint(
    checkpoint_cfg: dict[str, Any] | None = None,
    *,
    enabled: bool = True,
) -> list[ModelCheckpoint]:
    """Build a checkpoint callback that always writes ``last.ckpt``.

    The metric-specific callbacks intentionally keep ``save_last=False`` so that
    multiple best-metric callbacks do not race to write the same last checkpoint.
    This standalone callback owns the last checkpoint path.
    """
    if not enabled:
        return []
    cfg = dict(checkpoint_cfg or {})
    if cfg.get("save_last", True) is False:
        return []

    kwargs: dict[str, Any] = {}
    dirpath = cfg.get("last_dirpath", cfg.get("dirpath"))
    if dirpath is not None:
        kwargs["dirpath"] = dirpath
    return [
        ModelCheckpoint(
            save_top_k=0,
            save_last=True,
            auto_insert_metric_name=False,
            **kwargs,
        )
    ]


def build_periodic_checkpoints(
    checkpoint_cfg: dict[str, Any] | None = None,
    *,
    enabled: bool = True,
) -> list[ModelCheckpoint]:
    """Build epoch-numbered periodic checkpoints when explicitly requested."""
    if not enabled:
        return []
    cfg = dict(checkpoint_cfg or {})
    if not bool(cfg.get("save_periodic", False)):
        return []

    every_n_epochs = int(cfg.get("periodic_every_n_epochs", 0))
    if every_n_epochs <= 0:
        raise ValueError(
            f"periodic_every_n_epochs must be positive, got {every_n_epochs}."
        )

    kwargs: dict[str, Any] = {}
    dirpath = cfg.get("periodic_dirpath", cfg.get("dirpath"))
    if dirpath is not None:
        kwargs["dirpath"] = dirpath
    return [
        ModelCheckpoint(
            monitor=str(cfg.get("periodic_monitor", "epoch")),
            mode=str(cfg.get("periodic_mode", "max")),
            save_top_k=-1,
            save_last=False,
            every_n_epochs=every_n_epochs,
            filename=str(cfg.get("periodic_filename", "epoch-{epoch:03d}")),
            auto_insert_metric_name=False,
            **kwargs,
        )
    ]
