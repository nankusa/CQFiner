from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _batch_sample_ids(batch: Any) -> list[str]:
    sample_ids = getattr(batch, "sample_id", [])
    if isinstance(sample_ids, str):
        return [sample_ids]
    return [str(sample_id) for sample_id in sample_ids]


def _safe_float_name(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [float(row[key]) for row in rows]
    return [value for value in values if math.isfinite(value)]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float64).mean().item())


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float64).median().item())


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    if not values:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float64).sum().item())


def _split_distance_key(split_reference: str) -> str:
    if split_reference == "site_center":
        return "initial_dist_A"
    if split_reference == "ligand_atom":
        return "initial_nearest_ligand_atom_dist_A"
    raise ValueError(f"Unsupported split_reference: {split_reference}")


def _group_rows(
    rows: list[dict[str, Any]],
    *,
    split_reference: str,
    split_cutoff_A: float,
    group: str,
) -> list[dict[str, Any]]:
    if group == "all":
        return rows
    distance_key = _split_distance_key(split_reference)
    if group == "near":
        return [row for row in rows if float(row[distance_key]) <= split_cutoff_A]
    if group == "far":
        return [row for row in rows if float(row[distance_key]) > split_cutoff_A]
    raise ValueError(f"Unsupported near/far group: {group}")


def _site_group_reduced_mean(rows: list[dict[str, Any]], key: str) -> float:
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        value = float(row[key])
        if not math.isfinite(value):
            continue
        group_key = (str(row["sample_id"]), int(row["target_site_id"]))
        groups.setdefault(group_key, []).append(value)
    if not groups:
        return float("nan")
    group_means = [
        torch.tensor(values, dtype=torch.float64).mean().item()
        for values in groups.values()
    ]
    return float(torch.tensor(group_means, dtype=torch.float64).mean().item())


def _binary_dfl_metrics(
    rows: list[dict[str, Any]],
    *,
    split_reference: str,
    split_cutoff_A: float,
    stage: str,
) -> dict[str, float]:
    if stage not in {"stage0", "final"}:
        raise ValueError(f"Unsupported DFL stage: {stage}")
    if not rows:
        return {
            "binary_acc": float("nan"),
            "true_far_rate": float("nan"),
            "pred_near_rate": float("nan"),
            "pred_far_rate": float("nan"),
            "far_recall": float("nan"),
            "far_precision": float("nan"),
        }
    distance_key = _split_distance_key(split_reference)
    true_near = torch.tensor(
        [float(row[distance_key]) <= split_cutoff_A for row in rows],
        dtype=torch.bool,
    )
    pred_near = torch.tensor(
        [float(row[f"dfl_{stage}_pred_dist_A"]) <= split_cutoff_A for row in rows],
        dtype=torch.bool,
    )
    true_far = ~true_near
    pred_far = ~pred_near
    true_far_count = int(true_far.sum().item())
    pred_far_count = int(pred_far.sum().item())
    true_positive_far = int((true_far & pred_far).sum().item())
    return {
        "binary_acc": float((true_near == pred_near).float().mean().item()),
        "true_far_rate": float(true_far.float().mean().item()),
        "pred_near_rate": float(pred_near.float().mean().item()),
        "pred_far_rate": float(pred_far.float().mean().item()),
        "far_recall": float(true_positive_far / true_far_count)
        if true_far_count > 0
        else float("nan"),
        "far_precision": float(true_positive_far / pred_far_count)
        if pred_far_count > 0
        else float("nan"),
    }


class NearFarRecorder:
    def __init__(
        self,
        *,
        split_cutoff_A: float = 10.0,
        split_reference: str = "site_center",
        success_cutoff_A: float = 10.0,
        checkpoint_epoch: int = -1,
        checkpoint_global_step: int = -1,
        store_per_query: bool = True,
    ) -> None:
        self.split_cutoff_A = float(split_cutoff_A)
        if self.split_cutoff_A <= 0.0:
            raise ValueError(f"split_cutoff_A must be positive, got {split_cutoff_A}.")
        self.split_reference = str(split_reference).lower()
        if self.split_reference not in {"site_center", "ligand_atom"}:
            raise ValueError(f"Unsupported split_reference: {split_reference!r}.")
        self.success_cutoff_A = float(success_cutoff_A)
        if self.success_cutoff_A <= 0.0:
            raise ValueError(
                f"success_cutoff_A must be positive, got {success_cutoff_A}."
            )
        self.checkpoint_epoch = int(checkpoint_epoch)
        self.checkpoint_global_step = int(checkpoint_global_step)
        self.store_per_query = bool(store_per_query)
        self.rows: list[dict[str, Any]] = []

    def collect_batch(
        self,
        *,
        batch: Any,
        out: dict[str, Tensor],
        query_batch: Tensor,
        dataset_name: str,
        module: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        if module is None:
            raise RuntimeError("NearFarRecorder requires the Lightning module.")
        if context is None:
            raise RuntimeError("NearFarRecorder requires inference context.")
        if module.query_ranking_mode != "distance_distribution":
            raise ValueError(
                f"Near/far DFL metrics require distance_distribution mode, got {module.query_ranking_mode!r}."
            )

        required = {"query_pos_0", "final_pos", "out_init", "query_disp_pred"}
        missing = required - set(context)
        if missing:
            raise RuntimeError(
                f"NearFarRecorder context is missing keys: {sorted(missing)}"
            )

        query_pos_0 = context["query_pos_0"]
        final_pos = context["final_pos"]
        out_init = context["out_init"]
        query_disp_pred = context["query_disp_pred"]
        if "query_distance_logits" not in out_init:
            raise RuntimeError(
                "Near/far stage-0 DFL metrics require out_init['query_distance_logits']."
            )
        if "query_distance_logits" not in out:
            raise RuntimeError(
                "Near/far final DFL metrics require out['query_distance_logits']."
            )
        if (
            query_pos_0.shape != final_pos.shape
            or query_pos_0.shape != query_disp_pred.shape
        ):
            raise ValueError(
                "Near/far query tensor shape mismatch: "
                f"query_pos_0={tuple(query_pos_0.shape)}, final_pos={tuple(final_pos.shape)}, "
                f"query_disp_pred={tuple(query_disp_pred.shape)}."
            )
        if query_batch.shape != (query_pos_0.size(0),):
            raise ValueError(
                "Near/far query_batch shape mismatch: "
                f"expected {(query_pos_0.size(0),)}, got {tuple(query_batch.shape)}."
            )

        target_pos, target_site_ids, supervise = module._build_query_position_targets(
            batch, query_pos_0, query_batch
        )
        final_nearest_target_pos, final_nearest_site_ids, final_has_target = (
            module._assign_query_site_centers(
                final_pos,
                query_batch,
                batch,
            )
        )
        valid = (
            supervise
            & final_has_target
            & (target_site_ids >= 0)
            & (final_nearest_site_ids >= 0)
            & torch.isfinite(query_pos_0).all(dim=-1)
            & torch.isfinite(final_pos).all(dim=-1)
            & torch.isfinite(target_pos).all(dim=-1)
            & torch.isfinite(final_nearest_target_pos).all(dim=-1)
        )
        if not bool(valid.any().item()):
            raise RuntimeError(
                f"NearFarRecorder found no valid supervised queries for dataset {dataset_name}."
            )

        target_disp = module._build_query_displacement_targets(
            query_pos_0=query_pos_0,
            target_pos=target_pos,
        )
        disp_error = query_disp_pred - target_disp
        disp_mse = disp_error.square().mean(dim=-1)
        disp_l2 = torch.linalg.norm(disp_error, dim=-1)

        dfl_stage0_target = module._query_distance_target_distribution(
            query_pos_0, query_batch, batch
        )
        dfl_stage0_ce, dfl_stage0_valid = (
            module._query_distance_distribution_per_query_loss(
                out_init["query_distance_logits"],
                dfl_stage0_target,
            )
        )
        dfl_stage0_pred_dist, dfl_stage0_score = module._decode_query_distance_logits(
            out_init["query_distance_logits"],
        )
        dfl_final_target = module._query_distance_target_distribution(
            final_pos, query_batch, batch
        )
        dfl_final_ce, dfl_final_valid = (
            module._query_distance_distribution_per_query_loss(
                out["query_distance_logits"],
                dfl_final_target,
            )
        )
        dfl_final_pred_dist, dfl_final_score = module._decode_query_distance_logits(
            out["query_distance_logits"]
        )

        init_dist = torch.linalg.norm(target_pos - query_pos_0, dim=-1)
        final_assigned_dist = torch.linalg.norm(target_pos - final_pos, dim=-1)
        final_nearest_site_center_dist = torch.linalg.norm(
            final_nearest_target_pos - final_pos, dim=-1
        )
        initial_ligand_dist = module._nearest_ligand_distances_for_points(
            query_pos_0, query_batch, batch
        )
        final_ligand_dist = module._nearest_ligand_distances_for_points(
            final_pos, query_batch, batch
        )
        target_disp_norm = torch.linalg.norm(target_disp, dim=-1)
        pred_disp_norm = torch.linalg.norm(query_disp_pred, dim=-1)

        sample_ids = _batch_sample_ids(batch)
        query_indices = torch.arange(query_pos_0.size(0), device=query_pos_0.device)
        for sample_idx, sample_id in enumerate(sample_ids):
            sample_valid = (query_batch == sample_idx) & valid
            if not bool(sample_valid.any().item()):
                raise RuntimeError(
                    f"Sample {sample_id} in dataset {dataset_name} has no valid supervised near/far queries."
                )
            sample_global_indices = query_indices[query_batch == sample_idx].tolist()
            for local_query_idx, global_query_idx in enumerate(sample_global_indices):
                if not bool(valid[global_query_idx].item()):
                    continue
                self.rows.append(
                    {
                        "checkpoint_epoch": self.checkpoint_epoch,
                        "checkpoint_global_step": self.checkpoint_global_step,
                        "dataset": str(dataset_name),
                        "sample_id": str(sample_id),
                        "query_index": int(local_query_idx),
                        "target_site_id": int(
                            target_site_ids[global_query_idx].detach().cpu().item()
                        ),
                        "final_nearest_site_id": int(
                            final_nearest_site_ids[global_query_idx]
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "final_nearest_site_changed": int(
                            bool(
                                (
                                    final_nearest_site_ids[global_query_idx]
                                    != target_site_ids[global_query_idx]
                                ).item()
                            )
                        ),
                        "initial_dist_A": float(
                            init_dist[global_query_idx].detach().cpu().item()
                        ),
                        "final_assigned_dist_A": float(
                            final_assigned_dist[global_query_idx].detach().cpu().item()
                        ),
                        "final_nearest_site_center_dist_A": float(
                            final_nearest_site_center_dist[global_query_idx]
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "initial_nearest_ligand_atom_dist_A": float(
                            initial_ligand_dist[global_query_idx].detach().cpu().item()
                        ),
                        "final_nearest_ligand_atom_dist_A": float(
                            final_ligand_dist[global_query_idx].detach().cpu().item()
                        ),
                        "disp_mse": float(
                            disp_mse[global_query_idx].detach().cpu().item()
                        ),
                        "disp_l2_A": float(
                            disp_l2[global_query_idx].detach().cpu().item()
                        ),
                        "target_disp_norm_A": float(
                            target_disp_norm[global_query_idx].detach().cpu().item()
                        ),
                        "pred_disp_norm_A": float(
                            pred_disp_norm[global_query_idx].detach().cpu().item()
                        ),
                        "dfl_stage0_pred_dist_A": float(
                            dfl_stage0_pred_dist[global_query_idx].detach().cpu().item()
                        ),
                        "dfl_stage0_rank_score": float(
                            dfl_stage0_score[global_query_idx].detach().cpu().item()
                        ),
                        "dfl_stage0_ce": float(
                            dfl_stage0_ce[global_query_idx].detach().cpu().item()
                        ),
                        "dfl_stage0_valid": int(
                            bool(
                                dfl_stage0_valid[global_query_idx].detach().cpu().item()
                            )
                        ),
                        "dfl_final_pred_dist_A": float(
                            dfl_final_pred_dist[global_query_idx].detach().cpu().item()
                        ),
                        "dfl_final_rank_score": float(
                            dfl_final_score[global_query_idx].detach().cpu().item()
                        ),
                        "dfl_final_ce": float(
                            dfl_final_ce[global_query_idx].detach().cpu().item()
                        ),
                        "dfl_final_valid": int(
                            bool(
                                dfl_final_valid[global_query_idx].detach().cpu().item()
                            )
                        ),
                    }
                )

    def summary_rows(self) -> list[dict[str, Any]]:
        if not self.rows:
            raise RuntimeError("NearFarRecorder has no rows to summarize.")
        summary_rows: list[dict[str, Any]] = []
        grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            key = (
                int(row["checkpoint_epoch"]),
                int(row["checkpoint_global_step"]),
                str(row["dataset"]),
            )
            grouped.setdefault(key, []).append(row)

        success_name = _safe_float_name(self.success_cutoff_A)
        for (epoch, global_step, dataset_name), rows in sorted(grouped.items()):
            for group in ("all", "near", "far"):
                selected = _group_rows(
                    rows,
                    split_reference=self.split_reference,
                    split_cutoff_A=self.split_cutoff_A,
                    group=group,
                )
                row: dict[str, Any] = {
                    "checkpoint_epoch": int(epoch),
                    "checkpoint_global_step": int(global_step),
                    "dataset": dataset_name,
                    "split_reference": self.split_reference,
                    "split_cutoff_A": self.split_cutoff_A,
                    "group": group,
                    "num_queries": int(len(selected)),
                    "query_fraction": float(len(selected) / len(rows))
                    if rows
                    else float("nan"),
                    "initial_dist_mean_A": _mean(selected, "initial_dist_A"),
                    "initial_dist_median_A": _median(selected, "initial_dist_A"),
                    "final_assigned_dist_mean_A": _mean(
                        selected, "final_assigned_dist_A"
                    ),
                    "final_assigned_dist_median_A": _median(
                        selected, "final_assigned_dist_A"
                    ),
                    "final_nearest_site_center_dist_mean_A": _mean(
                        selected,
                        "final_nearest_site_center_dist_A",
                    ),
                    "final_nearest_site_center_dist_median_A": _median(
                        selected,
                        "final_nearest_site_center_dist_A",
                    ),
                    "final_nearest_site_changed_rate": _mean(
                        selected, "final_nearest_site_changed"
                    ),
                    "initial_nearest_ligand_atom_dist_mean_A": _mean(
                        selected,
                        "initial_nearest_ligand_atom_dist_A",
                    ),
                    "initial_nearest_ligand_atom_dist_median_A": _median(
                        selected,
                        "initial_nearest_ligand_atom_dist_A",
                    ),
                    "final_nearest_ligand_atom_dist_mean_A": _mean(
                        selected,
                        "final_nearest_ligand_atom_dist_A",
                    ),
                    "final_nearest_ligand_atom_dist_median_A": _median(
                        selected,
                        "final_nearest_ligand_atom_dist_A",
                    ),
                    "disp_mse_mean": _mean(selected, "disp_mse"),
                    "disp_mse_sum": _sum(selected, "disp_mse"),
                    "disp_mse_site_group_reduced": _site_group_reduced_mean(
                        selected, "disp_mse"
                    ),
                    "disp_l2_mean_A": _mean(selected, "disp_l2_A"),
                    "dfl_stage0_ce_mean": _mean(selected, "dfl_stage0_ce"),
                    "dfl_stage0_ce_sum": _sum(selected, "dfl_stage0_ce"),
                    "dfl_stage0_ce_site_group_reduced": _site_group_reduced_mean(
                        selected, "dfl_stage0_ce"
                    ),
                    "dfl_final_ce_mean": _mean(selected, "dfl_final_ce"),
                    "dfl_stage0_pred_dist_mean_A": _mean(
                        selected, "dfl_stage0_pred_dist_A"
                    ),
                    "dfl_final_pred_dist_mean_A": _mean(
                        selected, "dfl_final_pred_dist_A"
                    ),
                }
                if selected:
                    assigned = torch.tensor(
                        [float(item["final_assigned_dist_A"]) for item in selected],
                        dtype=torch.float64,
                    )
                    nearest = torch.tensor(
                        [
                            float(item["final_nearest_site_center_dist_A"])
                            for item in selected
                        ],
                        dtype=torch.float64,
                    )
                    row[f"disp_assigned_success_at_{success_name}A"] = float(
                        (assigned <= self.success_cutoff_A).double().mean().item()
                    )
                    row[f"disp_any_site_success_at_{success_name}A"] = float(
                        (nearest <= self.success_cutoff_A).double().mean().item()
                    )
                else:
                    row[f"disp_assigned_success_at_{success_name}A"] = float("nan")
                    row[f"disp_any_site_success_at_{success_name}A"] = float("nan")

                for stage in ("stage0", "final"):
                    metrics = _binary_dfl_metrics(
                        selected,
                        split_reference=self.split_reference,
                        split_cutoff_A=self.split_cutoff_A,
                        stage=stage,
                    )
                    for metric_name, metric_value in metrics.items():
                        row[f"dfl_{stage}_{metric_name}_at_split"] = metric_value
                summary_rows.append(row)
        return summary_rows

    def loss_contribution_rows(self) -> list[dict[str, Any]]:
        if not self.rows:
            raise RuntimeError("NearFarRecorder has no rows for loss contribution.")
        out: list[dict[str, Any]] = []
        grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            key = (
                int(row["checkpoint_epoch"]),
                int(row["checkpoint_global_step"]),
                str(row["dataset"]),
            )
            grouped.setdefault(key, []).append(row)
        for (epoch, global_step, dataset_name), rows in sorted(grouped.items()):
            near_rows = _group_rows(
                rows,
                split_reference=self.split_reference,
                split_cutoff_A=self.split_cutoff_A,
                group="near",
            )
            far_rows = _group_rows(
                rows,
                split_reference=self.split_reference,
                split_cutoff_A=self.split_cutoff_A,
                group="far",
            )
            all_disp_sum = _sum(rows, "disp_mse")
            all_dfl_sum = _sum(rows, "dfl_stage0_ce")
            far_disp_sum = _sum(far_rows, "disp_mse")
            far_dfl_sum = _sum(far_rows, "dfl_stage0_ce")
            out.append(
                {
                    "checkpoint_epoch": int(epoch),
                    "checkpoint_global_step": int(global_step),
                    "dataset": dataset_name,
                    "split_reference": self.split_reference,
                    "split_cutoff_A": self.split_cutoff_A,
                    "num_near_queries": int(len(near_rows)),
                    "num_far_queries": int(len(far_rows)),
                    "near_query_fraction": float(len(near_rows) / len(rows))
                    if rows
                    else float("nan"),
                    "far_query_fraction": float(len(far_rows) / len(rows))
                    if rows
                    else float("nan"),
                    "near_disp_mse_sum": _sum(near_rows, "disp_mse"),
                    "far_disp_mse_sum": far_disp_sum,
                    "far_disp_mse_sum_fraction": far_disp_sum / all_disp_sum
                    if all_disp_sum > 0.0
                    else float("nan"),
                    "near_dfl_stage0_ce_sum": _sum(near_rows, "dfl_stage0_ce"),
                    "far_dfl_stage0_ce_sum": far_dfl_sum,
                    "far_dfl_stage0_ce_sum_fraction": far_dfl_sum / all_dfl_sum
                    if all_dfl_sum > 0.0
                    else float("nan"),
                    "near_disp_mse_site_group_reduced": _site_group_reduced_mean(
                        near_rows, "disp_mse"
                    ),
                    "far_disp_mse_site_group_reduced": _site_group_reduced_mean(
                        far_rows, "disp_mse"
                    ),
                    "near_dfl_stage0_ce_site_group_reduced": _site_group_reduced_mean(
                        near_rows,
                        "dfl_stage0_ce",
                    ),
                    "far_dfl_stage0_ce_site_group_reduced": _site_group_reduced_mean(
                        far_rows,
                        "dfl_stage0_ce",
                    ),
                }
            )
        return out

    def reduced_loss_contribution_rows(self) -> list[dict[str, Any]]:
        if not self.rows:
            raise RuntimeError(
                "NearFarRecorder has no rows for reduced loss contribution."
            )
        distance_key = _split_distance_key(self.split_reference)
        by_epoch_dataset_site: dict[
            tuple[int, int, str, str, int], list[dict[str, Any]]
        ] = {}
        for row in self.rows:
            key = (
                int(row["checkpoint_epoch"]),
                int(row["checkpoint_global_step"]),
                str(row["dataset"]),
                str(row["sample_id"]),
                int(row["target_site_id"]),
            )
            by_epoch_dataset_site.setdefault(key, []).append(row)

        grouped: dict[tuple[int, int, str], list[dict[str, float]]] = {}
        for (
            epoch,
            global_step,
            dataset_name,
            _sample_id,
            _site_id,
        ), rows in by_epoch_dataset_site.items():
            total_count = len(rows)
            if total_count == 0:
                raise RuntimeError("Internal error: empty near/far site group.")
            near_disp_sum = 0.0
            far_disp_sum = 0.0
            near_dfl_sum = 0.0
            far_dfl_sum = 0.0
            near_count = 0
            far_count = 0
            for row in rows:
                is_near = float(row[distance_key]) <= self.split_cutoff_A
                disp = float(row["disp_mse"])
                dfl = float(row["dfl_stage0_ce"])
                if is_near:
                    near_count += 1
                    near_disp_sum += disp
                    near_dfl_sum += dfl
                else:
                    far_count += 1
                    far_disp_sum += disp
                    far_dfl_sum += dfl
            grouped.setdefault((epoch, global_step, dataset_name), []).append(
                {
                    "near_query_fraction": near_count / float(total_count),
                    "far_query_fraction": far_count / float(total_count),
                    "near_disp_reduced": near_disp_sum / float(total_count),
                    "far_disp_reduced": far_disp_sum / float(total_count),
                    "all_disp_reduced": (near_disp_sum + far_disp_sum)
                    / float(total_count),
                    "near_dfl_reduced": near_dfl_sum / float(total_count),
                    "far_dfl_reduced": far_dfl_sum / float(total_count),
                    "all_dfl_reduced": (near_dfl_sum + far_dfl_sum)
                    / float(total_count),
                }
            )

        out: list[dict[str, Any]] = []
        for (epoch, global_step, dataset_name), groups in sorted(grouped.items()):
            far_disp = (
                torch.tensor(
                    [row["far_disp_reduced"] for row in groups],
                    dtype=torch.float64,
                )
                .mean()
                .item()
            )
            all_disp = (
                torch.tensor(
                    [row["all_disp_reduced"] for row in groups],
                    dtype=torch.float64,
                )
                .mean()
                .item()
            )
            far_dfl = (
                torch.tensor(
                    [row["far_dfl_reduced"] for row in groups],
                    dtype=torch.float64,
                )
                .mean()
                .item()
            )
            all_dfl = (
                torch.tensor(
                    [row["all_dfl_reduced"] for row in groups],
                    dtype=torch.float64,
                )
                .mean()
                .item()
            )
            out.append(
                {
                    "checkpoint_epoch": int(epoch),
                    "checkpoint_global_step": int(global_step),
                    "dataset": dataset_name,
                    "split_reference": self.split_reference,
                    "split_cutoff_A": self.split_cutoff_A,
                    "num_site_groups": int(len(groups)),
                    "near_query_fraction_mean": float(
                        torch.tensor(
                            [row["near_query_fraction"] for row in groups],
                            dtype=torch.float64,
                        )
                        .mean()
                        .item()
                    ),
                    "far_query_fraction_mean": float(
                        torch.tensor(
                            [row["far_query_fraction"] for row in groups],
                            dtype=torch.float64,
                        )
                        .mean()
                        .item()
                    ),
                    "near_disp_reduced": float(
                        torch.tensor(
                            [row["near_disp_reduced"] for row in groups],
                            dtype=torch.float64,
                        )
                        .mean()
                        .item()
                    ),
                    "far_disp_reduced": float(far_disp),
                    "all_disp_reduced": float(all_disp),
                    "far_disp_reduced_fraction": float(far_disp / all_disp)
                    if all_disp > 0.0
                    else float("nan"),
                    "near_dfl_reduced": float(
                        torch.tensor(
                            [row["near_dfl_reduced"] for row in groups],
                            dtype=torch.float64,
                        )
                        .mean()
                        .item()
                    ),
                    "far_dfl_reduced": float(far_dfl),
                    "all_dfl_reduced": float(all_dfl),
                    "far_dfl_reduced_fraction": float(far_dfl / all_dfl)
                    if all_dfl > 0.0
                    else float("nan"),
                }
            )
        return out

    def write(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.store_per_query:
            self._write_rows(output / "nearfar_per_query.csv", self.rows)
        self._write_rows(output / "nearfar_summary_by_group.csv", self.summary_rows())
        self._write_rows(
            output / "nearfar_loss_contribution.csv", self.loss_contribution_rows()
        )
        self._write_rows(
            output / "nearfar_reduced_loss_contribution.csv",
            self.reduced_loss_contribution_rows(),
        )

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise RuntimeError(f"No rows to write for {path}.")
        fieldnames = list(rows[0].keys())
        for row in rows:
            extra = set(row) - set(fieldnames)
            if extra:
                raise ValueError(
                    f"Row for {path} has unexpected keys: {sorted(extra)}."
                )
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
