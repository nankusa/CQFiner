from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from scipy.spatial import ConvexHull, Delaunay

from src.utils.metrics import (
    build_target_sites,
    calc_dca_dcc_metrics,
    evaluate_mask_ap,
    extract_ligand_groups,
    success_rate_from_distances,
)


class PostprocessRecorder:
    """Compare site-center post-processing choices without changing model outputs."""

    def __init__(self) -> None:
        self._threshold: float | None = None
        self._records: dict[str, dict[str, list[Any]]] = {}

    @staticmethod
    def _empty_centers() -> np.ndarray:
        return np.zeros((0, 3), dtype=np.float32)

    @staticmethod
    def _empty_scores() -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)

    @staticmethod
    def _as_numpy_centers(value: Any, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"{label} centers must have shape [N, 3], got {arr.shape}."
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{label} centers contain non-finite values.")
        return arr

    @staticmethod
    def _as_numpy_scores(value: Any, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if not np.isfinite(arr).all():
            raise ValueError(f"{label} scores contain non-finite values.")
        return arr

    @staticmethod
    def _score_weighted_center(coords: np.ndarray, scores: np.ndarray) -> np.ndarray:
        weights = np.clip(scores.astype(np.float64, copy=False), a_min=0.0, a_max=None)
        denom = float(weights.sum())
        if denom <= 0.0:
            raise ValueError(
                "Cannot compute score-weighted query center because all group weights are non-positive."
            )
        return (
            (coords.astype(np.float64, copy=False) * weights[:, None]).sum(axis=0)
            / denom
        ).astype(np.float32)

    @staticmethod
    def _convex_hull_center(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Convex-hull coordinates must have shape [N, 3], got {coords.shape}."
            )
        if coords.shape[0] == 0:
            raise ValueError(
                "Cannot compute convex-hull center for an empty residue mask."
            )
        if not np.isfinite(coords).all():
            raise ValueError("Convex-hull coordinates contain non-finite values.")
        if coords.shape[0] < 4:
            return coords.mean(axis=0).astype(np.float32)

        hull = ConvexHull(coords)
        tetras = Delaunay(hull.points[hull.vertices])
        center = np.zeros((3,), dtype=np.float64)
        volume_sum = 0.0
        for simplex in tetras.simplices:
            tetra = tetras.points[simplex]
            a, b, c, d = tetra
            volume = abs(np.linalg.det(np.stack([a - d, b - d, c - d], axis=0))) / 6.0
            center += tetra.mean(axis=0) * volume
            volume_sum += volume
        if volume_sum <= 0.0:
            raise ValueError(
                "Convex-hull tetrahedralization produced zero total volume."
            )
        return (center / volume_sum).astype(np.float32)

    @classmethod
    def _mask_hull_centers(cls, *, host_pos: Tensor, masks: Any) -> np.ndarray:
        host_coords = host_pos.detach().float().cpu().numpy().astype(np.float32)
        mask_arr = np.asarray(masks, dtype=np.float32)
        if host_coords.ndim != 2 or host_coords.shape[1] != 3:
            raise ValueError(
                f"Host coordinates must have shape [N, 3], got {host_coords.shape}."
            )
        if mask_arr.ndim != 2:
            raise ValueError(
                f"Pocket masks must have shape [P, N], got {mask_arr.shape}."
            )
        if mask_arr.shape[1] != host_coords.shape[0]:
            raise ValueError(
                "Pocket-mask width must match host coordinate count: "
                f"{mask_arr.shape[1]} vs {host_coords.shape[0]}."
            )
        centers = []
        for mask_idx, mask in enumerate(mask_arr):
            selected = mask > 0
            if not np.any(selected):
                raise ValueError(
                    f"Predicted pocket mask {mask_idx} is empty; cannot compute convex-hull center."
                )
            centers.append(cls._convex_hull_center(host_coords[selected]))
        if not centers:
            return cls._empty_centers()
        return np.stack(centers, axis=0).astype(np.float32, copy=False)

    @classmethod
    def _query_nms_weighted(
        cls,
        *,
        coords: np.ndarray,
        scores: np.ndarray,
        nms_radius: float,
        module: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(coords, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Query coords must have shape [Q, 3], got {coords.shape}."
            )
        if scores.shape != (coords.shape[0],):
            raise ValueError(
                f"Query score shape mismatch: scores={scores.shape}, coords={coords.shape}."
            )
        if not np.isfinite(coords).all() or not np.isfinite(scores).all():
            raise ValueError("Query coords/scores contain non-finite values.")
        if coords.shape[0] == 0:
            return cls._empty_centers(), cls._empty_scores()

        order = np.argsort(-scores, kind="stable")
        weighted_centers: list[np.ndarray] = []
        group_scores: list[float] = []
        radius = float(max(nms_radius, 0.0))
        if radius <= 0.0:
            for idx in order:
                group = np.asarray([idx], dtype=np.int64)
                weighted_centers.append(coords[idx])
                group_scores.append(
                    float(module._aggregate_query_rank_scores(scores[group]))
                )
        else:
            suppressed = np.zeros((coords.shape[0],), dtype=bool)
            radius_sq = float(radius * radius)
            for idx in order:
                idx = int(idx)
                if suppressed[idx]:
                    continue
                diff = coords - coords[idx : idx + 1]
                dist_sq = np.sum(diff * diff, axis=1)
                group_mask = (~suppressed) & (dist_sq <= radius_sq)
                if not np.any(group_mask):
                    raise RuntimeError(
                        "NMS selected a seed but produced an empty group."
                    )
                group = np.nonzero(group_mask)[0].astype(np.int64, copy=False)
                weighted_centers.append(
                    cls._score_weighted_center(coords[group], scores[group])
                )
                group_scores.append(
                    float(module._aggregate_query_rank_scores(scores[group]))
                )
                suppressed |= group_mask
                suppressed[idx] = True

        if not weighted_centers:
            raise RuntimeError(
                "NMS produced no query groups for a non-empty query set."
            )
        return (
            np.stack(weighted_centers, axis=0).astype(np.float32, copy=False),
            np.asarray(group_scores, dtype=np.float32),
        )

    def _ensure_dataset(self, dataset_name: str) -> dict[str, list[Any]]:
        if dataset_name not in self._records:
            self._records[dataset_name] = {
                "query_top_score_centers": [],
                "query_weighted_centers": [],
                "query_scores": [],
                "residue_hull_centers": [],
                "residue_hull_scores": [],
                "ligands": [],
                "site_predictions": [],
                "site_targets": [],
            }
        return self._records[dataset_name]

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
            raise RuntimeError("PostprocessRecorder requires the Lightning module.")
        if context is None:
            raise RuntimeError("PostprocessRecorder requires inference context.")
        required = {"final_pos", "query_rank_scores", "site_prediction_outputs"}
        missing = required - set(context)
        if missing:
            raise RuntimeError(
                f"PostprocessRecorder context is missing keys: {sorted(missing)}"
            )

        final_pos = context["final_pos"]
        query_rank_scores = context["query_rank_scores"]
        site_prediction_outputs = context["site_prediction_outputs"]
        if query_batch.shape != (final_pos.size(0),):
            raise ValueError(
                "PostprocessRecorder query_batch shape mismatch: "
                f"expected {(final_pos.size(0),)}, got {tuple(query_batch.shape)}."
            )
        if query_rank_scores.shape != (final_pos.size(0),):
            raise ValueError(
                "PostprocessRecorder query_rank_scores shape mismatch: "
                f"expected {(final_pos.size(0),)}, got {tuple(query_rank_scores.shape)}."
            )
        if not isinstance(site_prediction_outputs, list):
            raise TypeError(
                "PostprocessRecorder expects site_prediction_outputs to be a list."
            )

        if self._threshold is None:
            self._threshold = float(module.val_dcc.threshold)

        target_mask_batch = getattr(
            batch,
            "target_mask_host_id_batch",
            torch.zeros(
                getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                ).size(0),
                dtype=torch.long,
                device=batch.x.device,
            ),
        )

        sample_ids = (
            batch.sample_id if isinstance(batch.sample_id, list) else [batch.sample_id]
        )
        outputs_by_sample: dict[int, dict[str, Any]] = {}
        for item_idx, item in enumerate(site_prediction_outputs):
            if not isinstance(item, dict):
                raise TypeError(
                    f"site_prediction_outputs[{item_idx}] must be a dict, got {type(item).__name__}."
                )
            if "sample_idx" not in item or "prediction" not in item:
                raise RuntimeError(
                    f"site_prediction_outputs[{item_idx}] requires sample_idx and prediction."
                )
            sample_idx = int(item["sample_idx"])
            if sample_idx in outputs_by_sample:
                raise ValueError(
                    f"Duplicate site prediction output for sample_idx={sample_idx} in {dataset_name}."
                )
            outputs_by_sample[sample_idx] = item

        records = self._ensure_dataset(str(dataset_name))
        for sample_idx, _sample_id in enumerate(sample_ids):
            if sample_idx not in outputs_by_sample:
                raise KeyError(
                    f"Missing site prediction output for sample_idx={sample_idx} in {dataset_name}."
                )
            query_sel = query_batch == sample_idx
            host_sel = batch.batch == sample_idx
            if not bool(host_sel.any().item()):
                raise ValueError(
                    f"Postprocess prediction output references sample {sample_idx} with no host nodes."
                )
            target_sel = batch.target_pos_batch == sample_idx
            mask_sel = target_mask_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx
            if query_sel.any():
                coords = (
                    final_pos[query_sel]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                scores = (
                    query_rank_scores[query_sel]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                top_centers, group_scores = module._apply_query_nms(coords, scores)
                weighted_centers, weighted_group_scores = self._query_nms_weighted(
                    coords=coords,
                    scores=scores,
                    nms_radius=float(module.query_eval_nms_radius),
                    module=module,
                )
                if group_scores.shape != weighted_group_scores.shape or not np.allclose(
                    group_scores,
                    weighted_group_scores,
                    rtol=1.0e-6,
                    atol=1.0e-7,
                ):
                    raise ValueError(
                        "Weighted query grouping produced scores that differ from the default query NMS path."
                    )
            else:
                top_centers = self._empty_centers()
                weighted_centers = self._empty_centers()
                group_scores = self._empty_scores()

            output_item = outputs_by_sample[sample_idx]
            prediction = output_item["prediction"]
            if not isinstance(prediction, dict):
                raise TypeError(
                    f"site_prediction_outputs[{sample_idx}]['prediction'] must be a dict, "
                    f"got {type(prediction).__name__}."
                )
            target = build_target_sites(
                num_targets=int(target_sel.sum().item()),
                num_host=int(host_sel.sum().item()),
                target_mask_site_id=getattr(
                    batch,
                    "target_mask_site_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
                target_mask_host_id=getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
            )
            sample_id = str(
                output_item.get(
                    "sample_id",
                    sample_ids[sample_idx]
                    if sample_idx < len(sample_ids)
                    else sample_idx,
                )
            )
            atom_residue_indices = output_item.get("atom_residue_indices")
            if atom_residue_indices is not None:
                atom_residue_indices = np.asarray(atom_residue_indices, dtype=np.int64)
            prediction, target = module._maybe_convert_site_eval_to_residue_level(
                prediction=prediction,
                target=target,
                sample_id=sample_id,
                dataset_name=dataset_name,
                atom_residue_indices=atom_residue_indices,
            )
            residue_centers = self._mask_hull_centers(
                host_pos=batch.pos[host_sel],
                masks=prediction["pocket_masks"],
            )
            residue_scores = self._as_numpy_scores(prediction["scores"], "residue_hull")
            if residue_centers.shape[0] != residue_scores.shape[0]:
                raise ValueError(
                    "Residue hull center/score count mismatch: "
                    f"{residue_centers.shape[0]} vs {residue_scores.shape[0]}."
                )

            ligands = extract_ligand_groups(
                ligand_pos=batch.ligand_pos[ligand_sel],
                ligand_ids=batch.ligand_id[ligand_sel],
            )
            records["query_top_score_centers"].append(top_centers)
            records["query_weighted_centers"].append(weighted_centers)
            records["query_scores"].append(group_scores)
            records["residue_hull_centers"].append(residue_centers)
            records["residue_hull_scores"].append(residue_scores)
            records["ligands"].append(ligands)
            records["site_predictions"].append(prediction)
            records["site_targets"].append(target)

    def _metric_rows(self) -> list[dict[str, Any]]:
        if self._threshold is None:
            raise RuntimeError("PostprocessRecorder has no collected batches.")
        rows: list[dict[str, Any]] = []
        pathways = [
            ("query_top_score", "query_top_score_centers", "query_scores"),
            ("query_weighted", "query_weighted_centers", "query_scores"),
            ("residue_hull", "residue_hull_centers", "residue_hull_scores"),
        ]
        for dataset_name, records in sorted(self._records.items()):
            ligands = records["ligands"]
            ap_iou_0p5 = evaluate_mask_ap(
                records["site_predictions"], records["site_targets"], iou_thr=0.5
            )
            for pathway, center_key, score_key in pathways:
                dcc_topn, dca_topn = calc_dca_dcc_metrics(
                    pred_centers_list=records[center_key],
                    pred_scores_list=records[score_key],
                    ligands_list=ligands,
                    top_n_plus=0,
                )
                dcc_plus2, dca_plus2 = calc_dca_dcc_metrics(
                    pred_centers_list=records[center_key],
                    pred_scores_list=records[score_key],
                    ligands_list=ligands,
                    top_n_plus=2,
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "pathway": pathway,
                        "n_samples": len(ligands),
                        "ap_iou_0.5": ap_iou_0p5,
                        "dcc": success_rate_from_distances(
                            dcc_topn, threshold=self._threshold
                        ),
                        "dca": success_rate_from_distances(
                            dca_topn, threshold=self._threshold
                        ),
                        "dcc_plus2": success_rate_from_distances(
                            dcc_plus2, threshold=self._threshold
                        ),
                        "dca_plus2": success_rate_from_distances(
                            dca_plus2, threshold=self._threshold
                        ),
                    }
                )
        return rows

    def write(self, log_dir: str | Path) -> None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        rows = self._metric_rows()
        if not rows:
            raise RuntimeError("PostprocessRecorder produced no metric rows.")
        pd.DataFrame(rows).to_csv(path / "postprocess_summary.csv", index=False)
