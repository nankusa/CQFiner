from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.utils.metrics import extract_ligand_groups


DEFAULT_SITE_SIZE_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-40", 21, 40),
    (">40", 41, None),
)


class SiteSizeRecorder:
    """Collect center-localization metrics stratified by target residue-mask size."""

    def __init__(
        self,
        *,
        bins: list[dict[str, Any]] | None = None,
    ) -> None:
        self.bins = self._parse_bins(bins)
        self._threshold: float | None = None
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _parse_bins(
        bins: list[dict[str, Any]] | None,
    ) -> tuple[tuple[str, int, int | None], ...]:
        if bins is None:
            return DEFAULT_SITE_SIZE_BINS
        parsed: list[tuple[str, int, int | None]] = []
        for idx, item in enumerate(bins):
            if not isinstance(item, dict):
                raise TypeError(f"site_size.bins[{idx}] must be a mapping.")
            for key in ("label", "min"):
                if key not in item:
                    raise ValueError(
                        f"site_size.bins[{idx}] missing required key: {key}"
                    )
            label = str(item["label"])
            lo = int(item["min"])
            hi = None if item.get("max") is None else int(item["max"])
            if lo < 0:
                raise ValueError(
                    f"site_size.bins[{idx}].min must be non-negative, got {lo}."
                )
            if hi is not None and hi < lo:
                raise ValueError(
                    f"site_size.bins[{idx}].max must be >= min, got {hi} < {lo}."
                )
            parsed.append((label, lo, hi))
        if not parsed:
            raise ValueError("site_size.bins must not be empty.")
        return tuple(parsed)

    def _bin_label(self, site_size: int) -> str:
        for label, lo, hi in self.bins:
            if site_size >= lo and (hi is None or site_size <= hi):
                return label
        raise ValueError(f"No site-size bin covers site_size={site_size}.")

    @staticmethod
    def _top_centers(
        pred_centers: np.ndarray, pred_scores: np.ndarray, top_k: int
    ) -> np.ndarray:
        pred_centers = np.asarray(pred_centers, dtype=np.float32)
        pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
        if pred_centers.ndim != 2 or pred_centers.shape[1] != 3:
            raise ValueError(
                f"pred_centers must have shape [P, 3], got {pred_centers.shape}."
            )
        if pred_scores.shape != (pred_centers.shape[0],):
            raise ValueError(
                f"pred_scores shape mismatch: {pred_scores.shape} vs centers={pred_centers.shape}."
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        if pred_centers.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        sorted_ids = np.argsort(pred_scores)[::-1]
        return pred_centers[sorted_ids[:top_k]]

    @staticmethod
    def _site_distances(
        pred_centers: np.ndarray, ligand_coords: np.ndarray
    ) -> tuple[float, float]:
        ligand_coords = np.asarray(ligand_coords, dtype=np.float32)
        if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3:
            raise ValueError(
                f"ligand_coords must have shape [A, 3], got {ligand_coords.shape}."
            )
        if ligand_coords.shape[0] == 0:
            raise ValueError(
                "Cannot compute site-size metrics for an empty ligand group."
            )
        if pred_centers.shape[0] == 0:
            return float("inf"), float("inf")
        ligand_center = ligand_coords.mean(axis=0)
        dcc = float(np.linalg.norm(pred_centers - ligand_center[None, :], axis=1).min())
        dca = float(
            np.linalg.norm(
                pred_centers[:, None, :] - ligand_coords[None, :, :], axis=-1
            ).min()
        )
        return dcc, dca

    @staticmethod
    def _sample_site_sizes(
        *,
        num_targets: int,
        target_mask_site_id: Tensor,
    ) -> np.ndarray:
        if num_targets <= 0:
            raise ValueError(
                f"num_targets must be positive for site-size metrics, got {num_targets}."
            )
        site_ids = (
            target_mask_site_id.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        if site_ids.size > 0:
            if int(site_ids.min()) < 0 or int(site_ids.max()) >= num_targets:
                raise ValueError(
                    f"target_mask_site_id contains out-of-range ids for num_targets={num_targets}: "
                    f"min={int(site_ids.min())}, max={int(site_ids.max())}."
                )
        return np.bincount(site_ids, minlength=num_targets).astype(np.int64, copy=False)

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
            raise RuntimeError("SiteSizeRecorder requires the Lightning module.")
        if context is None:
            raise RuntimeError("SiteSizeRecorder requires inference context.")
        required = {"final_pos", "query_rank_scores"}
        missing = required - set(context)
        if missing:
            raise RuntimeError(
                f"SiteSizeRecorder context is missing keys: {sorted(missing)}"
            )

        final_pos = context["final_pos"]
        query_rank_scores = context["query_rank_scores"]
        if query_batch.shape != (final_pos.size(0),):
            raise ValueError(
                "SiteSizeRecorder query_batch shape mismatch: "
                f"expected {(final_pos.size(0),)}, got {tuple(query_batch.shape)}."
            )
        if query_rank_scores.shape != (final_pos.size(0),):
            raise ValueError(
                "SiteSizeRecorder query_rank_scores shape mismatch: "
                f"expected {(final_pos.size(0),)}, got {tuple(query_rank_scores.shape)}."
            )
        if self._threshold is None:
            self._threshold = float(module.val_dcc.threshold)

        sample_ids = (
            batch.sample_id if isinstance(batch.sample_id, list) else [batch.sample_id]
        )
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

        for sample_idx, sample_id in enumerate(sample_ids):
            target_sel = batch.target_pos_batch == sample_idx
            query_sel = query_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx
            mask_sel = target_mask_batch == sample_idx

            num_targets = int(target_sel.sum().item())
            if num_targets <= 0:
                raise RuntimeError(
                    f"Sample {sample_id} in dataset {dataset_name} has no target sites."
                )
            ligands = extract_ligand_groups(
                ligand_pos=batch.ligand_pos[ligand_sel],
                ligand_ids=batch.ligand_id[ligand_sel],
            )
            if len(ligands) != num_targets:
                raise ValueError(
                    f"Sample {sample_id} target/ligand count mismatch: "
                    f"num_targets={num_targets}, ligand_groups={len(ligands)}."
                )
            site_sizes = self._sample_site_sizes(
                num_targets=num_targets,
                target_mask_site_id=getattr(
                    batch,
                    "target_mask_site_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
            )

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
                pred_centers, pred_scores = module._apply_query_nms(coords, scores)
            else:
                pred_centers = np.zeros((0, 3), dtype=np.float32)
                pred_scores = np.zeros((0,), dtype=np.float32)

            topn_centers = self._top_centers(
                pred_centers, pred_scores, top_k=num_targets
            )
            plus2_centers = self._top_centers(
                pred_centers, pred_scores, top_k=num_targets + 2
            )
            for site_idx, ligand_coords in enumerate(ligands):
                dcc, dca = self._site_distances(topn_centers, ligand_coords)
                dcc_plus2, dca_plus2 = self._site_distances(
                    plus2_centers, ligand_coords
                )
                site_size = int(site_sizes[site_idx])
                self.rows.append(
                    {
                        "dataset": str(dataset_name),
                        "sample_id": str(sample_id),
                        "site_id": int(site_idx),
                        "site_size": site_size,
                        "size_bin": self._bin_label(site_size),
                        "dcc_dist_A": dcc,
                        "dca_dist_A": dca,
                        "dcc_plus2_dist_A": dcc_plus2,
                        "dca_plus2_dist_A": dca_plus2,
                    }
                )

    @staticmethod
    def _success(values: list[float], threshold: float) -> float:
        if not values:
            return float("nan")
        finite_or_inf = np.asarray(values, dtype=np.float32)
        return float(np.mean(finite_or_inf < threshold))

    @staticmethod
    def _mean(values: list[float]) -> float:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return float("nan")
        return float(np.asarray(finite, dtype=np.float64).mean())

    def summary_rows(self) -> list[dict[str, Any]]:
        if self._threshold is None:
            raise RuntimeError("SiteSizeRecorder has no collected batches.")
        rows: list[dict[str, Any]] = []
        datasets = sorted({str(row["dataset"]) for row in self.rows})
        for dataset in datasets:
            dataset_rows = [row for row in self.rows if str(row["dataset"]) == dataset]
            bucket_defs = (("ALL", None),) + tuple(
                (label, label) for label, _, _ in self.bins
            )
            for bucket_label, row_bin in bucket_defs:
                bucket_rows = (
                    dataset_rows
                    if row_bin is None
                    else [
                        row
                        for row in dataset_rows
                        if str(row["size_bin"]) == str(row_bin)
                    ]
                )
                sample_ids = {str(row["sample_id"]) for row in bucket_rows}
                rows.append(
                    {
                        "dataset": dataset,
                        "size_bin": bucket_label,
                        "sample_count": len(sample_ids),
                        "site_count": len(bucket_rows),
                        "mean_site_size": self._mean(
                            [float(row["site_size"]) for row in bucket_rows]
                        ),
                        "dcc": self._success(
                            [float(row["dcc_dist_A"]) for row in bucket_rows],
                            self._threshold,
                        ),
                        "dca": self._success(
                            [float(row["dca_dist_A"]) for row in bucket_rows],
                            self._threshold,
                        ),
                        "dcc_plus2": self._success(
                            [float(row["dcc_plus2_dist_A"]) for row in bucket_rows],
                            self._threshold,
                        ),
                        "dca_plus2": self._success(
                            [float(row["dca_plus2_dist_A"]) for row in bucket_rows],
                            self._threshold,
                        ),
                    }
                )
        return rows

    def write(self, log_dir: str | Path) -> None:
        output_dir = Path(log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            raise RuntimeError("SiteSizeRecorder produced no per-site rows.")
        row_fieldnames = list(self.rows[0].keys())
        with (output_dir / "site_size_per_site.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row_fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        summary = self.summary_rows()
        if not summary:
            raise RuntimeError("SiteSizeRecorder produced no summary rows.")
        pd.DataFrame(summary).to_csv(output_dir / "site_size_summary.csv", index=False)
