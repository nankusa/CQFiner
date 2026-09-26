from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    MeanShift,
    estimate_bandwidth,
)
from torch import Tensor

from src.data.modal_paths import modal_path
from src.utils.metrics import (
    DCA,
    DCC,
    calc_dca_dcc_metrics,
    evaluate_mask_ap,
    success_rate_from_distances,
)


class EvalPostprocessMixin:
    @staticmethod
    def _aggregate_rank_scores(scores: np.ndarray, mode: str) -> float:
        if scores.size == 0:
            return 0.0
        if mode == "mean":
            return float(np.mean(scores))
        if mode == "max" or mode == "top1":
            return float(np.max(scores))
        if mode == "sum":
            return float(np.sum(scores))
        return float(np.sum(scores**2))

    def _aggregate_query_rank_scores(self, scores: np.ndarray) -> float:
        return self._aggregate_rank_scores(scores, self.query_nms_score_aggregation)

    def _aggregate_site_mask_rank_scores(self, scores: np.ndarray) -> float:
        return self._aggregate_rank_scores(scores, self.site_mask_score_aggregation)

    def _apply_query_nms(
        self,
        coords: np.ndarray,
        scores: np.ndarray,
        positive_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if coords.shape[0] == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if positive_mask is None:
            keep = np.ones(coords.shape[0], dtype=bool)
        else:
            keep = np.asarray(positive_mask, dtype=bool).reshape(-1)
        keep &= np.isfinite(scores)
        keep &= np.isfinite(coords).all(axis=1)
        if not np.any(keep):
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        coords = coords[keep]
        scores = scores[keep]
        order = np.argsort(-scores, kind="stable")
        if self.query_eval_nms_radius <= 0:
            kept_idx = order
            kept_scores = scores[kept_idx]
        else:
            suppressed = np.zeros(coords.shape[0], dtype=bool)
            kept: list[int] = []
            kept_scores_list: list[float] = []
            radius_sq = float(self.query_eval_nms_radius**2)
            for idx in order:
                if suppressed[idx]:
                    continue
                kept.append(int(idx))
                diff = coords - coords[idx : idx + 1]
                dist_sq = np.sum(diff * diff, axis=1)
                cluster_mask = (~suppressed) & (dist_sq <= radius_sq)
                if not np.any(cluster_mask):
                    cluster_mask = np.zeros_like(suppressed)
                    cluster_mask[idx] = True
                kept_scores_list.append(
                    self._aggregate_query_rank_scores(scores[cluster_mask])
                )
                suppressed |= cluster_mask
                suppressed[idx] = True
            kept_idx = np.asarray(kept, dtype=np.int64)
            kept_scores = np.asarray(kept_scores_list, dtype=np.float32)

        return coords[kept_idx].astype(np.float32, copy=False), kept_scores.astype(
            np.float32, copy=False
        )

    def _query_nms_groups(
        self,
        coords: np.ndarray,
        scores: np.ndarray,
        positive_mask: np.ndarray | None = None,
        nms_radius: float | None = None,
        score_aggregation: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        if coords.shape[0] == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if positive_mask is None:
            keep = np.ones(coords.shape[0], dtype=bool)
        else:
            keep = np.asarray(positive_mask, dtype=bool).reshape(-1)
        keep &= np.isfinite(scores)
        keep &= np.isfinite(coords).all(axis=1)
        if not np.any(keep):
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )

        original_idx = np.nonzero(keep)[0]
        coords = coords[keep]
        scores = scores[keep]
        order = np.argsort(-scores, kind="stable")
        centers: list[np.ndarray] = []
        kept_scores: list[float] = []
        groups: list[np.ndarray] = []
        radius = (
            self.query_eval_nms_radius
            if nms_radius is None
            else float(max(nms_radius, 0.0))
        )
        aggregation = (
            "square" if score_aggregation is None else str(score_aggregation).lower()
        )
        if radius <= 0:
            for idx in order:
                centers.append(coords[idx])
                kept_scores.append(
                    self._aggregate_rank_scores(
                        scores[np.asarray([idx], dtype=np.int64)], aggregation
                    )
                )
                groups.append(original_idx[np.asarray([idx], dtype=np.int64)])
        else:
            suppressed = np.zeros(coords.shape[0], dtype=bool)
            radius_sq = float(radius**2)
            for idx in order:
                if suppressed[idx]:
                    continue
                diff = coords - coords[idx : idx + 1]
                dist_sq = np.sum(diff * diff, axis=1)
                cluster_mask = (~suppressed) & (dist_sq <= radius_sq)
                if not np.any(cluster_mask):
                    cluster_mask = np.zeros_like(suppressed)
                    cluster_mask[idx] = True
                centers.append(coords[idx])
                kept_scores.append(
                    self._aggregate_rank_scores(scores[cluster_mask], aggregation)
                )
                groups.append(original_idx[cluster_mask])
                suppressed |= cluster_mask
                suppressed[idx] = True

        if not centers:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )
        return (
            np.stack(centers, axis=0).astype(np.float32, copy=False),
            np.asarray(kept_scores, dtype=np.float32),
            groups,
        )

    def _query_affinity_groups(
        self,
        coords: np.ndarray,
        scores: np.ndarray,
        affinity_probs: np.ndarray,
        positive_mask: np.ndarray | None = None,
        affinity_threshold: float | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if coords.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                [],
            )

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        affinity_probs = np.asarray(affinity_probs, dtype=np.float32)
        if coords.shape[0] != scores.shape[0]:
            raise ValueError(
                f"query affinity grouping coord/score size mismatch: {coords.shape[0]} vs {scores.shape[0]}."
            )
        if affinity_probs.shape != (coords.shape[0], coords.shape[0]):
            raise ValueError(
                "query affinity grouping expects affinity_probs [Q, Q], "
                f"got {tuple(affinity_probs.shape)} for Q={coords.shape[0]}."
            )
        if not np.isfinite(affinity_probs).all():
            raise ValueError(
                "query affinity grouping received non-finite affinity probabilities."
            )

        threshold = (
            self.site_mask_affinity_threshold
            if affinity_threshold is None
            else float(affinity_threshold)
        )
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"query affinity grouping threshold must be in (0, 1), got {threshold}."
            )

        if positive_mask is None:
            keep = np.ones(coords.shape[0], dtype=bool)
        else:
            keep = np.asarray(positive_mask, dtype=bool).reshape(-1)
        keep &= np.isfinite(scores)
        keep &= np.isfinite(coords).all(axis=1)
        if not np.any(keep):
            return (
                np.zeros((0,), dtype=np.float32),
                [],
            )

        original_idx = np.nonzero(keep)[0]
        coords = coords[keep]
        scores = scores[keep]
        affinity_probs = affinity_probs[np.ix_(original_idx, original_idx)]
        affinity_probs = 0.5 * (affinity_probs + affinity_probs.T)
        np.fill_diagonal(affinity_probs, 1.0)
        if not np.isfinite(affinity_probs).all():
            raise ValueError(
                "query affinity grouping produced non-finite symmetric affinities."
            )

        order = np.argsort(-scores, kind="stable")
        assigned = np.zeros((coords.shape[0],), dtype=bool)
        groups: list[np.ndarray] = []
        cohesions: list[float] = []
        for seed in order:
            seed = int(seed)
            if assigned[seed]:
                continue

            group = np.asarray([seed], dtype=np.int64)
            assigned[seed] = True

            while True:
                candidates = np.nonzero(~assigned)[0].astype(np.int64, copy=False)
                if candidates.size == 0:
                    break
                seed_affinity = affinity_probs[seed, candidates]
                group_affinity = affinity_probs[np.ix_(candidates, group)]
                topk = min(3, int(group.size))
                if topk == 1:
                    topk_group_affinity = group_affinity[:, 0]
                else:
                    topk_group_affinity = np.partition(
                        group_affinity, kth=group_affinity.shape[1] - topk, axis=1
                    )[:, -topk:].mean(axis=1)
                attach = (seed_affinity >= threshold) | (
                    topk_group_affinity >= threshold
                )
                if not np.any(attach):
                    break
                new_members = candidates[attach]
                assigned[new_members] = True
                group = np.concatenate([group, new_members], axis=0)

            group_scores = scores[group]
            group_order = np.argsort(-group_scores, kind="stable")
            group = group[group_order]
            groups.append(original_idx[group])
            if group.size == 1:
                cohesions.append(float(threshold))
            else:
                pair_affinity = affinity_probs[np.ix_(group, group)]
                upper = pair_affinity[np.triu_indices(group.size, k=1)]
                cohesions.append(float(np.mean(upper)))

        return np.asarray(cohesions, dtype=np.float32), groups

    def _query_distance_cluster_groups(
        self,
        coords: np.ndarray,
        scores: np.ndarray,
        positive_mask: np.ndarray | None = None,
        cluster_threshold: float | None = None,
        score_aggregation: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        if coords.shape[0] == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )

        coords = np.asarray(coords, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"query distance clustering expects coords [Q, 3], got {tuple(coords.shape)}."
            )
        if scores.shape != (coords.shape[0],):
            raise ValueError(
                f"query distance clustering score size mismatch: {scores.shape} vs Q={coords.shape[0]}."
            )

        if positive_mask is None:
            keep = np.ones(coords.shape[0], dtype=bool)
        else:
            keep = np.asarray(positive_mask, dtype=bool).reshape(-1)
        keep &= np.isfinite(scores)
        keep &= np.isfinite(coords).all(axis=1)
        if not np.any(keep):
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )

        threshold = (
            self.site_mask_cluster_threshold
            if cluster_threshold is None
            else float(cluster_threshold)
        )
        if threshold <= 0.0:
            raise ValueError(
                f"query distance clustering threshold must be positive, got {threshold}."
            )
        aggregation = (
            "square" if score_aggregation is None else str(score_aggregation).lower()
        )

        original_idx = np.nonzero(keep)[0]
        coords = coords[keep]
        scores = scores[keep]
        if coords.shape[0] == 1:
            return (
                coords.astype(np.float32, copy=False),
                np.asarray(
                    [self._aggregate_rank_scores(scores, aggregation)], dtype=np.float32
                ),
                [original_idx.astype(np.int64, copy=False)],
            )

        diff = coords[:, None, :] - coords[None, :, :]
        distance = np.sqrt(
            np.sum(diff * diff, axis=-1, dtype=np.float32), dtype=np.float32
        )
        if not np.isfinite(distance).all():
            raise ValueError("query distance clustering produced non-finite distances.")
        np.fill_diagonal(distance, 0.0)
        cluster_labels = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=threshold,
            compute_full_tree=True,
        ).fit_predict(distance)

        groups: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        site_scores: list[float] = []
        seed_scores: list[float] = []
        for label in np.unique(cluster_labels):
            cluster = np.nonzero(cluster_labels == label)[0].astype(
                np.int64, copy=False
            )
            order = np.argsort(-scores[cluster], kind="stable")
            cluster = cluster[order]
            groups.append(original_idx[cluster])
            centers.append(coords[cluster[0]])
            site_scores.append(
                self._aggregate_rank_scores(scores[cluster], aggregation)
            )
            seed_scores.append(float(scores[cluster[0]]))
        group_order = np.argsort(
            -np.asarray(seed_scores, dtype=np.float32), kind="stable"
        )
        return (
            np.stack(centers, axis=0).astype(np.float32, copy=False)[group_order],
            np.asarray(site_scores, dtype=np.float32)[group_order],
            [groups[int(idx)] for idx in group_order],
        )

    @staticmethod
    def _grasp_cluster_score(probs: np.ndarray, mode: str) -> float:
        if probs.size == 0:
            return 0.0
        if mode == "mean":
            return float(np.mean(probs))
        if mode == "sum":
            return float(np.sum(probs))
        if mode == "square":
            return float(np.sum(np.square(probs)))
        raise ValueError(f"Unsupported GrASP cluster score aggregation: {mode}")

    @staticmethod
    def _grasp_convex_hull_center(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.shape[0] < 4:
            return coords.mean(axis=0).astype(np.float32)
        try:
            hull = ConvexHull(coords)
            if hull.volume <= 1.0e-8:
                return coords.mean(axis=0).astype(np.float32)
            tetras = Delaunay(hull.points[hull.vertices])
            center = np.zeros((3,), dtype=np.float64)
            volume_sum = 0.0
            for simplex in tetras.simplices:
                tetra = tetras.points[simplex]
                a, b, c, d = tetra
                volume = (
                    abs(np.linalg.det(np.stack([a - d, b - d, c - d], axis=0))) / 6.0
                )
                center += tetra.mean(axis=0) * volume
                volume_sum += volume
            if volume_sum <= 1.0e-8:
                return coords.mean(axis=0).astype(np.float32)
            return (center / volume_sum).astype(np.float32)
        except Exception:
            return coords.mean(axis=0).astype(np.float32)

    @classmethod
    def _grasp_cluster_center(
        cls, coords: np.ndarray, probs: np.ndarray, centroid_type: str
    ) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        if coords.shape[0] == 0:
            return np.zeros((3,), dtype=np.float32)
        if centroid_type == "hull":
            return cls._grasp_convex_hull_center(coords)
        if centroid_type == "prob":
            weights = probs
        elif centroid_type == "square":
            weights = np.square(probs)
        elif centroid_type == "centroid":
            weights = np.ones_like(probs)
        else:
            raise ValueError(f"Unsupported GrASP centroid_type: {centroid_type}")
        denom = float(weights.sum())
        if denom <= 1.0e-8:
            return coords.mean(axis=0).astype(np.float32)
        return ((coords * weights[:, None]).sum(axis=0) / denom).astype(np.float32)

    def _aggregate_host_site_probs_for_grasp(
        self, mask_probs: Tensor, query_scores: Tensor
    ) -> Tensor:
        if mask_probs.ndim != 2 or mask_probs.size(0) == 0:
            return mask_probs.new_zeros(
                (mask_probs.size(1) if mask_probs.ndim == 2 else 0,)
            )
        mode = self.site_dcc_host_prob_aggregation
        if mode == "max":
            return mask_probs.max(dim=0).values
        if mode == "mean":
            return mask_probs.mean(dim=0)

        weights = (
            query_scores.detach()
            .to(device=mask_probs.device, dtype=mask_probs.dtype)
            .clamp(min=0.0)
        )
        if weights.numel() != mask_probs.size(0):
            weights = mask_probs.new_ones((mask_probs.size(0),))
        if weights.sum() <= 0:
            weights = mask_probs.new_ones((mask_probs.size(0),))
        weighted_probs = mask_probs * weights.unsqueeze(-1)
        if mode == "score_max":
            return weighted_probs.max(dim=0).values
        if mode == "score_mean":
            return weighted_probs.sum(dim=0) / weights.sum().clamp(min=1.0e-8)
        raise ValueError(f"Unsupported site_dcc_host_prob_aggregation: {mode}")

    def _grasp_site_centers_from_host_probs(
        self,
        host_pos: Tensor | None,
        host_probs: Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        if host_pos is None or host_probs.numel() == 0 or host_pos.size(0) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        coords = host_pos.detach().float().cpu().numpy().astype(np.float32)
        probs = host_probs.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        if coords.shape[0] != probs.shape[0]:
            raise ValueError(
                f"host position/probability size mismatch: {coords.shape[0]} vs {probs.shape[0]}"
            )

        valid = np.isfinite(probs) & np.isfinite(coords).all(axis=1)
        positive = valid & (probs > self.site_dcc_prob_threshold)
        pos_indices = np.nonzero(positive)[0]
        if pos_indices.size == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        pos_coords = coords[pos_indices]
        pos_probs = probs[pos_indices]
        if pos_indices.size == 1:
            labels = np.zeros((1,), dtype=np.int64)
        elif self.site_dcc_cluster_method == "dbscan":
            labels = DBSCAN(
                eps=self.site_dcc_cluster_eps,
                min_samples=self.site_dcc_cluster_min_samples,
            ).fit_predict(pos_coords)
        elif self.site_dcc_cluster_method == "meanshift":
            bandwidth = estimate_bandwidth(
                pos_coords, quantile=self.site_dcc_cluster_quantile
            )
            if bandwidth <= 0:
                bandwidth = max(self.site_dcc_cluster_eps, 1.0e-17)
            labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(
                pos_coords
            )
        else:
            labels = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.site_dcc_cluster_eps,
                linkage=self.site_dcc_cluster_method,
            ).fit_predict(pos_coords)

        centers: list[np.ndarray] = []
        scores: list[float] = []
        for cluster_id in np.unique(labels):
            cluster_id = int(cluster_id)
            if cluster_id < 0:
                continue
            member = labels == cluster_id
            if not np.any(member):
                continue
            cluster_coords = pos_coords[member]
            cluster_probs = pos_probs[member]
            centers.append(
                self._grasp_cluster_center(
                    cluster_coords, cluster_probs, self.site_dcc_centroid_type
                )
            )
            scores.append(
                self._grasp_cluster_score(
                    cluster_probs, self.site_dcc_score_aggregation
                )
            )

        if not centers:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        scores_arr = np.asarray(scores, dtype=np.float32)
        order = np.argsort(-scores_arr, kind="stable")
        return (
            np.stack(centers, axis=0).astype(np.float32, copy=False)[order],
            scores_arr[order],
        )

    @staticmethod
    def _batch_sample_ids(batch) -> list[str]:
        sample_ids = getattr(batch, "sample_id", [])
        if isinstance(sample_ids, str):
            return [sample_ids]
        return [str(sample_id) for sample_id in sample_ids]

    @staticmethod
    def _atom_masks_to_residue_masks(
        atom_masks: np.ndarray,
        atom_residue_indices: np.ndarray,
        num_residues: int,
    ) -> np.ndarray:
        atom_masks = np.asarray(atom_masks, dtype=bool)
        residue_masks = np.zeros((atom_masks.shape[0], num_residues), dtype=np.float32)
        if atom_masks.size == 0 or atom_residue_indices.size == 0 or num_residues <= 0:
            return residue_masks

        valid_atom = (atom_residue_indices >= 0) & (atom_residue_indices < num_residues)
        for mask_idx, atom_mask in enumerate(atom_masks):
            residue_ids = atom_residue_indices[atom_mask & valid_atom]
            if residue_ids.size > 0:
                residue_masks[mask_idx, np.unique(residue_ids)] = 1.0
        return residue_masks

    @staticmethod
    def _atom_probs_to_residue_probs(
        atom_probs: np.ndarray,
        atom_residue_indices: np.ndarray,
        num_residues: int,
    ) -> np.ndarray:
        atom_probs = np.asarray(atom_probs, dtype=np.float32)
        residue_probs = np.zeros((atom_probs.shape[0], num_residues), dtype=np.float32)
        if atom_probs.ndim != 2:
            raise ValueError(
                f"Expected atom probabilities [P, A], got {atom_probs.shape}."
            )
        if atom_residue_indices.ndim != 1:
            raise ValueError(
                f"Expected atom_residue_indices [A], got {atom_residue_indices.shape}."
            )
        if atom_probs.shape[1] != atom_residue_indices.shape[0]:
            raise ValueError(
                f"Atom probability width mismatch: probs={atom_probs.shape}, atom_residue_indices={atom_residue_indices.shape}."
            )
        if atom_probs.size == 0 or atom_residue_indices.size == 0 or num_residues <= 0:
            return residue_probs
        if not np.isfinite(atom_probs).all():
            raise ValueError(
                "Non-finite atom mask probabilities encountered during residue-level eval conversion."
            )
        if np.any((atom_probs < 0.0) | (atom_probs > 1.0)):
            bad_mask = (atom_probs < 0.0) | (atom_probs > 1.0)
            bad_values = atom_probs[bad_mask]
            raise ValueError(
                "Atom mask probabilities must be in [0, 1]: "
                f"shape={atom_probs.shape}, min={float(np.min(atom_probs)):.9g}, "
                f"max={float(np.max(atom_probs)):.9g}, bad_count={int(bad_values.size)}, "
                f"bad_min={float(np.min(bad_values)):.9g}, bad_max={float(np.max(bad_values)):.9g}."
            )

        valid_atom = (atom_residue_indices >= 0) & (atom_residue_indices < num_residues)
        valid_atom_ids = np.nonzero(valid_atom)[0]
        for atom_idx in valid_atom_ids:
            residue_idx = int(atom_residue_indices[atom_idx])
            residue_probs[:, residue_idx] = np.maximum(
                residue_probs[:, residue_idx], atom_probs[:, atom_idx]
            )
        return residue_probs

    def _resolved_eval_data_cfg(self, dataset_name: str | None = None) -> dict:
        base_cfg = dict(self.data_cfg)
        if dataset_name is None:
            return base_cfg
        for test_cfg in self.test_data_cfgs:
            name = str(test_cfg.get("name") or test_cfg.get("dataset_name") or "")
            if name == str(dataset_name):
                resolved = dict(base_cfg)
                resolved.update(test_cfg)
                return resolved
        return base_cfg

    def _uses_surface_atom_eval(self, dataset_name: str | None = None) -> bool:
        cfg = self._resolved_eval_data_cfg(dataset_name)
        mode = str(cfg.get("host_node_mode", "residue")).lower()
        return mode in {
            "atom",
            "surface",
            "surface_atom",
            "surface_atoms",
            "surface_atom_graph",
        }

    def _target_npz_path(self, sample_id: str, dataset_name: str | None = None) -> Path:
        cfg = self._resolved_eval_data_cfg(dataset_name)
        root = Path(cfg.get("root", self.data_cfg.get("root", "."))).expanduser()
        dataset = str(cfg.get("dataset_name", self.data_cfg.get("dataset_name", "")))
        target_modal = str(
            cfg.get("target_modal", self.data_cfg.get("target_modal", "pocket"))
        )
        return modal_path(root / dataset, target_modal, sample_id, ".npz")

    def _num_residues_for_sample(
        self,
        sample_id: str,
        atom_residue_indices: np.ndarray,
        dataset_name: str | None = None,
    ) -> int:
        cfg = self._resolved_eval_data_cfg(dataset_name)
        root = Path(cfg.get("root", self.data_cfg.get("root", "."))).expanduser()
        dataset = str(cfg.get("dataset_name", self.data_cfg.get("dataset_name", "")))
        dataset_root = root / dataset
        for modal in cfg.get(
            "embedding_modalities", self.data_cfg.get("embedding_modalities", ["esm"])
        ) or ["esm"]:
            emb_path = modal_path(dataset_root, str(modal), sample_id, ".npy")
            if emb_path.exists():
                return int(np.load(emb_path, mmap_mode="r").shape[0])
        return (
            int(atom_residue_indices.max() + 1) if atom_residue_indices.size > 0 else 0
        )

    def _maybe_convert_site_eval_to_residue_level(
        self,
        prediction: dict[str, np.ndarray],
        target: dict[str, np.ndarray],
        sample_id: str,
        dataset_name: str | None = None,
        atom_residue_indices: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if not self._uses_surface_atom_eval(dataset_name):
            return prediction, target

        if atom_residue_indices is None:
            target_path = self._target_npz_path(sample_id, dataset_name)
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Missing target npz for surface residue-level eval: {target_path}"
                )
            with np.load(target_path) as payload:
                atom_residue_indices = np.asarray(
                    payload["atom_residue_indices"], dtype=np.int64
                )
        else:
            atom_residue_indices = np.asarray(atom_residue_indices, dtype=np.int64)
        num_residues = self._num_residues_for_sample(
            sample_id, atom_residue_indices, dataset_name
        )

        pred_masks = np.asarray(prediction["pocket_masks"], dtype=np.float32)
        target_masks = np.asarray(target["pocket_masks"], dtype=np.float32)
        if pred_masks.shape[1] != atom_residue_indices.shape[0]:
            raise ValueError(
                f"surface_atom prediction mask width mismatch for {sample_id}: "
                f"{pred_masks.shape[1]} vs {atom_residue_indices.shape[0]} atom_residue_indices"
            )
        if target_masks.shape[1] != atom_residue_indices.shape[0]:
            raise ValueError(
                f"surface_atom target mask width mismatch for {sample_id}: "
                f"{target_masks.shape[1]} vs {atom_residue_indices.shape[0]} atom_residue_indices"
            )

        residue_prediction = dict(prediction)
        residue_target = dict(target)
        residue_prediction["pocket_masks"] = self._atom_masks_to_residue_masks(
            pred_masks,
            atom_residue_indices,
            num_residues,
        )
        if "pocket_mask_probs" in residue_prediction:
            pred_probs = np.asarray(
                residue_prediction["pocket_mask_probs"], dtype=np.float32
            )
            if pred_probs.shape != pred_masks.shape:
                raise ValueError(
                    f"surface_atom prediction probability shape mismatch for {sample_id}: "
                    f"probs={pred_probs.shape}, masks={pred_masks.shape}"
                )
            residue_prediction["pocket_mask_probs"] = self._atom_probs_to_residue_probs(
                pred_probs,
                atom_residue_indices,
                num_residues,
            )
        residue_target["pocket_masks"] = self._atom_masks_to_residue_masks(
            target_masks,
            atom_residue_indices,
            num_residues,
        )
        residue_target["res_mask"] = np.ones((num_residues,), dtype=bool)
        return residue_prediction, residue_target

    def _build_vn_dot_site_prediction(
        self,
        query_pos: Tensor,
        query_scores: Tensor,
        mask_logits: Tensor,
        mask_pair_mask: Tensor | None = None,
        host_pos: Tensor | None = None,
        mask_threshold: float | None = None,
        affinity_logits: Tensor | None = None,
        affinity_threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        num_queries = int(query_pos.size(0))
        num_host = int(mask_logits.size(1)) if mask_logits.ndim == 2 else 0
        threshold = (
            self.site_mask_threshold
            if mask_threshold is None
            else float(mask_threshold)
        )
        detached_mask_logits = (
            mask_logits.detach()
            if mask_logits.ndim == 2
            else query_pos.new_zeros((0, num_host))
        )
        mask_logits_for_fusion = detached_mask_logits
        mask_probs = torch.sigmoid(detached_mask_logits)
        if mask_pair_mask is not None:
            if mask_pair_mask.shape != mask_probs.shape:
                raise ValueError(
                    "VN-dot mask_pair_mask shape mismatch: "
                    f"expected {tuple(mask_probs.shape)}, got {tuple(mask_pair_mask.shape)}."
                )
            valid_mask_pairs = mask_pair_mask.to(
                device=mask_probs.device, dtype=torch.bool
            )
            mask_probs = mask_probs.masked_fill(
                ~valid_mask_pairs,
                0.0,
            )
            mask_logits_for_fusion = mask_logits_for_fusion.masked_fill(
                ~valid_mask_pairs, -20.0
            )
        host_probs = self._aggregate_host_site_probs_for_grasp(mask_probs, query_scores)
        site_centers, site_scores_for_dcc = self._grasp_site_centers_from_host_probs(
            host_pos, host_probs
        )
        if num_queries == 0:
            return {
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "centers": np.zeros((0, 3), dtype=np.float32),
                "pocket_masks": np.zeros((0, num_host), dtype=np.float32),
                "pocket_mask_probs": np.zeros((0, num_host), dtype=np.float32),
                "site_centers": site_centers,
                "site_scores": site_scores_for_dcc,
            }

        coords_np = query_pos.detach().float().cpu().numpy().astype(np.float32)
        scores_np = query_scores.detach().float().cpu().numpy().astype(np.float32)
        positive_mask = None
        if self.site_mask_query_score_threshold > 0:
            positive_mask = scores_np >= self.site_mask_query_score_threshold
        if self.site_mask_grouping == "affinity":
            if affinity_logits is None:
                raise ValueError("VN-dot affinity grouping requires affinity_logits.")
            if affinity_logits.shape != (num_queries, num_queries):
                raise ValueError(
                    "VN-dot affinity logits shape mismatch: "
                    f"expected {(num_queries, num_queries)}, got {tuple(affinity_logits.shape)}."
                )
            affinity_probs = (
                torch.sigmoid(affinity_logits.detach())
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            group_cohesions, groups = self._query_affinity_groups(
                coords_np,
                scores_np,
                affinity_probs=affinity_probs,
                positive_mask=positive_mask,
                affinity_threshold=affinity_threshold,
            )
            centers = None
            site_scores = None
        elif self.site_mask_grouping == "cluster":
            group_cohesions = None
            centers, site_scores, groups = self._query_distance_cluster_groups(
                coords_np,
                scores_np,
                positive_mask=positive_mask,
                cluster_threshold=self.site_mask_cluster_threshold,
                score_aggregation=self.site_mask_score_aggregation,
            )
        else:
            group_cohesions = None
            centers, site_scores, groups = self._query_nms_groups(
                coords_np,
                scores_np,
                positive_mask=positive_mask,
                nms_radius=self.site_mask_nms_radius,
                score_aggregation=self.site_mask_score_aggregation,
            )
        if not groups:
            return {
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "centers": np.zeros((0, 3), dtype=np.float32),
                "pocket_masks": np.zeros((0, num_host), dtype=np.float32),
                "pocket_mask_probs": np.zeros((0, num_host), dtype=np.float32),
                "site_centers": site_centers,
                "site_scores": site_scores_for_dcc,
            }

        masks: list[np.ndarray] = []
        mask_probs_out: list[np.ndarray] = []
        kept_centers: list[np.ndarray] = []
        kept_scores: list[float] = []
        affinity_grouping = self.site_mask_grouping == "affinity"
        learned_grouping = self.site_mask_grouping == "affinity"
        if affinity_grouping and group_cohesions.shape != (len(groups),):
            raise ValueError(
                "VN-dot affinity grouping cohesion count mismatch: "
                f"{tuple(group_cohesions.shape)} vs groups={len(groups)}."
            )
        for group_order_idx, group_idx_np in enumerate(groups):
            group = group_idx_np
            group_idx = torch.as_tensor(
                group, dtype=torch.long, device=mask_probs.device
            )
            group_query_scores = query_scores.detach()[group_idx].to(
                device=mask_probs.device
            )
            if learned_grouping:
                group_weights = torch.softmax(
                    group_query_scores.float() / 0.2, dim=0
                ).to(dtype=mask_probs.dtype)
                fused_logit = (
                    mask_logits_for_fusion[group_idx] * group_weights.unsqueeze(-1)
                ).sum(dim=0)
                fused_prob = torch.sigmoid(fused_logit)
                center = (
                    query_pos.detach()[group_idx]
                    .to(device=mask_probs.device, dtype=mask_probs.dtype)
                    .mul(group_weights.unsqueeze(-1))
                    .sum(dim=0)
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            else:
                if centers is None or site_scores is None:
                    raise ValueError(
                        "VN-dot NMS grouping requires centers and site_scores."
                    )
                center = centers[group_order_idx]
                site_score = site_scores[group_order_idx]
                group_weights = group_query_scores.to(
                    device=mask_probs.device, dtype=torch.float64
                ).clamp(min=0.0)
                if group_weights.sum() <= 0:
                    group_weights = torch.ones_like(group_weights)
                group_probs = mask_probs[group_idx].to(dtype=torch.float64)
                fused_prob = (group_probs * group_weights.unsqueeze(-1)).sum(
                    dim=0
                ) / group_weights.sum().clamp(min=1e-8)
            mask_sel = fused_prob > threshold
            if not mask_sel.any():
                continue
            if affinity_grouping:
                query_conf = torch.max(group_query_scores.float())
                cohesion = float(group_cohesions[group_order_idx])
                mask_conf = fused_prob[mask_sel].float().mean()
                site_score = float((query_conf * mask_conf).item()) * cohesion
            mask = mask_sel.detach().float().cpu().numpy().astype(np.float32)
            prob = fused_prob.detach().float().cpu().numpy().astype(np.float32)
            masks.append(mask)
            mask_probs_out.append(prob)
            kept_centers.append(center)
            kept_scores.append(float(site_score))

        if not masks:
            return {
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "centers": np.zeros((0, 3), dtype=np.float32),
                "pocket_masks": np.zeros((0, num_host), dtype=np.float32),
                "pocket_mask_probs": np.zeros((0, num_host), dtype=np.float32),
                "site_centers": site_centers,
                "site_scores": site_scores_for_dcc,
            }

        final_scores = np.asarray(kept_scores, dtype=np.float32)
        final_order = np.argsort(-final_scores, kind="stable")
        return {
            "scores": final_scores[final_order],
            "labels": np.zeros((len(masks),), dtype=np.int64),
            "centers": np.stack(kept_centers, axis=0).astype(np.float32, copy=False)[
                final_order
            ],
            "pocket_masks": np.stack(masks, axis=0).astype(np.float32, copy=False)[
                final_order
            ],
            "pocket_mask_probs": np.stack(mask_probs_out, axis=0).astype(
                np.float32, copy=False
            )[final_order],
            "site_centers": site_centers,
            "site_scores": site_scores_for_dcc,
        }

    @staticmethod
    def _gather_objects(local_obj):
        if not dist.is_available() or not dist.is_initialized():
            return local_obj
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_obj)
        merged = []
        for part in gathered:
            merged.extend(part)
        return merged

    def _log_epoch_metrics(self, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
            )
        if self.trainer.is_global_zero:
            for logger in self.trainer.loggers:
                logger.log_metrics(metrics, step=self.global_step)

    @staticmethod
    def _weighted_row_mean(rows: list[dict], key: str) -> float:
        if not rows:
            return float("nan")
        weights = np.asarray(
            [row.get("batch_size", 1) for row in rows], dtype=np.float32
        )
        values = np.asarray([row[key] for row in rows], dtype=np.float32)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            return float("nan")
        return float(np.sum(values * weights) / total_weight)

    def _dataset_name_from_dataloader_idx(self, dataloader_idx: int) -> str:
        datamodule = getattr(self.trainer, "datamodule", None)
        indices = (
            getattr(datamodule, "test_dataloader_indices", {})
            if datamodule is not None
            else {}
        )
        for name, idx in indices.items():
            if idx == dataloader_idx:
                return str(name)
        return f"loader_{dataloader_idx}"

    def _ensure_site_test_bucket(self, dataset_name: str) -> None:
        if dataset_name in self._test_loss_rows:
            return
        self._test_loss_rows[dataset_name] = []
        self._test_query_metric_rows[dataset_name] = []
        self._test_site_prediction_outputs[dataset_name] = []
        self._test_site_predictions[dataset_name] = []
        self._test_site_targets[dataset_name] = []
        self._test_pred_centers[dataset_name] = []
        self._test_pred_scores[dataset_name] = []
        self._test_ligands[dataset_name] = []
        self._test_host_pred_centers[dataset_name] = []
        self._test_host_pred_scores[dataset_name] = []
        self._test_host_ligands[dataset_name] = []
        self._test_query_dcc_metrics[dataset_name] = DCC(
            threshold=self.val_dcc.threshold
        ).to(self.device)
        self._test_query_dca_metrics[dataset_name] = DCA(
            threshold=self.val_dca.threshold
        ).to(self.device)

    def _build_site_epoch_metrics(
        self,
        query_metric_rows: list[dict],
        predictions,
        targets,
        pred_centers,
        pred_scores,
        ligands,
        host_pred_centers,
        host_pred_scores,
        host_ligands,
        prefix: str,
    ) -> dict[str, float]:
        epoch_metrics: dict[str, float] = {}
        if self.rank_based_eval and predictions:
            ap_iou_03 = evaluate_mask_ap(predictions, targets, iou_thr=0.3)
            ap_iou_05 = evaluate_mask_ap(predictions, targets, iou_thr=0.5)

            epoch_metrics[f"{prefix}/ap_iou_0.3"] = ap_iou_03
            epoch_metrics[f"{prefix}/ap_iou_0.5"] = ap_iou_05
            epoch_metrics[f"{prefix}/site_ap_iou_0.3"] = ap_iou_03
            epoch_metrics[f"{prefix}/site_ap_iou_0.5"] = ap_iou_05

        if self.rank_based_eval and pred_centers:
            dcc_topn, dca_topn = calc_dca_dcc_metrics(
                pred_centers_list=pred_centers,
                pred_scores_list=pred_scores,
                ligands_list=ligands,
                top_n_plus=0,
            )
            dcc_topn_plus_2, dca_topn_plus_2 = calc_dca_dcc_metrics(
                pred_centers_list=pred_centers,
                pred_scores_list=pred_scores,
                ligands_list=ligands,
                top_n_plus=2,
            )

            epoch_metrics[f"{prefix}/query_dcc_topn"] = success_rate_from_distances(
                dcc_topn,
                threshold=self.val_dcc.threshold,
            )
            epoch_metrics[f"{prefix}/query_dca_topn"] = success_rate_from_distances(
                dca_topn,
                threshold=self.val_dca.threshold,
            )
            epoch_metrics[f"{prefix}/query_dcc_topn_plus_2"] = (
                success_rate_from_distances(
                    dcc_topn_plus_2,
                    threshold=self.val_dcc.threshold,
                )
            )
            epoch_metrics[f"{prefix}/query_dca_topn_plus_2"] = (
                success_rate_from_distances(
                    dca_topn_plus_2,
                    threshold=self.val_dca.threshold,
                )
            )
        elif self.rank_based_eval:
            epoch_metrics[f"{prefix}/query_dcc_topn"] = 0.0
            epoch_metrics[f"{prefix}/query_dca_topn"] = 0.0
            epoch_metrics[f"{prefix}/query_dcc_topn_plus_2"] = 0.0
            epoch_metrics[f"{prefix}/query_dca_topn_plus_2"] = 0.0

        if self.rank_based_eval and host_pred_centers:
            host_dcc_topn, host_dca_topn = calc_dca_dcc_metrics(
                pred_centers_list=host_pred_centers,
                pred_scores_list=host_pred_scores,
                ligands_list=host_ligands,
                top_n_plus=0,
            )
            host_dcc_topn_plus_2, host_dca_topn_plus_2 = calc_dca_dcc_metrics(
                pred_centers_list=host_pred_centers,
                pred_scores_list=host_pred_scores,
                ligands_list=host_ligands,
                top_n_plus=2,
            )

            epoch_metrics[f"{prefix}/site_dcc_topn"] = success_rate_from_distances(
                host_dcc_topn,
                threshold=self.val_dcc.threshold,
            )
            epoch_metrics[f"{prefix}/site_dca_topn"] = success_rate_from_distances(
                host_dca_topn,
                threshold=self.val_dca.threshold,
            )
            epoch_metrics[f"{prefix}/site_dcc_topn_plus_2"] = (
                success_rate_from_distances(
                    host_dcc_topn_plus_2,
                    threshold=self.val_dcc.threshold,
                )
            )
            epoch_metrics[f"{prefix}/site_dca_topn_plus_2"] = (
                success_rate_from_distances(
                    host_dca_topn_plus_2,
                    threshold=self.val_dca.threshold,
                )
            )
        return epoch_metrics
