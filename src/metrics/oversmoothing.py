from __future__ import annotations

import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


OVERSMOOTHING_METRICS = [
    "n_nodes",
    "feature_norm_mean",
    "feature_norm_cv",
    "feature_variance",
    "pairwise_cosine_mean",
    "edge_cosine_mean",
    "pairwise_sq_l2_per_dim",
    "effective_rank",
    "effective_rank_ratio",
    "top1_spectrum_ratio",
    "dirichlet_energy",
    "dirichlet_energy_norm",
]

OVERSMOOTHING_FEATURE_TRANSFORMS = {"raw", "center_l2"}


def _normalize_feature_transform(name: str) -> str:
    transform = str(name).lower()
    if transform in {"center", "centered", "centered_l2", "center_then_l2"}:
        transform = "center_l2"
    if transform not in OVERSMOOTHING_FEATURE_TRANSFORMS:
        raise ValueError(
            "Unsupported oversmoothing feature transform: "
            f"{name!r}. Expected one of {sorted(OVERSMOOTHING_FEATURE_TRANSFORMS)}."
        )
    return transform


def _sample_feature_stats(
    features: Tensor,
    *,
    edge_index: Tensor | None,
    complete_graph_edges: bool,
    feature_transform: str,
) -> dict[str, float]:
    if features.ndim != 2:
        raise ValueError(
            f"Oversmoothing features must have shape [N, D], got {tuple(features.shape)}."
        )
    n_nodes = int(features.size(0))
    feature_dim = int(features.size(1))
    if n_nodes < 2:
        raise ValueError(
            f"Oversmoothing metrics require at least two nodes, got {n_nodes}."
        )
    if feature_dim <= 0:
        raise ValueError(
            f"Oversmoothing feature dimension must be positive, got {feature_dim}."
        )

    context = (
        torch.autocast(device_type=features.device.type, enabled=False)
        if features.device.type == "cuda"
        else nullcontext()
    )
    with context:
        x = features.detach().float()
        transform = _normalize_feature_transform(feature_transform)
        if transform == "center_l2":
            x = x - x.mean(dim=0, keepdim=True)
            transform_norms = torch.linalg.norm(x, dim=-1, keepdim=True)
            if bool((transform_norms <= 0).any().item()):
                raise ValueError(
                    "center_l2 oversmoothing transform produced a zero-norm node feature."
                )
            x = x / transform_norms
        norms = torch.linalg.norm(x, dim=-1)
        if bool((norms <= 0).any().item()):
            raise ValueError("Oversmoothing metrics received a zero-norm node feature.")
        norm_mean = float(norms.mean().item())
        norm_std = float(norms.std(unbiased=False).item())

        centered = x - x.mean(dim=0, keepdim=True)
        centered_sq_sum = centered.square().sum()
        feature_variance = centered_sq_sum / float((n_nodes - 1) * feature_dim)
        pairwise_sq_l2_per_dim = (2.0 * centered_sq_sum / float(n_nodes - 1)) / float(
            feature_dim
        )

        normalized = x / norms.unsqueeze(-1)
        normalized_sum = normalized.sum(dim=0)
        pairwise_cosine = (normalized_sum.dot(normalized_sum) - float(n_nodes)) / float(
            n_nodes * (n_nodes - 1)
        )

        covariance = centered.transpose(0, 1).matmul(centered) / float(n_nodes - 1)
        eigvals = torch.linalg.eigvalsh(covariance).clamp(min=0.0)
        eig_total = eigvals.sum()
        if float(eig_total.item()) <= 1.0e-12:
            effective_rank = torch.tensor(1.0, device=x.device)
            top1_ratio = torch.tensor(1.0, device=x.device)
        else:
            probs = eigvals / eig_total
            positive = probs > 0
            entropy = -(probs[positive] * torch.log(probs[positive])).sum()
            effective_rank = torch.exp(entropy)
            top1_ratio = eigvals.max() / eig_total
        rank_denominator = float(min(n_nodes - 1, feature_dim))

        if edge_index is not None:
            if edge_index.ndim != 2 or edge_index.size(0) != 2:
                raise ValueError(
                    f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
                )
            if edge_index.numel() == 0:
                raise ValueError(
                    "Oversmoothing edge metrics require at least one edge."
                )
            if (
                int(edge_index.min().item()) < 0
                or int(edge_index.max().item()) >= n_nodes
            ):
                raise ValueError(
                    "Oversmoothing edge_index contains indices outside the local feature range: "
                    f"min={int(edge_index.min().item())}, max={int(edge_index.max().item())}, n_nodes={n_nodes}."
                )
            edge_src, edge_dst = edge_index[0], edge_index[1]
            edge_diff = x[edge_src] - x[edge_dst]
            dirichlet_energy = edge_diff.square().sum(dim=-1).mean() / float(
                feature_dim
            )
            edge_cosine = (
                (normalized[edge_src] * normalized[edge_dst]).sum(dim=-1).mean()
            )
        elif complete_graph_edges:
            dirichlet_energy = pairwise_sq_l2_per_dim
            edge_cosine = pairwise_cosine
        else:
            raise ValueError(
                "Oversmoothing requires either edge_index or complete_graph_edges=true."
            )

        mean_sq_norm_per_dim = x.square().sum(dim=-1).mean() / float(feature_dim)
        dirichlet_energy_norm = dirichlet_energy / mean_sq_norm_per_dim.clamp(
            min=1.0e-12
        )

        return {
            "n_nodes": float(n_nodes),
            "feature_norm_mean": norm_mean,
            "feature_norm_cv": norm_std / max(norm_mean, 1.0e-12),
            "feature_variance": float(feature_variance.item()),
            "pairwise_cosine_mean": float(pairwise_cosine.item()),
            "edge_cosine_mean": float(edge_cosine.item()),
            "pairwise_sq_l2_per_dim": float(pairwise_sq_l2_per_dim.item()),
            "effective_rank": float(effective_rank.item()),
            "effective_rank_ratio": float((effective_rank / rank_denominator).item()),
            "top1_spectrum_ratio": float(top1_ratio.item()),
            "dirichlet_energy": float(dirichlet_energy.item()),
            "dirichlet_energy_norm": float(dirichlet_energy_norm.item()),
        }


def _local_host_edges(batch, sample_idx: int) -> Tensor:
    edge_index = getattr(batch, "edge_index", None)
    if edge_index is None:
        raise AttributeError("Host oversmoothing metrics require batch.edge_index.")
    host_batch = batch.batch
    host_sel = host_batch == sample_idx
    if not bool(host_sel.any().item()):
        raise ValueError(f"Sample {sample_idx} has no host nodes.")
    local_index = torch.full(
        (host_batch.size(0),), -1, dtype=torch.long, device=host_batch.device
    )
    local_index[host_sel.nonzero(as_tuple=False).view(-1)] = torch.arange(
        int(host_sel.sum().item()),
        dtype=torch.long,
        device=host_batch.device,
    )
    src_local = local_index[edge_index[0]]
    dst_local = local_index[edge_index[1]]
    edge_sel = (src_local >= 0) & (dst_local >= 0)
    if not bool(edge_sel.any().item()):
        raise ValueError(
            f"Sample {sample_idx} has no host-host edges for oversmoothing metrics."
        )
    return torch.stack([src_local[edge_sel], dst_local[edge_sel]], dim=0)


def _batch_sample_ids(batch) -> list[str]:
    sample_ids = getattr(batch, "sample_id", [])
    if isinstance(sample_ids, str):
        return [sample_ids]
    return [str(sample_id) for sample_id in sample_ids]


class OversmoothingRecorder:
    def __init__(
        self, targets: list[str], feature_transforms: list[str] | None = None
    ) -> None:
        normalized = [str(target).lower() for target in targets]
        if not normalized:
            raise ValueError("OversmoothingRecorder requires at least one target.")
        invalid = sorted(set(normalized) - {"host", "query"})
        if invalid:
            raise ValueError(f"Unsupported oversmoothing target(s): {invalid}.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"Duplicate oversmoothing targets are not allowed: {targets}."
            )
        transforms = [
            _normalize_feature_transform(transform)
            for transform in (feature_transforms or ["raw"])
        ]
        if len(set(transforms)) != len(transforms):
            raise ValueError(
                f"Duplicate oversmoothing feature transforms are not allowed: {feature_transforms}."
            )
        self.targets = normalized
        self.feature_transforms = transforms
        self.rows: list[dict[str, Any]] = []

    def collect_batch(
        self,
        *,
        batch,
        out: dict[str, Tensor],
        query_batch: Tensor,
        dataset_name: str,
        module: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        sample_ids = _batch_sample_ids(batch)
        if "host" in self.targets:
            features = out.get("host_scalar")
            if features is None:
                raise RuntimeError("Host oversmoothing requires out['host_scalar'].")
            for sample_idx in batch.batch.unique(sorted=True).tolist():
                sample_idx = int(sample_idx)
                host_sel = batch.batch == sample_idx
                edge_index = _local_host_edges(batch, sample_idx)
                for feature_transform in self.feature_transforms:
                    stats = _sample_feature_stats(
                        features[host_sel],
                        edge_index=edge_index,
                        complete_graph_edges=False,
                        feature_transform=feature_transform,
                    )
                    self.rows.append(
                        self._row(
                            dataset_name,
                            sample_ids,
                            sample_idx,
                            "host",
                            feature_transform,
                            stats,
                        )
                    )

        if "query" in self.targets:
            features = out.get("query_scalar")
            if features is None:
                raise RuntimeError("Query oversmoothing requires out['query_scalar'].")
            for sample_idx in query_batch.unique(sorted=True).tolist():
                sample_idx = int(sample_idx)
                query_sel = query_batch == sample_idx
                for feature_transform in self.feature_transforms:
                    stats = _sample_feature_stats(
                        features[query_sel],
                        edge_index=None,
                        complete_graph_edges=True,
                        feature_transform=feature_transform,
                    )
                    self.rows.append(
                        self._row(
                            dataset_name,
                            sample_ids,
                            sample_idx,
                            "query",
                            feature_transform,
                            stats,
                        )
                    )

    @staticmethod
    def _row(
        dataset_name: str,
        sample_ids: list[str],
        sample_idx: int,
        target: str,
        feature_transform: str,
        stats: dict[str, float],
    ) -> dict[str, Any]:
        sample_id = (
            sample_ids[sample_idx] if sample_idx < len(sample_ids) else str(sample_idx)
        )
        return {
            "dataset": dataset_name,
            "sample_id": sample_id,
            "target": target,
            "feature_transform": feature_transform,
            **stats,
        }

    def summary_rows(self) -> list[dict[str, Any]]:
        if not self.rows:
            raise RuntimeError("OversmoothingRecorder has no rows to summarize.")
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            key = (
                str(row["dataset"]),
                str(row["target"]),
                str(row["feature_transform"]),
            )
            grouped.setdefault(key, []).append(row)

        summaries: list[dict[str, Any]] = []
        for (dataset_name, target, feature_transform), rows in sorted(grouped.items()):
            summary: dict[str, Any] = {
                "dataset": dataset_name,
                "target": target,
                "feature_transform": feature_transform,
                "num_samples": len(rows),
            }
            for metric in OVERSMOOTHING_METRICS:
                values = torch.tensor(
                    [float(row[metric]) for row in rows], dtype=torch.float64
                )
                summary[metric] = float(values.mean().item())
                summary[f"{metric}_sample_std"] = float(
                    values.std(unbiased=False).item()
                )
            summaries.append(summary)
        return summaries

    def write(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._write_rows(output / "oversmoothing_per_sample.csv", self.rows)
        self._write_rows(output / "oversmoothing_summary.csv", self.summary_rows())

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
