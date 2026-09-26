from __future__ import annotations

import numpy as np


def _normalize_surface_option(option) -> dict:
    if option is None or option is False:
        return {"mode": "none"}
    if isinstance(option, str):
        return {"mode": option.lower()}
    if isinstance(option, (int, float, np.integer, np.floating)) and not isinstance(
        option, bool
    ):
        return {"mode": "threshold", "threshold": float(option)}
    if isinstance(option, dict):
        normalized = dict(option)
        normalized["mode"] = str(normalized.get("mode", "none")).lower()
        return normalized
    raise TypeError(f"Unsupported surface atom option type: {type(option).__name__}")


def normalize_surface_atom_sasa_filter(option) -> dict:
    cfg = _normalize_surface_option(option)
    mode = cfg["mode"]
    if mode in {"", "none", "false", "off", "all"}:
        return {"mode": "none"}
    if mode == "threshold":
        if "threshold" not in cfg:
            raise ValueError(
                "surface_atom_sasa_filter mode='threshold' requires 'threshold'."
            )
        return {"mode": "threshold", "threshold": float(cfg["threshold"])}
    if mode in {"topk", "topk_per_residue", "per_residue_topk"}:
        value = cfg.get("topk_per_residue", cfg.get("k", cfg.get("topk")))
        if value is None:
            raise ValueError(
                "surface_atom_sasa_filter mode='topk_per_residue' requires 'topk_per_residue'."
            )
        return {"mode": "topk_per_residue", "topk_per_residue": int(value)}
    raise ValueError(f"Unsupported surface_atom_sasa_filter mode: {mode}")


def normalize_surface_atom_downsample(option) -> dict:
    cfg = _normalize_surface_option(option)
    mode = cfg["mode"]
    if mode in {"", "none", "false", "off", "all"}:
        return {"mode": "none"}
    if mode == "voxel":
        voxel_size = float(cfg.get("voxel_size", cfg.get("size", 2.0)))
        if voxel_size <= 0:
            raise ValueError("surface_atom_downsample voxel_size must be positive.")
        normalized = {"mode": "voxel", "voxel_size": voxel_size}
        if cfg.get("max_nodes") is not None:
            normalized["max_nodes"] = int(cfg["max_nodes"])
        return normalized
    if mode in {"fps", "farthest_point"}:
        max_nodes = cfg.get("max_nodes", cfg.get("num_nodes"))
        if max_nodes is None:
            raise ValueError("surface_atom_downsample mode='fps' requires 'max_nodes'.")
        return {"mode": "fps", "max_nodes": int(max_nodes)}
    if mode in {"topk_sasa", "sasa_topk"}:
        max_nodes = cfg.get("max_nodes", cfg.get("topk", cfg.get("k")))
        if max_nodes is None:
            raise ValueError(
                "surface_atom_downsample mode='topk_sasa' requires 'max_nodes'."
            )
        return {"mode": "topk_sasa", "max_nodes": int(max_nodes)}
    raise ValueError(f"Unsupported surface_atom_downsample mode: {mode}")


def _topk_indices_by_sasa(indices: np.ndarray, sasa: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if indices.size <= k:
        return np.sort(indices.astype(np.int64, copy=False))
    order = np.lexsort((indices, -sasa[indices]))
    return np.sort(indices[order[:k]].astype(np.int64, copy=False))


def _apply_surface_atom_sasa_filter(
    indices: np.ndarray,
    sasa: np.ndarray,
    atom_residue_indices: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    mode = cfg["mode"]
    if mode == "none":
        return indices
    if mode == "threshold":
        return indices[sasa[indices] >= float(cfg["threshold"])]
    if mode == "topk_per_residue":
        k = int(cfg["topk_per_residue"])
        kept = []
        for residue_idx in np.unique(atom_residue_indices[indices]):
            residue_indices = indices[atom_residue_indices[indices] == residue_idx]
            kept.append(_topk_indices_by_sasa(residue_indices, sasa, k))
        if not kept:
            return np.zeros((0,), dtype=np.int64)
        return np.sort(np.concatenate(kept).astype(np.int64, copy=False))
    raise ValueError(f"Unsupported normalized SASA filter mode: {mode}")


def _voxel_downsample_indices(
    indices: np.ndarray, coords: np.ndarray, sasa: np.ndarray, voxel_size: float
) -> np.ndarray:
    if indices.size == 0:
        return indices
    voxels = np.floor(coords[indices] / float(voxel_size)).astype(np.int64)
    best_by_voxel: dict[tuple[int, int, int], int] = {}
    for atom_idx, voxel in zip(indices.tolist(), voxels.tolist()):
        key = tuple(int(x) for x in voxel)
        best = best_by_voxel.get(key)
        if (
            best is None
            or sasa[atom_idx] > sasa[best]
            or (sasa[atom_idx] == sasa[best] and atom_idx < best)
        ):
            best_by_voxel[key] = int(atom_idx)
    return np.sort(np.fromiter(best_by_voxel.values(), dtype=np.int64))


def _fps_downsample_indices(
    indices: np.ndarray, coords: np.ndarray, sasa: np.ndarray, max_nodes: int
) -> np.ndarray:
    if max_nodes <= 0:
        return np.zeros((0,), dtype=np.int64)
    if indices.size <= max_nodes:
        return np.sort(indices.astype(np.int64, copy=False))
    selected = [int(indices[np.argmax(sasa[indices])])]
    min_dist_sq = np.sum((coords[indices] - coords[selected[0]]) ** 2, axis=1)
    selected_mask = indices == selected[0]
    for _ in range(1, max_nodes):
        scores = min_dist_sq.copy()
        scores[selected_mask] = -1.0
        next_local = int(np.argmax(scores))
        next_idx = int(indices[next_local])
        if scores[next_local] < 0:
            break
        selected.append(next_idx)
        selected_mask[next_local] = True
        dist_sq = np.sum((coords[indices] - coords[next_idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
    return np.sort(np.asarray(selected, dtype=np.int64))


def _apply_surface_atom_downsample(
    indices: np.ndarray,
    coords: np.ndarray,
    sasa: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    mode = cfg["mode"]
    if mode == "none":
        return indices
    if mode == "voxel":
        selected = _voxel_downsample_indices(
            indices, coords, sasa, float(cfg["voxel_size"])
        )
        max_nodes = cfg.get("max_nodes")
        if max_nodes is not None and selected.size > int(max_nodes):
            selected = _topk_indices_by_sasa(selected, sasa, int(max_nodes))
        return selected
    if mode == "fps":
        return _fps_downsample_indices(indices, coords, sasa, int(cfg["max_nodes"]))
    if mode == "topk_sasa":
        return _topk_indices_by_sasa(indices, sasa, int(cfg["max_nodes"]))
    raise ValueError(f"Unsupported normalized downsample mode: {mode}")


def select_surface_atom_indices(
    coords: np.ndarray,
    sasa: np.ndarray,
    atom_residue_indices: np.ndarray,
    sasa_filter=None,
    downsample=None,
    sample_id: str | None = None,
) -> np.ndarray:
    sasa_cfg = normalize_surface_atom_sasa_filter(sasa_filter)
    downsample_cfg = normalize_surface_atom_downsample(downsample)
    indices = np.arange(coords.shape[0], dtype=np.int64)
    indices = _apply_surface_atom_sasa_filter(
        indices, sasa, atom_residue_indices, sasa_cfg
    )
    indices = _apply_surface_atom_downsample(indices, coords, sasa, downsample_cfg)
    if indices.size == 0:
        label = f" for {sample_id}" if sample_id else ""
        raise ValueError(
            f"Surface atom selection removed all host nodes{label}. "
            f"sasa_filter={sasa_cfg}, downsample={downsample_cfg}"
        )
    return np.sort(indices.astype(np.int64, copy=False))
