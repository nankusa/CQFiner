#!/usr/bin/env python3
"""Quantify lock-and-key style protein-ligand shape complementarity."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
from click.core import ParameterSource
from tqdm import tqdm

from .config_utils import dataset_root_from_config, load_yaml_config
from .modal_paths import modal_dir


VDW_RADII = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}


@dataclass
class LigandEntry:
    sample_id: str
    ligand_id: int
    xyz: np.ndarray
    radii: np.ndarray

    @property
    def num_atoms(self) -> int:
        return int(self.xyz.shape[0])


def _resolve(ctx: click.Context, name: str, cli_value, cfg_value):
    source = ctx.get_parameter_source(name)
    if source != ParameterSource.DEFAULT:
        return cli_value
    return cfg_value if cfg_value is not None else cli_value


def _radii(atomic_numbers: np.ndarray) -> np.ndarray:
    return np.asarray(
        [VDW_RADII.get(int(z), 1.70) for z in atomic_numbers], dtype=np.float32
    )


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _select_local_host(
    protein_xyz: np.ndarray,
    protein_radii: np.ndarray,
    ligand_xyz: np.ndarray,
    ligand_radii: np.ndarray,
    loose_gap_cutoff: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    if ligand_xyz.shape[0] == 0 or protein_xyz.shape[0] == 0:
        return protein_xyz, protein_radii

    center = ligand_xyz.mean(axis=0)
    ligand_extent = np.linalg.norm(ligand_xyz - center, axis=1).max()
    cutoff = (
        ligand_extent
        + float(ligand_radii.max())
        + float(protein_radii.max())
        + loose_gap_cutoff
    )
    keep = np.linalg.norm(protein_xyz - center, axis=1) <= cutoff
    if not np.any(keep):
        return protein_xyz, protein_radii
    return protein_xyz[keep], protein_radii[keep]


def _min_surface_gap(
    ligand_xyz: np.ndarray,
    ligand_radii: np.ndarray,
    protein_xyz: np.ndarray,
    protein_radii: np.ndarray,
) -> np.ndarray:
    protein_xyz, protein_radii = _select_local_host(
        protein_xyz=protein_xyz,
        protein_radii=protein_radii,
        ligand_xyz=ligand_xyz,
        ligand_radii=ligand_radii,
    )
    distances = np.linalg.norm(
        ligand_xyz[:, None, :] - protein_xyz[None, :, :], axis=-1
    )
    gaps = distances - (ligand_radii[:, None] + protein_radii[None, :])
    return gaps.min(axis=1)


def _pose_metrics(
    ligand_xyz: np.ndarray,
    ligand_radii: np.ndarray,
    protein_xyz: np.ndarray,
    protein_radii: np.ndarray,
) -> dict[str, float]:
    min_gap = _min_surface_gap(
        ligand_xyz=ligand_xyz,
        ligand_radii=ligand_radii,
        protein_xyz=protein_xyz,
        protein_radii=protein_radii,
    )
    support = float(((min_gap >= -0.1) & (min_gap <= 1.5)).mean())
    tight = float(((min_gap >= -0.1) & (min_gap <= 0.8)).mean())
    clash = float((min_gap < -0.3).mean())
    loose = float((min_gap > 2.0).mean())
    return {
        "support_frac": support,
        "tight_frac": tight,
        "clash_frac": clash,
        "loose_frac": loose,
        "mean_gap": float(min_gap.mean()),
        "median_gap": float(np.median(min_gap)),
        "score": float(support - 2.0 * clash - 0.5 * loose),
    }


def _contact_host_stats(
    ligand_xyz: np.ndarray,
    protein_xyz: np.ndarray,
    atom_sasa: np.ndarray,
    contact_distance: float = 4.5,
) -> tuple[int, float]:
    distances = np.linalg.norm(
        ligand_xyz[:, None, :] - protein_xyz[None, :, :], axis=-1
    )
    host_contact_mask = distances.min(axis=0) <= contact_distance
    if not np.any(host_contact_mask):
        return 0, float("nan")
    return int(host_contact_mask.sum()), float(atom_sasa[host_contact_mask].mean())


def _metric_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _metric_win_rate(
    decoys: list[float], native: float, higher_is_better: bool
) -> float:
    if not decoys:
        return float("nan")
    decoys_arr = np.asarray(decoys, dtype=np.float32)
    if higher_is_better:
        return float((decoys_arr <= native).mean())
    return float((decoys_arr >= native).mean())


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x = x[valid]
    y = y[valid]
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _quantile_summary(
    values: np.ndarray, gains: np.ndarray, wins: np.ndarray
) -> list[dict[str, float]]:
    quantiles = np.quantile(values, [0.25, 0.5, 0.75])
    bins = [
        -np.inf,
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
        np.inf,
    ]
    summary = []
    for idx in range(4):
        mask = (values > bins[idx]) & (values <= bins[idx + 1])
        summary.append(
            {
                "bin_index": idx + 1,
                "count": int(mask.sum()),
                "gain_mean": float(np.nanmean(gains[mask]))
                if np.any(mask)
                else float("nan"),
                "gain_median": float(np.nanmedian(gains[mask]))
                if np.any(mask)
                else float("nan"),
                "win_rate_ge_0.95": float(np.nanmean(wins[mask] >= 0.95))
                if np.any(mask)
                else float("nan"),
            }
        )
    return summary


def _find_mismatch_candidates(
    ligand_pool: list[LigandEntry],
    ligand_sizes: np.ndarray,
    sample_id: str,
    ligand_xyz: np.ndarray,
    ligand_size: int,
    size_tolerance: int,
    max_decoys: int,
    rng: np.random.Generator,
) -> list[LigandEntry]:
    if max_decoys <= 0:
        return []

    lower = max(1, ligand_size - size_tolerance)
    upper = ligand_size + size_tolerance
    candidate_indices = np.flatnonzero(
        (ligand_sizes >= lower) & (ligand_sizes <= upper)
    )
    if candidate_indices.size == 0:
        return []

    rng.shuffle(candidate_indices)
    candidates: list[LigandEntry] = []
    for idx in candidate_indices:
        entry = ligand_pool[int(idx)]
        if (
            entry.sample_id == sample_id
            and entry.xyz.shape == ligand_xyz.shape
            and np.allclose(entry.xyz, ligand_xyz)
        ):
            continue
        candidates.append(entry)
        if len(candidates) >= max_decoys:
            break
    return candidates


def _build_ligand_pool(
    target_paths: list[Path],
) -> tuple[list[LigandEntry], np.ndarray]:
    pool: list[LigandEntry] = []
    for path in target_paths:
        sample_id = path.stem
        target_npz = np.load(path)
        ligand_ids = np.asarray(target_npz["ligand_ids"])
        ligand_xyz_all = np.asarray(target_npz["ligand_coords"], dtype=np.float32)
        ligand_radii_all = _radii(target_npz["ligand_atomic_numbers"])
        for ligand_id in np.unique(ligand_ids):
            mask = ligand_ids == ligand_id
            pool.append(
                LigandEntry(
                    sample_id=sample_id,
                    ligand_id=int(ligand_id),
                    xyz=ligand_xyz_all[mask].copy(),
                    radii=ligand_radii_all[mask].copy(),
                )
            )
    ligand_sizes = np.asarray([entry.num_atoms for entry in pool], dtype=np.int32)
    return pool, ligand_sizes


def _summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    def arr(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float32)

    lig_atoms = arr("lig_atoms")
    contact_host_atoms = arr("contact_host_atoms")
    rot_score_gain = arr("native_score") - arr("rot_score_mean")
    mismatch_score_gain = arr("native_score") - arr("mismatch_score_mean")
    rot_score_win = arr("rot_score_win_rate")
    mismatch_score_win = arr("mismatch_score_win_rate")

    return {
        "pairs": len(rows),
        "native_score_mean": float(np.nanmean(arr("native_score"))),
        "rotation_score_mean": float(np.nanmean(arr("rot_score_mean"))),
        "mismatch_score_mean": float(np.nanmean(arr("mismatch_score_mean"))),
        "native_support_mean": float(np.nanmean(arr("native_support_frac"))),
        "rotation_support_mean": float(np.nanmean(arr("rot_support_mean"))),
        "mismatch_support_mean": float(np.nanmean(arr("mismatch_support_mean"))),
        "native_clash_mean": float(np.nanmean(arr("native_clash_frac"))),
        "rotation_clash_mean": float(np.nanmean(arr("rot_clash_mean"))),
        "mismatch_clash_mean": float(np.nanmean(arr("mismatch_clash_mean"))),
        "rotation_score_win_ge_0.95": float(np.nanmean(rot_score_win >= 0.95)),
        "mismatch_score_win_ge_0.75": float(np.nanmean(mismatch_score_win >= 0.75)),
        "mismatch_score_win_ge_0.875": float(np.nanmean(mismatch_score_win >= 0.875)),
        "corr_lig_atoms_rot_score_gain": _corr(lig_atoms, rot_score_gain),
        "corr_contact_host_atoms_rot_score_gain": _corr(
            contact_host_atoms, rot_score_gain
        ),
        "corr_lig_atoms_mismatch_score_gain": _corr(lig_atoms, mismatch_score_gain),
        "corr_contact_host_atoms_mismatch_score_gain": _corr(
            contact_host_atoms, mismatch_score_gain
        ),
        "lig_atom_quartiles": [
            float(x) for x in np.quantile(lig_atoms, [0.25, 0.5, 0.75])
        ],
        "contact_host_atom_quartiles": [
            float(x) for x in np.quantile(contact_host_atoms, [0.25, 0.5, 0.75])
        ],
        "lig_atom_rotation_gain_bins": _quantile_summary(
            lig_atoms, rot_score_gain, rot_score_win
        ),
        "lig_atom_mismatch_gain_bins": _quantile_summary(
            lig_atoms, mismatch_score_gain, mismatch_score_win
        ),
        "contact_host_rotation_gain_bins": _quantile_summary(
            contact_host_atoms, rot_score_gain, rot_score_win
        ),
        "contact_host_mismatch_gain_bins": _quantile_summary(
            contact_host_atoms, mismatch_score_gain, mismatch_score_win
        ),
    }


def _print_summary(summary: dict[str, object]) -> None:
    click.echo(f"pairs: {summary['pairs']}")
    click.echo(f"native_score_mean: {summary['native_score_mean']:.4f}")
    click.echo(f"rotation_score_mean: {summary['rotation_score_mean']:.4f}")
    click.echo(f"mismatch_score_mean: {summary['mismatch_score_mean']:.4f}")
    click.echo(f"native_support_mean: {summary['native_support_mean']:.4f}")
    click.echo(f"rotation_support_mean: {summary['rotation_support_mean']:.4f}")
    click.echo(f"mismatch_support_mean: {summary['mismatch_support_mean']:.4f}")
    click.echo(f"native_clash_mean: {summary['native_clash_mean']:.4f}")
    click.echo(f"rotation_clash_mean: {summary['rotation_clash_mean']:.4f}")
    click.echo(f"mismatch_clash_mean: {summary['mismatch_clash_mean']:.4f}")
    click.echo(
        f"rotation_score_win_ge_0.95: {summary['rotation_score_win_ge_0.95']:.4f}"
    )
    click.echo(
        f"mismatch_score_win_ge_0.75: {summary['mismatch_score_win_ge_0.75']:.4f}"
    )
    click.echo(
        f"mismatch_score_win_ge_0.875: {summary['mismatch_score_win_ge_0.875']:.4f}"
    )
    click.echo(
        f"corr_lig_atoms_rot_score_gain: {summary['corr_lig_atoms_rot_score_gain']:.4f}"
    )
    click.echo(
        f"corr_contact_host_atoms_rot_score_gain: {summary['corr_contact_host_atoms_rot_score_gain']:.4f}"
    )
    click.echo(
        f"corr_lig_atoms_mismatch_score_gain: {summary['corr_lig_atoms_mismatch_score_gain']:.4f}"
    )
    click.echo(
        f"corr_contact_host_atoms_mismatch_score_gain: {summary['corr_contact_host_atoms_mismatch_score_gain']:.4f}"
    )


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file.",
)
@click.option(
    "--path",
    "dataset_root",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--target-modal",
    default="pocket",
    type=str,
    help="Target modality directory under dataset root.",
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    help="Optional limit on the number of target files to scan.",
)
@click.option(
    "--rotation-decoys",
    default=16,
    type=int,
    help="Random rotations per native ligand pose.",
)
@click.option(
    "--mismatch-decoys",
    default=8,
    type=int,
    help="Size-matched ligand swaps per native ligand pose.",
)
@click.option(
    "--mismatch-size-tol",
    default=4,
    type=int,
    help="Allowed atom-count difference for mismatch decoys.",
)
@click.option("--seed", default=0, type=int)
@click.option(
    "--output-csv",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional per-pair result CSV.",
)
@click.option(
    "--summary-json",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional summary JSON.",
)
@click.option(
    "--show-progress/--no-progress", default=True, help="Display a tqdm progress bar."
)
@click.pass_context
def main(
    ctx: click.Context,
    config: Path | None,
    dataset_root: Path | None,
    target_modal: str,
    max_samples: int | None,
    rotation_decoys: int,
    mismatch_decoys: int,
    mismatch_size_tol: int,
    seed: int,
    output_csv: Path | None,
    summary_json: Path | None,
    show_progress: bool,
) -> None:
    cfg = load_yaml_config(config)
    data_cfg = cfg.get("data", {})
    dataset_root = _resolve(
        ctx, "dataset_root", dataset_root, dataset_root_from_config(cfg)
    )
    target_modal = _resolve(
        ctx, "target_modal", target_modal, data_cfg.get("target_modal", "pocket")
    )

    if dataset_root is None:
        raise click.ClickException(
            "Dataset root is required. Provide --path or set data.root + data.dataset_name in --config."
        )

    dataset_root = dataset_root.expanduser().resolve()
    target_root = modal_dir(dataset_root, target_modal)
    if not target_root.exists():
        raise click.ClickException(f"Missing target modality directory: {target_root}")

    target_paths = sorted(target_root.glob("*.npz"))
    if max_samples is not None:
        target_paths = target_paths[:max_samples]
    if not target_paths:
        raise click.ClickException(f"No target files found under {target_root}")

    rng = np.random.default_rng(seed)
    ligand_pool, ligand_sizes = _build_ligand_pool(target_paths)

    rows: list[dict[str, float]] = []
    iterator = (
        tqdm(target_paths, desc="shape_complementarity")
        if show_progress
        else target_paths
    )
    for path in iterator:
        sample_id = path.stem
        target_npz = np.load(path)

        protein_xyz = np.asarray(target_npz["atom_coords"], dtype=np.float32)
        protein_radii = _radii(target_npz["atom_atomic_numbers"])
        atom_sasa = np.asarray(target_npz["atom_sasa"], dtype=np.float32)

        ligand_ids = np.asarray(target_npz["ligand_ids"])
        ligand_xyz_all = np.asarray(target_npz["ligand_coords"], dtype=np.float32)
        ligand_radii_all = _radii(target_npz["ligand_atomic_numbers"])

        for ligand_id in np.unique(ligand_ids):
            ligand_mask = ligand_ids == ligand_id
            ligand_xyz = ligand_xyz_all[ligand_mask]
            ligand_radii = ligand_radii_all[ligand_mask]
            native = _pose_metrics(
                ligand_xyz=ligand_xyz,
                ligand_radii=ligand_radii,
                protein_xyz=protein_xyz,
                protein_radii=protein_radii,
            )

            contact_host_atoms, mean_contact_sasa = _contact_host_stats(
                ligand_xyz=ligand_xyz,
                protein_xyz=protein_xyz,
                atom_sasa=atom_sasa,
            )

            center = ligand_xyz.mean(axis=0, keepdims=True)
            rotation_scores = []
            rotation_supports = []
            rotation_clashes = []
            for _ in range(rotation_decoys):
                rotation = _random_rotation(rng)
                rotated_xyz = (ligand_xyz - center) @ rotation.T + center
                metrics = _pose_metrics(
                    ligand_xyz=rotated_xyz,
                    ligand_radii=ligand_radii,
                    protein_xyz=protein_xyz,
                    protein_radii=protein_radii,
                )
                rotation_scores.append(metrics["score"])
                rotation_supports.append(metrics["support_frac"])
                rotation_clashes.append(metrics["clash_frac"])

            mismatch_scores = []
            mismatch_supports = []
            mismatch_clashes = []
            mismatch_candidates = _find_mismatch_candidates(
                ligand_pool=ligand_pool,
                ligand_sizes=ligand_sizes,
                sample_id=sample_id,
                ligand_xyz=ligand_xyz,
                ligand_size=int(ligand_xyz.shape[0]),
                size_tolerance=mismatch_size_tol,
                max_decoys=mismatch_decoys,
                rng=rng,
            )
            for candidate in mismatch_candidates:
                candidate_center = candidate.xyz.mean(axis=0, keepdims=True)
                recentered_xyz = candidate.xyz - candidate_center + center
                metrics = _pose_metrics(
                    ligand_xyz=recentered_xyz,
                    ligand_radii=candidate.radii,
                    protein_xyz=protein_xyz,
                    protein_radii=protein_radii,
                )
                mismatch_scores.append(metrics["score"])
                mismatch_supports.append(metrics["support_frac"])
                mismatch_clashes.append(metrics["clash_frac"])

            rows.append(
                {
                    "sample_id": sample_id,
                    "ligand_id": int(ligand_id),
                    "lig_atoms": int(ligand_xyz.shape[0]),
                    "host_atoms": int(protein_xyz.shape[0]),
                    "contact_host_atoms": contact_host_atoms,
                    "mean_contact_sasa": mean_contact_sasa,
                    "native_score": native["score"],
                    "native_support_frac": native["support_frac"],
                    "native_tight_frac": native["tight_frac"],
                    "native_clash_frac": native["clash_frac"],
                    "native_loose_frac": native["loose_frac"],
                    "native_mean_gap": native["mean_gap"],
                    "native_median_gap": native["median_gap"],
                    "rot_score_mean": _metric_mean(rotation_scores),
                    "rot_support_mean": _metric_mean(rotation_supports),
                    "rot_clash_mean": _metric_mean(rotation_clashes),
                    "rot_score_win_rate": _metric_win_rate(
                        rotation_scores, native["score"], higher_is_better=True
                    ),
                    "rot_support_win_rate": _metric_win_rate(
                        rotation_supports, native["support_frac"], higher_is_better=True
                    ),
                    "rot_clash_win_rate": _metric_win_rate(
                        rotation_clashes, native["clash_frac"], higher_is_better=False
                    ),
                    "mismatch_score_mean": _metric_mean(mismatch_scores),
                    "mismatch_support_mean": _metric_mean(mismatch_supports),
                    "mismatch_clash_mean": _metric_mean(mismatch_clashes),
                    "mismatch_score_win_rate": _metric_win_rate(
                        mismatch_scores, native["score"], higher_is_better=True
                    ),
                    "mismatch_support_win_rate": _metric_win_rate(
                        mismatch_supports, native["support_frac"], higher_is_better=True
                    ),
                    "mismatch_clash_win_rate": _metric_win_rate(
                        mismatch_clashes, native["clash_frac"], higher_is_better=False
                    ),
                }
            )

    if output_csv is not None:
        output_csv = output_csv.expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = _summarize(rows)
    _print_summary(summary)

    if summary_json is not None:
        summary_json = summary_json.expanduser().resolve()
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        with summary_json.open("w") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
