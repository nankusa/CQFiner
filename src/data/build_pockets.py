#!/usr/bin/env python3
from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from .build_benchmark_binding_info import _build_one as build_benchmark_pocket
from .build_binding_info import _build_one as build_distance_pocket
from .build_plan import DATASET_SPECS, validate_data_root
from .modal_paths import modal_path


POCKET_MODAL = "pocket.__building__"
REQUIRED_FIELDS = {
    "binding_residues",
    "binding_site_masks",
    "binding_site_centers",
    "atom_coords",
    "atom_atomic_numbers",
    "atom_res_names",
    "atom_res_ids",
    "atom_chains",
    "atom_residue_keys",
    "atom_residue_indices",
    "atom_sasa",
    "atom_depths",
    "ligand_coords",
    "ligand_atomic_numbers",
    "ligand_ids",
}


def _init_worker() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def validate_pocket(path: Path, sample_id: str) -> tuple[int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Pocket builder did not create {path}")
    with np.load(path, allow_pickle=False) as pocket:
        missing = sorted(REQUIRED_FIELDS - set(pocket.files))
        if missing:
            raise KeyError(f"Pocket {path} is missing fields: {missing}")
        atom_coords = np.asarray(pocket["atom_coords"])
        atom_count = int(atom_coords.shape[0])
        if atom_coords.ndim != 2 or atom_coords.shape[1] != 3 or atom_count == 0:
            raise ValueError(
                f"Pocket {sample_id} atom_coords must have shape [A, 3] with A>0, got {atom_coords.shape}."
            )
        if not np.isfinite(atom_coords).all():
            raise ValueError(
                f"Pocket {sample_id} atom_coords contains non-finite values."
            )
        for key in (
            "atom_atomic_numbers",
            "atom_res_names",
            "atom_res_ids",
            "atom_chains",
            "atom_residue_keys",
            "atom_residue_indices",
            "atom_sasa",
            "atom_depths",
            "binding_residues",
        ):
            value = np.asarray(pocket[key])
            if value.ndim != 1 or value.shape[0] != atom_count:
                raise ValueError(
                    f"Pocket {sample_id} field {key!r} must have shape [{atom_count}], got {value.shape}."
                )
        for key in ("atom_sasa", "atom_depths"):
            if not np.isfinite(np.asarray(pocket[key])).all():
                raise ValueError(
                    f"Pocket {sample_id} field {key!r} contains non-finite values."
                )
        atom_residue_indices = np.asarray(pocket["atom_residue_indices"])
        if (atom_residue_indices < 0).any():
            raise ValueError(
                f"Pocket {sample_id} atom_residue_indices contains negative values."
            )

        site_masks = np.asarray(pocket["binding_site_masks"])
        centers = np.asarray(pocket["binding_site_centers"])
        site_count = int(site_masks.shape[0])
        if site_masks.ndim != 2 or site_masks.shape[1] != atom_count or site_count == 0:
            raise ValueError(
                f"Pocket {sample_id} binding_site_masks must have shape [S, {atom_count}] with S>0, got {site_masks.shape}."
            )
        if centers.shape != (site_count, 3) or not np.isfinite(centers).all():
            raise ValueError(
                f"Pocket {sample_id} binding_site_centers must have shape [{site_count}, 3] and be finite."
            )
        empty_sites = np.flatnonzero(site_masks.sum(axis=1) <= 0).tolist()
        if empty_sites:
            raise ValueError(
                f"Pocket {sample_id} has empty binding site masks: {empty_sites}"
            )
        expected_union = np.any(site_masks > 0, axis=0)
        if not np.array_equal(
            np.asarray(pocket["binding_residues"]) > 0, expected_union
        ):
            raise ValueError(
                f"Pocket {sample_id} binding_residues does not equal the union of binding_site_masks."
            )

        ligand_coords = np.asarray(pocket["ligand_coords"])
        ligand_z = np.asarray(pocket["ligand_atomic_numbers"])
        ligand_ids = np.asarray(pocket["ligand_ids"])
        ligand_count = int(ligand_coords.shape[0])
        if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3 or ligand_count == 0:
            raise ValueError(
                f"Pocket {sample_id} ligand_coords must have shape [L, 3] with L>0."
            )
        if ligand_z.shape != (ligand_count,) or ligand_ids.shape != (ligand_count,):
            raise ValueError(f"Pocket {sample_id} ligand field lengths do not match.")
        if not np.isfinite(ligand_coords).all():
            raise ValueError(
                f"Pocket {sample_id} ligand_coords contains non-finite values."
            )
        unique_ligand_ids = np.unique(ligand_ids)
        if not np.array_equal(unique_ligand_ids, np.arange(site_count)):
            raise ValueError(
                f"Pocket {sample_id} ligand_ids must cover sites 0..{site_count - 1}, got {unique_ligand_ids.tolist()}."
            )
    return atom_count, site_count, ligand_count


def _build_one(
    data_root: str, dataset: str, sample_id: str, pocket_source: str
) -> tuple[str, int, int, int, int]:
    dataset_root = Path(data_root) / dataset
    sample_dir = dataset_root / "protein_ligand" / sample_id
    if pocket_source == "ligand_distance":
        built = build_distance_pocket(
            sample_dir=sample_dir,
            dataset_root=dataset_root,
            output_modal=POCKET_MODAL,
            threshold=4.0,
            surface_sasa_threshold=1e-4,
            overwrite=False,
            use_residue_depths=False,
        )
    elif pocket_source == "explicit_residues":
        built = build_benchmark_pocket(
            sample_dir=sample_dir,
            dataset_root=dataset_root,
            output_modal=POCKET_MODAL,
            surface_sasa_threshold=1e-4,
            overwrite=False,
            use_residue_depths=False,
        )
    else:
        raise ValueError(f"Unsupported pocket_source={pocket_source!r} for {dataset}.")
    if built is not True:
        raise RuntimeError(
            f"Pocket builder returned {built!r} for {dataset}/{sample_id}."
        )
    path = modal_path(dataset_root, POCKET_MODAL, sample_id, ".npz")
    atom_count, site_count, ligand_count = validate_pocket(path, sample_id)
    return sample_id, path.stat().st_size, atom_count, site_count, ligand_count


def _build_dataset(
    data_root: Path,
    dataset: str,
    sample_ids: list[str],
    pocket_source: str,
    workers: int,
) -> None:
    dataset_root = data_root / dataset
    final_dir = dataset_root / "pocket"
    staging_dir = dataset_root / POCKET_MODAL
    if final_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing pocket directory: {final_dir}"
        )
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=False)

    total_bytes = 0
    total_atoms = 0
    total_sites = 0
    total_ligand_atoms = 0
    completed = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = {
            executor.submit(
                _build_one, str(data_root), dataset, sample_id, pocket_source
            ): sample_id
            for sample_id in sample_ids
        }
        for future in as_completed(futures):
            requested_sample_id = futures[future]
            try:
                sample_id, size, atom_count, site_count, ligand_count = future.result()
                total_bytes += size
                total_atoms += atom_count
                total_sites += site_count
                total_ligand_atoms += ligand_count
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Pocket construction failed for {dataset}/{requested_sample_id}") from error
            completed += 1
            if completed % 250 == 0 or completed == len(sample_ids):
                print(f"[pocket:{dataset}] {completed}/{len(sample_ids)}", flush=True)

    if failures:
        details = "\n".join(failures)
        raise RuntimeError(
            f"Pocket construction failed for {len(failures)} {dataset} samples:\n{details}"
        )
    built_files = sorted(staging_dir.glob("*.npz"))
    if len(built_files) != len(sample_ids):
        raise RuntimeError(
            f"Pocket file count mismatch for {dataset}: built={len(built_files)}, expected={len(sample_ids)}."
        )
    staging_dir.rename(final_dir)
    print(
        f"[pocket:{dataset}] complete samples={len(sample_ids)} bytes={total_bytes} "
        f"atoms={total_atoms} sites={total_sites} ligand_atoms={total_ligand_atoms}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly build all SurfQNet pocket targets."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--datasets", nargs="+", choices=[s.name for s in DATASET_SPECS],
                        default=[s.name for s in DATASET_SPECS])
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError(f"workers must be positive, got {args.workers}.")
    data_root = args.data_root.expanduser().resolve()
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("Duplicate dataset selections")
    specs = [s for s in DATASET_SPECS if s.name in args.datasets]
    ids_by_dataset = validate_data_root(data_root, specs)
    print(f"[pocket] data_root={data_root} workers={args.workers}", flush=True)
    for spec in specs:
        _build_dataset(
            data_root,
            spec.name,
            ids_by_dataset[spec.name],
            spec.pocket_source,
            args.workers,
        )
    print(
        f"[pocket] complete datasets={len(specs)} samples={sum(map(len, ids_by_dataset.values()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
