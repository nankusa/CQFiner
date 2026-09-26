#!/usr/bin/env python3
"""Build benchmark pocket targets from explicit pocket residue files.

Expected sample layout under ``<dataset>/<input_dir>/<sample_id>/``:

- ``protein.pdb``
- ``ligand_*.pdb|sdf|mol2``
- ``pocket_*.txt``

Each ``pocket_*.txt`` stores one residue identifier per line. The output format
matches ``src.data.build_binding_info`` so the rest of SiteFlow can reuse the
same graph builder and evaluation code.
"""

from __future__ import annotations

from pathlib import Path
import re

import click
import numpy as np
from click.core import ParameterSource
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .build_binding_info import (
    _extract_ligands,
    _extract_surface_atoms,
    _write_target,
)
from .config_utils import load_yaml_config, resolve_feature_build_config
from .modal_paths import modal_path


def _iter_sample_dirs(sample_root: Path) -> list[Path]:
    return sorted(
        [entry for entry in sample_root.iterdir() if entry.is_dir()],
        key=lambda path: path.name,
    )


def _natural_key(text: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def _site_index_from_name(path: Path) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _extract_pocket_files(sample_dir: Path) -> dict[int, Path]:
    pocket_files = {}
    for path in sorted(
        sample_dir.glob("pocket*.txt"), key=lambda item: _natural_key(item.name)
    ):
        site_idx = _site_index_from_name(path)
        if site_idx is None:
            continue
        pocket_files[site_idx] = path
    return pocket_files


def _parse_pocket_residue_ids(path: Path) -> set[str]:
    residue_ids: set[str] = set()
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            for token in re.split(r"[\s,]+", line):
                token = token.strip()
                if token:
                    residue_ids.add(token)
    return residue_ids


def _build_site_mask(
    atom_residue_keys: np.ndarray,
    atom_res_ids: np.ndarray,
    atom_chains: np.ndarray,
    residue_ids: set[str],
) -> np.ndarray:
    if not residue_ids:
        return np.zeros((atom_residue_keys.shape[0],), dtype=np.float32)

    chain_residue_ids = set()
    plain_residue_ids = set()
    for residue_id in residue_ids:
        if ":" in residue_id:
            chain_id, residue_key = residue_id.split(":", 1)
            if not chain_id or not residue_key:
                raise ValueError(f"Invalid chain-aware pocket residue id: {residue_id}")
            chain_residue_ids.add((chain_id, residue_key))
        else:
            plain_residue_ids.add(residue_id)

    residue_key_mask = np.zeros((atom_residue_keys.shape[0],), dtype=bool)
    if chain_residue_ids:
        atom_chain_values = atom_chains.astype(str)
        atom_residue_values = atom_residue_keys.astype(str)
        for chain_id, residue_key in chain_residue_ids:
            residue_key_mask |= (atom_chain_values == chain_id) & (
                atom_residue_values == residue_key
            )

    if plain_residue_ids:
        residue_key_mask |= np.isin(
            atom_residue_keys.astype(str),
            np.asarray(sorted(plain_residue_ids), dtype=object),
        )
    if residue_key_mask.any() or chain_residue_ids:
        return residue_key_mask.astype(np.float32)

    try:
        residue_int_ids = {int(value) for value in plain_residue_ids}
    except ValueError:
        return residue_key_mask.astype(np.float32)
    return np.isin(
        atom_res_ids.astype(np.int32),
        np.asarray(sorted(residue_int_ids), dtype=np.int32),
    ).astype(np.float32)


def _build_one(
    sample_dir: Path,
    dataset_root: Path,
    output_modal: str,
    surface_sasa_threshold: float,
    overwrite: bool,
    use_residue_depths: bool,
) -> bool:
    sample_id = sample_dir.name
    output_path = modal_path(dataset_root, output_modal, sample_id, ".npz")
    if output_path.exists() and not overwrite:
        return True

    protein_path = sample_dir / "protein.pdb"
    if not protein_path.exists():
        return False

    (
        atom_coords,
        atom_atomic_numbers,
        atom_res_names,
        atom_res_ids,
        atom_chains,
        atom_residue_keys,
        atom_residue_indices,
        atom_sasa,
        atom_depths,
    ) = _extract_surface_atoms(
        protein_path,
        use_residue_depths=use_residue_depths,
        surface_sasa_threshold=surface_sasa_threshold,
    )
    if atom_coords.shape[0] == 0:
        return False

    ligands = _extract_ligands(sample_dir)
    if len(ligands) == 0:
        return False

    ligand_by_index = {
        _site_index_from_name(path): (path, coords, atomic_numbers)
        for path, coords, atomic_numbers in ligands
        if _site_index_from_name(path) is not None
    }
    pocket_files = _extract_pocket_files(sample_dir)
    shared_indices = sorted(set(ligand_by_index) & set(pocket_files))
    if not shared_indices:
        return False

    site_masks = []
    site_centers = []
    ligand_coords_all = []
    ligand_atomic_numbers_all = []
    ligand_ids_all = []

    for ligand_id, site_idx in enumerate(shared_indices):
        _, lig_coords, lig_atomic_numbers = ligand_by_index[site_idx]
        residue_ids = _parse_pocket_residue_ids(pocket_files[site_idx])
        mask = _build_site_mask(
            atom_residue_keys=atom_residue_keys,
            atom_res_ids=atom_res_ids,
            atom_chains=atom_chains,
            residue_ids=residue_ids,
        )
        site_masks.append(mask)
        site_centers.append(lig_coords.mean(axis=0).astype(np.float32))

        ligand_coords_all.append(lig_coords)
        ligand_atomic_numbers_all.append(lig_atomic_numbers)
        ligand_ids_all.extend([ligand_id] * lig_coords.shape[0])

    payload = {
        "binding_residues": np.any(np.stack(site_masks, axis=0), axis=0).astype(
            np.float32
        ),
        "binding_site_masks": np.stack(site_masks, axis=0).astype(np.float32),
        "binding_site_centers": np.asarray(site_centers, dtype=np.float32),
        "atom_coords": atom_coords.astype(np.float32),
        "atom_atomic_numbers": atom_atomic_numbers.astype(np.int32),
        "atom_res_names": atom_res_names,
        "atom_res_ids": atom_res_ids,
        "atom_chains": atom_chains,
        "atom_residue_keys": atom_residue_keys,
        "atom_residue_indices": atom_residue_indices.astype(np.int32),
        "atom_sasa": atom_sasa.astype(np.float32),
        "atom_depths": atom_depths.astype(np.float32),
        "ligand_coords": np.concatenate(ligand_coords_all, axis=0).astype(np.float32),
        "ligand_atomic_numbers": np.concatenate(
            ligand_atomic_numbers_all, axis=0
        ).astype(np.int32),
        "ligand_ids": np.asarray(ligand_ids_all, dtype=np.int32),
    }
    _write_target(output_path, payload)
    return True


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
    "--input-dir",
    default="protein_ligand",
    type=str,
    help="Sample directory under dataset root.",
)
@click.option(
    "--output-modal",
    default="pocket",
    type=str,
    help="Output modality directory under dataset root.",
)
@click.option(
    "--surface-sasa-threshold",
    default=1e-4,
    type=float,
    help="Keep only protein heavy atoms with SASA above this threshold as host graph nodes.",
)
@click.option(
    "--use-residue-depths/--no-use-residue-depths",
    default=None,
    help="Whether to compute residue depths via Bio.PDB.ResidueDepth/MSMS during pocket target building.",
)
@click.option("--n-jobs", default=1, type=int)
@click.option("--overwrite/--skip-existing", default=False)
@click.pass_context
def main(
    ctx: click.Context,
    config: Path | None,
    dataset_root: Path | None,
    input_dir: str,
    output_modal: str,
    surface_sasa_threshold: float,
    use_residue_depths: bool | None,
    n_jobs: int,
    overwrite: bool,
):
    cfg = load_yaml_config(config)
    cfg_values = resolve_feature_build_config(cfg)

    def resolve(name: str, cli_value, cfg_value):
        source = ctx.get_parameter_source(name)
        if source != ParameterSource.DEFAULT:
            return cli_value
        return cfg_value if cfg_value is not None else cli_value

    dataset_root = resolve("dataset_root", dataset_root, cfg_values["dataset_root"])
    input_dir = resolve("input_dir", input_dir, cfg_values["input_dir"])
    output_modal = resolve("output_modal", output_modal, cfg_values["target_modal"])
    surface_sasa_threshold = resolve(
        "surface_sasa_threshold",
        surface_sasa_threshold,
        cfg_values["surface_sasa_threshold"],
    )
    use_residue_depths = resolve(
        "use_residue_depths", use_residue_depths, cfg_values["use_residue_depths"]
    )
    n_jobs = resolve("n_jobs", n_jobs, cfg_values["n_jobs"])
    overwrite = resolve("overwrite", overwrite, cfg_values["overwrite"])

    if dataset_root is None:
        raise click.ClickException(
            "Dataset root is required. Provide --path or set data.root + data.dataset_name in --config."
        )

    dataset_root = Path(dataset_root).expanduser().resolve()
    sample_root = dataset_root / input_dir
    if not sample_root.exists():
        raise click.ClickException(f"Input directory does not exist: {sample_root}")

    sample_dirs = _iter_sample_dirs(sample_root)
    if n_jobs <= 1:
        results = [
            _build_one(
                sample_dir=sample_dir,
                dataset_root=dataset_root,
                output_modal=output_modal,
                surface_sasa_threshold=surface_sasa_threshold,
                overwrite=overwrite,
                use_residue_depths=bool(use_residue_depths),
            )
            for sample_dir in tqdm(sample_dirs, desc="build_benchmark_targets")
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_build_one)(
                sample_dir=sample_dir,
                dataset_root=dataset_root,
                output_modal=output_modal,
                surface_sasa_threshold=surface_sasa_threshold,
                overwrite=overwrite,
                use_residue_depths=bool(use_residue_depths),
            )
            for sample_dir in tqdm(sample_dirs, desc="build_benchmark_targets")
        )

    ok = int(sum(bool(result) for result in results))
    click.echo(f"done {ok}/{len(sample_dirs)}")


if __name__ == "__main__":
    main()
