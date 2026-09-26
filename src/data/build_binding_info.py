#!/usr/bin/env python3
"""Build pocket targets into <dataset>/<output_modal>/<sample>.npz."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.SeqUtils import seq1
from click.core import ParameterSource
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .config_utils import load_yaml_config, resolve_feature_build_config
from .modal_paths import modal_dir, modal_path


_DEPTH_WARNING_EMITTED = False

_ELEMENT_SYMBOLS = [
    "X",
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
    "RF",
    "DB",
    "SG",
    "BH",
    "HS",
    "MT",
    "DS",
    "RG",
    "CN",
    "NH",
    "FL",
    "MC",
    "LV",
    "TS",
    "OG",
]
_ATOMIC_NUMBER_BY_SYMBOL = {
    symbol.upper(): idx for idx, symbol in enumerate(_ELEMENT_SYMBOLS)
}


def _iter_sample_dirs(sample_root: Path) -> list[Path]:
    return sorted(
        [entry for entry in sample_root.iterdir() if entry.is_dir()],
        key=lambda path: path.name,
    )


def _normalize_element_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip()
    if not symbol:
        return "X"
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[:1].upper() + symbol[1:].lower()


def _infer_pdb_element(line: str) -> str:
    element = _normalize_element_symbol(line[76:78].strip())
    if element != "X":
        return element

    atom_name = line[12:16].strip()
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return "X"
    if len(letters) >= 2 and letters[:2].upper() in _ATOMIC_NUMBER_BY_SYMBOL:
        return _normalize_element_symbol(letters[:2])
    return _normalize_element_symbol(letters[:1])


def _read_pdb_atoms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    atomic_numbers = []
    with path.open("r") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                continue
            try:
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
                symbol = _infer_pdb_element(line).upper()
                atomic_numbers.append(_ATOMIC_NUMBER_BY_SYMBOL.get(symbol, 0))
            except ValueError:
                continue
    return np.asarray(coords, dtype=np.float32), np.asarray(
        atomic_numbers, dtype=np.int32
    )


def _read_sdf_atoms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r") as handle:
        lines = handle.readlines()

    if len(lines) < 4:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    counts_line = lines[3]
    if "V3000" in counts_line:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    try:
        atom_count = int(counts_line[:3])
    except ValueError:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    coords = []
    atomic_numbers = []
    atom_start = 4
    atom_end = atom_start + atom_count
    for idx in range(atom_start, min(atom_end, len(lines))):
        line = lines[idx]
        try:
            coords.append(
                [
                    float(line[0:10]),
                    float(line[10:20]),
                    float(line[20:30]),
                ]
            )
            symbol = _normalize_element_symbol(line[31:34].strip()).upper()
            atomic_numbers.append(_ATOMIC_NUMBER_BY_SYMBOL.get(symbol, 0))
        except ValueError:
            continue
    return np.asarray(coords, dtype=np.float32), np.asarray(
        atomic_numbers, dtype=np.int32
    )


def _read_mol2_atoms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    atomic_numbers = []
    in_atom_block = False
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom_block:
                break
            if not in_atom_block:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                atom_type = parts[5] if len(parts) > 5 else ""
                symbol = _normalize_element_symbol(atom_type.split(".")[0]).upper()
                atomic_numbers.append(_ATOMIC_NUMBER_BY_SYMBOL.get(symbol, 0))
            except ValueError:
                continue
    return np.asarray(coords, dtype=np.float32), np.asarray(
        atomic_numbers, dtype=np.int32
    )


def _read_ligand_atoms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return _read_pdb_atoms(path)
    if suffix == ".sdf":
        return _read_sdf_atoms(path)
    if suffix == ".mol2":
        return _read_mol2_atoms(path)
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)


def _filter_heavy_ligand_atoms(
    path: Path, coords: np.ndarray, atomic_numbers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if coords.shape[0] != atomic_numbers.shape[0]:
        raise ValueError(
            f"Ligand atom count mismatch in {path}: {coords.shape[0]} coords vs {atomic_numbers.shape[0]} elements."
        )
    if coords.shape[0] == 0:
        return coords, atomic_numbers
    heavy = atomic_numbers != 1
    if not heavy.any():
        raise ValueError(f"No heavy ligand atoms found in {path}.")
    return coords[heavy].astype(np.float32, copy=False), atomic_numbers[heavy].astype(
        np.int32, copy=False
    )


def _extract_ligands(sample_dir: Path) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    lig_paths = sorted(
        [
            path
            for path in sample_dir.iterdir()
            if path.is_file() and path.name.lower().startswith("ligand")
        ],
        key=lambda path: path.name,
    )
    ligands: list[tuple[Path, np.ndarray, np.ndarray]] = []
    for path in lig_paths:
        coords, atomic_numbers = _read_ligand_atoms(path)
        coords, atomic_numbers = _filter_heavy_ligand_atoms(
            path, coords, atomic_numbers
        )
        if coords.shape[0] > 0:
            ligands.append((path, coords, atomic_numbers))
    return ligands


def _residue_key(residue) -> str:
    residue_number = str(residue.id[1]).strip()
    insertion_code = str(residue.id[2]).strip()
    return f"{residue_number}{insertion_code}" if insertion_code else residue_number


def _is_heavy_atom(atom) -> bool:
    element = (getattr(atom, "element", "") or "").strip().upper()
    name = atom.get_name().strip().upper()
    if element == "H":
        return False
    if name.startswith("H"):
        return False
    if name[:1].isdigit() and len(name) > 1 and name[1] == "H":
        return False
    return True


def _atom_atomic_number(atom) -> int:
    symbol = _normalize_element_symbol(
        (getattr(atom, "element", "") or "").strip()
    ).upper()
    if symbol == "X":
        name = atom.get_name().strip().upper()
        letters = "".join(ch for ch in name if ch.isalpha())
        if len(letters) >= 2 and letters[:2].upper() in _ATOMIC_NUMBER_BY_SYMBOL:
            symbol = _normalize_element_symbol(letters[:2]).upper()
        elif letters:
            symbol = _normalize_element_symbol(letters[:1]).upper()
    return int(_ATOMIC_NUMBER_BY_SYMBOL.get(symbol, 0))


def _calculate_residue_depths(structure) -> dict[tuple[str, int], float] | None:
    global _DEPTH_WARNING_EMITTED

    model = structure[0]
    residue_depth = ResidueDepth(model)

    depth_map: dict[tuple[str, int], float] = {}
    for chain in model:
        for residue in chain:
            key = (chain.id, residue.id[1])
            try:
                depth_tuple = residue_depth[chain.id, residue.id]
                depth_map[key] = float(depth_tuple[0])
            except KeyError:
                depth_map[key] = np.nan
    return depth_map


def _extract_surface_atoms(
    protein_path: Path, use_residue_depths: bool, surface_sasa_threshold: float
):
    structure = PDBParser(QUIET=True).get_structure("protein", str(protein_path))
    ShrakeRupley().compute(structure, level="A")
    depth_map = _calculate_residue_depths(structure) if use_residue_depths else None

    atom_coords = []
    atom_atomic_numbers = []
    atom_res_names = []
    atom_res_ids = []
    atom_chains = []
    atom_residue_keys = []
    atom_residue_indices = []
    atom_sasa = []
    atom_depths = []
    residue_index = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                try:
                    _ = seq1(residue.resname)
                except KeyError:
                    continue
                if "CA" not in residue:
                    continue
                residue_key = _residue_key(residue)
                if not use_residue_depths:
                    residue_depth = 0.0
                elif depth_map is None:
                    residue_depth = np.nan
                else:
                    residue_depth = depth_map.get((chain.id, residue.id[1]), np.nan)

                for atom in residue.get_atoms():
                    if not _is_heavy_atom(atom):
                        continue
                    sasa = float(getattr(atom, "sasa", 0.0) or 0.0)
                    if sasa <= surface_sasa_threshold:
                        continue
                    atom_coords.append(atom.get_coord())
                    atom_atomic_numbers.append(_atom_atomic_number(atom))
                    atom_res_names.append(residue.resname)
                    atom_res_ids.append(residue.id[1])
                    atom_chains.append(chain.id)
                    atom_residue_keys.append(residue_key)
                    atom_residue_indices.append(residue_index)
                    atom_sasa.append(sasa)
                    atom_depths.append(residue_depth)
                residue_index += 1

    return (
        np.asarray(atom_coords, dtype=np.float32),
        np.asarray(atom_atomic_numbers, dtype=np.int32),
        np.asarray(atom_res_names),
        np.asarray(atom_res_ids, dtype=np.int32),
        np.asarray(atom_chains),
        np.asarray(atom_residue_keys),
        np.asarray(atom_residue_indices, dtype=np.int32),
        np.asarray(atom_sasa, dtype=np.float32),
        np.asarray(atom_depths, dtype=np.float32),
    )


def _compute_surface_atom_masks(
    atom_coords: np.ndarray,
    ligands: list[tuple[Path, np.ndarray, np.ndarray]],
    threshold: float,
):
    bind_masks = []
    centers = []
    ligand_coords_all = []
    ligand_atomic_numbers_all = []
    ligand_ids_all = []
    for ligand_id, (_, lig_coords, lig_atomic_numbers) in enumerate(ligands):
        distances = np.linalg.norm(
            atom_coords[:, None, :] - lig_coords[None, :, :], axis=-1
        )
        min_atom_distances = distances.min(axis=-1)
        mask = min_atom_distances <= threshold
        bind_masks.append(mask.astype(np.float32))
        centers.append(lig_coords.mean(axis=0).astype(np.float32))
        ligand_coords_all.append(lig_coords)
        ligand_atomic_numbers_all.append(lig_atomic_numbers)
        ligand_ids_all.extend([ligand_id] * lig_coords.shape[0])
    return (
        bind_masks,
        centers,
        ligand_coords_all,
        ligand_atomic_numbers_all,
        ligand_ids_all,
    )


def _write_target(output_path: Path, payload: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **payload)
    tmp_path.replace(output_path)


def _build_one(
    sample_dir: Path,
    dataset_root: Path,
    output_modal: str,
    threshold: float,
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

    (
        bind_masks,
        centers,
        ligand_coords_all,
        ligand_atomic_numbers_all,
        ligand_ids_all,
    ) = _compute_surface_atom_masks(
        atom_coords=atom_coords,
        ligands=ligands,
        threshold=threshold,
    )

    payload = {
        "binding_residues": np.any(np.stack(bind_masks, axis=0), axis=0).astype(
            np.float32
        ),
        "binding_site_masks": np.stack(bind_masks, axis=0).astype(np.float32),
        "binding_site_centers": np.asarray(centers, dtype=np.float32),
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
    "--threshold",
    default=4.0,
    type=float,
    help="Protein-atom to ligand distance threshold in Angstrom.",
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
    help="Whether to compute residue depths via Bio.PDB.ResidueDepth/MSMS.",
)
@click.option("--n-jobs", default=1, type=int)
@click.option(
    "--overwrite/--skip-existing",
    default=False,
    help="Overwrite existing target files.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config: Path | None,
    dataset_root: Path,
    input_dir: str,
    output_modal: str,
    threshold: float,
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
    threshold = resolve("threshold", threshold, cfg_values["threshold"])
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

    dataset_root = dataset_root.expanduser().resolve()
    sample_root = dataset_root / input_dir
    if not sample_root.exists():
        raise click.ClickException(f"Sample root does not exist: {sample_root}")

    output_dir = modal_dir(dataset_root, output_modal)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = _iter_sample_dirs(sample_root)

    if n_jobs <= 1:
        ok = [
            _build_one(
                sample_dir,
                dataset_root=dataset_root,
                output_modal=output_modal,
                threshold=threshold,
                surface_sasa_threshold=surface_sasa_threshold,
                overwrite=overwrite,
                use_residue_depths=use_residue_depths,
            )
            for sample_dir in tqdm(sample_dirs, desc="build_binding")
        ]
    else:
        ok = Parallel(n_jobs=n_jobs)(
            delayed(_build_one)(
                sample_dir,
                dataset_root=dataset_root,
                output_modal=output_modal,
                threshold=threshold,
                surface_sasa_threshold=surface_sasa_threshold,
                overwrite=overwrite,
                use_residue_depths=use_residue_depths,
            )
            for sample_dir in tqdm(sample_dirs, desc="build_binding")
        )

    click.echo(f"done {sum(ok)}/{len(ok)}")


if __name__ == "__main__":
    main()
