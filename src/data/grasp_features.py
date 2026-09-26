from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from scipy.spatial import cKDTree


_RESIDUE_FEATURES = {
    "ALA": [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "ARG": [
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
    ],
    "ASN": [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "ASP": [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
    ],
    "CYS": [
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
    ],
    "GLN": [
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "GLU": [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
    ],
    "GLY": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "HIS": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
    ],
    "ILE": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "LEU": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "LYS": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
    ],
    "MET": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "PHE": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "PRO": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "SER": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "THR": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "TRP": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
    "TYR": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
    ],
    "VAL": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ],
}

_ATOM_FEATURES = {
    "C": [1, 0, 0, 0],
    "N": [0, 1, 0, 0],
    "O": [0, 0, 1, 0],
    "S": [0, 0, 0, 1],
}

_HYBRIDIZATION_FEATURES = {
    "SP2": [1, 0],
    "SP3": [0, 1],
}

_DEGREE_RADII = np.arange(2.0, 11.0, 1.0, dtype=np.float32)
_DENSITY_VOLUMES = ((4.0 / 3.0) * np.pi * (_DEGREE_RADII**3)).astype(np.float32)


@lru_cache(maxsize=1)
def _feature_factory():
    return ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )


def _safe_residue_name(name: str) -> str:
    cleaned = "".join(ch for ch in (name or "").upper() if ch.isalpha())
    return cleaned[:3]


def _is_standard_residue(name: str) -> bool:
    try:
        seq1(name)
        return True
    except KeyError:
        return False


def _protein_mol_from_pdb(protein_path: Path, *, prepare_chemistry: bool = True):
    mol = Chem.MolFromPDBFile(str(protein_path), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(
            f"Failed to parse protein structure with RDKit: {protein_path}"
        )
    if prepare_chemistry:
        Chem.SanitizeMol(mol, catchErrors=True)
        try:
            Chem.GetSymmSSSR(mol)
        except Exception:  # noqa: BLE001
            pass
        mol.UpdatePropertyCache(strict=False)
        Chem.rdmolops.SetHybridization(mol)
    return mol


def _feature_membership(mol, feature_name: str) -> set[int]:
    atom_ids: set[int] = set()
    try:
        features = _feature_factory().GetFeaturesForMol(mol, includeOnly=feature_name)
    except Exception:  # noqa: BLE001
        return atom_ids
    for feature in features:
        atom_ids.update(int(atom_id) for atom_id in feature.GetAtomIds())
    return atom_ids


def _cumulative_radial_density(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.zeros((0, 9), dtype=np.float32)
    tree = cKDTree(coords)
    counts = []
    for radius in _DEGREE_RADII:
        count = tree.query_ball_point(coords, float(radius), return_length=True)
        counts.append(np.asarray(count, dtype=np.float32) - 1.0)
    cumulative = np.stack(counts, axis=1)
    return cumulative / _DENSITY_VOLUMES[None, :]


def _surface_degree_feature(coords: np.ndarray, cutoff: float) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    tree = cKDTree(coords)
    counts = tree.query_ball_point(coords, float(cutoff), return_length=True)
    degree = np.asarray(counts, dtype=np.float32) - 1.0
    return degree[:, None]


def _binary_pair(flag: bool) -> list[float]:
    return [1.0, 0.0] if flag else [0.0, 1.0]


def _heavy_atom_records(protein_path: Path) -> list[dict]:
    mol = _protein_mol_from_pdb(protein_path, prepare_chemistry=True)
    conformer = mol.GetConformer()
    acceptors = _feature_membership(mol, "Acceptor")
    donors = _feature_membership(mol, "Donor")
    hydrophobes = _feature_membership(mol, "Hydrophobe")
    lumped_hydrophobes = _feature_membership(mol, "LumpedHydrophobe")

    records: list[dict] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        residue_info = atom.GetPDBResidueInfo()
        if residue_info is None:
            continue
        residue_name = _safe_residue_name(residue_info.GetResidueName())
        if not _is_standard_residue(residue_name):
            continue
        atom_name = residue_info.GetName().strip().upper()
        if atom_name == "":
            atom_name = atom.GetSymbol().upper()
        position = conformer.GetAtomPosition(atom.GetIdx())
        records.append(
            {
                "atom_idx": int(atom.GetIdx()),
                "atomic_num": int(atom.GetAtomicNum()),
                "res_name": residue_name,
                "res_id": int(residue_info.GetResidueNumber()),
                "chain_id": residue_info.GetChainId().strip() or " ",
                "icode": residue_info.GetInsertionCode().strip()
                if residue_info.GetInsertionCode()
                else "",
                "atom_name": atom_name,
                "coord": np.asarray(
                    [position.x, position.y, position.z], dtype=np.float32
                ),
                "formal_charge": float(atom.GetFormalCharge()),
                "num_bonds_w_heavy_atoms": float(
                    atom.GetTotalDegree() - atom.GetTotalNumHs(includeNeighbors=True)
                ),
                "ring": _binary_pair(_safe_atom_flag(atom, "ring")),
                "aromatic": _binary_pair(_safe_atom_flag(atom, "aromatic")),
                "mass": float(atom.GetMass()),
                "hybridization": _HYBRIDIZATION_FEATURES.get(
                    str(atom.GetHybridization()), [0.0, 0.0]
                ),
                "acceptor": _binary_pair(atom.GetIdx() in acceptors),
                "donor": _binary_pair(atom.GetIdx() in donors),
                "hydrophobe": _binary_pair(atom.GetIdx() in hydrophobes),
                "lumped_hydrophobe": _binary_pair(atom.GetIdx() in lumped_hydrophobes),
            }
        )
    return records


def _identity_atom_records(protein_path: Path) -> list[dict]:
    mol = _protein_mol_from_pdb(protein_path, prepare_chemistry=False)
    conformer = mol.GetConformer()

    records: list[dict] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        residue_info = atom.GetPDBResidueInfo()
        if residue_info is None:
            continue
        residue_name = _safe_residue_name(residue_info.GetResidueName())
        if not _is_standard_residue(residue_name):
            continue
        atom_name = residue_info.GetName().strip().upper()
        if atom_name == "":
            atom_name = atom.GetSymbol().upper()
        position = conformer.GetAtomPosition(atom.GetIdx())
        records.append(
            {
                "atom_idx": int(atom.GetIdx()),
                "atomic_num": int(atom.GetAtomicNum()),
                "res_name": residue_name,
                "res_id": int(residue_info.GetResidueNumber()),
                "chain_id": residue_info.GetChainId().strip() or " ",
                "icode": residue_info.GetInsertionCode().strip()
                if residue_info.GetInsertionCode()
                else "",
                "atom_name": atom_name,
                "coord": np.asarray(
                    [position.x, position.y, position.z], dtype=np.float32
                ),
            }
        )
    return records


def _base_grasp_feature(
    record: dict, radial_density: np.ndarray, sasa: float
) -> np.ndarray:
    residue_features = _RESIDUE_FEATURES.get(record["res_name"], [0.0] * 28)
    atom_features = _ATOM_FEATURES.get(
        Chem.GetPeriodicTable().GetElementSymbol(record["atomic_num"]).upper(),
        [0.0] * 4,
    )
    values = (
        residue_features
        + atom_features
        + radial_density.tolist()
        + [float(sasa)]
        + [record["formal_charge"]]
        + [record["num_bonds_w_heavy_atoms"]]
        + record["ring"]
        + record["aromatic"]
        + [record["mass"]]
        + record["hybridization"]
        + record["acceptor"]
        + record["donor"]
        + record["hydrophobe"]
        + record["lumped_hydrophobe"]
    )
    return np.asarray(values, dtype=np.float32)


def _safe_atom_flag(atom, flag: str) -> bool:
    try:
        if flag == "ring":
            return bool(atom.IsInRing())
        if flag == "aromatic":
            return bool(atom.GetIsAromatic())
    except Exception:  # noqa: BLE001
        return False
    raise ValueError(f"Unsupported atom flag: {flag}")


def _atom_atomic_number_from_biopython(atom) -> int:
    element_symbol = (getattr(atom, "element", "") or "").strip()
    if element_symbol:
        return int(Chem.GetPeriodicTable().GetAtomicNumber(element_symbol.title()))
    return int(
        Chem.GetPeriodicTable().GetAtomicNumber(
            atom.get_name().strip()[:1].title() or "X"
        )
    )


def _full_pdb_heavy_atom_metadata(protein_path: Path) -> list[dict]:
    structure = PDBParser(QUIET=True).get_structure("protein", str(protein_path))
    atoms: list[dict] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                residue_name = _safe_residue_name(residue.resname)
                if not _is_standard_residue(residue_name):
                    continue
                if "CA" not in residue:
                    continue
                for atom in residue.get_atoms():
                    atomic_num = _atom_atomic_number_from_biopython(atom)
                    if atomic_num <= 1:
                        continue
                    atoms.append(
                        {
                            "chain_id": chain.id.strip() or " ",
                            "res_id": int(residue.id[1]),
                            "icode": str(residue.id[2]).strip(),
                            "res_name": residue_name,
                            "atom_name": atom.get_name().strip().upper(),
                            "atomic_num": atomic_num,
                            "coord": np.asarray(atom.get_coord(), dtype=np.float32),
                        }
                    )
    return atoms


def _surface_atom_metadata_from_target(
    protein_path: Path,
    atom_coords: np.ndarray,
    atom_atomic_numbers: np.ndarray,
    atom_res_names: np.ndarray,
    atom_res_ids: np.ndarray,
    atom_chains: np.ndarray,
) -> list[dict]:
    full_atoms = _full_pdb_heavy_atom_metadata(protein_path)
    grouped: dict[tuple[str, int, str, int], list[int]] = {}
    for idx, atom in enumerate(full_atoms):
        key = (atom["chain_id"], atom["res_id"], atom["res_name"], atom["atomic_num"])
        grouped.setdefault(key, []).append(idx)

    used = np.zeros(len(full_atoms), dtype=bool)
    surface_atoms: list[dict] = []
    for coord, atomic_num, res_name, res_id, chain_id in zip(
        atom_coords,
        atom_atomic_numbers,
        atom_res_names,
        atom_res_ids,
        atom_chains,
        strict=True,
    ):
        key = (
            str(chain_id).strip() or " ",
            int(res_id),
            _safe_residue_name(str(res_name)),
            int(atomic_num),
        )
        candidate_ids = grouped.get(key, [])
        best_idx = -1
        best_dist = np.inf
        for candidate_idx in candidate_ids:
            if used[candidate_idx]:
                continue
            distance = float(np.linalg.norm(full_atoms[candidate_idx]["coord"] - coord))
            if distance < best_dist:
                best_idx = candidate_idx
                best_dist = distance
        if best_idx < 0 or best_dist > 1.0e-3:
            raise ValueError(
                "Failed to recover surface atom identity from protein.pdb while building GrASP features "
                f"for key={key} with distance {best_dist:.4e}."
            )
        used[best_idx] = True
        surface_atoms.append(full_atoms[best_idx])
    return surface_atoms


def _match_surface_atoms(records: list[dict], surface_atoms: list[dict]) -> list[int]:
    grouped: dict[tuple[str, int, str, str, str], list[int]] = {}
    for idx, record in enumerate(records):
        key = (
            record["chain_id"],
            record["res_id"],
            record["icode"],
            record["res_name"],
            record["atom_name"],
        )
        grouped.setdefault(key, []).append(idx)

    used = np.zeros(len(records), dtype=bool)
    matched: list[int] = []
    for atom in surface_atoms:
        key = (
            atom["chain_id"],
            atom["res_id"],
            atom["icode"],
            atom["res_name"],
            atom["atom_name"],
        )
        candidate_ids = grouped.get(key, [])
        best_idx = -1
        best_dist = np.inf
        for candidate_idx in candidate_ids:
            if used[candidate_idx]:
                continue
            distance = float(
                np.linalg.norm(records[candidate_idx]["coord"] - atom["coord"])
            )
            if distance < best_dist:
                best_idx = candidate_idx
                best_dist = distance
        if best_idx < 0:
            raise ValueError(
                "Failed to match surface atom to full-protein RDKit atom while building GrASP features "
                f"for key={key}."
            )
        used[best_idx] = True
        matched.append(best_idx)
    return matched


def _prepare_surface_grasp_alignment(
    protein_path: Path,
    atom_coords: np.ndarray,
    atom_atomic_numbers: np.ndarray,
    atom_res_names: np.ndarray,
    atom_res_ids: np.ndarray,
    atom_chains: np.ndarray,
    *,
    validate_only: bool = False,
) -> tuple[list[dict], list[dict], list[int]]:
    records = (
        _identity_atom_records(protein_path)
        if validate_only
        else _heavy_atom_records(protein_path)
    )
    if not records:
        raise ValueError(
            f"No heavy atom records found while building GrASP features for {protein_path}"
        )

    surface_atoms = _surface_atom_metadata_from_target(
        protein_path=protein_path,
        atom_coords=atom_coords,
        atom_atomic_numbers=atom_atomic_numbers,
        atom_res_names=atom_res_names,
        atom_res_ids=atom_res_ids,
        atom_chains=atom_chains,
    )

    surface_indices = _match_surface_atoms(records=records, surface_atoms=surface_atoms)
    return records, surface_atoms, surface_indices


def validate_surface_grasp_sample(
    protein_path: Path,
    atom_coords: np.ndarray,
    atom_atomic_numbers: np.ndarray,
    atom_res_names: np.ndarray,
    atom_res_ids: np.ndarray,
    atom_chains: np.ndarray,
) -> tuple[bool, str | None]:
    try:
        _prepare_surface_grasp_alignment(
            protein_path=protein_path,
            atom_coords=atom_coords,
            atom_atomic_numbers=atom_atomic_numbers,
            atom_res_names=atom_res_names,
            atom_res_ids=atom_res_ids,
            atom_chains=atom_chains,
            validate_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


def build_surface_grasp_features(
    protein_path: Path,
    atom_coords: np.ndarray,
    atom_atomic_numbers: np.ndarray,
    atom_res_names: np.ndarray,
    atom_res_ids: np.ndarray,
    atom_chains: np.ndarray,
    atom_sasa: np.ndarray,
    host_host_cutoff: float,
) -> np.ndarray:
    records, _, surface_indices = _prepare_surface_grasp_alignment(
        protein_path=protein_path,
        atom_coords=atom_coords,
        atom_atomic_numbers=atom_atomic_numbers,
        atom_res_names=atom_res_names,
        atom_res_ids=atom_res_ids,
        atom_chains=atom_chains,
    )
    full_coords = np.stack([record["coord"] for record in records], axis=0).astype(
        np.float32
    )
    radial_density = _cumulative_radial_density(full_coords)

    feature_rows = [
        _base_grasp_feature(
            records[record_idx],
            radial_density[record_idx],
            float(atom_sasa[surface_idx]),
        )
        for surface_idx, record_idx in enumerate(surface_indices)
    ]
    features = np.stack(feature_rows, axis=0).astype(np.float32)
    degree = _surface_degree_feature(
        atom_coords.astype(np.float32), cutoff=host_host_cutoff
    )
    return np.concatenate([features, degree], axis=1).astype(np.float32)
