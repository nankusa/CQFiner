from pathlib import Path
import ast
import csv
import re

import numpy as np
import torch
from torch import Tensor
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree

from .grasp_features import build_surface_grasp_features
from .modal_paths import modal_path
from .plm import _load_residue_embeddings
from .surface import (
    normalize_surface_atom_downsample,
    normalize_surface_atom_sasa_filter,
    select_surface_atom_indices,
)
from .unified import NodeType, make_unified_data


RES_IDS = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "X": 20,
}

ATOM_IDS = {
    "X": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "S": 4,
}


def normalize_host_node_mode(mode: str | None) -> str:
    normalized = str(mode or "residue").lower()
    aliases = {
        "res": "residue",
        "residue": "residue",
        "residue_graph": "residue",
        "ca": "residue",
        "atom": "surface_atom",
        "surface": "surface_atom",
        "surface_atom": "surface_atom",
        "surface_atoms": "surface_atom",
        "surface_atom_graph": "surface_atom",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported host_node_mode: {mode}. Use 'residue' or 'surface_atom'."
        )
    return aliases[normalized]


def _res_type_ids(res_names: np.ndarray) -> Tensor:
    ids = np.array(
        [RES_IDS.get(str(x), RES_IDS["X"]) for x in res_names], dtype=np.int64
    )
    return torch.from_numpy(ids).long()


def _atom_type_ids_and_z(atomic_numbers: np.ndarray) -> tuple[Tensor, Tensor]:
    mapped = np.zeros_like(atomic_numbers, dtype=np.int64)
    mapped[atomic_numbers == 6] = ATOM_IDS["C"]
    mapped[atomic_numbers == 7] = ATOM_IDS["N"]
    mapped[atomic_numbers == 8] = ATOM_IDS["O"]
    mapped[atomic_numbers == 16] = ATOM_IDS["S"]
    atom_z = torch.from_numpy(atomic_numbers.astype(np.int64)).long()
    atom_type_ids = torch.from_numpy(mapped).long()
    return atom_type_ids, atom_z


def _dataset_root_from_sample_dir(sample_dir: Path, input_dir: str) -> Path:
    for parent in sample_dir.parents:
        if parent.name == input_dir:
            return parent.parent
    raise ValueError(f"Could not infer dataset root from sample dir: {sample_dir}")


def _natural_key(text: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def _residue_key(resseq: int, icode: str | None = None) -> str:
    suffix = str(icode or "").strip()
    return f"{int(resseq)}{suffix}" if suffix else str(int(resseq))


def _is_heavy_atom(atom) -> bool:
    element = (getattr(atom, "element", "") or "").strip().upper()
    name = atom.get_name().strip().upper()
    if element == "H" or name.startswith("H"):
        return False
    if name[:1].isdigit() and len(name) > 1 and name[1] == "H":
        return False
    return True


def _parse_residue_ca_records(protein_path: Path) -> dict[str, np.ndarray]:
    structure = PDBParser(QUIET=True).get_structure("protein", str(protein_path))
    coords = []
    res_names = []
    res_ids = []
    res_keys = []
    chains = []
    residue_atom_coords = []
    protein_atom_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                protein_atom_coords.extend(atom.get_coord() for atom in residue.get_atoms())
                if "CA" not in residue:
                    continue
                try:
                    _ = seq1(residue.resname)
                except KeyError:
                    continue
                coords.append(residue["CA"].get_coord())
                res_names.append(residue.resname)
                res_ids.append(residue.id[1])
                res_keys.append(_residue_key(residue.id[1], residue.id[2]))
                chains.append(chain.id.strip() or "A")
                atom_coords = [
                    atom.get_coord()
                    for atom in residue.get_atoms()
                    if _is_heavy_atom(atom)
                ]
                residue_atom_coords.append(np.asarray(atom_coords, dtype=np.float32))

    if not coords:
        raise ValueError(f"No standard CA residues found in {protein_path}")

    return {
        "coords": np.asarray(coords, dtype=np.float32),
        "res_names": np.asarray(res_names),
        "res_ids": np.asarray(res_ids, dtype=np.int32),
        "res_keys": np.asarray(res_keys),
        "chains": np.asarray(chains),
        "atom_coords_by_residue": residue_atom_coords,
        "protein_atom_coords": np.asarray(protein_atom_coords, dtype=np.float32),
    }


def _ligand_heavy_atoms(target_npz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"ligand_coords", "ligand_atomic_numbers", "ligand_ids"}
    missing = sorted(required.difference(target_npz))
    if missing:
        raise KeyError(
            f"Target npz must contain {missing} to define ligand heavy-atom geometric centers."
        )

    ligand_coords = np.asarray(target_npz["ligand_coords"], dtype=np.float32)
    ligand_z = np.asarray(target_npz["ligand_atomic_numbers"], dtype=np.int64)
    ligand_ids = np.asarray(target_npz["ligand_ids"], dtype=np.int64)
    if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3:
        raise ValueError(
            f"ligand_coords must have shape [N, 3], got {ligand_coords.shape}."
        )
    if (
        ligand_coords.shape[0] != ligand_z.shape[0]
        or ligand_coords.shape[0] != ligand_ids.shape[0]
    ):
        raise ValueError(
            "ligand_coords/ligand_atomic_numbers/ligand_ids length mismatch: "
            f"{ligand_coords.shape[0]} vs {ligand_z.shape[0]} vs {ligand_ids.shape[0]}."
        )
    heavy = ligand_z != 1
    ligand_coords = ligand_coords[heavy]
    ligand_z = ligand_z[heavy]
    ligand_ids = ligand_ids[heavy]
    if ligand_ids.size == 0:
        raise ValueError(
            "No ligand heavy atoms found; cannot define ligand geometric centers."
        )
    if (ligand_ids < 0).any():
        raise ValueError("ligand_ids must be non-negative site ids.")
    return (
        ligand_coords.astype(np.float32, copy=False),
        ligand_z.astype(np.int64, copy=False),
        ligand_ids,
    )


def _ligand_centers_by_site(
    ligand_coords: np.ndarray, ligand_ids: np.ndarray
) -> np.ndarray:
    unique_ids = np.unique(ligand_ids)
    expected_ids = np.arange(int(unique_ids[-1]) + 1, dtype=np.int64)
    if not np.array_equal(unique_ids, expected_ids):
        raise ValueError(
            f"ligand_ids must be contiguous from 0, got {unique_ids.tolist()}."
        )

    centers = np.zeros((expected_ids.shape[0], 3), dtype=np.float32)
    for site_idx in expected_ids.tolist():
        site_ligand = ligand_coords[ligand_ids == site_idx]
        if site_ligand.size == 0:
            raise ValueError(
                f"No ligand heavy atoms found for site {site_idx}; cannot define ligand geometric center."
            )
        centers[site_idx] = site_ligand.mean(axis=0)
    return centers


def _validate_atom_residue_indices(
    atom_residue_indices: np.ndarray, num_residues: int, sample_id: str
) -> np.ndarray:
    atom_residue_indices = atom_residue_indices.astype(np.int64, copy=False)
    valid = (atom_residue_indices >= 0) & (atom_residue_indices < num_residues)
    if atom_residue_indices.size > 0 and not valid.all():
        bad = atom_residue_indices[~valid][:5].tolist()
        raise ValueError(
            f"Surface atom/residue index mismatch in {sample_id}: residue_count={num_residues}, "
            f"invalid atom_residue_indices examples={bad}"
        )
    return valid


def _build_host_host_edge_index(
    host_pos: Tensor,
    *,
    cutoff: float,
    max_neighbors: int,
    sample_id: str,
) -> Tensor:
    cutoff = float(cutoff)
    max_neighbors = int(max_neighbors)
    if cutoff <= 0.0:
        raise ValueError(
            f"host_host_cutoff must be positive for {sample_id}, got {cutoff}."
        )
    if max_neighbors <= 0:
        raise ValueError(
            f"graph_max_neighbors must be positive for {sample_id}, got {max_neighbors}."
        )
    if host_pos.ndim != 2 or host_pos.size(1) != 3:
        raise ValueError(
            f"host_pos must have shape [N, 3] for {sample_id}, got {tuple(host_pos.shape)}."
        )

    coords = host_pos.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(coords).all():
        raise ValueError(f"host_pos contains non-finite coordinates for {sample_id}.")

    tree = cKDTree(coords)
    neighbors_by_target = tree.query_ball_point(coords, r=cutoff)
    edge_src: list[int] = []
    edge_dst: list[int] = []
    for target_idx, candidates in enumerate(neighbors_by_target):
        source_idx = np.asarray(
            [idx for idx in candidates if idx != target_idx], dtype=np.int64
        )
        if source_idx.size == 0:
            continue
        delta = coords[source_idx] - coords[target_idx]
        dist_sq = np.einsum("ij,ij->i", delta, delta)
        order = np.lexsort((source_idx, dist_sq))
        selected = source_idx[order[:max_neighbors]]
        edge_src.extend(int(idx) for idx in selected)
        edge_dst.extend([target_idx] * int(selected.size))

    if not edge_src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([edge_src, edge_dst], dtype=torch.long)


def _normalize_position_token(value) -> str:
    token = str(value).strip().strip("'\"")
    if token.endswith(".0"):
        token = token[:-2]
    return token


def _parse_position_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [_normalize_position_token(item) for item in raw if str(item).strip()]

    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = re.split(r"[\s,;+]+", text.strip("[](){}"))
    if isinstance(parsed, (int, float, str)):
        parsed = [parsed]
    return [_normalize_position_token(item) for item in parsed if str(item).strip()]


def _read_pdb_to_uniprot_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [chunk.strip() for chunk in line.split(",", 1)]
            if len(parts) != 2:
                continue
            pdb_residue, uniprot_position = parts
            if pdb_residue and uniprot_position:
                mapping[_normalize_position_token(pdb_residue)] = (
                    _normalize_position_token(uniprot_position)
                )
    return mapping


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _site_index_from_name(path_or_name: Path | str) -> int | None:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    match = re.search(r"(\d+)(?=\.[^.]+$|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def _selected_unisite_site_names(sample_dir: Path) -> list[str]:
    ligand_files = sorted(
        [
            path.name
            for path in sample_dir.iterdir()
            if path.is_file() and path.name.lower().startswith("ligand")
        ],
    )
    manifest_rows = _read_csv_rows(sample_dir / "site_manifest.csv")
    if not manifest_rows:
        return [
            f"site{site_idx}"
            for site_idx in (_site_index_from_name(name) for name in ligand_files)
            if site_idx is not None
        ]

    output_to_site = {
        row.get("output_ligand", ""): row.get("site", "")
        for row in manifest_rows
        if row.get("status", "").lower() == "ok"
        and row.get("output_ligand")
        and row.get("site")
    }
    selected = [output_to_site[name] for name in ligand_files if name in output_to_site]
    if selected:
        return selected
    return [
        row.get("site", "")
        for row in sorted(
            manifest_rows, key=lambda row: _natural_key(row.get("site", ""))
        )
        if row.get("status", "").lower() == "ok" and row.get("site")
    ]


def _site_masks_from_unisite_metadata(
    sample_dir: Path,
    residue_records: dict[str, np.ndarray],
) -> np.ndarray | None:
    rows = _read_csv_rows(sample_dir / "source_info.csv")
    if not rows:
        return None

    selected_site_names = _selected_unisite_site_names(sample_dir)
    if selected_site_names:
        row_by_site = {row.get("site", ""): row for row in rows}
        rows = [
            row_by_site[site_name]
            for site_name in selected_site_names
            if site_name in row_by_site
        ]
        if not rows:
            return None
    else:
        rows = sorted(rows, key=lambda row: _natural_key(row.get("site", "")))

    pdb_to_uniprot = _read_pdb_to_uniprot_mapping(sample_dir / "source.mapping")
    res_keys = np.asarray(
        [_normalize_position_token(item) for item in residue_records["res_keys"]]
    )
    res_uniprot = np.asarray([pdb_to_uniprot.get(key, "") for key in res_keys])

    site_masks = []
    for row in rows:
        mask = np.zeros((res_keys.shape[0],), dtype=np.float32)
        uniprot_positions = set(_parse_position_list(row.get("site_position_uniprot")))
        if uniprot_positions and np.any(res_uniprot != ""):
            mask[
                np.isin(
                    res_uniprot, np.asarray(sorted(uniprot_positions), dtype=object)
                )
            ] = 1.0

        if not mask.any():
            pdb_positions = set(_parse_position_list(row.get("site_position_pdb")))
            if pdb_positions:
                mask = _residue_position_mask(residue_records, pdb_positions)
        site_masks.append(mask)

    site_masks_np = np.stack(site_masks, axis=0).astype(np.float32)
    _validate_nonempty_site_masks(
        site_masks_np, sample_dir.name, "source_info metadata"
    )
    return site_masks_np


def _parse_pocket_residue_ids(path: Path) -> set[str]:
    residue_ids: set[str] = set()
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            for token in re.split(r"[\s,;+]+", line):
                token = _normalize_position_token(token)
                if token:
                    residue_ids.add(token)
    return residue_ids


def _residue_position_mask(
    residue_records: dict[str, np.ndarray], residue_ids: set[str]
) -> np.ndarray:
    res_keys = np.asarray(
        [_normalize_position_token(item) for item in residue_records["res_keys"]]
    )
    res_ids = np.asarray(
        [_normalize_position_token(item) for item in residue_records["res_ids"]]
    )
    chains = np.asarray([str(item).strip() for item in residue_records["chains"]])
    mask = np.zeros((res_keys.shape[0],), dtype=np.float32)

    for residue_id in residue_ids:
        token = _normalize_position_token(residue_id)
        if ":" in token:
            chain_token, position_token = token.split(":", 1)
            chain_token = chain_token.strip()
            position_token = _normalize_position_token(position_token)
            if not chain_token or not position_token:
                raise ValueError(
                    f"Invalid chain-qualified residue token: {residue_id!r}"
                )
            mask[
                (
                    (chains == chain_token)
                    & ((res_keys == position_token) | (res_ids == position_token))
                )
            ] = 1.0
        else:
            mask[(res_keys == token) | (res_ids == token)] = 1.0
    return mask


def _validate_nonempty_site_masks(
    site_masks: np.ndarray, sample_id: str, source: str
) -> None:
    empty_sites = np.nonzero(site_masks.sum(axis=1) <= 0)[0].tolist()
    if empty_sites:
        raise ValueError(
            f"{source} produced empty residue site mask(s) for {sample_id}: {empty_sites}"
        )


def _site_masks_from_pocket_files(
    sample_dir: Path,
    residue_records: dict[str, np.ndarray],
) -> np.ndarray | None:
    pocket_files_by_index = {
        _site_index_from_name(path): path
        for path in sample_dir.glob("pocket*.txt")
        if _site_index_from_name(path) is not None
    }
    ligand_indices = [
        _site_index_from_name(path)
        for path in sorted(
            [
                path
                for path in sample_dir.iterdir()
                if path.is_file() and path.name.lower().startswith("ligand")
            ],
            key=lambda path: path.name,
        )
    ]
    shared_indices = [idx for idx in ligand_indices if idx in pocket_files_by_index]
    if shared_indices:
        pocket_files = [pocket_files_by_index[idx] for idx in shared_indices]
    else:
        pocket_files = sorted(
            pocket_files_by_index.values(), key=lambda path: _natural_key(path.name)
        )
    if not pocket_files:
        return None

    site_masks = []
    for pocket_file in pocket_files:
        residue_ids = _parse_pocket_residue_ids(pocket_file)
        site_masks.append(_residue_position_mask(residue_records, residue_ids))
    site_masks_np = np.stack(site_masks, axis=0).astype(np.float32)
    _validate_nonempty_site_masks(
        site_masks_np, sample_dir.name, "pocket residue files"
    )
    return site_masks_np


def _load_residue_site_masks(
    sample_dir: Path,
    residue_records: dict[str, np.ndarray],
    target_npz,
    atom_residue_indices: np.ndarray,
    expected_sites: int,
    sample_id: str,
) -> np.ndarray:
    site_masks = _site_masks_from_unisite_metadata(sample_dir, residue_records)
    if site_masks is None:
        site_masks = _site_masks_from_pocket_files(sample_dir, residue_records)
    if site_masks is not None:
        if site_masks.shape[0] != expected_sites:
            raise ValueError(
                f"Residue site count mismatch in {sample_id}: metadata has {site_masks.shape[0]} sites, "
                f"target file has {expected_sites} centers/ligands."
            )
        return site_masks

    target_site_masks = _residue_site_masks_from_target_npz(
        target_npz=target_npz,
        atom_residue_indices=atom_residue_indices,
        num_residues=int(residue_records["coords"].shape[0]),
        expected_sites=expected_sites,
        sample_id=sample_id,
    )
    if target_site_masks is not None:
        return target_site_masks
    if site_masks is None:
        raise ValueError(
            f"Cannot build residue-level pocket masks for {sample_id}: expected source_info.csv/source.mapping "
            "or pocket*.txt metadata, or target npz binding_site_masks compatible with residues/atoms."
        )
    raise ValueError(f"Cannot build residue-level pocket masks for {sample_id}.")


def _residue_site_masks_from_target_npz(
    target_npz,
    atom_residue_indices: np.ndarray,
    num_residues: int,
    expected_sites: int,
    sample_id: str,
) -> np.ndarray | None:
    if "binding_site_masks" in target_npz:
        site_masks = np.asarray(target_npz["binding_site_masks"], dtype=np.float32)
    elif "binding_residues" in target_npz:
        site_masks = np.repeat(
            np.asarray(target_npz["binding_residues"], dtype=np.float32)[None, :],
            expected_sites,
            axis=0,
        )
    else:
        return None

    if site_masks.ndim != 2:
        raise ValueError(
            f"binding_site_masks for {sample_id} must be 2D, got shape={site_masks.shape}."
        )
    if site_masks.shape[0] != expected_sites:
        raise ValueError(
            f"binding_site_masks site count mismatch in {sample_id}: "
            f"{site_masks.shape[0]} vs expected_sites={expected_sites}."
        )
    if site_masks.shape[1] == num_residues:
        residue_masks = site_masks.astype(np.float32, copy=False)
    elif site_masks.shape[1] == atom_residue_indices.shape[0]:
        residue_masks = np.zeros((expected_sites, num_residues), dtype=np.float32)
        positive_site, positive_atom = np.nonzero(site_masks > 0)
        if positive_site.size > 0:
            residue_idx = atom_residue_indices[positive_atom]
            residue_masks[positive_site, residue_idx] = 1.0
    else:
        raise ValueError(
            f"binding_site_masks width for {sample_id} matches neither residue nor atom count: "
            f"width={site_masks.shape[1]}, residues={num_residues}, atoms={atom_residue_indices.shape[0]}."
        )

    empty_sites = np.nonzero(residue_masks.sum(axis=1) <= 0)[0].tolist()
    if empty_sites:
        raise ValueError(
            f"Target npz binding_site_masks for {sample_id} contain empty site(s): {empty_sites}"
        )
    return residue_masks.astype(np.float32, copy=False)


def _aggregate_atom_features_to_residues(
    features: np.ndarray,
    atom_residue_indices: np.ndarray,
    num_residues: int,
) -> np.ndarray:
    if features.size == 0:
        return np.zeros((num_residues, 0), dtype=np.float32)

    out = np.zeros((num_residues, features.shape[1]), dtype=np.float32)
    counts = np.zeros((num_residues, 1), dtype=np.float32)
    np.add.at(out, atom_residue_indices, features.astype(np.float32, copy=False))
    np.add.at(counts, atom_residue_indices, 1.0)
    return out / np.maximum(counts, 1.0)


def build_protein_unified_graph(
    sample_dir: Path,
    input_dir: str = "protein_ligand",
    target_modal: str = "pocket",
    embedding_modalities: tuple[str, ...] = ("esm",),
    host_feature_mode: str = "esm",
    host_node_mode: str = "residue",
    host_feature_dim: int | None = None,
    host_host_cutoff: float = 12.0,
    graph_max_neighbors: int = 64,
    surface_atom_sasa_filter=None,
    surface_atom_downsample=None,
    build_edges: bool = True,
):
    dataset_root = _dataset_root_from_sample_dir(sample_dir, input_dir=input_dir)
    sample_id = sample_dir.name
    host_feature_mode = str(host_feature_mode).lower()
    if host_feature_mode not in {"esm", "zeros", "grasp", "none"}:
        raise ValueError(f"Unsupported host_feature_mode: {host_feature_mode}")
    host_node_mode = normalize_host_node_mode(host_node_mode)

    target_path = modal_path(dataset_root, target_modal, sample_id, ".npz")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target file for {sample_id}: {target_path}")
    with np.load(target_path, allow_pickle=False) as target_file:
        target_npz = {key: target_file[key] for key in target_file.files}

    required_surface_fields = {"atom_coords", "atom_sasa", "atom_residue_indices"}
    missing_surface_fields = sorted(required_surface_fields.difference(target_npz))
    if missing_surface_fields:
        raise KeyError(
            f"Missing {missing_surface_fields} in {target_path}. Rebuild pocket targets with the surface-atom "
            "src.data.build_binding_info pipeline."
        )

    raw_surface_pos_np = target_npz["atom_coords"].astype(np.float32, copy=False)
    raw_surface_sasa_np = target_npz["atom_sasa"].astype(np.float32, copy=False)
    atom_residue_indices_np = target_npz["atom_residue_indices"].astype(
        np.int64, copy=False
    )
    residue_records = _parse_residue_ca_records(sample_dir / "protein.pdb")
    num_residues = int(residue_records["coords"].shape[0])
    _validate_atom_residue_indices(atom_residue_indices_np, num_residues, sample_id)
    surface_atom_source_indices_np = np.arange(
        raw_surface_pos_np.shape[0], dtype=np.int64
    )
    if host_node_mode == "surface_atom":
        surface_atom_source_indices_np = select_surface_atom_indices(
            coords=raw_surface_pos_np,
            sasa=raw_surface_sasa_np,
            atom_residue_indices=atom_residue_indices_np,
            sasa_filter=surface_atom_sasa_filter,
            downsample=surface_atom_downsample,
            sample_id=sample_id,
        )
        selected_surface_pos_np = raw_surface_pos_np[surface_atom_source_indices_np]
        selected_surface_sasa_np = raw_surface_sasa_np[surface_atom_source_indices_np]
    else:
        selected_surface_pos_np = raw_surface_pos_np
        selected_surface_sasa_np = raw_surface_sasa_np
    surface_pos = torch.from_numpy(selected_surface_pos_np).float()
    surface_depth = torch.from_numpy(selected_surface_sasa_np).float()

    ligand_coords_np, ligand_z_np, ligand_ids_np = _ligand_heavy_atoms(target_npz)
    target_centers_np = _ligand_centers_by_site(ligand_coords_np, ligand_ids_np)
    expected_sites = int(target_centers_np.shape[0])
    residue_site_masks_np = None
    if host_node_mode == "residue":
        residue_site_masks_np = _load_residue_site_masks(
            sample_dir=sample_dir,
            residue_records=residue_records,
            target_npz=target_npz,
            atom_residue_indices=atom_residue_indices_np,
            expected_sites=expected_sites,
            sample_id=sample_id,
        )

    if host_node_mode == "surface_atom":
        host_pos = surface_pos
        selected_atom_residue_indices_np = atom_residue_indices_np[
            surface_atom_source_indices_np
        ]
        host_label = torch.from_numpy(
            target_npz["binding_residues"][surface_atom_source_indices_np]
        ).float()
        host_depth = surface_depth
        atom_type_id, host_z = _atom_type_ids_and_z(
            target_npz["atom_atomic_numbers"][surface_atom_source_indices_np]
        )
        residue_type_id = _res_type_ids(
            target_npz["atom_res_names"][surface_atom_source_indices_np]
        )
        atom_residue_indices = torch.from_numpy(selected_atom_residue_indices_np).long()
        host_atom_residue_id = atom_residue_indices
        host_atom_source_id = torch.from_numpy(surface_atom_source_indices_np).long()
    else:
        host_pos = torch.from_numpy(residue_records["coords"]).float()
        host_label = torch.from_numpy(
            np.any(residue_site_masks_np > 0, axis=0).astype(np.float32)
        ).float()
        host_depth = torch.ones((num_residues,), dtype=torch.float32)
        atom_type_id = torch.full((num_residues,), ATOM_IDS["X"], dtype=torch.long)
        host_z = torch.zeros((num_residues,), dtype=torch.long)
        residue_type_id = _res_type_ids(residue_records["res_names"])
        atom_residue_indices = torch.from_numpy(atom_residue_indices_np).long()
        host_atom_residue_id = None
        host_atom_source_id = None

    if host_feature_mode == "none":
        host_embed = torch.empty((host_pos.size(0), 0), dtype=torch.float32)
    elif host_feature_mode == "zeros":
        resolved_feature_dim = host_feature_dim
        if resolved_feature_dim is None:
            raise ValueError(
                "host_feature_dim is required when host_feature_mode='zeros' so the ablated host_x "
                "still matches model.input_dim."
            )
        host_embed = torch.zeros(
            (host_pos.size(0), int(resolved_feature_dim)), dtype=torch.float32
        )
    elif host_feature_mode == "grasp":
        resolved_feature_dim = host_feature_dim
        if resolved_feature_dim is None:
            raise ValueError(
                "host_feature_dim is required when host_feature_mode='grasp'."
            )
        grasp_features = build_surface_grasp_features(
            protein_path=sample_dir / "protein.pdb",
            atom_coords=target_npz["atom_coords"],
            atom_atomic_numbers=target_npz["atom_atomic_numbers"],
            atom_res_names=target_npz["atom_res_names"],
            atom_res_ids=target_npz["atom_res_ids"],
            atom_chains=target_npz["atom_chains"],
            atom_sasa=target_npz["atom_sasa"],
            host_host_cutoff=float(host_host_cutoff),
        )
        if grasp_features.shape[1] != int(resolved_feature_dim):
            raise ValueError(
                "GrASP host feature dim mismatch: "
                f"built {grasp_features.shape[1]} dims but data.host_feature_dim={resolved_feature_dim}."
            )
        if host_node_mode == "surface_atom":
            grasp_features = grasp_features[surface_atom_source_indices_np]
        elif host_node_mode == "residue":
            grasp_features = _aggregate_atom_features_to_residues(
                grasp_features,
                atom_residue_indices_np,
                host_pos.size(0),
            )
        host_embed = torch.from_numpy(grasp_features).float()
    else:
        if host_node_mode == "residue":
            embedding_parts = _load_residue_embeddings(
                dataset_root=dataset_root,
                embedding_modalities=embedding_modalities,
                sample_id=sample_id,
                expected_residues=host_pos.size(0),
            )
        else:
            embedding_parts = []
            for modal in embedding_modalities:
                emb_path = modal_path(dataset_root, modal, sample_id, ".npy")
                if not emb_path.exists():
                    raise FileNotFoundError(
                        f"Missing embedding for {sample_id}, modal={modal}: {emb_path}"
                    )
                emb = torch.from_numpy(np.load(emb_path)).float()
                if atom_residue_indices.numel() > 0 and int(
                    atom_residue_indices.max().item()
                ) >= emb.size(0):
                    raise ValueError(
                        f"Atom/residue embedding mismatch in {sample_id} for modal={modal}: "
                        f"max residue index {int(atom_residue_indices.max().item())} vs {emb.size(0)} residue embeddings"
                    )
                embedding_parts.append(emb[atom_residue_indices])

        if not embedding_parts:
            raise ValueError(
                f"No embedding modalities found for {sample_id}. Expected at least one embedding under "
                f"{dataset_root} matching {embedding_modalities}."
            )
        host_embed = torch.cat(embedding_parts, dim=-1)
    host_x = host_embed
    x = host_x
    z = host_z
    pos = host_pos
    edge_index = torch.empty((2, 0), dtype=torch.long)
    if build_edges:
        edge_index = _build_host_host_edge_index(
            host_pos=host_pos,
            cutoff=float(host_host_cutoff),
            max_neighbors=int(graph_max_neighbors),
            sample_id=sample_id,
        )
    node_type = torch.full((x.size(0),), NodeType.HOST, dtype=torch.long)

    target_pos = torch.from_numpy(target_centers_np).float()
    target_id = torch.arange(target_pos.size(0), dtype=torch.long)
    if host_node_mode == "residue":
        site_masks_np = residue_site_masks_np
    else:
        site_masks_np = (
            target_npz["binding_site_masks"]
            if "binding_site_masks" in target_npz
            else None
        )
        if site_masks_np is None:
            site_masks_np = np.repeat(
                target_npz["binding_residues"][None, :], target_pos.size(0), axis=0
            )
        site_masks_np = site_masks_np[:, surface_atom_source_indices_np]
    site_mask_site_id, site_mask_host_id = np.nonzero(site_masks_np > 0)
    target_mask_site_id = torch.from_numpy(site_mask_site_id).long()
    target_mask_host_id = torch.from_numpy(site_mask_host_id).long()
    ligand_pos = torch.from_numpy(ligand_coords_np).float()
    ligand_z = torch.from_numpy(ligand_z_np).long()
    ligand_id = torch.from_numpy(ligand_ids_np).long()

    data = make_unified_data(
        z=z,
        x=x,
        atom_type_id=atom_type_id,
        residue_type_id=residue_type_id,
        pos=pos,
        edge_index=edge_index,
        node_type=node_type,
        base_feat_dim=0,
        host_label=host_label,
        host_depth=host_depth,
        target_pos=target_pos,
        target_id=target_id,
        target_mask_site_id=target_mask_site_id,
        target_mask_host_id=target_mask_host_id,
        ligand_pos=ligand_pos,
        ligand_z=ligand_z,
        ligand_id=ligand_id,
        sample_id=sample_id,
        surface_pos=surface_pos,
        surface_depth=surface_depth,
        protein_atom_pos=torch.from_numpy(residue_records["protein_atom_coords"]),
        host_atom_residue_id=host_atom_residue_id,
        host_atom_source_id=host_atom_source_id,
    )
    data.graph_type = "residue" if host_node_mode == "residue" else "atom"
    return data
