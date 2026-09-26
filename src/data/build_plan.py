from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetBuildSpec:
    name: str
    all_ids_file: str
    pocket_source: str


DATASET_SPECS = (
    DatasetBuildSpec("unisite", "all_ids_unisite_0p9", "ligand_distance"),
    DatasetBuildSpec("holo4k", "all_ids_holo4k", "explicit_residues"),
    DatasetBuildSpec("coach420", "all_ids_coach420", "explicit_residues"),

)

EXPECTED_UNLISTED = {
    "unisite": {"P32722"},
}


def read_sample_ids(data_root: Path, spec: DatasetBuildSpec) -> list[str]:
    split_path = data_root / spec.name / "splits" / spec.all_ids_file
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing canonical all-ids file: {split_path}")
    with split_path.open("r") as handle:
        sample_ids = [line.strip() for line in handle if line.strip()]
    if not sample_ids:
        raise ValueError(f"Canonical all-ids file is empty: {split_path}")
    invalid_ids = [
        sample_id for sample_id in sample_ids if Path(sample_id).name != sample_id
    ]
    if invalid_ids:
        raise ValueError(f"Invalid sample ids in {split_path}: {invalid_ids[:10]}")
    duplicates = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate sample ids in {split_path}: {duplicates[:10]}")
    return sample_ids


def validate_data_root(data_root: Path, specs=DATASET_SPECS) -> dict[str, list[str]]:
    data_root = data_root.resolve()
    ids_by_dataset: dict[str, list[str]] = {}
    for spec in specs:
        dataset_root = data_root / spec.name
        sample_ids = read_sample_ids(data_root, spec)
        ids_by_dataset[spec.name] = sample_ids
        listed = set(sample_ids)
        raw_ids = {
            path.name
            for path in (dataset_root / "protein_ligand").iterdir()
            if path.is_dir()
        }
        esm_ids = {path.stem for path in (dataset_root / "esm").glob("*.npy")}
        missing_raw = sorted(listed - raw_ids)
        missing_esm = sorted(listed - esm_ids)
        if missing_raw or missing_esm:
            raise FileNotFoundError(
                f"Incomplete inputs for {spec.name}: missing_raw={missing_raw[:10]}, missing_esm={missing_esm[:10]}"
            )
        expected_unlisted = EXPECTED_UNLISTED.get(spec.name, set())
        unlisted_raw = raw_ids - listed
        unlisted_esm = esm_ids - listed
        if unlisted_raw != expected_unlisted or unlisted_esm != expected_unlisted:
            raise ValueError(
                f"Unexpected unlisted inputs for {spec.name}: raw={sorted(unlisted_raw)}, "
                f"esm={sorted(unlisted_esm)}, expected={sorted(expected_unlisted)}"
            )
    return ids_by_dataset
