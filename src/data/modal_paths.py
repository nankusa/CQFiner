from pathlib import Path


def modal_dir(dataset_root: Path, modal: str) -> Path:
    return dataset_root / modal


def modal_path(dataset_root: Path, modal: str, sample_id: str, suffix: str) -> Path:
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return modal_dir(dataset_root, modal) / f"{sample_id}{suffix}"
