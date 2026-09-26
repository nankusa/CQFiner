from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .modal_paths import modal_path


def _load_residue_embeddings(
    dataset_root: Path,
    embedding_modalities: tuple[str, ...],
    sample_id: str,
    expected_residues: int,
) -> list[Tensor]:
    embedding_parts = []
    for modal in embedding_modalities:
        emb_path = modal_path(dataset_root, modal, sample_id, ".npy")
        if not emb_path.exists():
            continue
        emb = torch.from_numpy(np.load(emb_path)).float()
        if emb.size(0) != expected_residues:
            raise ValueError(
                f"Residue embedding mismatch in {sample_id} for modal={modal}: "
                f"{emb.size(0)} embeddings vs {expected_residues} CA residues"
            )
        embedding_parts.append(emb)
    return embedding_parts
