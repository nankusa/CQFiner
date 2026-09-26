from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .graph_builder import build_protein_unified_graph, normalize_host_node_mode
from .modal_paths import modal_path
from .unified import NodeType


class UnifiedStructureDataset(Dataset):
    """Load coordinates, targets and features; connectivity is built by the model."""

    def __init__(
        self,
        root: str | Path,
        sample_names: list[str],
        host_node_mode: str,
        host_feature_mode: str,
        host_feature_dim: int,
        embedding_modalities: list[str] | tuple[str, ...] | None = None,
        input_dir: str = "protein_ligand",
        target_modal: str = "pocket",
        surface_atom_sasa_filter=None,
        surface_atom_downsample=None,
    ) -> None:
        self.root = Path(root)
        self.host_node_mode = normalize_host_node_mode(host_node_mode)
        self.graph_type = "residue" if self.host_node_mode == "residue" else "atom"
        self.input_dir = str(input_dir)
        self.target_modal = str(target_modal)
        self.surface_atom_sasa_filter = surface_atom_sasa_filter
        self.surface_atom_downsample = surface_atom_downsample
        self.host_feature_mode = str(host_feature_mode).lower()
        if self.host_feature_mode not in {"esm", "zeros"}:
            raise ValueError(
                f"Unsupported host_feature_mode={host_feature_mode!r}. "
                "SurfQNet supports only 'esm' and 'zeros'."
            )
        self.host_feature_dim = int(host_feature_dim)
        if self.host_feature_dim <= 0:
            raise ValueError(
                f"host_feature_dim must be positive, got {self.host_feature_dim}."
            )

        self.embedding_modalities = tuple(embedding_modalities or ())
        if self.host_feature_mode == "esm" and not self.embedding_modalities:
            raise ValueError(
                "host_feature_mode='esm' requires at least one embedding modality."
            )
        if self.host_feature_mode == "zeros" and self.embedding_modalities:
            raise ValueError(
                "host_feature_mode='zeros' requires embedding_modalities to be empty."
            )

        self.sample_names = list(sample_names)
        if not self.sample_names:
            raise RuntimeError("The requested split contains no samples.")
        invalid_ids = [
            sample_id
            for sample_id in self.sample_names
            if Path(sample_id).name != sample_id or not sample_id
        ]
        if invalid_ids:
            raise ValueError(f"Split contains invalid sample ids: {invalid_ids[:10]}")
        duplicate_ids = sorted(
            sample_id
            for sample_id, count in Counter(self.sample_names).items()
            if count > 1
        )
        if duplicate_ids:
            raise ValueError(
                f"Split contains duplicate sample ids: {duplicate_ids[:10]}"
            )

        self._validate_required_files()

    def _embedding_path(self, modality: str, sample_id: str) -> Path:
        return modal_path(self.root, modality, sample_id, ".npy")

    def _validate_required_files(self) -> None:
        missing_inputs = [
            str(path)
            for sample_id in self.sample_names
            for path in (
                self.root / self.input_dir / sample_id / "protein.pdb",
                modal_path(self.root, self.target_modal, sample_id, ".npz"),
            )
            if not path.is_file()
        ]
        if missing_inputs:
            raise FileNotFoundError(
                f"Missing {len(missing_inputs)} structure/target files under {self.root}; "
                f"examples={missing_inputs[:10]}"
            )
        if self.host_feature_mode == "esm":
            missing_embeddings = [
                f"{modality}/{sample_id}.npy"
                for sample_id in self.sample_names
                for modality in self.embedding_modalities
                if not self._embedding_path(modality, sample_id).is_file()
            ]
            if missing_embeddings:
                raise FileNotFoundError(
                    f"Missing {len(missing_embeddings)} embedding files under {self.root}; "
                    f"examples={missing_embeddings[:10]}"
                )

    def __len__(self) -> int:
        return len(self.sample_names)

    @staticmethod
    def _require_tensor(graph, key: str, sample_id: str) -> Tensor:
        value = getattr(graph, key, None)
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Graph {sample_id} field {key!r} must be a Tensor, got {type(value).__name__}."
            )
        return value

    def _validate_graph(self, graph, sample_id: str) -> int:
        stored_sample_id = getattr(graph, "sample_id", None)
        if stored_sample_id != sample_id:
            raise ValueError(
                f"Structure/sample id mismatch for {sample_id}: "
                f"stored={stored_sample_id!r}, expected={sample_id!r}."
            )
        stored_graph_type = getattr(graph, "graph_type", None)
        if stored_graph_type != self.graph_type:
            raise ValueError(
                f"Graph type mismatch for {sample_id}: stored={stored_graph_type!r}, expected={self.graph_type!r}."
            )

        pos = self._require_tensor(graph, "pos", sample_id)
        if pos.ndim != 2 or pos.size(1) != 3 or pos.size(0) == 0:
            raise ValueError(
                f"Graph {sample_id} pos must have shape [N, 3] with N>0, got {tuple(pos.shape)}."
            )
        if not bool(torch.isfinite(pos).all().item()):
            raise ValueError(f"Graph {sample_id} pos contains non-finite values.")
        num_nodes = int(pos.size(0))

        x = self._require_tensor(graph, "x", sample_id)
        if x.ndim != 2 or tuple(x.shape) != (num_nodes, 0):
            raise ValueError(
                f"Graph {sample_id} must contain structural-only x with shape [{num_nodes}, 0], got {tuple(x.shape)}."
            )
        for key in (
            "z",
            "atom_type_id",
            "residue_type_id",
            "node_type",
            "host_label",
            "host_depth",
        ):
            value = self._require_tensor(graph, key, sample_id)
            if value.ndim != 1 or value.numel() != num_nodes:
                raise ValueError(
                    f"Graph {sample_id} field {key!r} must have shape [{num_nodes}], got {tuple(value.shape)}."
                )
        node_type = graph.node_type
        if not bool((node_type == NodeType.HOST).all().item()):
            raise ValueError(
                f"Structure {sample_id} contains non-host nodes before query sampling."
            )

        edge_index = self._require_tensor(graph, "edge_index", sample_id)
        if (
            edge_index.dtype != torch.long
            or edge_index.ndim != 2
            or edge_index.size(0) != 2
        ):
            raise ValueError(
                f"Graph {sample_id} edge_index must be torch.long with shape [2, E]."
            )
        if edge_index.numel() > 0:
            raise ValueError(
                f"Structure {sample_id} must not contain precomputed edges."
            )

        for key in (
            "target_pos",
            "target_id",
            "target_mask_site_id",
            "target_mask_host_id",
            "ligand_pos",
            "ligand_z",
            "ligand_id",
            "surface_pos",
            "surface_depth",
        ):
            self._require_tensor(graph, key, sample_id)
        if (
            graph.target_pos.ndim != 2
            or graph.target_pos.size(1) != 3
            or graph.target_pos.size(0) == 0
        ):
            raise ValueError(
                f"Graph {sample_id} target_pos must have shape [S, 3] with S>0."
            )
        if (
            graph.ligand_pos.ndim != 2
            or graph.ligand_pos.size(1) != 3
            or graph.ligand_pos.size(0) == 0
        ):
            raise ValueError(
                f"Graph {sample_id} ligand_pos must have shape [L, 3] with L>0."
            )
        if (
            graph.target_mask_site_id.numel() == 0
            or graph.target_mask_host_id.numel() == 0
        ):
            raise ValueError(
                f"Graph {sample_id} contains no positive host/site assignments."
            )
        if graph.target_mask_site_id.numel() != graph.target_mask_host_id.numel():
            raise ValueError(
                f"Graph {sample_id} target mask index lengths do not match."
            )
        if int(graph.target_mask_site_id.max().item()) >= int(graph.target_pos.size(0)):
            raise ValueError(f"Graph {sample_id} target_mask_site_id is out of range.")
        if int(graph.target_mask_host_id.max().item()) >= num_nodes:
            raise ValueError(f"Graph {sample_id} target_mask_host_id is out of range.")

        if self.graph_type == "atom":
            atom_residue_id = self._require_tensor(
                graph, "host_atom_residue_id", sample_id
            )
            atom_source_id = self._require_tensor(
                graph, "host_atom_source_id", sample_id
            )
            if atom_residue_id.ndim != 1 or atom_residue_id.numel() != num_nodes:
                raise ValueError(
                    f"Graph {sample_id} host_atom_residue_id must have shape [{num_nodes}]."
                )
            if atom_source_id.ndim != 1 or atom_source_id.numel() != num_nodes:
                raise ValueError(
                    f"Graph {sample_id} host_atom_source_id must have shape [{num_nodes}]."
                )
            if atom_residue_id.numel() and int(atom_residue_id.min().item()) < 0:
                raise ValueError(
                    f"Graph {sample_id} host_atom_residue_id contains negative indices."
                )
        return num_nodes

    def _load_embeddings(self, graph, sample_id: str, num_nodes: int) -> Tensor:
        parts: list[Tensor] = []
        atom_residue_id = getattr(graph, "host_atom_residue_id", None)
        for modality in self.embedding_modalities:
            path = self._embedding_path(modality, sample_id)
            array = np.load(path, allow_pickle=False)
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError(
                    f"Embedding {path} must have shape [R, D] with R,D>0, got {array.shape}."
                )
            if not np.isfinite(array).all():
                raise ValueError(f"Embedding {path} contains non-finite values.")
            embedding = torch.from_numpy(array).float()
            if self.graph_type == "residue":
                if embedding.size(0) != num_nodes:
                    raise ValueError(
                        f"Residue graph/embedding length mismatch for {sample_id}: "
                        f"graph={num_nodes}, {modality}={embedding.size(0)}."
                    )
                parts.append(embedding)
            else:
                if int(atom_residue_id.max().item()) >= int(embedding.size(0)):
                    raise ValueError(
                        f"Atom graph/embedding mapping mismatch for {sample_id}: "
                        f"max residue index={int(atom_residue_id.max().item())}, {modality} rows={embedding.size(0)}."
                    )
                parts.append(embedding[atom_residue_id])
        features = torch.cat(parts, dim=-1)
        if features.size(1) != self.host_feature_dim:
            raise ValueError(
                f"Host feature width mismatch for {sample_id}: loaded={features.size(1)}, "
                f"configured={self.host_feature_dim}."
            )
        return features

    def __getitem__(self, index: int):
        sample_id = self.sample_names[index]
        graph = build_protein_unified_graph(
            sample_dir=self.root / self.input_dir / sample_id,
            input_dir=self.input_dir,
            target_modal=self.target_modal,
            embedding_modalities=(),
            host_feature_mode="none",
            host_node_mode=self.host_node_mode,
            surface_atom_sasa_filter=self.surface_atom_sasa_filter,
            surface_atom_downsample=self.surface_atom_downsample,
            build_edges=False,
        )
        num_nodes = self._validate_graph(graph, sample_id)
        if self.host_feature_mode == "esm":
            graph.x = self._load_embeddings(graph, sample_id, num_nodes)
        else:
            graph.x = torch.zeros(
                (num_nodes, self.host_feature_dim), dtype=torch.float32
            )
        return graph
