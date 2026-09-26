from __future__ import annotations

from collections import Counter
from pathlib import Path

import lightning as pl
from torch_geometric.loader import DataLoader

from .dataset import UnifiedStructureDataset
from .graph_builder import normalize_host_node_mode


class UnifiedStructureDataModule(pl.LightningDataModule):
    """Load structural inputs for online graph construction in train/val/test."""

    def __init__(
        self,
        root: str,
        dataset_name: str = "unisite",
        split_tag: str = "unisite_0p9",
        embedding_modalities: list[str] | None = None,
        host_feature_mode: str = "esm",
        host_node_mode: str = "residue",
        host_feature_dim: int = 1280,
        num_query_nodes: int = 100,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        test_datasets: list[dict] | None = None,
        input_dir: str = "protein_ligand",
        target_modal: str = "pocket",
        surface_atom_sasa_filter=None,
        surface_atom_downsample=None,
        **unexpected_kwargs,
    ) -> None:
        super().__init__()
        if unexpected_kwargs:
            unexpected = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"Unexpected datamodule argument(s): {unexpected}")
        self.root = Path(root)
        self.input_dir = str(input_dir)
        self.target_modal = str(target_modal)
        self.surface_atom_sasa_filter = surface_atom_sasa_filter
        self.surface_atom_downsample = surface_atom_downsample
        self.dataset_name = str(dataset_name)
        self.split_tag = str(split_tag)
        self.embedding_modalities = (
            ["esm"] if embedding_modalities is None else list(embedding_modalities)
        )
        self.host_feature_mode = str(host_feature_mode).lower()
        if self.host_feature_mode not in {"esm", "zeros"}:
            raise ValueError(
                f"Unsupported host_feature_mode: {self.host_feature_mode!r}."
            )
        self.host_node_mode = normalize_host_node_mode(host_node_mode)
        self.host_feature_dim = int(host_feature_dim)
        if self.host_feature_dim <= 0:
            raise ValueError(
                f"host_feature_dim must be positive, got {self.host_feature_dim}."
            )
        self.num_query_nodes = int(num_query_nodes)
        if self.num_query_nodes <= 0:
            raise ValueError(
                f"num_query_nodes must be positive, got {self.num_query_nodes}."
            )
        if self.host_feature_mode == "esm" and not self.embedding_modalities:
            raise ValueError("host_feature_mode='esm' requires embedding_modalities.")
        if self.host_feature_mode == "zeros" and self.embedding_modalities:
            raise ValueError(
                "host_feature_mode='zeros' requires embedding_modalities to be empty."
            )
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {self.num_workers}."
            )
        self.test_datasets = [dict(config) for config in (test_datasets or [])]
        self._test_dataloader_indices: dict[str, int] = {}

    def _resolve_dataset_cfg(self, overrides: dict | None = None) -> dict:
        config = {
            "root": self.root,
            "input_dir": self.input_dir,
            "target_modal": self.target_modal,
            "surface_atom_sasa_filter": self.surface_atom_sasa_filter,
            "surface_atom_downsample": self.surface_atom_downsample,
            "dataset_name": self.dataset_name,
            "split_tag": self.split_tag,
            "embedding_modalities": list(self.embedding_modalities),
            "host_feature_mode": self.host_feature_mode,
            "host_node_mode": self.host_node_mode,
            "host_feature_dim": self.host_feature_dim,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if overrides:
            allowed = set(config) | {"name", "split_mode"}
            unknown = sorted(set(overrides) - allowed)
            if unknown:
                raise TypeError(f"Unexpected dataset override key(s): {unknown}")
            config.update(overrides)
        config["root"] = Path(config["root"])
        config["dataset_name"] = str(config["dataset_name"])
        config["split_tag"] = str(config.get("split_tag") or config["dataset_name"])
        config["embedding_modalities"] = list(config.get("embedding_modalities") or [])
        config["host_feature_mode"] = str(config["host_feature_mode"]).lower()
        if config["host_feature_mode"] not in {"esm", "zeros"}:
            raise ValueError(
                f"Unsupported host_feature_mode: {config['host_feature_mode']!r}."
            )
        if config["host_feature_mode"] == "esm" and not config["embedding_modalities"]:
            raise ValueError("host_feature_mode='esm' requires embedding_modalities.")
        if config["host_feature_mode"] == "zeros" and config["embedding_modalities"]:
            raise ValueError(
                "host_feature_mode='zeros' requires embedding_modalities to be empty."
            )
        config["host_node_mode"] = normalize_host_node_mode(config["host_node_mode"])
        config["host_feature_dim"] = int(config["host_feature_dim"])
        if config["host_feature_dim"] <= 0:
            raise ValueError(
                f"host_feature_dim must be positive, got {config['host_feature_dim']}."
            )
        config["batch_size"] = int(config["batch_size"])
        config["num_workers"] = int(config["num_workers"])
        if config["batch_size"] <= 0:
            raise ValueError(
                f"batch_size must be positive, got {config['batch_size']}."
            )
        if config["num_workers"] < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {config['num_workers']}."
            )
        config["pin_memory"] = bool(config["pin_memory"])
        return config

    @staticmethod
    def _normalize_split_mode(value: str | None) -> str:
        mode = "test" if value is None else str(value).lower()
        if mode not in {"train", "valid", "test"}:
            raise ValueError(
                f"Unsupported split_mode={value!r}; expected train, valid, or test."
            )
        return mode

    @staticmethod
    def _read_ids(path: Path) -> list[str]:
        if not path.is_file():
            raise FileNotFoundError(f"Split file does not exist: {path}")
        with path.open("r") as handle:
            sample_ids = [line.strip() for line in handle if line.strip()]
        if not sample_ids:
            raise ValueError(f"Split file is empty: {path}")
        duplicates = sorted(
            sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"Split file contains duplicate sample ids: {path}; examples={duplicates[:10]}"
            )
        return sample_ids

    def _dataset(self, mode: str, cfg: dict | None = None) -> UnifiedStructureDataset:
        resolved = self._resolve_dataset_cfg(cfg)
        dataset_root = resolved["root"] / resolved["dataset_name"]
        split_path = dataset_root / "splits" / f"{mode}_ids_{resolved['split_tag']}"
        return UnifiedStructureDataset(
            root=dataset_root,
            sample_names=self._read_ids(split_path),
            host_node_mode=resolved["host_node_mode"],
            host_feature_mode=resolved["host_feature_mode"],
            host_feature_dim=resolved["host_feature_dim"],
            embedding_modalities=resolved["embedding_modalities"],
            input_dir=resolved["input_dir"],
            target_modal=resolved["target_modal"],
            surface_atom_sasa_filter=resolved["surface_atom_sasa_filter"],
            surface_atom_downsample=resolved["surface_atom_downsample"],
        )

    def _dataloader(
        self, mode: str, cfg: dict | None = None, shuffle: bool = False
    ) -> DataLoader:
        resolved = self._resolve_dataset_cfg(cfg)
        return DataLoader(
            self._dataset(mode, cfg=resolved),
            batch_size=resolved["batch_size"],
            shuffle=shuffle,
            num_workers=resolved["num_workers"],
            pin_memory=resolved["pin_memory"],
            follow_batch=[
                "target_pos",
                "ligand_pos",
                "target_mask_host_id",
                "surface_pos",
                "protein_atom_pos",
            ],
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("valid", shuffle=False)

    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return dict(self._test_dataloader_indices)

    def test_dataloader(self) -> list[DataLoader]:
        if not self.test_datasets:
            self._test_dataloader_indices = {}
            return []
        loaders: list[DataLoader] = []
        indices: dict[str, int] = {}
        for index, config in enumerate(self.test_datasets):
            resolved = self._resolve_dataset_cfg(config)
            name = str(config.get("name") or resolved["dataset_name"])
            mode = self._normalize_split_mode(config.get("split_mode"))
            if name in indices:
                raise ValueError(f"Duplicate test dataset name: {name!r}.")
            indices[name] = index
            loaders.append(self._dataloader(mode, cfg=resolved, shuffle=False))
        self._test_dataloader_indices = indices
        return loaders
