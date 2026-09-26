from .dataset import UnifiedStructureDataset
from .unified import KEYS, NodeType

__all__ = ["UnifiedStructureDataModule", "UnifiedStructureDataset", "KEYS", "NodeType"]


def __getattr__(name: str):
    if name == "UnifiedStructureDataModule":
        from .datamodule import UnifiedStructureDataModule

        return UnifiedStructureDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
