from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data


@dataclass
class NodeType:
    HOST: int = 0
    QUERY: int = 1


@dataclass
class UnifiedKeys:
    z: str = "z"
    x: str = "x"
    atom_type_id: str = "atom_type_id"
    residue_type_id: str = "residue_type_id"
    pos: str = "pos"
    edge_index: str = "edge_index"
    node_type: str = "node_type"
    base_feat_dim: str = "base_feat_dim"
    host_mask: str = "host_mask"
    query_mask: str = "query_mask"
    host_label: str = "host_label"
    host_depth: str = "host_depth"
    target_pos: str = "target_pos"
    target_batch: str = "target_batch"
    target_id: str = "target_id"
    target_mask_site_id: str = "target_mask_site_id"
    target_mask_host_id: str = "target_mask_host_id"
    ligand_pos: str = "ligand_pos"
    ligand_z: str = "ligand_z"
    ligand_id: str = "ligand_id"
    ligand_batch: str = "ligand_batch"
    surface_pos: str = "surface_pos"
    surface_depth: str = "surface_depth"
    protein_atom_pos: str = "protein_atom_pos"
    host_atom_residue_id: str = "host_atom_residue_id"
    host_atom_source_id: str = "host_atom_source_id"
    sample_id: str = "sample_id"


KEYS = UnifiedKeys()


def make_unified_data(
    z: Tensor,
    x: Tensor,
    atom_type_id: Tensor,
    residue_type_id: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    node_type: Tensor,
    base_feat_dim: int,
    host_label: Tensor,
    host_depth: Tensor,
    target_pos: Tensor,
    target_id: Tensor,
    target_mask_site_id: Tensor,
    target_mask_host_id: Tensor,
    ligand_pos: Tensor,
    ligand_z: Tensor,
    ligand_id: Tensor,
    sample_id: str,
    surface_pos: Tensor | None = None,
    surface_depth: Tensor | None = None,
    host_atom_residue_id: Tensor | None = None,
    host_atom_source_id: Tensor | None = None,
    protein_atom_pos: Tensor | None = None,
) -> Data:
    data = Data()
    data[KEYS.z] = z.long()
    data[KEYS.x] = x.float()
    data[KEYS.atom_type_id] = atom_type_id.long()
    data[KEYS.residue_type_id] = residue_type_id.long()
    data[KEYS.pos] = pos.float()
    data[KEYS.edge_index] = edge_index.long()
    data[KEYS.node_type] = node_type.long()
    data[KEYS.base_feat_dim] = torch.tensor([base_feat_dim], dtype=torch.long)
    data[KEYS.host_mask] = node_type == NodeType.HOST
    data[KEYS.query_mask] = node_type == NodeType.QUERY

    # Host labels only; query nodes use 0 placeholder.
    labels = torch.zeros(x.size(0), dtype=torch.float32)
    labels[data[KEYS.host_mask]] = host_label.float()
    data[KEYS.host_label] = labels

    depths = torch.zeros(x.size(0), dtype=torch.float32)
    depths[data[KEYS.host_mask]] = host_depth.float()
    data[KEYS.host_depth] = depths

    data[KEYS.target_pos] = target_pos.float()
    data[KEYS.target_batch] = torch.zeros(target_pos.size(0), dtype=torch.long)
    data[KEYS.target_id] = target_id.long()
    data[KEYS.target_mask_site_id] = target_mask_site_id.long()
    data[KEYS.target_mask_host_id] = target_mask_host_id.long()

    data[KEYS.ligand_pos] = ligand_pos.float()
    data[KEYS.ligand_z] = ligand_z.long()
    data[KEYS.ligand_id] = ligand_id.long()
    data[KEYS.ligand_batch] = torch.zeros(ligand_pos.size(0), dtype=torch.long)
    if surface_pos is not None:
        data[KEYS.surface_pos] = surface_pos.float()
        if surface_depth is None:
            surface_depth = torch.ones(surface_pos.size(0), dtype=torch.float32)
        data[KEYS.surface_depth] = surface_depth.float()
    if host_atom_residue_id is not None:
        data[KEYS.host_atom_residue_id] = host_atom_residue_id.long()
    if host_atom_source_id is not None:
        data[KEYS.host_atom_source_id] = host_atom_source_id.long()
    if protein_atom_pos is not None:
        data[KEYS.protein_atom_pos] = protein_atom_pos.float()
    data[KEYS.sample_id] = sample_id
    return data
