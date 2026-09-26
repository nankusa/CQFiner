from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.norm import LayerNorm as PyGLayerNorm

from src.model.layers import Dense
from .query_host_attn import (
    QUERY_TO_HOST_EDGE_TYPE,
    QueryToHostAttention,
    aggregate_with_query_to_host_as_one,
    normalize_query_to_host_aggregation,
)


def unsorted_segment_sum(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    out_shape = (num_segments,) + tuple(data.shape[1:])
    result = data.new_zeros(out_shape)
    index = segment_ids.view(-1, *([1] * (data.dim() - 1))).expand_as(data)
    result.scatter_add_(0, index, data)
    return result


def unsorted_segment_mean(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    out_shape = (num_segments,) + tuple(data.shape[1:])
    result = data.new_zeros(out_shape)
    count = data.new_zeros(out_shape)
    index = segment_ids.view(-1, *([1] * (data.dim() - 1))).expand_as(data)
    result.scatter_add_(0, index, data)
    count.scatter_add_(0, index, torch.ones_like(data))
    return result / count.clamp(min=1)


class CoorsNorm(nn.Module):
    def __init__(self, eps: float = 1e-8, scale_init: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.full((1,), float(scale_init)))

    def forward(self, coords: Tensor) -> Tensor:
        norm = coords.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return coords / norm * self.scale


def _segment_aggr(
    data: Tensor, segment_ids: Tensor, num_segments: int, aggr: str
) -> Tensor:
    if aggr == "sum":
        return unsorted_segment_sum(data, segment_ids, num_segments)
    if aggr == "mean":
        return unsorted_segment_mean(data, segment_ids, num_segments)
    raise ValueError(f"Unsupported aggregation: {aggr}")


class PairNorm(nn.Module):
    def __init__(self, scale: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = float(scale)
        self.eps = float(eps)
        if self.scale <= 0.0:
            raise ValueError(f"PairNorm scale must be positive, got {scale}.")

    def _write_normalized(
        self, out: Tensor, x: Tensor, mask: Tensor, label: str
    ) -> None:
        group_x = x[mask]
        if group_x.size(0) <= 1:
            raise ValueError(
                f"PairNorm requires at least two nodes per group, got {group_x.size(0)} for {label}."
            )
        centered = group_x - group_x.mean(dim=0, keepdim=True)
        scale = torch.sqrt(centered.pow(2).sum(dim=-1).mean())
        scale_value = float(scale.detach().cpu().item())
        if not bool(torch.isfinite(scale).item()) or scale_value <= self.eps:
            raise FloatingPointError(
                f"Degenerate PairNorm scale for {label}: {scale_value}."
            )
        out[mask] = centered * (self.scale / scale)

    def forward(self, x: Tensor, batch: Tensor, group: Tensor | None = None) -> Tensor:
        if batch.dim() != 1 or batch.size(0) != x.size(0):
            raise ValueError(
                f"PairNorm batch shape mismatch: batch={tuple(batch.shape)}, x={tuple(x.shape)}."
            )
        if group is not None and (group.dim() != 1 or group.size(0) != x.size(0)):
            raise ValueError(
                f"PairNorm group shape mismatch: group={tuple(group.shape)}, x={tuple(x.shape)}."
            )
        if batch.numel() == 0:
            raise ValueError("PairNorm requires at least one node.")

        out = torch.empty_like(x)
        for sample_idx in batch.unique(sorted=True):
            sample_label = int(sample_idx.detach().cpu().item())
            sample_mask = batch == sample_idx
            if group is None:
                self._write_normalized(
                    out, x, sample_mask, label=f"graph {sample_label}"
                )
                continue

            sample_group = group[sample_mask]
            for group_idx in sample_group.unique(sorted=True):
                group_label = int(group_idx.detach().cpu().item())
                group_mask = sample_mask & (group == group_idx)
                self._write_normalized(
                    out,
                    x,
                    group_mask,
                    label=f"graph {sample_label}, group {group_label}",
                )
        return out


class EGNNLayer(nn.Module):
    """Adapted from repo/vnegnn EGNN with the same edge / node / coord update structure."""

    def __init__(
        self,
        hidden_dim: int,
        edge_input_dim: int = 0,
        activation: Optional[Callable] = None,
        residual: bool = True,
        attention: bool = False,
        normalize: bool | None = None,
        node_aggr: str = "mean",
        coords_agg: str = "mean",
        dropout: float = 0.1,
        norm_feats: bool = False,
        norm_coords: bool | None = None,
        norm_coors_scale_init: float = 1e-2,
        tanh: bool = False,
        initialization_gain: float = 1.0,
        degree_norm: str = "none",
        degree_norm_scope: str = "global",
        query_to_host_aggregation: str = "sum",
        query_to_host_coord_update: bool = False,
    ) -> None:
        super().__init__()
        act = activation if activation is not None else nn.SiLU()
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.attention = attention
        self.node_aggr = str(node_aggr).lower()
        self.coords_agg = coords_agg
        self.degree_norm = str(degree_norm).lower()
        if self.degree_norm not in {"none", "symmetric"}:
            raise ValueError(f"Unsupported EGNN degree_norm: {degree_norm}")
        self.degree_norm_scope = str(degree_norm_scope).lower()
        if self.degree_norm_scope not in {"global", "relation"}:
            raise ValueError(f"Unsupported EGNN degree_norm_scope: {degree_norm_scope}")
        if self.degree_norm == "none" and self.degree_norm_scope != "global":
            raise ValueError(
                "EGNN degree_norm_scope must be 'global' when degree_norm='none'."
            )
        self.query_to_host_aggregation = normalize_query_to_host_aggregation(
            query_to_host_aggregation
        )
        self.query_to_host_coord_update = bool(query_to_host_coord_update)
        self.query_to_host_attention = None
        if self.query_to_host_aggregation == "attn":
            if self.degree_norm != "none":
                raise ValueError(
                    "EGNN query-to-host attention aggregation does not support degree_norm."
                )
            self.query_to_host_attention = QueryToHostAttention(hidden_dim)
        self.tanh = tanh
        self.epsilon = 1e-8
        self.initialization_gain = float(initialization_gain)
        if norm_coords is None:
            norm_coords = True if normalize is None else bool(normalize)
        self.node_norm = PyGLayerNorm(hidden_dim) if norm_feats else nn.Identity()
        self.coord_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init)
            if norm_coords
            else nn.Identity()
        )

        input_edge_dim = 2 * hidden_dim + 1 + edge_input_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.Dropout(dropout),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )

        coord_layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            act,
            nn.Linear(hidden_dim, 1),
        ]
        if tanh:
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)

        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        else:
            self.att_mlp = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=self.initialization_gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _coord2radial(self, edge_index: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = coord_diff.norm(dim=-1, keepdim=True)
        return radial, self.coord_norm(coord_diff)

    def _edge_model(
        self, source: Tensor, target: Tensor, radial: Tensor, edge_attr: Tensor | None
    ) -> Tensor:
        source = self.node_norm(source)
        target = self.node_norm(target)
        if edge_attr is None:
            edge_input = torch.cat([source, target, radial], dim=-1)
        else:
            edge_input = torch.cat([source, target, radial, edge_attr], dim=-1)
        edge_feat = self.edge_mlp(edge_input)
        if self.att_mlp is not None:
            edge_feat = edge_feat * self.att_mlp(edge_feat)
        return edge_feat

    @staticmethod
    def _validate_node_type(node_type: Tensor, num_nodes: int, label: str) -> None:
        if node_type.dim() != 1 or node_type.size(0) != num_nodes:
            raise ValueError(
                f"{label} node_type shape mismatch: node_type={tuple(node_type.shape)}, num_nodes={num_nodes}."
            )
        if node_type.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
            torch.uint8,
        }:
            raise TypeError(
                f"{label} node_type must be an integer tensor, got {node_type.dtype}."
            )
        if bool((node_type < 0).any().item()):
            raise ValueError(f"{label} node_type must be non-negative.")

    def _global_degree_norm_weight(
        self, edge_index: Tensor, num_nodes: int, aggr: str
    ) -> Tensor:
        row, col = edge_index
        recv_deg = (
            torch.bincount(row, minlength=num_nodes)
            .to(device=row.device, dtype=torch.float32)
            .clamp_min(1.0)
        )
        send_deg = (
            torch.bincount(col, minlength=num_nodes)
            .to(device=col.device, dtype=torch.float32)
            .clamp_min(1.0)
        )
        weight = torch.rsqrt(recv_deg[row] * send_deg[col])
        if aggr == "mean":
            weight = weight * recv_deg[row]
        return weight.unsqueeze(-1)

    def _relation_degree_norm_weight(
        self, edge_index: Tensor, num_nodes: int, aggr: str, node_type: Tensor
    ) -> Tensor:
        self._validate_node_type(node_type, num_nodes, "EGNN relation degree norm")
        row, col = edge_index
        row_type = node_type[row]
        col_type = node_type[col]
        global_recv_deg = (
            torch.bincount(row, minlength=num_nodes)
            .to(device=row.device, dtype=torch.float32)
            .clamp_min(1.0)
        )
        recv_deg_for_edge = torch.empty(
            row.size(0), device=row.device, dtype=torch.float32
        )
        send_deg_for_edge = torch.empty(
            row.size(0), device=row.device, dtype=torch.float32
        )

        relation_pairs = torch.unique(
            torch.stack([row_type, col_type], dim=1), dim=0, sorted=True
        )
        for pair in relation_pairs:
            relation_mask = (row_type == pair[0]) & (col_type == pair[1])
            relation_row = row[relation_mask]
            relation_col = col[relation_mask]
            relation_recv_deg = (
                torch.bincount(relation_row, minlength=num_nodes)
                .to(device=row.device, dtype=torch.float32)
                .clamp_min(1.0)
            )
            relation_send_deg = (
                torch.bincount(relation_col, minlength=num_nodes)
                .to(device=row.device, dtype=torch.float32)
                .clamp_min(1.0)
            )
            recv_deg_for_edge[relation_mask] = relation_recv_deg[relation_row]
            send_deg_for_edge[relation_mask] = relation_send_deg[relation_col]

        weight = torch.rsqrt(recv_deg_for_edge * send_deg_for_edge)
        if aggr == "mean":
            weight = weight * global_recv_deg[row]
        return weight.unsqueeze(-1)

    def _degree_norm_weight(
        self,
        edge_index: Tensor,
        num_nodes: int,
        aggr: str,
        node_type: Tensor | None = None,
    ) -> Tensor | None:
        if self.degree_norm == "none":
            return None
        if self.degree_norm != "symmetric":
            raise ValueError(f"Unsupported EGNN degree_norm: {self.degree_norm}")
        aggr = str(aggr).lower()
        if aggr not in {"sum", "mean"}:
            raise ValueError(
                f"degree_norm='symmetric' supports sum/mean aggregation, got {aggr}."
            )

        if self.degree_norm_scope == "global":
            return self._global_degree_norm_weight(edge_index, num_nodes, aggr)
        if self.degree_norm_scope == "relation":
            if node_type is None:
                raise ValueError(
                    "node_type is required when EGNN degree_norm_scope='relation'."
                )
            return self._relation_degree_norm_weight(
                edge_index, num_nodes, aggr, node_type=node_type
            )
        raise ValueError(
            f"Unsupported EGNN degree_norm_scope: {self.degree_norm_scope}"
        )

    def _coord_model(
        self,
        pos: Tensor,
        edge_index: Tensor,
        coord_diff: Tensor,
        edge_feat: Tensor,
        edge_weight: Tensor | None = None,
        edge_type: Tensor | None = None,
        query_to_host_weight: Tensor | None = None,
    ) -> Tensor:
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if edge_weight is not None:
            trans = trans * edge_weight.to(device=trans.device, dtype=trans.dtype)
        active_mask = None
        if query_to_host_weight is not None:
            if edge_type is None:
                raise ValueError(
                    "EGNN query-to-host coordinate aggregation requires edge_type."
                )
            if not self.query_to_host_coord_update:
                active_mask = edge_type != QUERY_TO_HOST_EDGE_TYPE
            agg = aggregate_with_query_to_host_as_one(
                trans,
                row,
                dim_size=pos.size(0),
                aggr=self.coords_agg,
                edge_type=edge_type,
                edge_weight=query_to_host_weight,
                active_mask=active_mask,
            )
        else:
            agg = _segment_aggr(trans, row, pos.size(0), self.coords_agg)
        return pos + agg

    def _node_model(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_feat: Tensor,
        edge_weight: Tensor | None = None,
        edge_type: Tensor | None = None,
        query_to_host_weight: Tensor | None = None,
    ) -> Tensor:
        row, _ = edge_index
        if edge_weight is not None:
            edge_feat = edge_feat * edge_weight.to(
                device=edge_feat.device, dtype=edge_feat.dtype
            )
        if query_to_host_weight is not None:
            if edge_type is None:
                raise ValueError(
                    "EGNN query-to-host node aggregation requires edge_type."
                )
            agg = aggregate_with_query_to_host_as_one(
                edge_feat,
                row,
                dim_size=x.size(0),
                aggr=self.node_aggr,
                edge_type=edge_type,
                edge_weight=query_to_host_weight,
            )
        else:
            agg = _segment_aggr(edge_feat, row, x.size(0), self.node_aggr)
        node_input = torch.cat([x, agg], dim=-1)
        out = self.node_mlp(node_input)
        if self.residual:
            out = x + out
        return out

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        node_type: Tensor | None = None,
        edge_type: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if edge_index.numel() == 0:
            return x, pos
        row, col = edge_index
        query_to_host_weight = None
        if self.query_to_host_attention is not None:
            if edge_type is None:
                raise ValueError(
                    "EGNN query-to-host attention aggregation requires edge_type."
                )
            if edge_type.shape != (edge_index.size(1),):
                raise ValueError(
                    "EGNN edge_type shape mismatch: "
                    f"expected {(edge_index.size(1),)}, got {tuple(edge_type.shape)}."
                )
        radial, coord_diff = self._coord2radial(edge_index, pos)
        edge_feat = self._edge_model(x[row], x[col], radial, edge_attr)
        if self.query_to_host_attention is not None:
            query_to_host_weight = self.query_to_host_attention.edge_weights(
                edge_attr=edge_feat,
                edge_index=edge_index,
                edge_type=edge_type,
            )
        node_edge_weight = self._degree_norm_weight(
            edge_index, x.size(0), self.node_aggr, node_type=node_type
        )
        coord_edge_weight = self._degree_norm_weight(
            edge_index, x.size(0), self.coords_agg, node_type=node_type
        )
        next_pos = self._coord_model(
            pos,
            edge_index,
            coord_diff,
            edge_feat,
            edge_weight=coord_edge_weight,
            edge_type=edge_type,
            query_to_host_weight=query_to_host_weight,
        )
        next_x = self._node_model(
            x,
            edge_index,
            edge_feat,
            edge_weight=node_edge_weight,
            edge_type=edge_type,
            query_to_host_weight=query_to_host_weight,
        )
        return next_x, next_pos


class EGNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        activation: Optional[Callable] = None,
        edge_input_dim: int = 0,
        use_time_conditioning: bool = False,
        residual: bool = True,
        attention: bool = False,
        normalize: bool | None = None,
        node_aggr: str = "mean",
        coords_agg: str = "mean",
        dropout: float = 0.1,
        norm_feats: bool = False,
        norm_coords: bool | None = None,
        norm_coors_scale_init: float = 1e-2,
        tanh: bool = False,
        initialization_gain: float = 1.0,
        degree_norm: str = "none",
        degree_norm_scope: str = "global",
        pairnorm: str = "none",
        pairnorm_scale: float = 1.0,
        pairnorm_scope: str = "all",
        query_to_host_aggregation: str = "sum",
        query_to_host_coord_update: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_time_conditioning = use_time_conditioning
        self.query_to_host_aggregation = normalize_query_to_host_aggregation(
            query_to_host_aggregation
        )
        self.query_to_host_coord_update = bool(query_to_host_coord_update)
        self.degree_norm_scope = str(degree_norm_scope).lower()
        if self.degree_norm_scope not in {"global", "relation"}:
            raise ValueError(f"Unsupported EGNN degree_norm_scope: {degree_norm_scope}")
        self.pairnorm_mode = str(pairnorm).lower()
        if self.pairnorm_mode in {"", "none", "off", "false"}:
            self.pairnorm_mode = "none"
        elif self.pairnorm_mode in {"pn", "pairnorm"}:
            self.pairnorm_mode = "pairnorm"
        else:
            raise ValueError(f"Unsupported EGNN pairnorm: {pairnorm}")
        self.pairnorm_scope = str(pairnorm_scope).lower()
        if self.pairnorm_scope not in {"all", "node_type"}:
            raise ValueError(f"Unsupported EGNN pairnorm_scope: {pairnorm_scope}")
        if self.pairnorm_mode == "none" and self.pairnorm_scope != "all":
            raise ValueError("EGNN pairnorm_scope must be 'all' when pairnorm='none'.")
        self.pairnorm = (
            PairNorm(scale=pairnorm_scale) if self.pairnorm_mode == "pairnorm" else None
        )

        act = activation if activation is not None else nn.SiLU()
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_input_dim=edge_input_dim,
                    activation=act,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    node_aggr=node_aggr,
                    coords_agg=coords_agg,
                    dropout=dropout,
                    norm_feats=norm_feats,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    tanh=tanh,
                    initialization_gain=initialization_gain,
                    degree_norm=degree_norm,
                    degree_norm_scope=self.degree_norm_scope,
                    query_to_host_aggregation=self.query_to_host_aggregation,
                    query_to_host_coord_update=self.query_to_host_coord_update,
                )
                for _ in range(num_layers)
            ]
        )
        self.time_mlp = None
        if use_time_conditioning:
            self.time_mlp = nn.Sequential(
                Dense(hidden_dim + 1, hidden_dim, activation=nn.SiLU()),
                Dense(hidden_dim, hidden_dim),
            )

    def _forward_impl(
        self,
        pos: Tensor,
        edge_index: Tensor,
        q: Tensor,
        t: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        node_type: Tensor | None = None,
        edge_type: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        if edge_index.numel() == 0 or q.numel() == 0:
            intermediates = (
                [(q, pos) for _ in self.layers] if return_intermediate else []
            )
            return q, pos, intermediates
        requires_node_type = (
            self.pairnorm is not None and self.pairnorm_scope == "node_type"
        ) or any(
            layer.degree_norm != "none" and layer.degree_norm_scope == "relation"
            for layer in self.layers
        )
        if requires_node_type:
            if node_type is None:
                raise ValueError(
                    "node_type is required for EGNN node-type-aware normalization."
                )
            EGNNLayer._validate_node_type(node_type, q.size(0), "EGNNEncoder")
        elif node_type is not None:
            EGNNLayer._validate_node_type(node_type, q.size(0), "EGNNEncoder")
        if self.pairnorm is not None:
            if batch is None:
                raise ValueError("batch is required when EGNN pairnorm is enabled.")
            if batch.dim() != 1 or batch.size(0) != q.size(0):
                raise ValueError(
                    f"Expected batch to have {q.size(0)} rows, got {tuple(batch.shape)}."
                )

        if self.time_mlp is not None:
            if t is None:
                raise ValueError("t is required when use_time_conditioning=True")
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            if t.size(0) != q.size(0):
                raise ValueError(
                    f"Expected t to have {q.size(0)} rows, got {t.size(0)}"
                )
            t = t.to(device=q.device, dtype=q.dtype)

        current_q = q
        current_pos = pos
        intermediates: list[tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            layer_q = current_q
            if self.time_mlp is not None:
                layer_q = self.time_mlp(torch.cat([layer_q, t], dim=-1))
            current_q, current_pos = layer(
                x=layer_q,
                pos=current_pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_type=node_type,
                edge_type=edge_type,
            )
            if self.pairnorm is not None:
                pairnorm_group = (
                    node_type if self.pairnorm_scope == "node_type" else None
                )
                current_q = self.pairnorm(current_q, batch, group=pairnorm_group)
            if return_intermediate:
                intermediates.append((current_q, current_pos))
        return current_q, current_pos, intermediates

    def forward(
        self,
        pos: Tensor,
        edge_index: Tensor,
        q: Tensor,
        t: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        node_type: Tensor | None = None,
        edge_type: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        current_q, current_pos, _ = self._forward_impl(
            pos=pos,
            edge_index=edge_index,
            q=q,
            t=t,
            edge_attr=edge_attr,
            batch=batch,
            node_type=node_type,
            edge_type=edge_type,
            return_intermediate=False,
        )
        return current_q, current_pos

    def forward_with_intermediates(
        self,
        pos: Tensor,
        edge_index: Tensor,
        q: Tensor,
        t: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        node_type: Tensor | None = None,
        edge_type: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        return self._forward_impl(
            pos=pos,
            edge_index=edge_index,
            q=q,
            t=t,
            edge_attr=edge_attr,
            batch=batch,
            node_type=node_type,
            edge_type=edge_type,
            return_intermediate=True,
        )
