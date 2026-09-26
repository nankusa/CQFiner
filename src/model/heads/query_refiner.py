from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

from src.graph.interaction import interaction_edges

from src.model.gnn.egnn import EGNNEncoder


@dataclass
class RefinementState:
    host_scalar: Tensor
    query_scalar: Tensor
    query_pos: Tensor


@dataclass
class RefinementOutput:
    final: RefinementState
    layers: list[RefinementState]


class QueryEGNNRefiner(nn.Module):
    """Fixed-topology EGNN refinement; protein coordinates always remain fixed.

    One-way refinement keeps protein features fixed. In bidirectional mode they
    receive query messages and carry those updates into subsequent layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        graph_mode: str = "full",
        cutoff: float = 8.0,
        max_neighbors: int = 32,
        dropout: float = 0.1,
        query_query_edges: bool = False,
        bidirectional: bool = False,
        identity_init: bool = True,
        node_aggr: str = "mean",
        norm_feats: bool = False,
        norm_coords: bool = True,
        norm_coors_scale_init: float = 1.0e-2,
        initialization_gain: float = 1.0,
        degree_norm: str = "none",
        degree_norm_scope: str = "global",
        pairnorm: str = "none",
        pairnorm_scale: float = 1.0,
        pairnorm_scope: str = "all",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.graph_mode = str(graph_mode).lower()
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.query_query_edges = bool(query_query_edges)
        if self.query_query_edges:
            raise ValueError("SurfQNet refinement does not use query-query edges.")
        self.bidirectional = bool(bidirectional)
        self.identity_init = bool(identity_init)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")
        if self.num_layers <= 0:
            raise ValueError(
                f"QueryEGNNRefiner num_layers must be positive, got {self.num_layers}."
            )
        if self.graph_mode not in {"full", "cutoff"}:
            raise ValueError(
                f"QueryEGNNRefiner graph_mode must be full or cutoff, got {graph_mode!r}."
            )
        if self.graph_mode == "cutoff" and self.cutoff <= 0.0:
            raise ValueError(
                f"QueryEGNNRefiner cutoff must be positive in cutoff mode, got {self.cutoff}."
            )
        if self.max_neighbors <= 0:
            raise ValueError(
                f"QueryEGNNRefiner max_neighbors must be positive, got {self.max_neighbors}."
            )

        self.encoder = EGNNEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_time_conditioning=False,
            node_aggr=node_aggr,
            dropout=dropout,
            norm_feats=norm_feats,
            norm_coords=norm_coords,
            norm_coors_scale_init=norm_coors_scale_init,
            initialization_gain=initialization_gain,
            degree_norm=degree_norm,
            degree_norm_scope=degree_norm_scope,
            pairnorm=pairnorm,
            pairnorm_scale=pairnorm_scale,
            pairnorm_scope=pairnorm_scope,
        )
        if self.identity_init:
            self._init_identity_updates()

    def _init_identity_updates(self) -> None:
        for layer in self.encoder.layers:
            coord_last = layer.coord_mlp[-1]
            if not isinstance(coord_last, nn.Linear):
                raise TypeError(
                    "QueryEGNNRefiner identity initialization expects the last coord_mlp module to be Linear, "
                    f"got {type(coord_last).__name__}."
                )
            nn.init.zeros_(coord_last.weight)
            nn.init.zeros_(coord_last.bias)
            node_last = layer.node_mlp[-1]
            if not isinstance(node_last, nn.Linear):
                raise TypeError(
                    "QueryEGNNRefiner identity initialization expects the last node_mlp module to be Linear, "
                    f"got {type(node_last).__name__}."
                )
            nn.init.zeros_(node_last.weight)
            nn.init.zeros_(node_last.bias)

    def _build_edge_index(self, host_pos, query_pos, host_batch, query_batch):
        return interaction_edges(
            host_pos,
            query_pos,
            host_batch,
            query_batch,
            mode=self.graph_mode,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            nearest=True,
            bidirectional=self.bidirectional,
        )

    def forward(
        self,
        host_scalar: Tensor,
        host_pos: Tensor,
        host_batch: Tensor,
        query_scalar: Tensor,
        query_pos: Tensor,
        query_batch: Tensor,
        return_intermediate: bool = False,
    ) -> RefinementOutput:
        if self.encoder.pairnorm is not None:
            raise ValueError(
                "QueryEGNNRefiner with host context does not support PairNorm."
            )
        if host_scalar.ndim != 2:
            raise ValueError(
                f"QueryEGNNRefiner expects host_scalar [H, D], got {tuple(host_scalar.shape)}."
            )
        if query_scalar.ndim != 2:
            raise ValueError(
                f"QueryEGNNRefiner expects query_scalar [Q, D], got {tuple(query_scalar.shape)}."
            )
        if host_scalar.size(-1) != self.hidden_dim:
            raise ValueError(
                "QueryEGNNRefiner host feature dimension mismatch: "
                f"expected {self.hidden_dim}, got {host_scalar.size(-1)}."
            )
        if query_scalar.size(-1) != self.hidden_dim:
            raise ValueError(
                "QueryEGNNRefiner query feature dimension mismatch: "
                f"expected {self.hidden_dim}, got {query_scalar.size(-1)}."
            )
        if host_pos.shape != (host_scalar.size(0), 3):
            raise ValueError(
                "QueryEGNNRefiner host_pos shape mismatch: "
                f"expected {(host_scalar.size(0), 3)}, got {tuple(host_pos.shape)}."
            )
        if query_pos.shape != (query_scalar.size(0), 3):
            raise ValueError(
                "QueryEGNNRefiner query_pos shape mismatch: "
                f"expected {(query_scalar.size(0), 3)}, got {tuple(query_pos.shape)}."
            )
        if host_batch.shape != (host_scalar.size(0),):
            raise ValueError(
                "QueryEGNNRefiner host_batch shape mismatch: "
                f"expected {(host_scalar.size(0),)}, got {tuple(host_batch.shape)}."
            )
        if query_batch.shape != (query_scalar.size(0),):
            raise ValueError(
                "QueryEGNNRefiner query_batch shape mismatch: "
                f"expected {(query_scalar.size(0),)}, got {tuple(query_batch.shape)}."
            )
        if host_scalar.size(0) == 0:
            raise ValueError("QueryEGNNRefiner received no host nodes.")
        if query_scalar.size(0) == 0:
            raise ValueError("QueryEGNNRefiner received no query nodes.")
        if not torch.isfinite(host_pos).all():
            raise ValueError("QueryEGNNRefiner received non-finite host coordinates.")
        if not torch.isfinite(query_pos).all():
            raise ValueError("QueryEGNNRefiner received non-finite query coordinates.")

        edge_index = self._build_edge_index(
            host_pos=host_pos,
            query_pos=query_pos,
            host_batch=host_batch,
            query_batch=query_batch,
        )
        if edge_index.numel() == 0:
            raise ValueError(
                "QueryEGNNRefiner built no host-to-query or query-to-query edges."
            )

        num_host = int(host_scalar.size(0))
        current_scalar = torch.cat([host_scalar, query_scalar], dim=0)
        current_pos = torch.cat([host_pos, query_pos], dim=0)
        intermediates: list[RefinementState] = []
        for layer in self.encoder.layers:
            current_scalar, current_pos = layer(
                x=current_scalar,
                pos=current_pos,
                edge_index=edge_index,
                edge_attr=None,
                node_type=None,
            )
            if not self.bidirectional:
                current_scalar = torch.cat(
                    [host_scalar, current_scalar[num_host:]], dim=0
                )
            current_pos = torch.cat([host_pos, current_pos[num_host:]], dim=0)
            if return_intermediate:
                intermediates.append(
                    RefinementState(
                        current_scalar[:num_host],
                        current_scalar[num_host:],
                        current_pos[num_host:],
                    )
                )
        refined_scalar = current_scalar[num_host:]
        refined_pos = current_pos[num_host:]
        if not torch.isfinite(refined_pos).all():
            raise FloatingPointError(
                "QueryEGNNRefiner produced non-finite query coordinates."
            )
        for layer_idx, state in enumerate(intermediates, start=1):
            if not torch.isfinite(state.query_pos).all():
                raise FloatingPointError(
                    f"QueryEGNNRefiner layer {layer_idx} produced non-finite query coordinates."
                )
        if return_intermediate:
            if len(intermediates) != self.num_layers:
                raise RuntimeError(
                    "QueryEGNNRefiner intermediate output count mismatch: "
                    f"expected {self.num_layers}, got {len(intermediates)}."
                )
        return RefinementOutput(
            RefinementState(current_scalar[:num_host], refined_scalar, refined_pos),
            intermediates,
        )
