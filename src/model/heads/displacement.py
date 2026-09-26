from __future__ import annotations


import torch
import torch.nn as nn
from torch import Tensor

from src.model.layers import Dense, GaussianRBF, scatter_add
from src.model.gnn.visnet import ExpNormalSmearing


class EGNNDisplacementHead(nn.Module):
    """Predicts equivariant displacements by weighting relative edge directions."""

    def __init__(
        self,
        hidden_dim: int,
        cutoff: float,
        n_rbf: int,
        basis_type: str = "gaussian",
    ) -> None:
        super().__init__()
        self.basis_type = basis_type
        if basis_type == "gaussian":
            self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        elif basis_type == "expnorm":
            self.rbf = ExpNormalSmearing(cutoff=cutoff, num_rbf=n_rbf, trainable=False)
        else:
            raise ValueError(f"Unsupported basis_type: {basis_type}")
        self.edge_mlp = nn.Sequential(
            Dense(3 * hidden_dim + n_rbf, hidden_dim, activation=nn.SiLU()),
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()),
            Dense(hidden_dim, 1),
        )

    def forward(
        self,
        q: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_gate: Tensor | None = None,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return pos.new_zeros(pos.size(0), 3)

        src, dst = edge_index[0], edge_index[1]
        rel = pos[dst] - pos[src]
        dist = torch.norm(rel, dim=-1).clamp(min=1e-12)
        direction = rel / dist.unsqueeze(-1)
        rbf = self.rbf(dist.unsqueeze(-1))
        if rbf.dim() > 2:
            rbf = rbf.view(rbf.size(0), -1)

        edge_feat = torch.cat(
            [
                q[src],
                q[dst],
                q[src] - q[dst],
                rbf,
            ],
            dim=-1,
        )
        weights = self.edge_mlp(edge_feat)
        if edge_gate is not None:
            weights = weights * edge_gate
        disp = weights * direction
        return scatter_add(disp, src, q.size(0))


class ViSNetDisplacementHead(nn.Module):
    """Uses ViSNet's learned edge features directly for EGNN-style displacement aggregation."""

    def __init__(
        self,
        hidden_dim: int,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        aggregation = str(aggregation).lower()
        if aggregation not in {"sum", "mean"}:
            raise ValueError(
                f"Unsupported ViSNet displacement aggregation: {aggregation!r}"
            )
        self.aggregation = aggregation
        self.edge_mlp = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()),
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()),
            Dense(hidden_dim, 1),
        )

    def forward(
        self,
        edge_attr: Tensor,
        edge_vec: Tensor,
        edge_index: Tensor,
        num_nodes: int,
        edge_gate: Tensor | None = None,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return edge_vec.new_zeros((num_nodes, 3))

        src = edge_index[0]
        dist = torch.norm(edge_vec, dim=-1).clamp(min=1e-12)
        direction = edge_vec / dist.unsqueeze(-1)

        weights = self.edge_mlp(edge_attr)
        if edge_gate is not None:
            weights = weights * edge_gate
        disp = weights * direction
        out = scatter_add(disp, src, num_nodes)
        if self.aggregation == "sum":
            return out
        if self.aggregation == "mean":
            counts = torch.bincount(src, minlength=num_nodes).to(
                device=out.device, dtype=out.dtype
            )
            return out / counts.clamp(min=1.0).unsqueeze(-1)
        raise ValueError(
            f"Unsupported ViSNet displacement aggregation: {self.aggregation!r}"
        )
