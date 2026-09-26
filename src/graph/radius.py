"""Online radius edges in SurfQNet's [receiver, sender] convention."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch_cluster import radius, radius_graph


def _validate_positions(
    pos: Tensor, batch: Tensor, cutoff: float, max_neighbors: int
) -> None:
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"cutoff must be finite and positive, got {cutoff}.")
    if (
        isinstance(max_neighbors, bool)
        or not isinstance(max_neighbors, int)
        or max_neighbors <= 0
    ):
        raise ValueError(
            f"max_neighbors must be a positive integer, got {max_neighbors}."
        )
    if pos.ndim != 2 or pos.size(1) != 3 or not pos.is_floating_point():
        raise ValueError(
            f"Expected floating-point positions [N, 3], got {pos.shape}/{pos.dtype}."
        )
    if not bool(torch.isfinite(pos).all()):
        raise ValueError("Online graph positions contain non-finite values.")
    if (
        batch.shape != (pos.size(0),)
        or batch.dtype != torch.long
        or batch.device != pos.device
    ):
        raise ValueError(
            "Online graph requires a long batch vector matching positions on the same device."
        )
    if batch.numel() and (
        bool((batch < 0).any()) or bool((batch[1:] < batch[:-1]).any())
    ):
        raise ValueError("Online graph batch IDs must be non-negative and sorted.")


@torch.no_grad()
def host_edges(
    pos: Tensor, batch: Tensor, *, cutoff: float, max_neighbors: int
) -> Tensor:
    _validate_positions(pos, batch, cutoff, max_neighbors)
    if pos.size(0) == 0:
        raise ValueError("Cannot construct a protein graph without host nodes.")
    return radius_graph(
        pos.float(),
        r=cutoff,
        batch=batch,
        batch_size=int(batch[-1]) + 1,
        loop=False,
        max_num_neighbors=max_neighbors,
        flow="target_to_source",
        num_workers=1,
    )


@torch.no_grad()
def host_query_edges(
    host_pos: Tensor,
    query_pos: Tensor,
    host_batch: Tensor,
    query_batch: Tensor,
    *,
    cutoff: float,
    max_neighbors: int,
) -> Tensor:
    _validate_positions(host_pos, host_batch, cutoff, max_neighbors)
    _validate_positions(query_pos, query_batch, cutoff, max_neighbors)
    if host_pos.device != query_pos.device:
        raise ValueError("Host and query positions must share a device.")
    if host_pos.size(0) == 0:
        raise ValueError("Cannot construct a protein graph without host nodes.")
    if not bool(torch.isin(query_batch, host_batch.unique()).all()):
        raise ValueError("Query batch contains a sample without host nodes.")
    # Search only host candidates, so query senders cannot consume the neighbor budget.
    edges = radius(
        x=host_pos.float(),
        y=query_pos.float(),
        r=cutoff,
        batch_x=host_batch,
        batch_y=query_batch,
        batch_size=int(host_batch[-1]) + 1,
        max_num_neighbors=max_neighbors,
        num_workers=1,
    )
    edges[0] += host_pos.size(0)
    return edges
