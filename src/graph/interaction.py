"""Protein-query connectivity shared by the ViSNet and EGNN stages."""

import torch
from torch import Tensor

from .radius import _validate_positions, host_query_edges


@torch.no_grad()
def interaction_edges(
    host_pos: Tensor,
    query_pos: Tensor,
    host_batch: Tensor,
    query_batch: Tensor,
    *,
    mode: str,
    cutoff: float,
    max_neighbors: int,
    nearest: bool = False,
    bidirectional: bool = False,
) -> Tensor:
    """Return HQ edges, optionally followed by their exact reversed pairs.

    A local neighbor cap counts protein neighbors per query. The ViSNet stage
    uses native radius ordering; refinement preserves the original nearest-k
    policy. Full connectivity has no degree cap in either stage.
    """
    _validate_positions(host_pos, host_batch, cutoff, max_neighbors)
    _validate_positions(query_pos, query_batch, cutoff, max_neighbors)
    if host_pos.device != query_pos.device:
        raise ValueError("Protein and query coordinates must share a device.")
    if not host_pos.size(0) or not query_pos.size(0):
        raise ValueError("Protein-query graphs require both node sets.")
    if not torch.isin(query_batch, host_batch.unique()).all():
        raise ValueError("A query belongs to a sample without protein nodes.")
    if mode not in {"cutoff", "full"}:
        raise ValueError(f"Unknown interaction mode: {mode}")
    if mode == "cutoff" and not nearest:
        edges = host_query_edges(
            host_pos,
            query_pos,
            host_batch,
            query_batch,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        )
    else:
        chunks = []
        for sample in query_batch.unique(sorted=True).tolist():
            hosts = (host_batch == sample).nonzero(as_tuple=True)[0]
            queries = (query_batch == sample).nonzero(as_tuple=True)[0]
            if mode == "full":
                receivers = queries.repeat_interleave(hosts.numel())
                senders = hosts.repeat(queries.numel())
            else:
                distance = torch.cdist(
                    query_pos[queries].float(), host_pos[hosts].float()
                )
                values, indices = distance.masked_fill(
                    distance > cutoff, float("inf")
                ).topk(
                    k=min(max_neighbors, hosts.numel()),
                    dim=1,
                    largest=False,
                    sorted=False,
                )
                valid = torch.isfinite(values)
                receivers = queries[:, None].expand_as(indices)[valid]
                senders = hosts[indices[valid]]
            chunks.append(torch.stack((receivers + host_pos.size(0), senders)))
        edges = torch.cat(chunks, dim=1)
    if bidirectional:
        edges = torch.cat((edges, edges.flip(0)), dim=1)
    return edges
