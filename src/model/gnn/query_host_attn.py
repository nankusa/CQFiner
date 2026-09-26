from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import softmax


HOST_HOST_EDGE_TYPE = 0
HOST_TO_QUERY_EDGE_TYPE = 1
QUERY_TO_HOST_EDGE_TYPE = 2
QUERY_TO_QUERY_EDGE_TYPE = 3


def normalize_query_to_host_aggregation(value: str) -> str:
    mode = str(value).lower()
    if mode in {"sum", "add", "none", "off", "false"}:
        return "sum"
    if mode in {"attn", "attention", "softmax"}:
        return "attn"
    raise ValueError(f"Unsupported query_to_host_aggregation: {value!r}.")


def _scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    if src.numel() == 0:
        return out
    return out.index_add(0, index, src)


def _broadcast_edge_weight(edge_weight: Tensor, target: Tensor) -> Tensor:
    weight = edge_weight
    while weight.dim() < target.dim():
        weight = weight.unsqueeze(-1)
    return weight.to(device=target.device, dtype=target.dtype)


def aggregate_with_query_to_host_as_one(
    messages: Tensor,
    receiver: Tensor,
    *,
    dim_size: int,
    aggr: str,
    edge_type: Tensor | None = None,
    edge_weight: Tensor | None = None,
    active_mask: Tensor | None = None,
) -> Tensor:
    if receiver.dim() != 1 or receiver.size(0) != messages.size(0):
        raise ValueError(
            "aggregate_with_query_to_host_as_one receiver shape mismatch: "
            f"receiver={tuple(receiver.shape)}, messages={tuple(messages.shape)}."
        )
    if dim_size <= 0:
        raise ValueError(f"aggregate dim_size must be positive, got {dim_size}.")
    aggr = str(aggr).lower()
    if aggr not in {"sum", "mean"}:
        raise ValueError(f"Unsupported aggregation: {aggr}.")
    if edge_weight is not None and edge_weight.size(0) != messages.size(0):
        raise ValueError(
            "edge_weight length mismatch: "
            f"edge_weight={tuple(edge_weight.shape)}, messages={tuple(messages.shape)}."
        )
    if edge_type is not None and edge_type.shape != receiver.shape:
        raise ValueError(
            f"edge_type shape mismatch: expected {tuple(receiver.shape)}, got {tuple(edge_type.shape)}."
        )
    if active_mask is not None:
        if active_mask.shape != receiver.shape:
            raise ValueError(
                f"active_mask shape mismatch: expected {tuple(receiver.shape)}, got {tuple(active_mask.shape)}."
            )
        receiver = receiver[active_mask]
        messages = messages[active_mask]
        edge_type = None if edge_type is None else edge_type[active_mask]
        edge_weight = None if edge_weight is None else edge_weight[active_mask]

    if messages.size(0) == 0:
        return messages.new_zeros((dim_size,) + tuple(messages.shape[1:]))

    weighted_messages = messages
    if edge_weight is not None:
        weighted_messages = messages * _broadcast_edge_weight(edge_weight, messages)
    summed = _scatter_sum(weighted_messages, receiver, dim_size)
    if aggr == "sum":
        return summed

    if edge_type is None:
        counts = torch.bincount(receiver, minlength=dim_size).to(
            device=messages.device, dtype=messages.dtype
        )
    else:
        query_to_host = edge_type == QUERY_TO_HOST_EDGE_TYPE
        non_query_to_host = ~query_to_host
        counts = torch.bincount(receiver[non_query_to_host], minlength=dim_size).to(
            device=messages.device,
            dtype=messages.dtype,
        )
        q2h_counts = torch.bincount(receiver[query_to_host], minlength=dim_size).to(
            device=messages.device
        )
        counts = counts + (q2h_counts > 0).to(
            device=messages.device, dtype=messages.dtype
        )

    shape = (counts.size(0),) + (1,) * (summed.dim() - 1)
    return summed / counts.clamp(min=1).view(shape)


class QueryToHostAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.logit_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.logit_mlp:
            if isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.logit_mlp[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError(
                f"Expected final query-to-host attention module to be Linear, got {type(final).__name__}."
            )
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def edge_weights(
        self, edge_attr: Tensor, edge_index: Tensor, edge_type: Tensor | None
    ) -> Tensor:
        if edge_attr.ndim != 2 or edge_attr.size(-1) != self.hidden_dim:
            raise ValueError(
                f"query-to-host attention expects edge_attr [E, {self.hidden_dim}], got {tuple(edge_attr.shape)}."
            )
        if (
            edge_index.ndim != 2
            or edge_index.size(0) != 2
            or edge_index.size(1) != edge_attr.size(0)
        ):
            raise ValueError(
                "query-to-host attention edge_index shape mismatch: "
                f"edge_index={tuple(edge_index.shape)}, edge_attr={tuple(edge_attr.shape)}."
            )
        if edge_type is None:
            raise ValueError("query-to-host attention requires edge_type.")
        if edge_type.shape != (edge_attr.size(0),):
            raise ValueError(
                f"query-to-host attention edge_type shape mismatch: expected {(edge_attr.size(0),)}, got {tuple(edge_type.shape)}."
            )

        weights = edge_attr.new_ones((edge_attr.size(0), 1))
        query_to_host = edge_type == QUERY_TO_HOST_EDGE_TYPE
        if not query_to_host.any():
            return weights

        receiver = edge_index[0]
        logits = self.logit_mlp(edge_attr[query_to_host]).squeeze(-1)
        alpha = softmax(logits, receiver[query_to_host]).to(
            device=weights.device, dtype=weights.dtype
        )
        weights[query_to_host] = alpha.unsqueeze(-1)
        return weights
