import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch.utils.checkpoint import checkpoint

from .query_host_attn import (
    QueryToHostAttention,
    aggregate_with_query_to_host_as_one,
    normalize_query_to_host_aggregation,
)


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, distances: Tensor) -> Tensor:
        cutoffs = 0.5 * ((distances * math.pi / self.cutoff).cos() + 1.0)
        return cutoffs * (distances < self.cutoff).float()


class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.means = nn.Parameter(means)
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> tuple[Tensor, Tensor]:
        start_value = torch.exp(torch.tensor(-self.cutoff))
        means = torch.linspace(start_value, 1.0, self.num_rbf)
        beta = (2.0 / self.num_rbf * (1.0 - start_value)).pow(-2)
        betas = beta.repeat(self.num_rbf)
        return means, betas

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        if isinstance(self.means, nn.Parameter):
            self.means.data.copy_(means)
        else:
            self.means.copy_(means)
        if isinstance(self.betas, nn.Parameter):
            self.betas.data.copy_(betas)
        else:
            self.betas.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        return (
            self.cutoff_fn(dist)
            * (-self.betas * (((self.alpha * (-dist)).exp() - self.means) ** 2)).exp()
        )


class Sphere(nn.Module):
    def __init__(self, lmax: int = 1) -> None:
        super().__init__()
        if lmax not in (1, 2):
            raise ValueError(f"Sphere only supports lmax=1 or 2 (got {lmax}).")
        self.lmax = lmax

    def forward(self, edge_vec: Tensor) -> Tensor:
        x = edge_vec[..., 0]
        y = edge_vec[..., 1]
        z = edge_vec[..., 2]
        sh_1 = torch.stack([x, y, z], dim=-1)
        if self.lmax == 1:
            return sh_1

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))
        sh_2 = torch.stack([sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)
        return torch.cat([sh_1, sh_2], dim=-1)


class VecLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        trainable: bool,
        norm_type: Optional[str] = "max_min",
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.eps = 1e-12

        weight = torch.ones(hidden_channels)
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def reset_parameters(self) -> None:
        if isinstance(self.weight, nn.Parameter):
            nn.init.ones_(self.weight)
        else:
            self.weight.fill_(1.0)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1, keepdim=True)
        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist
        max_val = dist.max(dim=-1).values
        min_val = dist.min(dim=-1).values
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        return dist.relu() * direct

    def forward(self, vec: Tensor) -> Tensor:
        if vec.size(1) not in (3, 8):
            raise ValueError(
                f"VecLayerNorm only supports 3 or 8 vector channels (got {vec.size(1)})."
            )

        if self.norm_type == "max_min":
            if vec.size(1) == 3:
                vec = self.max_min_norm(vec)
            else:
                vec1, vec2 = torch.split(vec, [3, 5], dim=1)
                vec = torch.cat(
                    [self.max_min_norm(vec1), self.max_min_norm(vec2)], dim=1
                )

        return vec * self.weight.unsqueeze(0).unsqueeze(0)


class InputNeighborEmbedding(MessagePassing):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        node_aggr: str = "sum",
        query_to_host_aggregation: str = "sum",
    ) -> None:
        super().__init__(aggr="add", node_dim=0, flow="target_to_source")
        self.hidden_channels = int(hidden_channels)
        self.node_aggr = str(node_aggr).lower()
        if self.node_aggr not in {"sum", "mean"}:
            raise ValueError(f"Unsupported ViSNet node_aggr: {node_aggr!r}.")
        self.query_to_host_aggregation = normalize_query_to_host_aggregation(
            query_to_host_aggregation
        )
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)
        self.query_to_host_attention = None
        if self.query_to_host_aggregation == "attn":
            self.query_to_host_attention = QueryToHostAttention(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.zero_()
        self.combine.bias.data.zero_()
        if self.query_to_host_attention is not None:
            self.query_to_host_attention.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        edge_type: Tensor | None = None,
    ) -> Tensor:
        weights = self.distance_proj(edge_attr) * self.cutoff(edge_weight).unsqueeze(-1)
        src, dst = edge_index[0], edge_index[1]
        if self.query_to_host_attention is None:
            edge_messages = self.message(x[dst], weights)
            x_neighbors = aggregate_with_query_to_host_as_one(
                edge_messages,
                src,
                dim_size=x.size(0),
                aggr=self.node_aggr,
            )
        else:
            if edge_type is None:
                raise ValueError(
                    "ViSNet query-to-host attention in neighbor embedding requires edge_type."
                )
            edge_messages = self.message(x[dst], weights)
            query_to_host_weight = self.query_to_host_attention.edge_weights(
                edge_attr=edge_messages,
                edge_index=edge_index,
                edge_type=edge_type,
            )
            x_neighbors = aggregate_with_query_to_host_as_one(
                edge_messages,
                src,
                dim_size=x.size(0),
                aggr=self.node_aggr,
                edge_type=edge_type,
                edge_weight=query_to_host_weight,
            )
        return self.combine(torch.cat([x, x_neighbors], dim=-1))

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class EdgeEmbedding(nn.Module):
    def __init__(self, num_rbf: int, hidden_channels: int) -> None:
        super().__init__()
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.zero_()

    def forward(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        x: Tensor,
    ) -> Tensor:
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        return (x_i + x_j) * self.edge_proj(edge_attr)


class ViSMessagePassing(MessagePassing):
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
        node_aggr: str = "sum",
        query_to_host_aggregation: str = "sum",
    ) -> None:
        super().__init__(aggr="add", node_dim=0, flow="target_to_source")
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        self.node_aggr = str(node_aggr).lower()
        if self.node_aggr not in {"sum", "mean"}:
            raise ValueError(f"Unsupported ViSNet node_aggr: {node_aggr!r}.")
        self.query_to_host_aggregation = normalize_query_to_host_aggregation(
            query_to_host_aggregation
        )

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels=hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )
        self.cutoff = CosineCutoff(cutoff)
        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
        if not last_layer:
            self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        else:
            self.s_proj = None
        if not last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.query_to_host_attention = None
        if self.query_to_host_aggregation == "attn":
            self.query_to_host_attention = QueryToHostAttention(hidden_channels)
        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor) -> Tensor:
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        linear_layers = [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.dk_proj,
            self.dv_proj,
            self.o_proj,
        ]
        if self.s_proj is not None:
            linear_layers.append(self.s_proj)
        for layer in linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.zero_()
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)
        if self.query_to_host_attention is not None:
            self.query_to_host_attention.reset_parameters()

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        edge_dir: Tensor,
        edge_type: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(edge_attr)).reshape(
            -1, self.num_heads, self.head_dim
        )
        dv = self.act(self.dv_proj(edge_attr)).reshape(
            -1, self.num_heads, self.head_dim
        )

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)

        src, dst = edge_index[0], edge_index[1]
        edge_x_msg, edge_vec_msg = self.message(
            q_i=q[src],
            k_j=k[dst],
            v_j=v[dst],
            vec_j=vec[dst],
            dk=dk,
            dv=dv,
            r_ij=edge_weight,
            d_ij=edge_dir,
        )
        query_to_host_weight = None
        if self.query_to_host_attention is not None:
            if edge_type is None:
                raise ValueError(
                    "ViSNet query-to-host attention aggregation requires edge_type."
                )
            query_to_host_weight = self.query_to_host_attention.edge_weights(
                edge_attr=edge_attr,
                edge_index=edge_index,
                edge_type=edge_type,
            )
        x_msg = aggregate_with_query_to_host_as_one(
            edge_x_msg,
            src,
            dim_size=x.size(0),
            aggr=self.node_aggr,
            edge_type=edge_type,
            edge_weight=query_to_host_weight,
        )
        vec_msg = aggregate_with_query_to_host_as_one(
            edge_vec_msg,
            src,
            dim_size=x.size(0),
            aggr=self.node_aggr,
            edge_type=edge_type,
            edge_weight=query_to_host_weight,
        )

        o1, o2, o3 = torch.split(self.o_proj(x_msg), self.hidden_channels, dim=-1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_msg

        if self.last_layer:
            return dx, dvec, None

        vec_i = vec[edge_index[0]]
        vec_j = vec[edge_index[1]]
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), edge_dir)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -edge_dir)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(edge_attr)) * w_dot
        return dx, dvec, df_ij

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Tensor,
        dv: Tensor,
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> tuple[Tensor, Tensor]:
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = (v_j * dv * attn.unsqueeze(2)).view(-1, self.hidden_channels)
        if self.last_layer:
            return v_j, torch.zeros_like(vec_j)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=-1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
        return v_j, vec_j

    def aggregate(
        self,
        features: tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> tuple[Tensor, Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return x, vec


class ViSNetEncoder(nn.Module):
    """ViSNet-style equivariant encoder adapted to SiteFlow's scalar node features."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        cutoff: float,
        n_rbf: int = 64,
        num_heads: int = 8,
        lmax: int = 1,
        trainable_rbf: bool = False,
        vecnorm_type: Optional[str] = "max_min",
        trainable_vecnorm: bool = False,
        node_aggr: str = "sum",
        query_to_host_aggregation: str = "sum",
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lmax = lmax
        self.activation_checkpoint = bool(activation_checkpoint)
        self.node_aggr = str(node_aggr).lower()
        if self.node_aggr not in {"sum", "mean"}:
            raise ValueError(f"Unsupported ViSNet node_aggr: {node_aggr!r}.")
        self.query_to_host_aggregation = normalize_query_to_host_aggregation(
            query_to_host_aggregation
        )
        self.vector_channels = ((lmax + 1) ** 2) - 1
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(
            cutoff=cutoff,
            num_rbf=n_rbf,
            trainable=trainable_rbf,
        )
        self.neighbor_embedding = InputNeighborEmbedding(
            hidden_channels=hidden_dim,
            num_rbf=n_rbf,
            cutoff=cutoff,
            node_aggr=self.node_aggr,
            query_to_host_aggregation=self.query_to_host_aggregation,
        )
        self.edge_embedding = EdgeEmbedding(
            num_rbf=n_rbf,
            hidden_channels=hidden_dim,
        )
        self.layers = nn.ModuleList(
            [
                ViSMessagePassing(
                    num_heads=num_heads,
                    hidden_channels=hidden_dim,
                    cutoff=cutoff,
                    vecnorm_type=vecnorm_type,
                    trainable_vecnorm=trainable_vecnorm,
                    last_layer=(idx == num_layers - 1),
                    node_aggr=self.node_aggr,
                    query_to_host_aggregation=self.query_to_host_aggregation,
                )
                for idx in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.vec_out_norm = VecLayerNorm(
            hidden_channels=hidden_dim,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(
        self,
        pos: Tensor,
        edge_index: Tensor,
        q: Tensor,
        edge_type: Tensor | None = None,
        vec: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        num_nodes = q.size(0)
        expected_vec_shape = (num_nodes, self.vector_channels, self.hidden_dim)
        if vec is None:
            vec = q.new_zeros(expected_vec_shape)
        else:
            if tuple(vec.shape) != expected_vec_shape:
                raise ValueError(
                    f"ViSNet vector state shape mismatch: expected {expected_vec_shape}, got {tuple(vec.shape)}."
                )
            if vec.device != q.device:
                raise ValueError(
                    f"ViSNet vector state device mismatch: q={q.device}, vec={vec.device}."
                )
            if vec.dtype != q.dtype:
                raise TypeError(
                    f"ViSNet vector state dtype mismatch: q={q.dtype}, vec={vec.dtype}."
                )
        if edge_index.numel() == 0:
            return (
                self.out_norm(q),
                self.vec_out_norm(vec),
                {
                    "edge_attr": q.new_zeros((0, self.hidden_dim)),
                    "edge_weight": q.new_zeros((0,)),
                    "edge_vec": pos.new_zeros((0, 3)),
                },
            )

        src, dst = edge_index[0], edge_index[1]
        edge_vec = pos[dst] - pos[src]
        edge_weight = torch.norm(edge_vec, dim=-1).clamp(min=1e-12)
        edge_dir = edge_vec / edge_weight.unsqueeze(-1)
        edge_dir = self.sphere(edge_dir)
        edge_rbf = self.distance_expansion(edge_weight)

        q = self.neighbor_embedding(
            q, edge_index, edge_weight, edge_rbf, edge_type=edge_type
        )
        edge_attr = self.edge_embedding(edge_index, edge_rbf, q)

        for layer in self.layers[:-1]:
            if self.activation_checkpoint and self.training:

                def layer_forward(
                    q_arg: Tensor, vec_arg: Tensor, edge_attr_arg: Tensor,
                    layer_module=layer,
                ):
                    return layer_module(
                        q_arg,
                        vec_arg,
                        edge_index,
                        edge_weight,
                        edge_attr_arg,
                        edge_dir,
                        edge_type=edge_type,
                    )

                dq, dvec, dedge = checkpoint(
                    layer_forward,
                    q,
                    vec,
                    edge_attr,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                dq, dvec, dedge = layer(
                    q,
                    vec,
                    edge_index,
                    edge_weight,
                    edge_attr,
                    edge_dir,
                    edge_type=edge_type,
                )
            q = q + dq
            vec = vec + dvec
            if dedge is not None:
                edge_attr = edge_attr + dedge

        dq, dvec, _ = self.layers[-1](
            q, vec, edge_index, edge_weight, edge_attr, edge_dir, edge_type=edge_type
        )
        q = self.out_norm(q + dq)
        vec = self.vec_out_norm(vec + dvec)
        return (
            q,
            vec,
            {
                "edge_attr": edge_attr,
                "edge_weight": edge_weight,
                "edge_vec": edge_vec,
            },
        )
