from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
from src.model.layers import Dense


class QueryAdaptiveRBF(nn.Module):
    """Gaussian RBF basis whose distance scale is predicted by each query."""

    def __init__(
        self, num_rbf: int, min_radius: float, max_radius: float, init_radius: float
    ) -> None:
        super().__init__()
        self.num_rbf = int(num_rbf)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.init_radius = float(init_radius)
        if self.num_rbf <= 0:
            raise ValueError(f"num_rbf must be positive, got {self.num_rbf}.")
        if self.min_radius <= 0.0:
            raise ValueError(f"min_radius must be positive, got {self.min_radius}.")
        if self.max_radius <= self.min_radius:
            raise ValueError(
                f"max_radius must be larger than min_radius, got {self.max_radius} <= {self.min_radius}."
            )
        if not self.min_radius < self.init_radius < self.max_radius:
            raise ValueError(
                f"init_radius must be strictly between min_radius and max_radius, got min={self.min_radius}, init={self.init_radius}, max={self.max_radius}."
            )
        centers = torch.linspace(0.0, 1.0, self.num_rbf)
        self.register_buffer("centers", centers)
        delta = 1.0 if self.num_rbf == 1 else 1.0 / float(self.num_rbf - 1)
        self.gamma = 1.0 / (delta * delta)

    def forward(self, distances: Tensor, query_radius: Tensor) -> Tensor:
        if distances.ndim != 2:
            raise ValueError(
                f"QueryAdaptiveRBF expects distances [Q, H], got {tuple(distances.shape)}."
            )
        if query_radius.shape != (distances.size(0), 1):
            raise ValueError(
                f"QueryAdaptiveRBF query_radius shape mismatch: expected {(distances.size(0), 1)}, got {tuple(query_radius.shape)}."
            )
        scaled_distance = distances / query_radius
        return torch.exp(
            -self.gamma
            * torch.square(scaled_distance.unsqueeze(-1) - self.centers.to(distances))
        )


class VNDirectSiteMaskHead(nn.Module):
    """Predict a residue mask for every VN with distance-conditioned similarity."""

    loss_type = "per_vn_mask"

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int | None = None,
        dropout: float = 0.0,
        distance_num_rbf: int = 32,
        distance_cutoff: float = 30.0,
        use_distance_gate: bool = True,
        distance_rbf_mode: str = "adaptive",
        adaptive_rbf_min_radius: float = 2.0,
        adaptive_rbf_max_radius: float = 30.0,
        adaptive_rbf_init_radius: float = 12.0,
        use_affinity: bool = False,
        residue_encoder: str = "none",
    ) -> None:
        super().__init__()
        if distance_rbf_mode != "adaptive" or use_affinity or residue_encoder != "none":
            raise ValueError(
                "SurfQNet mask head uses query-adaptive distance gates without an extra encoder or affinity head."
            )
        proj_dim = int(projection_dim if projection_dim is not None else hidden_dim)
        if proj_dim <= 0:
            raise ValueError(f"projection_dim must be positive, got {proj_dim}.")
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")
        self.projection_dim = proj_dim
        self.scale = proj_dim ** (-0.5)
        self.distance_num_rbf = int(distance_num_rbf)
        if self.distance_num_rbf <= 0:
            raise ValueError(
                f"distance_num_rbf must be positive, got {self.distance_num_rbf}."
            )
        self.distance_cutoff = float(distance_cutoff)
        if self.distance_cutoff <= 0.0:
            raise ValueError(
                f"distance_cutoff must be positive, got {self.distance_cutoff}."
            )
        self.use_distance_gate = bool(use_distance_gate)
        self.distance_rbf_mode = str(distance_rbf_mode).lower()
        if self.distance_rbf_mode in {"query_adaptive", "adaptive_query"}:
            self.distance_rbf_mode = "adaptive"
        if self.distance_rbf_mode in {"legacy", "expnormal", "fixed_expnormal"}:
            self.distance_rbf_mode = "fixed"
        if self.distance_rbf_mode not in {"adaptive", "fixed"}:
            raise ValueError(
                f"Unsupported VN-dot distance_rbf_mode: {distance_rbf_mode!r}."
            )
        self.adaptive_rbf_min_radius = float(adaptive_rbf_min_radius)
        self.adaptive_rbf_max_radius = float(adaptive_rbf_max_radius)
        self.adaptive_rbf_init_radius = float(adaptive_rbf_init_radius)
        self.use_affinity = bool(use_affinity)
        residue_encoder = str(residue_encoder).lower()
        if residue_encoder in {"", "none", "identity", "off", "false"}:
            residue_encoder = "none"
        if residue_encoder in {"detr", "detr_encoder", "transformer"}:
            residue_encoder = "unisite"
        if residue_encoder not in {"none", "unisite"}:
            raise ValueError(
                f"Unsupported VN-dot residue_encoder: {residue_encoder!r}."
            )
        self.residue_encoder = residue_encoder
        self.query_proj = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()),
            nn.Dropout(dropout),
            Dense(hidden_dim, proj_dim),
        )
        self.host_proj = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()),
            nn.Dropout(dropout),
            Dense(hidden_dim, proj_dim),
        )
        self.distance_rbf = None
        self.distance_gate = None
        self.distance_bias = None
        self.radius_head = None
        if self.use_distance_gate:
            self.distance_rbf = QueryAdaptiveRBF(
                num_rbf=self.distance_num_rbf,
                min_radius=self.adaptive_rbf_min_radius,
                max_radius=self.adaptive_rbf_max_radius,
                init_radius=self.adaptive_rbf_init_radius,
            )
            self.radius_head = nn.Sequential(
                Dense(proj_dim, hidden_dim, activation=nn.SiLU()),
                nn.Dropout(dropout),
                Dense(hidden_dim, 1),
            )
            if isinstance(self.radius_head[-1], Dense):
                nn.init.zeros_(self.radius_head[-1].weight)
                radius_span = (
                    self.adaptive_rbf_max_radius - self.adaptive_rbf_min_radius
                )
                init_ratio = (
                    self.adaptive_rbf_init_radius - self.adaptive_rbf_min_radius
                ) / radius_span
                nn.init.constant_(
                    self.radius_head[-1].bias, math.log(init_ratio / (1.0 - init_ratio))
                )
            self.distance_gate = nn.Sequential(
                Dense(self.distance_num_rbf, hidden_dim, activation=nn.SiLU()),
                nn.Dropout(dropout),
                Dense(hidden_dim, proj_dim),
            )
            self.distance_bias = nn.Sequential(
                Dense(self.distance_num_rbf, hidden_dim, activation=nn.SiLU()),
                nn.Dropout(dropout),
                Dense(hidden_dim, 1),
            )
            self.distance_bias.requires_grad_(False)
            if isinstance(self.distance_gate[-1], Dense):
                nn.init.zeros_(self.distance_gate[-1].weight)
                nn.init.zeros_(self.distance_gate[-1].bias)
            if isinstance(self.distance_bias[-1], Dense):
                nn.init.zeros_(self.distance_bias[-1].weight)
                nn.init.zeros_(self.distance_bias[-1].bias)

    def _predict_radius(self, query_embed: Tensor) -> Tensor:
        if self.radius_head is None:
            raise RuntimeError("Adaptive RBF radius head is not initialized.")
        radius_unit = torch.sigmoid(self.radius_head(query_embed))
        radius_span = self.adaptive_rbf_max_radius - self.adaptive_rbf_min_radius
        return self.adaptive_rbf_min_radius + radius_span * radius_unit

    def _distance_gated_logits(
        self,
        query_embed: Tensor,
        query_pos: Tensor,
        query_radius: Tensor | None,
        host_embed: Tensor,
        host_pos: Tensor,
    ) -> Tensor:
        if self.distance_rbf is None or self.distance_gate is None:
            raise RuntimeError(
                "Distance gate is enabled but distance_rbf or distance_gate is not initialized."
            )
        distances = torch.cdist(query_pos.float(), host_pos.float()).to(
            dtype=query_embed.dtype
        )
        if query_radius is None:
            raise RuntimeError("Adaptive distance RBF requires query_radius.")
        rbf = self.distance_rbf(distances, query_radius)
        distance_gate = 2.0 * torch.sigmoid(self.distance_gate(rbf))
        return (query_embed[:, None, :] * host_embed[None, :, :] * distance_gate).sum(
            dim=-1
        ) * self.scale

    def sample_mask_logits(
        self,
        host_scalar: Tensor,
        query_scalar: Tensor,
        host_pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        if host_scalar.ndim != 2 or query_scalar.ndim != 2:
            raise ValueError(
                f"VNDirectSiteMaskHead.sample_mask_logits expects host/query scalar tensors with shape [N, D], got {tuple(host_scalar.shape)} and {tuple(query_scalar.shape)}."
            )
        if host_scalar.size(-1) != query_scalar.size(-1):
            raise ValueError(
                f"VNDirectSiteMaskHead host/query feature dimension mismatch: {host_scalar.size(-1)} vs {query_scalar.size(-1)}."
            )
        if query_scalar.size(0) == 0:
            return host_scalar.new_zeros((0, host_scalar.size(0)))
        host_embed = self.host_proj(host_scalar)
        query_embed = self.query_proj(query_scalar)
        if self.use_distance_gate:
            if self.distance_rbf is None or self.distance_gate is None:
                raise RuntimeError(
                    "Distance gate is enabled but distance_rbf or distance_gate is not initialized."
                )
            if host_pos is None or query_pos is None:
                raise ValueError(
                    "VNDirectSiteMaskHead distance gate requires host_pos and query_pos."
                )
            if host_pos.shape != (host_scalar.size(0), 3):
                raise ValueError(
                    f"VNDirectSiteMaskHead host_pos shape mismatch: expected {(host_scalar.size(0), 3)}, got {tuple(host_pos.shape)}."
                )
            if query_pos.shape != (query_scalar.size(0), 3):
                raise ValueError(
                    f"VNDirectSiteMaskHead query_pos shape mismatch: expected {(query_scalar.size(0), 3)}, got {tuple(query_pos.shape)}."
                )
            query_radius = (
                self._predict_radius(query_embed).to(dtype=query_embed.dtype)
                if self.distance_rbf_mode == "adaptive"
                else None
            )
            return self._distance_gated_logits(
                query_embed=query_embed,
                query_pos=query_pos,
                query_radius=query_radius,
                host_embed=host_embed,
                host_pos=host_pos,
            )
        return torch.matmul(query_embed, host_embed.transpose(0, 1)) * self.scale

    def forward(
        self,
        host_scalar: Tensor,
        host_batch: Tensor | None,
        query_scalar: Tensor | None = None,
        query_batch: Tensor | None = None,
        host_pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if query_scalar is None:
            raise ValueError("VNDirectSiteMaskHead requires query_scalar.")
        if host_batch is None or query_batch is None:
            raise ValueError("Explicit protein/query batch indices are required.")
        if self.use_distance_gate and (host_pos is None or query_pos is None):
            raise ValueError(
                "VNDirectSiteMaskHead distance gate requires host_pos and query_pos."
            )
        site_host_scalar = host_scalar
        sample_ids = host_batch.unique(sorted=True)
        batch_size = int(sample_ids.numel())
        max_host = 0
        max_queries = 0
        for sample_id in sample_ids.tolist():
            max_host = max(max_host, int((host_batch == sample_id).sum().item()))
            max_queries = max(max_queries, int((query_batch == sample_id).sum().item()))
        site_host_mask = torch.zeros(
            (batch_size, max_host), dtype=torch.bool, device=host_scalar.device
        )
        site_query_mask = torch.zeros(
            (batch_size, max_queries), dtype=torch.bool, device=host_scalar.device
        )
        for out_idx, sample_id in enumerate(sample_ids.tolist()):
            host_sel = host_batch == sample_id
            query_sel = query_batch == sample_id
            num_host = int(host_sel.sum().item())
            num_query = int(query_sel.sum().item())
            if num_host == 0:
                continue
            site_host_mask[out_idx, :num_host] = True
            if num_query == 0:
                continue
            site_query_mask[out_idx, :num_query] = True
        return {
            "site_host_scalar": site_host_scalar,
            "site_host_mask": site_host_mask,
            "site_query_mask": site_query_mask,
            "site_sample_ids": sample_ids,
        }
