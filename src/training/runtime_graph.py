from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
from src.graph.radius import host_edges
from src.graph.interaction import interaction_edges
from src.model.gnn.query_host_attn import (
    HOST_HOST_EDGE_TYPE,
    HOST_TO_QUERY_EDGE_TYPE,
    QUERY_TO_HOST_EDGE_TYPE,
)


class RuntimeGraphMixin:
    @staticmethod
    def _normalize_runtime_host_host_filter(
        value: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if value is None:
            return {
                "enabled": False,
                "min_host_nodes": 0,
                "cutoff": None,
                "max_neighbors": None,
            }
        if not isinstance(value, dict):
            raise TypeError(
                f"runtime_host_host_filter must be a mapping, got {type(value)!r}."
            )
        enabled = bool(value.get("enabled", False))
        min_host_nodes = int(value.get("min_host_nodes", 0))
        if min_host_nodes < 0:
            raise ValueError(
                f"runtime_host_host_filter.min_host_nodes must be non-negative, got {min_host_nodes}."
            )
        cutoff = value.get("cutoff")
        cutoff = None if cutoff is None else float(cutoff)
        if cutoff is not None and cutoff <= 0.0:
            raise ValueError(
                f"runtime_host_host_filter.cutoff must be positive, got {cutoff}."
            )
        max_neighbors = value.get("max_neighbors")
        max_neighbors = None if max_neighbors is None else int(max_neighbors)
        if max_neighbors is not None and max_neighbors <= 0:
            raise ValueError(
                f"runtime_host_host_filter.max_neighbors must be positive, got {max_neighbors}."
            )
        if enabled and cutoff is None and (max_neighbors is None):
            raise ValueError(
                "runtime_host_host_filter.enabled=true requires cutoff and/or max_neighbors."
            )
        return {
            "enabled": enabled,
            "min_host_nodes": min_host_nodes,
            "cutoff": cutoff,
            "max_neighbors": max_neighbors,
        }

    @staticmethod
    def _validate_host_host_edge_index(
        host_host: Tensor, *, host_count: int, device: torch.device
    ) -> None:
        if host_host is None:
            raise ValueError("A host-host edge tensor is required.")
        if not isinstance(host_host, Tensor):
            raise TypeError(
                f"batch.edge_index must be a Tensor, got {type(host_host)!r}."
            )
        if host_host.device != device:
            raise ValueError(
                f"batch.edge_index device mismatch: {host_host.device} vs expected {device}."
            )
        if host_host.dtype != torch.long:
            raise TypeError(
                f"batch.edge_index must have dtype torch.long, got {host_host.dtype}."
            )
        if host_host.ndim != 2 or host_host.size(0) != 2:
            raise ValueError(
                f"batch.edge_index must have shape [2, E], got {tuple(host_host.shape)}."
            )
        if host_host.numel() == 0:
            return
        min_idx = int(host_host.min().item())
        max_idx = int(host_host.max().item())
        if min_idx < 0 or max_idx >= host_count:
            raise ValueError(
                f"batch.edge_index must contain host-only indices in [0, num_host_nodes): min={min_idx}, max={max_idx}, num_host_nodes={host_count}."
            )
        if bool((host_host[0] == host_host[1]).any().item()):
            raise ValueError("batch.edge_index must not contain host-host self loops.")

    @staticmethod
    def _edge_receiver_rank(edge_index: Tensor, distances: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=edge_index.device)
        dst = edge_index[0]
        if dst.numel() > 1:
            if not bool((dst[1:] >= dst[:-1]).all().item()):
                raise RuntimeError(
                    "Runtime host-host filtering requires edges sorted by receiver node."
                )
            same_receiver = dst[1:] == dst[:-1]
            if bool(
                ((distances[1:] + 1e-06 < distances[:-1]) & same_receiver).any().item()
            ):
                raise RuntimeError(
                    "Runtime host-host filtering requires edges sorted by distance per receiver."
                )
        start = torch.ones((dst.numel(),), dtype=torch.bool, device=dst.device)
        if dst.numel() > 1:
            start[1:] = dst[1:] != dst[:-1]
        start_idx = start.nonzero(as_tuple=False).view(-1)
        group_end = torch.cat(
            [
                start_idx[1:],
                torch.tensor([dst.numel()], dtype=torch.long, device=dst.device),
            ]
        )
        counts = group_end - start_idx
        return torch.arange(
            dst.numel(), dtype=torch.long, device=dst.device
        ) - torch.repeat_interleave(start_idx, counts)

    def _filter_runtime_host_host_edges(
        self, host_pos: Tensor, host_batch: Tensor, host_host: Tensor
    ) -> Tensor:
        policy = self._normalize_runtime_host_host_filter(
            getattr(self, "runtime_host_host_filter", None)
        )
        if not bool(policy["enabled"]) or host_host.numel() == 0:
            return host_host
        dst, src = (host_host[0], host_host[1])
        src_sample = host_batch[src]
        dst_sample = host_batch[dst]
        if not bool((src_sample == dst_sample).all().item()):
            raise RuntimeError(
                "Runtime host-host filtering received cross-sample host-host edges."
            )
        host_counts = torch.bincount(
            host_batch, minlength=int(host_batch.max().item()) + 1
        )
        large_sample = host_counts >= int(policy["min_host_nodes"])
        active_edge = large_sample[dst_sample]
        if not bool(active_edge.any().item()):
            return host_host
        distances = torch.norm(host_pos[src].float() - host_pos[dst].float(), dim=-1)
        keep = torch.ones(
            (host_host.size(1),), dtype=torch.bool, device=host_host.device
        )
        cutoff = policy["cutoff"]
        if cutoff is not None:
            keep[active_edge] &= distances[active_edge] <= float(cutoff)
        filtered = host_host[:, keep]
        filtered_distances = distances[keep]
        filtered_active_edge = active_edge[keep]
        max_neighbors = policy["max_neighbors"]
        if max_neighbors is not None and filtered.numel() > 0:
            order = torch.argsort(filtered_distances, stable=True)
            order = order[torch.argsort(filtered[0, order], stable=True)]
            filtered = filtered[:, order]
            filtered_distances = filtered_distances[order]
            filtered_active_edge = filtered_active_edge[order]
            receiver_rank = self._edge_receiver_rank(filtered, filtered_distances)
            keep_rank = ~filtered_active_edge | (receiver_rank < int(max_neighbors))
            filtered = filtered[:, keep_rank]
        if filtered.size(1) == 0:
            raise RuntimeError(
                "Runtime host-host filtering removed all host-host edges from the batch."
            )
        return filtered

    def _build_runtime_edge_indices(
        self,
        host_pos: Tensor,
        query_pos: Tensor,
        host_batch: Tensor,
        query_batch: Tensor,
        host_host: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del host_host
        if self.use_query_to_query_edges:
            raise ValueError("SurfQNet does not use query-query edges.")
        host_host = host_edges(
            host_pos,
            host_batch,
            cutoff=self.host_host_cutoff,
            max_neighbors=self.graph_max_neighbors,
        )
        host_host = self._filter_runtime_host_host_edges(
            host_pos, host_batch, host_host
        )
        edges = [host_host]
        types = [torch.full_like(host_host[0], HOST_HOST_EDGE_TYPE)]
        if self.use_host_to_query_edges:
            host_query = interaction_edges(
                host_pos,
                query_pos,
                host_batch,
                query_batch,
                mode=self.interaction_graph_mode,
                cutoff=self.host_query_cutoff,
                max_neighbors=self.graph_max_neighbors,
                bidirectional=self.use_query_to_host_edges,
            )
            # Full connectivity controls edge enumeration. ViSNet's configured
            # radial envelope is independent (60 Å in the full-graph ablation).
            edges.append(host_query)
            edge_types = torch.where(
                host_query[0] >= host_pos.size(0),
                HOST_TO_QUERY_EDGE_TYPE,
                QUERY_TO_HOST_EDGE_TYPE,
            )
            types.append(edge_types)
        return (host_host, torch.cat(edges, dim=1), torch.cat(types))

    def _forward(
        self,
        batch,
        query_pos: Tensor,
        query_x: Tensor,
        query_batch: Tensor,
        t_query: Tensor,
    ):
        host_x = batch.x
        host_pos = batch.pos
        host_batch = batch.batch
        host_atom_type_id = batch.atom_type_id
        host_residue_type_id = batch.residue_type_id
        _, interaction_edge_index, interaction_edge_type = (
            self._build_runtime_edge_indices(
                host_pos=host_pos,
                query_pos=query_pos,
                host_batch=host_batch,
                query_batch=query_batch,
                host_host=None,
            )
        )
        out = self.model(
            host_x=host_x,
            query_x=query_x,
            host_atom_type_id=host_atom_type_id,
            host_residue_type_id=host_residue_type_id,
            host_pos=host_pos,
            query_pos=query_pos,
            interaction_edge_index=interaction_edge_index,
            interaction_edge_type=interaction_edge_type,
            t_query=t_query,
            host_batch=host_batch,
            query_batch=query_batch,
        )
        result = {
            "host_scalar": out["host_scalar"],
            "query_scalar": out["query_scalar"],
            "host_logits": out["host_final_logits"],
            "host_stage_logits": out["host_stage_logits"],
            "query_disp": out["query_disp"],
            "query_site_embed": out.get("query_site_embed", out["mask_embed"]),
        }
        if "query_distance_logits" in out:
            result["query_distance_logits"] = out["query_distance_logits"]
        if "query_conf_logits" in out:
            result["query_conf_logits"] = out["query_conf_logits"]
        for key in (
            "site_logits",
            "site_mask_logits",
            "site_mask_pair_mask",
            "site_host_scalar",
            "site_host_mask",
            "site_query_mask",
            "site_sample_ids",
            "site_aux_outputs",
            "site_contrastive_embed",
        ):
            if key in out:
                result[key] = out[key]
        return result

    def _query_refiner_enabled(self) -> bool:
        return bool(getattr(self.model, "query_refiner_enabled", False))

    def _refine_query_outputs(
        self,
        out: dict[str, Tensor],
        host_pos: Tensor,
        host_batch: Tensor,
        query_pos: Tensor,
        query_batch: Tensor,
        return_intermediate: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self._query_refiner_enabled():
            raise RuntimeError(
                "_refine_query_outputs requires model.query_refiner_enabled=true."
            )
        if not hasattr(self.model, "refine_queries"):
            raise RuntimeError(
                "model.query_refiner_enabled=true but model has no refine_queries method."
            )
        refined_out = self.model.refine_queries(
            out=out,
            host_pos=host_pos,
            host_batch=host_batch,
            query_pos=query_pos,
            query_batch=query_batch,
            return_intermediate=return_intermediate,
        )
        refined_pos = refined_out.get("query_refined_pos")
        if refined_pos is None:
            raise RuntimeError("query refiner did not return query_refined_pos.")
        if refined_pos.shape != query_pos.shape:
            raise ValueError(
                f"query refiner returned coordinate shape mismatch: expected {tuple(query_pos.shape)}, got {tuple(refined_pos.shape)}."
            )
        return (refined_pos, refined_out)

    def _predict_query_eval_state(
        self, batch, query_pos_0: Tensor, query_x: Tensor, query_batch: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero_t = self._constant_query_times(query_pos_0, 0.0)
        init_out = self._forward(
            batch,
            query_pos=query_pos_0,
            query_x=query_x,
            query_batch=query_batch,
            t_query=zero_t,
        )
        if self._query_refiner_enabled():
            stage0_pos, _ = self._apply_query_displacement(
                query_pos_0, init_out["query_disp"]
            )
            return self._refine_query_outputs(
                out=init_out,
                host_pos=batch.pos,
                host_batch=batch.batch,
                query_pos=stage0_pos.detach(),
                query_batch=query_batch,
            )
        if self.eval_forward_passes == 1:
            return (query_pos_0, init_out)
        final_pos, _ = self._apply_query_displacement(
            query_pos_0, init_out["query_disp"]
        )
        final_out = self._forward(
            batch,
            query_pos=final_pos,
            query_x=query_x,
            query_batch=query_batch,
            t_query=zero_t,
        )
        return (final_pos, final_out)

    def _assign_targets_per_sample(
        self,
        query_pos: Tensor,
        query_batch: Tensor,
        target_pos: Tensor,
        target_batch: Tensor,
        target_id: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assigned_pos = query_pos.clone()
        assigned_ids = torch.full(
            (query_pos.size(0),), -1, dtype=torch.long, device=query_pos.device
        )
        valid = torch.zeros(
            (query_pos.size(0),), dtype=torch.bool, device=query_pos.device
        )
        for sample_idx in query_batch.unique(sorted=True).tolist():
            query_sel = query_batch == sample_idx
            target_sel = target_batch == sample_idx
            if not query_sel.any() or not target_sel.any():
                continue
            sample_query_pos = query_pos[query_sel]
            sample_target_pos = target_pos[target_sel]
            valid_query = torch.isfinite(sample_query_pos).all(dim=-1)
            valid_target = torch.isfinite(sample_target_pos).all(dim=-1)
            if not valid_query.any() or not valid_target.any():
                continue
            query_indices = query_sel.nonzero(as_tuple=False).view(-1)[valid_query]
            target_indices = target_sel.nonzero(as_tuple=False).view(-1)[valid_target]
            dists = torch.cdist(
                sample_query_pos[valid_query].float(),
                sample_target_pos[valid_target].float(),
            )
            nearest = dists.argmin(dim=-1)
            matched_target_indices = target_indices[nearest]
            assigned_pos[query_indices] = target_pos[matched_target_indices]
            assigned_ids[query_indices] = target_id[matched_target_indices]
            valid[query_indices] = True
        return (assigned_pos, assigned_ids, valid)
