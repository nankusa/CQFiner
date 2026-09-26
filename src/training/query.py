from __future__ import annotations


import torch
from torch import Tensor
from src.sampling.volume import sample_atom_volume, sample_seed


class QuerySamplingMixin:
    _EVAL_QUERY_SAMPLING_MODES = {"fps", "random", "curvature"}

    @staticmethod
    def _farthest_point_indices(points: Tensor, num_points: int) -> Tensor:
        if num_points <= 0 or points.size(0) == 0:
            return torch.empty((0,), dtype=torch.long, device=points.device)

        num_available = int(points.size(0))
        if num_available == 1:
            return torch.zeros((num_points,), dtype=torch.long, device=points.device)

        selected = torch.empty((num_points,), dtype=torch.long, device=points.device)
        centroid = points.mean(dim=0, keepdim=True)
        min_dists = torch.full(
            (num_available,), float("inf"), dtype=points.dtype, device=points.device
        )
        current = torch.norm(points - centroid, dim=-1).argmax()

        for step in range(num_points):
            selected[step] = current
            current_point = points[current : current + 1]
            min_dists = torch.minimum(
                min_dists, torch.cdist(points, current_point).squeeze(-1)
            )
            current = min_dists.argmax()

        return selected

    def _curvature_weighted_indices(
        self, surface_pos: Tensor, num_points: int
    ) -> Tensor:
        if num_points <= 0:
            return torch.empty((0,), dtype=torch.long, device=surface_pos.device)
        if surface_pos.size(0) < 3:
            raise ValueError(
                "Curvature-weighted surface sampling requires at least 3 surface points."
            )

        num_neighbors = min(
            self.query_surface_normal_neighbors, int(surface_pos.size(0))
        )
        if num_neighbors < 3:
            raise ValueError(
                f"Curvature-weighted surface sampling requires at least 3 neighbors, got {num_neighbors}."
            )

        distances = torch.cdist(surface_pos.float(), surface_pos.float())
        neighbor_idx = distances.topk(k=num_neighbors, largest=False).indices
        neighborhoods = surface_pos[neighbor_idx]
        centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered) / max(
            num_neighbors - 1, 1
        )
        eigvals = torch.linalg.eigvalsh(cov.float()).clamp_min(0.0)
        curvature = eigvals[:, 0] / eigvals.sum(dim=-1).clamp_min(
            torch.finfo(eigvals.dtype).eps
        )
        if not torch.isfinite(curvature).all():
            raise ValueError(
                "Curvature-weighted surface sampling produced non-finite curvature values."
            )
        weights = curvature.to(device=surface_pos.device, dtype=torch.float32)
        weights = weights + torch.finfo(weights.dtype).eps
        if not torch.isfinite(weights).all() or float(weights.sum().item()) <= 0.0:
            raise ValueError(
                "Curvature-weighted surface sampling produced invalid sampling weights."
            )
        replacement = int(surface_pos.size(0)) < int(num_points)
        return torch.multinomial(
            weights, num_samples=int(num_points), replacement=replacement
        )

    @staticmethod
    def _normalize_vectors(vectors: Tensor) -> Tensor:
        return vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def _estimate_surface_outward_normals(
        self,
        surface_pos: Tensor,
        surface_idx: Tensor,
        sample_centroid: Tensor,
    ) -> Tensor:
        if surface_idx.numel() == 0:
            raise ValueError(
                "Cannot estimate surface normals without selected surface indices."
            )

        selected_pos = surface_pos[surface_idx]
        if surface_pos.size(0) < 3:
            raise ValueError(
                "Cannot estimate surface normals from fewer than 3 surface points."
            )

        num_neighbors = min(
            self.query_surface_normal_neighbors, int(surface_pos.size(0))
        )
        distances = torch.cdist(selected_pos.float(), surface_pos.float())
        neighbor_idx = distances.topk(k=num_neighbors, largest=False).indices
        neighborhoods = surface_pos[neighbor_idx]
        centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered) / max(
            num_neighbors - 1, 1
        )
        eigvecs = torch.linalg.eigh(cov.float()).eigenvectors
        normals = eigvecs[..., 0].to(device=surface_pos.device, dtype=surface_pos.dtype)

        radial = selected_pos - sample_centroid
        flip = (normals * radial).sum(dim=-1, keepdim=True) < 0
        normals = torch.where(flip, -normals, normals)

        normal_norm = normals.norm(dim=-1, keepdim=True)
        degenerate = normal_norm.squeeze(-1) < 1e-8
        if degenerate.any():
            raise ValueError("Degenerate surface normal estimate.")
        return self._normalize_vectors(normals)

    def _apply_surface_query_offset(
        self,
        surface_pos: Tensor,
        surface_idx: Tensor,
        base_pos: Tensor,
        sample_centroid: Tensor,
        randomize: bool,
    ) -> Tensor:
        outward = self._estimate_surface_outward_normals(
            surface_pos=surface_pos,
            surface_idx=surface_idx,
            sample_centroid=sample_centroid,
        )
        if randomize:
            offset_span = max(
                self.query_surface_max_offset - self.query_surface_min_offset, 0.0
            )
            if offset_span > 0:
                offset = (
                    self.query_surface_min_offset
                    + torch.rand(
                        (base_pos.size(0), 1),
                        device=base_pos.device,
                        dtype=base_pos.dtype,
                    )
                    * offset_span
                )
            else:
                offset = torch.full(
                    (base_pos.size(0), 1),
                    self.query_surface_min_offset,
                    device=base_pos.device,
                    dtype=base_pos.dtype,
                )
            tangent = torch.randn_like(base_pos)
            tangent = tangent - (tangent * outward).sum(dim=-1, keepdim=True) * outward
            tangent_norm = tangent.norm(dim=-1, keepdim=True)
            valid_tangent = tangent_norm.squeeze(-1) >= 1e-8
            if not valid_tangent.all():
                raise ValueError(
                    "Cannot jitter surface query positions with degenerate tangent directions."
                )
            tangent = tangent / tangent_norm.clamp(min=1e-8)
            tangent_scale = torch.randn(
                (base_pos.size(0), 1), device=base_pos.device, dtype=base_pos.dtype
            )
            tangent = tangent * tangent_scale * self.query_surface_tangent_jitter
            return base_pos + outward * offset + tangent

        offset_value = 0.5 * (
            self.query_surface_min_offset + self.query_surface_max_offset
        )
        offset = torch.full(
            (base_pos.size(0), 1),
            offset_value,
            device=base_pos.device,
            dtype=base_pos.dtype,
        )
        return base_pos + outward * offset

    def _eval_query_position_cache_key(
        self,
        *,
        sample_id: str,
        num_points: int,
        apply_offset: bool,
        sample_surface_pos: Tensor,
        sample_host_pos: Tensor,
    ) -> tuple:
        return (
            sample_id,
            int(num_points),
            bool(apply_offset),
            int(sample_surface_pos.size(0)),
            int(sample_host_pos.size(0)),
            round(self.query_surface_min_offset, 6),
            round(self.query_surface_max_offset, 6),
            int(self.query_surface_normal_neighbors),
        )

    def _sample_surface_positions(
        self,
        batch,
        num_points: int,
        apply_offset: bool,
        sampling_mode: str = "fps",
        randomize_offset: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        sampling_mode = str(sampling_mode).lower()
        if sampling_mode not in self._EVAL_QUERY_SAMPLING_MODES:
            raise ValueError(
                f"Unsupported surface query sampling mode: {sampling_mode}"
            )
        if randomize_offset is None:
            randomize_offset = sampling_mode == "random"
        host_pos = batch.pos
        host_batch = batch.batch
        if (
            not hasattr(batch, "surface_pos")
            or not hasattr(batch, "surface_pos_batch")
            or not hasattr(batch, "surface_depth")
        ):
            raise AttributeError(
                "Surface query sampling requires batch.surface_pos, batch.surface_pos_batch, and batch.surface_depth."
            )
        source_pos = batch.surface_pos
        source_batch = batch.surface_pos_batch
        source_depth = batch.surface_depth
        sampled_pos_chunks = []
        sampled_batch_chunks = []
        sample_ids = self._batch_sample_ids(batch)
        use_eval_cache = self.eval_query_position_cache and sampling_mode == "fps"

        for sample_idx in host_batch.unique(sorted=True).tolist():
            host_sel = host_batch == sample_idx
            source_sel = source_batch == sample_idx
            sample_pos = host_pos[host_sel]
            sample_source_pos = source_pos[source_sel]
            if sample_source_pos.size(0) == 0:
                raise ValueError(f"Sample {sample_idx} has no surface_pos entries.")

            if source_depth.size(0) != source_pos.size(0):
                raise ValueError(
                    f"surface_depth length mismatch: {source_depth.size(0)} vs surface_pos {source_pos.size(0)}."
                )
            sample_depth = source_depth[source_sel].to(device=sample_source_pos.device)
            surface_sel = torch.isfinite(sample_depth) & (sample_depth > 0)
            if not surface_sel.any():
                raise ValueError(
                    f"Sample {sample_idx} has no valid surface points after surface_depth filtering."
                )
            sample_surface_pos = sample_source_pos[surface_sel]

            if sample_pos.size(0) == 0:
                raise ValueError(
                    f"Sample {sample_idx} has no host positions for centroid calculation."
                )
            centroid = sample_pos.mean(dim=0, keepdim=True)
            sample_id = (
                sample_ids[sample_idx]
                if sample_idx < len(sample_ids)
                else str(sample_idx)
            )
            cache_key = None
            if use_eval_cache:
                cache_key = self._eval_query_position_cache_key(
                    sample_id=sample_id,
                    num_points=num_points,
                    apply_offset=apply_offset,
                    sample_surface_pos=sample_surface_pos,
                    sample_host_pos=sample_pos,
                )
                cached_pos = self._eval_query_position_cache.get(cache_key)
                if cached_pos is not None:
                    sampled_pos = cached_pos.to(
                        device=sample_pos.device, dtype=sample_pos.dtype
                    )
                    sampled_pos_chunks.append(sampled_pos)
                    sampled_batch_chunks.append(
                        torch.full(
                            (num_points,),
                            sample_idx,
                            dtype=torch.long,
                            device=sample_pos.device,
                        )
                    )
                    continue

            if sampling_mode == "fps":
                sample_idx_local = self._farthest_point_indices(
                    sample_surface_pos, num_points
                )
            elif sampling_mode == "random":
                sample_idx_local = torch.randint(
                    low=0,
                    high=sample_surface_pos.size(0),
                    size=(num_points,),
                    device=sample_surface_pos.device,
                )
            elif sampling_mode == "curvature":
                sample_idx_local = self._curvature_weighted_indices(
                    sample_surface_pos, num_points
                )
            else:
                raise ValueError(
                    f"Unsupported surface query sampling mode: {sampling_mode}"
                )
            sampled_pos = sample_surface_pos[sample_idx_local]
            if apply_offset:
                sampled_pos = self._apply_surface_query_offset(
                    surface_pos=sample_surface_pos,
                    surface_idx=sample_idx_local,
                    base_pos=sampled_pos,
                    sample_centroid=centroid,
                    randomize=bool(randomize_offset),
                )

            if cache_key is not None:
                self._eval_query_position_cache[cache_key] = (
                    sampled_pos.detach().float().cpu()
                )
            sampled_pos_chunks.append(sampled_pos)
            sampled_batch_chunks.append(
                torch.full(
                    (num_points,),
                    sample_idx,
                    dtype=torch.long,
                    device=sample_pos.device,
                )
            )

        if not sampled_pos_chunks:
            raise ValueError("Surface query sampling produced no query positions.")

        return torch.cat(sampled_pos_chunks, dim=0), torch.cat(
            sampled_batch_chunks, dim=0
        )

    def _jitter_query_base_positions(
        self, base_pos: Tensor, sample_reference_pos: Tensor
    ) -> Tensor:
        if base_pos.numel() == 0:
            return base_pos
        if sample_reference_pos.numel() == 0:
            raise ValueError(
                "Cannot jitter query positions without reference positions."
            )
        centroid = sample_reference_pos.mean(dim=0, keepdim=True)
        normals = base_pos - centroid
        normal_norm = normals.norm(dim=-1, keepdim=True)
        degenerate = normal_norm.squeeze(-1) < 1e-8
        if degenerate.any():
            raise ValueError(
                "Cannot jitter query positions with degenerate radial normals."
            )
        normals = self._normalize_vectors(normals)

        offset_span = max(
            self.query_surface_max_offset - self.query_surface_min_offset, 0.0
        )
        offset = self.query_surface_min_offset
        if offset_span > 0:
            offset = (
                offset
                + torch.rand(
                    (base_pos.size(0), 1), device=base_pos.device, dtype=base_pos.dtype
                )
                * offset_span
            )
        else:
            offset = torch.full(
                (base_pos.size(0), 1),
                offset,
                device=base_pos.device,
                dtype=base_pos.dtype,
            )

        tangent = torch.randn_like(base_pos)
        tangent = tangent - (tangent * normals).sum(dim=-1, keepdim=True) * normals
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        valid_tangent = tangent_norm.squeeze(-1) >= 1e-8
        if not valid_tangent.all():
            raise ValueError(
                "Cannot jitter query positions with degenerate tangent directions."
            )
        tangent = tangent / tangent_norm.clamp(min=1e-8)
        tangent_scale = torch.randn(
            (base_pos.size(0), 1), device=base_pos.device, dtype=base_pos.dtype
        )
        tangent = tangent * tangent_scale * self.query_surface_tangent_jitter
        return base_pos + normals * offset + tangent

    def _sample_random_query_positions_from_pool(
        self,
        pool_pos: Tensor,
        sample_reference_pos: Tensor,
        num_points: int,
    ) -> Tensor:
        if num_points <= 0:
            return pool_pos.new_zeros((0, 3))
        if pool_pos.size(0) == 0:
            raise ValueError("Cannot sample query positions from an empty pool.")
        if sample_reference_pos.size(0) == 0:
            raise ValueError(
                "Cannot sample query positions without reference positions."
            )
        choice = torch.randint(
            0, pool_pos.size(0), (num_points,), device=pool_pos.device
        )
        return self._jitter_query_base_positions(
            pool_pos[choice], sample_reference_pos=sample_reference_pos
        )

    def _sample_site_balanced_query_positions(
        self, batch
    ) -> tuple[Tensor, Tensor, Tensor]:
        host_pos = batch.pos
        host_batch = batch.batch
        required_mask_fields = (
            "target_mask_site_id",
            "target_mask_host_id",
            "target_mask_host_id_batch",
        )
        missing_mask_fields = [
            field for field in required_mask_fields if not hasattr(batch, field)
        ]
        if missing_mask_fields:
            raise AttributeError(
                f"Site-balanced query sampling requires {', '.join(missing_mask_fields)}."
            )
        target_mask_site_id = batch.target_mask_site_id
        target_mask_host_id = batch.target_mask_host_id
        target_mask_batch = batch.target_mask_host_id_batch
        if target_mask_site_id.size(0) != target_mask_host_id.size(0):
            raise ValueError(
                f"target_mask_site_id length mismatch: {target_mask_site_id.size(0)} vs target_mask_host_id {target_mask_host_id.size(0)}."
            )
        if target_mask_batch.size(0) != target_mask_host_id.size(0):
            raise ValueError(
                f"target_mask_host_id_batch length mismatch: {target_mask_batch.size(0)} vs target_mask_host_id {target_mask_host_id.size(0)}."
            )

        pos_chunks = []
        batch_chunks = []
        site_id_chunks = []
        positive_fraction = 0.75

        for sample_idx in host_batch.unique(sorted=True).tolist():
            host_idx = (host_batch == sample_idx).nonzero(as_tuple=False).view(-1)
            sample_host_pos = host_pos[host_idx]
            if sample_host_pos.size(0) == 0:
                raise ValueError(
                    f"Sample {sample_idx} has no host positions for site-balanced query sampling."
                )

            mask_sel = target_mask_batch == sample_idx
            sample_site_ids = torch.unique(target_mask_site_id[mask_sel], sorted=True)
            sample_site_ids = sample_site_ids[sample_site_ids >= 0]
            num_sites = int(sample_site_ids.numel())
            if num_sites == 0:
                raise ValueError(
                    f"Sample {sample_idx} has no target sites for site-balanced query sampling."
                )

            sample_points = []
            sample_site_targets = []
            positive_budget = (
                int(round(self.num_query_nodes * positive_fraction))
                if num_sites > 0
                else 0
            )
            positive_budget = (
                min(max(positive_budget, num_sites), self.num_query_nodes)
                if num_sites > 0
                else 0
            )

            if num_sites > 0 and positive_budget > 0:
                counts = torch.full(
                    (num_sites,),
                    positive_budget // num_sites,
                    dtype=torch.long,
                    device=host_pos.device,
                )
                remainder = positive_budget - int(counts.sum().item())
                if remainder > 0:
                    perm = torch.randperm(num_sites, device=host_pos.device)[:remainder]
                    counts[perm] += 1

                for site_offset, site_id in enumerate(sample_site_ids.tolist()):
                    count = int(counts[site_offset].item())
                    if count <= 0:
                        continue
                    site_mask = mask_sel & (target_mask_site_id == site_id)
                    local_host_ids = target_mask_host_id[site_mask]
                    valid = (local_host_ids >= 0) & (local_host_ids < host_idx.size(0))
                    local_host_ids = torch.unique(local_host_ids[valid], sorted=True)
                    if local_host_ids.numel() == 0:
                        raise ValueError(
                            f"Sample {sample_idx} site {site_id} has no valid target mask host ids."
                        )
                    pool = sample_host_pos[local_host_ids]
                    points = self._sample_random_query_positions_from_pool(
                        pool_pos=pool,
                        sample_reference_pos=sample_host_pos,
                        num_points=count,
                    )
                    sample_points.append(points)
                    sample_site_targets.append(
                        torch.full(
                            (points.size(0),),
                            site_id,
                            dtype=torch.long,
                            device=host_pos.device,
                        )
                    )

            remaining = self.num_query_nodes - sum(
                points.size(0) for points in sample_points
            )
            if remaining > 0:
                occupied_local = target_mask_host_id[mask_sel]
                valid_occupied = (occupied_local >= 0) & (
                    occupied_local < host_idx.size(0)
                )
                occupied_local = torch.unique(
                    occupied_local[valid_occupied], sorted=True
                )
                if occupied_local.numel() == 0:
                    raise ValueError(
                        f"Sample {sample_idx} has no occupied target mask host ids for background exclusion."
                    )
                keep = torch.ones(
                    host_idx.size(0), dtype=torch.bool, device=host_pos.device
                )
                keep[occupied_local] = False
                background_pool = sample_host_pos[keep]
                if background_pool.size(0) == 0:
                    raise ValueError(
                        f"Sample {sample_idx} has no background host positions for query sampling."
                    )
                points = self._sample_random_query_positions_from_pool(
                    pool_pos=background_pool,
                    sample_reference_pos=sample_host_pos,
                    num_points=remaining,
                )
                sample_points.append(points)
                sample_site_targets.append(
                    torch.full(
                        (points.size(0),), -1, dtype=torch.long, device=host_pos.device
                    )
                )

            if not sample_points:
                raise ValueError(
                    f"Sample {sample_idx} produced no site-balanced query positions."
                )
            sample_query_pos = torch.cat(sample_points, dim=0)
            sample_query_site_ids = torch.cat(sample_site_targets, dim=0)
            if sample_query_pos.size(0) > self.num_query_nodes:
                sample_query_pos = sample_query_pos[: self.num_query_nodes]
                sample_query_site_ids = sample_query_site_ids[: self.num_query_nodes]
            pos_chunks.append(sample_query_pos)
            batch_chunks.append(
                torch.full(
                    (sample_query_pos.size(0),),
                    sample_idx,
                    dtype=torch.long,
                    device=host_pos.device,
                )
            )
            site_id_chunks.append(sample_query_site_ids)

        if not pos_chunks:
            raise ValueError(
                "Site-balanced query sampling produced no query positions."
            )

        return (
            torch.cat(pos_chunks, dim=0),
            torch.cat(batch_chunks, dim=0),
            torch.cat(site_id_chunks, dim=0),
        )

    def _sample_query_positions(
        self, batch, training: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.query_sampling == "atom_volume":
            return self._sample_atom_volume_positions(batch, training)
        sampling_mode = (
            "random" if training else str(self.eval_query_sampling_mode).lower()
        )
        if sampling_mode not in self._EVAL_QUERY_SAMPLING_MODES:
            raise ValueError(f"Unsupported eval_query_sampling_mode: {sampling_mode}")
        num_query_nodes = (
            self.num_query_nodes if training else self.eval_num_query_nodes
        )
        query_pos, query_batch = self._sample_surface_positions(
            batch=batch,
            num_points=num_query_nodes,
            apply_offset=True,
            sampling_mode=sampling_mode,
            randomize_offset=training,
        )
        query_site_ids = torch.full(
            (query_pos.size(0),), -1, dtype=torch.long, device=query_pos.device
        )
        return query_pos, query_batch, query_site_ids

    def _sample_atom_volume_positions(self, batch, training: bool) -> tuple[Tensor, Tensor, Tensor]:
        if not hasattr(batch, "protein_atom_pos") or not hasattr(batch, "protein_atom_pos_batch"):
            raise AttributeError("Atom-volume sampling requires full protein_atom_pos and its batch vector.")
        sample_ids = self._batch_sample_ids(batch)
        count = self.num_query_nodes if training else self.eval_num_query_nodes
        positions, batches = [], []
        epoch = int(self.current_epoch) if training else None
        for sample in batch.batch.unique(sorted=True).tolist():
            if sample >= len(sample_ids):
                raise ValueError("Missing sample id for deterministic atom-volume sampling.")
            atoms = batch.protein_atom_pos[batch.protein_atom_pos_batch == sample]
            generator = torch.Generator(device=atoms.device)
            generator.manual_seed(sample_seed(self.query_sampling_seed, sample_ids[sample], epoch))
            points = sample_atom_volume(atoms, count, self.query_volume_radius, generator)
            positions.append(points)
            batches.append(torch.full((count,),sample,dtype=torch.long,device=points.device))
        if not positions:
            raise ValueError("Atom-volume sampling received no proteins.")
        query_pos, query_batch = torch.cat(positions), torch.cat(batches)
        return query_pos, query_batch, torch.full_like(query_batch, -1)

    def _build_query_features(self, batch, query_batch: Tensor | None = None) -> Tensor:
        host_x = batch.x
        host_batch = batch.batch
        shared_query_embed = self.shared_query_embed.to(
            device=host_x.device, dtype=host_x.dtype
        )
        if query_batch is not None:
            if query_batch.numel() == 0:
                return host_x.new_zeros((0, host_x.size(-1)))
            return shared_query_embed.expand(query_batch.size(0), -1)
        query_features = [
            shared_query_embed.expand(self.num_query_nodes, -1)
            for _ in host_batch.unique(sorted=True).tolist()
        ]
        if not query_features:
            return host_x.new_zeros((0, host_x.size(-1)))
        return torch.cat(query_features, dim=0)

    @staticmethod
    def _constant_query_times(query_pos: Tensor, value: float) -> Tensor:
        return torch.full(
            (query_pos.size(0), 1),
            value,
            device=query_pos.device,
            dtype=query_pos.dtype,
        )

    def _build_query_position_targets(
        self,
        batch,
        query_pos: Tensor,
        query_batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            query_pos.numel() == 0
            or batch.target_pos.numel() == 0
            or batch.ligand_pos.numel() == 0
        ):
            empty_ids = torch.full(
                (query_pos.size(0),), -1, dtype=torch.long, device=query_pos.device
            )
            empty_mask = torch.zeros(
                (query_pos.size(0),), dtype=torch.bool, device=query_pos.device
            )
            return query_pos.clone(), empty_ids, empty_mask

        target_ids = getattr(batch, "target_id", None)
        if target_ids is None or target_ids.size(0) != batch.target_pos.size(0):
            target_ids = torch.arange(
                batch.target_pos.size(0), device=query_pos.device, dtype=torch.long
            )
        else:
            target_ids = target_ids.to(device=query_pos.device, dtype=torch.long)

        target_pos, target_site_ids, has_target = self._assign_targets_per_sample(
            query_pos=query_pos,
            query_batch=query_batch,
            target_pos=batch.target_pos,
            target_batch=batch.target_pos_batch,
            target_id=target_ids,
        )
        near_ligand = (
            self._nearest_ligand_distances_for_points(query_pos, query_batch, batch)
            < self.query_disp_supervision_cutoff
        )
        supervise = has_target & near_ligand
        return target_pos, target_site_ids, supervise

    @staticmethod
    def _build_query_displacement_targets(
        query_pos_0: Tensor,
        target_pos: Tensor,
    ) -> Tensor:
        return target_pos - query_pos_0

    def _nearest_ligand_distances_for_points(
        self, points: Tensor, point_batch: Tensor, batch
    ) -> Tensor:
        return self._nearest_target_distances(
            points=points,
            point_batch=point_batch,
            target_pos=batch.ligand_pos,
            target_batch=batch.ligand_pos_batch,
        )

    @staticmethod
    def _apply_query_displacement(
        query_pos: Tensor, pred_disp: Tensor
    ) -> tuple[Tensor, Tensor]:
        disp = torch.nan_to_num(pred_disp, nan=0.0, posinf=0.0, neginf=0.0)
        final_pos = query_pos + disp
        invalid = ~torch.isfinite(final_pos).all(dim=-1)
        if invalid.any():
            final_pos[invalid] = query_pos[invalid]
            disp[invalid] = 0.0
        return final_pos, disp
