from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import Tensor


class QueryLossMixin:
    def _nearest_target_distances(
        self,
        points: Tensor,
        point_batch: Tensor,
        target_pos: Tensor,
        target_batch: Tensor,
        fill_value: float = 99.0,
    ) -> Tensor:
        distances = torch.full(
            (points.size(0),),
            fill_value,
            dtype=points.dtype,
            device=points.device,
        )
        for sample_idx in point_batch.unique(sorted=True).tolist():
            point_sel = point_batch == sample_idx
            target_sel = target_batch == sample_idx
            if not point_sel.any() or not target_sel.any():
                continue
            sample_points = points[point_sel]
            sample_targets = target_pos[target_sel]
            valid_points = torch.isfinite(sample_points).all(dim=-1)
            valid_targets = torch.isfinite(sample_targets).all(dim=-1)
            if not valid_points.any() or not valid_targets.any():
                continue

            point_indices = point_sel.nonzero(as_tuple=False).view(-1)[valid_points]
            dist = torch.cdist(
                sample_points[valid_points], sample_targets[valid_targets]
            )
            distances[point_indices] = dist.min(dim=-1).values
        return distances

    def _query_distance_support(self, reference: Tensor) -> Tensor:
        if self.query_distance_num_bins <= 1:
            return torch.zeros((1,), device=reference.device, dtype=reference.dtype)
        return (
            torch.arange(
                self.query_distance_num_bins,
                device=reference.device,
                dtype=reference.dtype,
            )
            * self.query_distance_bin_size
        )

    def _assign_query_site_centers(
        self,
        query_pos: Tensor,
        query_batch: Tensor,
        batch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if query_pos.numel() == 0 or batch.target_pos.numel() == 0:
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

        return self._assign_targets_per_sample(
            query_pos=query_pos,
            query_batch=query_batch,
            target_pos=batch.target_pos,
            target_batch=batch.target_pos_batch,
            target_id=target_ids,
        )

    def _query_distance_target_distribution(
        self,
        points: Tensor,
        point_batch: Tensor,
        batch,
    ) -> Tensor:
        targets = points.new_zeros((points.size(0), self.query_distance_num_bins))
        if targets.numel() == 0 or points.numel() == 0 or batch.target_pos.numel() == 0:
            return targets

        center_pos, center_site_ids, has_target = self._assign_query_site_centers(
            points, point_batch, batch
        )
        valid = (
            has_target
            & (center_site_ids >= 0)
            & torch.isfinite(points).all(dim=-1)
            & torch.isfinite(center_pos).all(dim=-1)
        )
        if not valid.any():
            return targets

        distances = torch.norm(center_pos[valid] - points[valid], dim=-1).clamp(
            min=0.0, max=self.query_distance_max
        )
        scaled = distances / self.query_distance_bin_size
        left = (
            torch.floor(scaled)
            .long()
            .clamp(min=0, max=self.query_distance_num_bins - 1)
        )
        right = torch.clamp(left + 1, max=self.query_distance_num_bins - 1)
        right_w = scaled - left.to(dtype=scaled.dtype)
        left_w = 1.0 - right_w
        valid_idx = valid.nonzero(as_tuple=False).view(-1)

        targets[valid_idx, left] = left_w.to(dtype=targets.dtype)
        same_bin = right == left
        if same_bin.any():
            targets[valid_idx[same_bin], right[same_bin]] = 1.0
        diff_bin = ~same_bin
        if diff_bin.any():
            targets[valid_idx[diff_bin], right[diff_bin]] = right_w[diff_bin].to(
                dtype=targets.dtype
            )

        return targets

    def _query_distance_distribution_per_query_loss(
        self, logits: Tensor, target_dist: Tensor
    ) -> tuple[Tensor, Tensor]:
        if logits.numel() == 0:
            return logits.new_zeros((0,)), torch.zeros(
                (0,), dtype=torch.bool, device=logits.device
            )
        valid = target_dist.sum(dim=-1) > 0
        loss_dtype = torch.float32
        per_query = torch.zeros(
            (logits.size(0),), dtype=loss_dtype, device=logits.device
        )
        if valid.any():
            log_probs = F.log_softmax(logits.to(dtype=loss_dtype), dim=-1)
            target = target_dist.to(device=log_probs.device, dtype=loss_dtype)
            per_query[valid] = -(target[valid] * log_probs[valid]).sum(dim=-1)
        return per_query, valid

    def _query_distance_distribution_loss(
        self, logits: Tensor, target_dist: Tensor
    ) -> Tensor:
        per_query, valid = self._query_distance_distribution_per_query_loss(
            logits, target_dist
        )
        if not valid.any():
            return logits.sum() * 0.0
        return per_query[valid].mean()

    def _query_confidence_targets(
        self,
        points: Tensor,
        point_batch: Tensor,
        batch,
    ) -> tuple[Tensor, Tensor]:
        if points.numel() == 0:
            return points.new_zeros((0,)), torch.zeros(
                (0,), dtype=torch.bool, device=points.device
            )
        if not hasattr(batch, "target_pos") or batch.target_pos.numel() == 0:
            return points.new_zeros((points.size(0),)), torch.zeros(
                (points.size(0),), dtype=torch.bool, device=points.device
            )

        distances = self._nearest_target_distances(
            points=points,
            point_batch=point_batch,
            target_pos=batch.target_pos,
            target_batch=batch.target_pos_batch,
        )
        valid = torch.isfinite(distances) & (distances < 99.0)
        targets = distances.detach().clone()
        close = targets <= self.confidence_gamma
        targets[close] = 1.0 - targets[close] / (self.confidence_gamma * 2.0)
        targets[~close] = self.confidence_c0
        return targets, valid

    def _query_confidence_per_query_loss(
        self,
        confidence: Tensor,
        points: Tensor,
        point_batch: Tensor,
        batch,
    ) -> tuple[Tensor, Tensor]:
        confidence = confidence.reshape(-1)
        if confidence.numel() == 0:
            return confidence.new_zeros((0,)), torch.zeros(
                (0,), dtype=torch.bool, device=confidence.device
            )
        targets, valid = self._query_confidence_targets(points, point_batch, batch)
        loss_dtype = torch.float32
        confidence_for_loss = confidence.to(dtype=loss_dtype)
        targets = targets.to(device=confidence.device, dtype=loss_dtype)
        valid = valid.to(device=confidence.device)
        per_query = torch.zeros(
            (confidence.size(0),), dtype=loss_dtype, device=confidence.device
        )
        if valid.any():
            loss = F.mse_loss(
                confidence_for_loss[valid], targets[valid], reduction="none"
            )
            weights = torch.full_like(loss, self.confidence_negative_weight)
            positive = targets[valid] > (self.confidence_c0 + 1.0e-8)
            weights = torch.where(
                positive,
                torch.full_like(weights, self.confidence_positive_weight),
                weights,
            )
            per_query[valid] = loss * weights
        return per_query, valid

    def _query_ranking_per_query_loss(
        self,
        out: dict[str, Tensor],
        ranking_pos: Tensor,
        query_batch: Tensor,
        batch,
    ) -> tuple[Tensor, Tensor]:
        if self.query_ranking_mode == "confidence":
            confidence = out.get("query_conf_logits")
            if confidence is None:
                reference = out.get("query_distance_logits", out["host_logits"])
                return reference.new_zeros((ranking_pos.size(0),)), torch.zeros(
                    (ranking_pos.size(0),),
                    dtype=torch.bool,
                    device=reference.device,
                )
            return self._query_confidence_per_query_loss(
                confidence, ranking_pos, query_batch, batch
            )

        logits = out.get("query_distance_logits")
        if logits is None:
            reference = out.get("query_conf_logits", out["host_logits"])
            return reference.new_zeros((ranking_pos.size(0),)), torch.zeros(
                (ranking_pos.size(0),),
                dtype=torch.bool,
                device=reference.device,
            )
        query_distance_targets = self._query_distance_target_distribution(
            ranking_pos, query_batch, batch
        )
        return self._query_distance_distribution_per_query_loss(
            logits, query_distance_targets
        )

    def _reduce_query_loss_per_site(
        self,
        loss_per_query: Tensor,
        query_batch: Tensor,
        site_ids: Tensor,
        valid_mask: Tensor,
        *,
        include_background: bool,
    ) -> Tensor:
        if loss_per_query.numel() == 0:
            return loss_per_query.sum() * 0.0

        active = valid_mask & torch.isfinite(loss_per_query)
        if not active.any():
            return loss_per_query.sum() * 0.0

        total = loss_per_query.new_zeros(())
        num_groups = 0
        for sample_idx in query_batch[active].unique(sorted=True).tolist():
            sample_active = active & (query_batch == sample_idx)
            sample_site_ids = torch.unique(site_ids[sample_active], sorted=True)
            if not include_background:
                sample_site_ids = sample_site_ids[sample_site_ids >= 0]
            for site_id in sample_site_ids.tolist():
                group_sel = sample_active & (site_ids == site_id)
                if not group_sel.any():
                    continue
                total = total + loss_per_query[group_sel].mean()
                num_groups += 1

        if num_groups == 0:
            return loss_per_query.sum() * 0.0
        return total / float(num_groups)

    def _query_site_contrastive_loss(
        self,
        embeddings: Tensor,
        query_batch: Tensor,
        site_ids: Tensor,
        supervise: Tensor,
    ) -> Tensor:
        if embeddings.numel() == 0:
            return embeddings.sum() * 0.0

        labels = site_ids.clone()
        labels[~supervise] = -1
        embeddings = F.normalize(embeddings, dim=-1, eps=1e-6)
        temperature = self.query_contrastive_temperature
        total = embeddings.new_zeros(())
        contributing_query_count = 0

        for sample_idx in query_batch.unique(sorted=True).tolist():
            sample_sel = query_batch == sample_idx
            sample_embeddings = embeddings[sample_sel]
            sample_labels = labels[sample_sel]
            num_query = sample_embeddings.size(0)
            if num_query < 2:
                continue

            logits = (
                torch.matmul(sample_embeddings, sample_embeddings.transpose(0, 1))
                / temperature
            )
            logits = logits - logits.max(dim=-1, keepdim=True).values.detach()

            eye = torch.eye(
                num_query, dtype=torch.bool, device=sample_embeddings.device
            )
            positive_mask = (
                (sample_labels.view(-1, 1) == sample_labels.view(1, -1))
                & (sample_labels.view(-1, 1) >= 0)
                & (~eye)
            )
            if not positive_mask.any():
                continue

            denom_mask = ~eye
            exp_logits = torch.exp(logits) * denom_mask.to(dtype=logits.dtype)
            log_prob = logits - torch.log(
                exp_logits.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            )

            positive_count = positive_mask.sum(dim=-1)
            contributing_mask = positive_count > 0
            if not contributing_mask.any():
                continue

            mean_log_prob_pos = (positive_mask.to(dtype=logits.dtype) * log_prob).sum(
                dim=-1
            ) / positive_count.clamp(min=1).to(dtype=logits.dtype)
            total = total - mean_log_prob_pos[contributing_mask].sum()
            contributing_query_count += int(contributing_mask.sum().item())

        if contributing_query_count == 0:
            return embeddings.sum() * 0.0
        return total / float(contributing_query_count)
