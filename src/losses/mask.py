from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from .detr import DetrMaskLossMixin


class SiteLossMixin(DetrMaskLossMixin):
    def _empty_site_detection_losses(self, reference: Tensor) -> dict[str, Tensor]:
        zero = reference.sum() * 0.0
        return {
            "site_detection_loss": zero,
            "site_cls_loss": zero,
            "site_mask_loss": zero,
            "site_dice_loss": zero,
            "site_contrastive_loss": zero,
            "site_affinity_loss": zero,
        }

    def _build_site_detection_targets(self, batch) -> list[dict[str, Tensor]]:
        device = batch.x.device
        dtype = batch.x.dtype
        target_mask_site_id = getattr(
            batch,
            "target_mask_site_id",
            torch.empty(0, dtype=torch.long, device=device),
        )
        target_mask_host_id = getattr(
            batch,
            "target_mask_host_id",
            torch.empty(0, dtype=torch.long, device=device),
        )
        target_mask_batch = getattr(
            batch,
            "target_mask_host_id_batch",
            torch.zeros(target_mask_host_id.size(0), dtype=torch.long, device=device),
        )
        targets: list[dict[str, Tensor]] = []
        for sample_idx in batch.batch.unique(sorted=True).tolist():
            host_sel = batch.batch == sample_idx
            num_host = int(host_sel.sum().item())
            target_sel = batch.target_pos_batch == sample_idx
            num_targets = int(target_sel.sum().item())
            labels = torch.zeros((num_targets,), dtype=torch.long, device=device)
            masks = torch.zeros((num_targets, num_host), dtype=dtype, device=device)
            if num_targets > 0 and num_host > 0 and (target_mask_site_id.numel() > 0):
                mask_sel = target_mask_batch == sample_idx
                site_ids = target_mask_site_id[mask_sel].long()
                host_ids = target_mask_host_id[mask_sel].long()
                valid = (
                    (site_ids >= 0)
                    & (site_ids < num_targets)
                    & (host_ids >= 0)
                    & (host_ids < num_host)
                )
                if valid.any():
                    masks[site_ids[valid], host_ids[valid]] = 1.0
            has_mask = masks.sum(dim=-1) > 0
            targets.append(
                {
                    "labels": labels[has_mask],
                    "pocket_masks": masks[has_mask],
                    "res_mask": torch.ones(
                        (num_host,), dtype=torch.bool, device=device
                    ),
                }
            )
        return targets

    def _site_detector_loss_type(self) -> str:
        site_detector = getattr(self.model, "site_detector", None)
        if site_detector is not None:
            return str(getattr(site_detector, "loss_type", "detr"))
        return str(getattr(self.model, "site_detector_type", "detr"))

    @staticmethod
    def _target_ids_for_batch(batch) -> Tensor:
        target_ids = getattr(batch, "target_id", None)
        if target_ids is None or target_ids.size(0) != batch.target_pos.size(0):
            return torch.arange(
                batch.target_pos.size(0),
                dtype=torch.long,
                device=batch.target_pos.device,
            )
        return target_ids.to(device=batch.target_pos.device, dtype=torch.long)

    def _dense_site_masks_tensor(
        self, batch, sample_idx: int, num_targets: int, num_host: int
    ) -> Tensor:
        masks = batch.x.new_zeros((num_targets, num_host))
        if num_targets == 0 or num_host == 0:
            return masks
        target_mask_site_id = getattr(
            batch,
            "target_mask_site_id",
            torch.empty(0, dtype=torch.long, device=batch.x.device),
        )
        target_mask_host_id = getattr(
            batch,
            "target_mask_host_id",
            torch.empty(0, dtype=torch.long, device=batch.x.device),
        )
        target_mask_batch = getattr(
            batch,
            "target_mask_host_id_batch",
            torch.zeros(
                target_mask_host_id.size(0), dtype=torch.long, device=batch.x.device
            ),
        )
        if target_mask_site_id.numel() == 0 or target_mask_host_id.numel() == 0:
            return masks
        mask_sel = target_mask_batch == sample_idx
        site_ids = target_mask_site_id[mask_sel].long()
        host_ids = target_mask_host_id[mask_sel].long()
        valid = (
            (site_ids >= 0)
            & (site_ids < num_targets)
            & (host_ids >= 0)
            & (host_ids < num_host)
        )
        if valid.any():
            masks[site_ids[valid], host_ids[valid]] = 1.0
        return masks

    def _vn_site_mask_losses(
        self,
        out: dict[str, Tensor],
        batch,
        query_batch: Tensor | None,
        query_site_ids: Tensor | None,
        query_supervise: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> dict[str, Tensor]:
        reference = out.get("query_distance_logits", out["host_logits"])
        if query_batch is None or query_site_ids is None:
            raise ValueError(
                "Mask supervision requires query batch and target site IDs."
            )
        host_scalar = out.get("site_host_scalar", out.get("host_scalar"))
        query_scalar = out.get("query_scalar")
        if host_scalar is None or query_scalar is None:
            raise RuntimeError(
                "VN-dot site-mask loss requires host_scalar and query_scalar in model outputs."
            )
        site_detector = getattr(self.model, "site_detector", None)
        if site_detector is None or not hasattr(site_detector, "sample_mask_logits"):
            raise RuntimeError(
                "VN-dot site-mask loss requires VNDirectSiteMaskHead.sample_mask_logits."
            )
        site_query_mask = out.get("site_query_mask")
        site_sample_ids = out.get("site_sample_ids", batch.batch.unique(sorted=True))
        target_ids = self._target_ids_for_batch(batch).to(device=batch.x.device)
        affinity_loss_weight = float(
            getattr(self, "site_mask_affinity_loss_weight", 0.0)
        )
        site_affinity_loss = reference.sum() * 0.0
        if affinity_loss_weight != 0.0:
            raise ValueError("Query affinity is not part of the production objective.")
        mask_loss_total = host_scalar.new_zeros(())
        dice_loss_total = host_scalar.new_zeros(())
        matched_query_count = 0
        for out_idx, sample_idx in enumerate(site_sample_ids.tolist()):
            sample_idx = int(sample_idx)
            host_sel = batch.batch == sample_idx
            query_sel = query_batch == sample_idx
            target_sel = batch.target_pos_batch == sample_idx
            num_host = int(host_sel.sum().item())
            num_targets = int(target_sel.sum().item())
            if not query_sel.any() or num_host == 0 or num_targets == 0:
                continue
            query_indices = query_sel.nonzero(as_tuple=False).view(-1)
            num_pred = int(query_indices.numel())
            if site_query_mask is not None:
                num_pred = min(num_pred, int(site_query_mask.size(1)))
            if num_pred == 0:
                continue
            query_indices = query_indices[:num_pred]
            valid_queries = torch.ones(
                (num_pred,), dtype=torch.bool, device=query_indices.device
            )
            if site_query_mask is not None:
                valid_queries &= site_query_mask[out_idx, :num_pred].to(
                    device=query_indices.device, dtype=torch.bool
                )
            sample_query_site_ids = query_site_ids[query_indices].to(
                device=query_indices.device, dtype=torch.long
            )
            sample_target_ids = target_ids[target_sel].to(
                device=query_indices.device, dtype=torch.long
            )
            target_masks = self._dense_site_masks_tensor(
                batch=batch,
                sample_idx=sample_idx,
                num_targets=num_targets,
                num_host=num_host,
            ).to(device=host_scalar.device, dtype=host_scalar.dtype)
            site_match = sample_query_site_ids.view(-1, 1) == sample_target_ids.view(
                1, -1
            )
            has_target_mask = target_masks.sum(dim=-1) > 0
            site_match = site_match & has_target_mask.to(device=site_match.device).view(
                1, -1
            )
            valid_queries &= sample_query_site_ids >= 0
            if query_supervise is not None:
                sample_query_supervise = query_supervise[query_indices].to(
                    device=query_indices.device, dtype=torch.bool
                )
                valid_queries &= sample_query_supervise
            valid_queries &= site_match.any(dim=1)
            target_rows = site_match.to(dtype=torch.long).argmax(dim=1)
            if not valid_queries.any():
                continue
            selected_query_indices = query_indices[valid_queries]
            selected_query_scalar = query_scalar[selected_query_indices]
            selected_query_pos = None
            if query_pos is not None:
                selected_query_pos = query_pos[selected_query_indices].to(
                    device=host_scalar.device
                )
            selected_pred_masks = site_detector.sample_mask_logits(
                host_scalar=host_scalar[host_sel],
                query_scalar=selected_query_scalar,
                host_pos=batch.pos[host_sel].to(device=host_scalar.device),
                query_pos=selected_query_pos,
            )
            selected_target_masks = target_masks[target_rows[valid_queries]]
            sample_count = int(selected_pred_masks.size(0))
            mask_loss_total = mask_loss_total + self._site_mask_loss(
                selected_pred_masks, selected_target_masks
            ) * float(sample_count)
            dice_loss_total = dice_loss_total + self._site_dice_loss(
                selected_pred_masks, selected_target_masks
            ) * float(sample_count)
            matched_query_count += sample_count
        if matched_query_count == 0:
            site_mask_loss = reference.sum() * 0.0
            site_dice_loss = reference.sum() * 0.0
        else:
            site_mask_loss = mask_loss_total / float(matched_query_count)
            site_dice_loss = dice_loss_total / float(matched_query_count)
        site_cls_loss = site_mask_loss * 0.0
        site_detection_loss = (
            site_mask_loss + site_dice_loss + site_affinity_loss * affinity_loss_weight
        )
        return {
            "site_detection_loss": site_detection_loss,
            "site_cls_loss": site_cls_loss,
            "site_mask_loss": site_mask_loss,
            "site_dice_loss": site_dice_loss,
            "site_contrastive_loss": site_mask_loss * 0.0,
            "site_affinity_loss": site_affinity_loss,
        }

    def _decode_query_distance_logits(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        if logits.numel() == 0:
            empty = logits.new_zeros((0,))
            return (empty, empty)
        pred_field = torch.softmax(logits, dim=-1)
        support = self._query_distance_support(logits)
        pred_distance = torch.sum(pred_field * support.unsqueeze(0), dim=-1)
        rank_scores = torch.exp(-pred_distance / max(self.query_distance_max, 1e-08))
        return (pred_distance, rank_scores)

    def _query_rank_scores_from_out(self, out: dict[str, Tensor]) -> Tensor:
        if self.query_ranking_mode == "confidence":
            confidence = out.get("query_conf_logits")
            if confidence is None:
                return out["host_logits"].new_zeros((0,))
            return confidence.reshape(-1)
        logits = out.get("query_distance_logits")
        if logits is None:
            return out["host_logits"].new_zeros((0,))
        _, rank_scores = self._decode_query_distance_logits(logits)
        return rank_scores

    def _site_detection_losses(
        self,
        out,
        batch,
        *,
        query_batch=None,
        query_site_ids=None,
        query_supervise=None,
        query_pos=None,
    ):
        if not self._loss_is_enabled("site_detection"):
            return self._empty_site_detection_losses(out["host_logits"])
        if self._site_detector_loss_type() == "detr":
            return self._detr_mask_losses(out, batch)
        if self._site_detector_loss_type() != "per_vn_mask":
            raise ValueError(
                "The production site objective requires the residue mask head."
            )
        if self.site_mask_affinity_loss_weight != 0:
            raise ValueError(
                "The production objective does not include query-query affinity."
            )
        return self._vn_site_mask_losses(
            out,
            batch,
            query_batch=query_batch,
            query_site_ids=query_site_ids,
            query_supervise=query_supervise,
            query_pos=query_pos,
        )

    @staticmethod
    def _site_mask_loss(
        pred_mask_logits: Tensor,
        target_masks: Tensor,
        num_masks: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if pred_mask_logits.numel() == 0:
            return pred_mask_logits.sum() * 0.0
        if pred_mask_logits.shape != target_masks.shape:
            raise ValueError(
                f"site mask loss shape mismatch: {tuple(pred_mask_logits.shape)} vs {tuple(target_masks.shape)}."
            )
        per_residue = F.binary_cross_entropy_with_logits(
            pred_mask_logits, target_masks, reduction="none"
        )
        if valid_mask is None:
            per_mask = per_residue.mean(dim=-1)
        else:
            if valid_mask.shape != pred_mask_logits.shape:
                raise ValueError(
                    "site mask valid_mask shape mismatch: "
                    f"expected {tuple(pred_mask_logits.shape)}, got {tuple(valid_mask.shape)}."
                )
            valid = valid_mask.to(device=pred_mask_logits.device, dtype=torch.bool)
            valid_count = valid.sum(dim=-1)
            if not bool((valid_count > 0).all().item()):
                raise RuntimeError(
                    "site mask loss received a selected mask row with no valid residue pairs."
                )
            per_mask = (per_residue * valid.to(dtype=per_residue.dtype)).sum(
                dim=-1
            ) / valid_count.to(dtype=per_residue.dtype)
        if num_masks is not None:
            normalizer = num_masks.to(
                device=pred_mask_logits.device, dtype=per_mask.dtype
            )
            return per_mask.sum() / normalizer
        return per_mask.mean()

    @staticmethod
    def _site_dice_loss(
        pred_mask_logits: Tensor,
        target_masks: Tensor,
        num_masks: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if pred_mask_logits.numel() == 0:
            return pred_mask_logits.sum() * 0.0
        if pred_mask_logits.shape != target_masks.shape:
            raise ValueError(
                f"site dice loss shape mismatch: {tuple(pred_mask_logits.shape)} vs {tuple(target_masks.shape)}."
            )
        pred = pred_mask_logits.sigmoid()
        if valid_mask is None:
            valid_pred = pred
            valid_target = target_masks
        else:
            if valid_mask.shape != pred_mask_logits.shape:
                raise ValueError(
                    "site dice valid_mask shape mismatch: "
                    f"expected {tuple(pred_mask_logits.shape)}, got {tuple(valid_mask.shape)}."
                )
            valid = valid_mask.to(device=pred_mask_logits.device, dtype=torch.bool)
            valid_count = valid.sum(dim=-1)
            if not bool((valid_count > 0).all().item()):
                raise RuntimeError(
                    "site dice loss received a selected mask row with no valid residue pairs."
                )
            valid_weight = valid.to(dtype=pred.dtype)
            valid_pred = pred * valid_weight
            valid_target = target_masks * valid_weight
        numerator = 2.0 * (valid_pred * valid_target).sum(dim=-1)
        denominator = valid_pred.sum(dim=-1) + valid_target.sum(dim=-1)
        per_mask = 1.0 - (numerator + 1.0) / (denominator + 1.0)
        if num_masks is not None:
            normalizer = num_masks.to(
                device=pred_mask_logits.device, dtype=per_mask.dtype
            )
            return per_mask.sum() / normalizer
        return per_mask.mean()
