"""UniSite-style Hungarian matching and DETR mask losses, restored from the formal source snapshot."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


class DetrMaskLossMixin:
    def _detr_mask_losses(self, out, batch):
        if self.model.site_detector.query_source != "learned":
            raise ValueError("This DETR objective requires learned mask queries.")
        targets = self._build_site_detection_targets(batch)
        if not targets:
            raise ValueError("DETR received an empty protein batch.")
        losses = self._detr_site_detection_losses_for_outputs(
            out["site_logits"], out["site_mask_logits"], out["site_query_mask"], targets
        )
        if self.model.site_detector.aux_loss:
            aux_outputs = out["site_aux_outputs"]
            if len(aux_outputs) != self.model.site_detector.num_decoder_layers - 1:
                raise ValueError("DETR auxiliary output count must match decoder depth.")
            for aux in aux_outputs:
                aux_losses = self._detr_site_detection_losses_for_outputs(
                    aux["site_logits"], aux["site_mask_logits"], out["site_query_mask"], targets
                )
                losses = tuple(a + b for a, b in zip(losses, aux_losses))
        cls, mask, dice = losses
        return {
            "site_detection_loss": cls + mask + dice,
            "site_cls_loss": cls,
            "site_mask_loss": mask,
            "site_dice_loss": dice,
            "site_contrastive_loss": cls * 0.0,
            "site_affinity_loss": cls * 0.0,
        }

    @staticmethod
    def _site_pairwise_mask_cost(pred_logits: Tensor, target_masks: Tensor) -> Tensor:
        if pred_logits.numel() == 0 or target_masks.numel() == 0:
            return pred_logits.new_zeros((pred_logits.size(0), target_masks.size(0)))
        pred = pred_logits.float()
        target = target_masks.float()
        positive_cost = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction="none")
        negative_cost = F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred), reduction="none")
        normalizer = float(max(pred.size(-1), 1))
        return (
            torch.einsum("qm,tm->qt", positive_cost, target)
            + torch.einsum("qm,tm->qt", negative_cost, 1.0 - target)
        ) / normalizer


    @staticmethod
    def _site_pairwise_dice_cost(pred_logits: Tensor, target_masks: Tensor) -> Tensor:
        if pred_logits.numel() == 0 or target_masks.numel() == 0:
            return pred_logits.new_zeros((pred_logits.size(0), target_masks.size(0)))
        pred = pred_logits.sigmoid().float()
        target = target_masks.float()
        numerator = 2.0 * torch.einsum("qm,tm->qt", pred, target)
        denominator = pred.sum(dim=-1, keepdim=True) + target.sum(dim=-1).unsqueeze(0)
        return 1.0 - (numerator + 1.0) / (denominator + 1.0)


    def _match_site_queries(
        self,
        pred_logits: Tensor,
        pred_mask_logits: Tensor,
        target_labels: Tensor,
        target_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if pred_logits.size(0) == 0 or target_labels.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=pred_logits.device)
            return empty, empty

        with torch.no_grad():
            class_prob = pred_logits.softmax(dim=-1)
            class_cost = -class_prob[:, target_labels]
            mask_cost = self._site_pairwise_mask_cost(pred_mask_logits, target_masks)
            dice_cost = self._site_pairwise_dice_cost(pred_mask_logits, target_masks)
            cost = class_cost + mask_cost + dice_cost
            row_idx, col_idx = linear_sum_assignment(cost.detach().float().cpu().numpy())
        return (
            torch.as_tensor(row_idx, dtype=torch.long, device=pred_logits.device),
            torch.as_tensor(col_idx, dtype=torch.long, device=pred_logits.device),
        )


    @staticmethod
    def _site_target_mask_count(targets: list[dict[str, Tensor]], reference: Tensor) -> Tensor:
        count = reference.new_tensor([sum(int(target["labels"].numel()) for target in targets)], dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            count = count / float(dist.get_world_size())
        return count.clamp(min=1.0).squeeze(0)


    def _detr_site_detection_losses_for_outputs(
        self,
        site_logits: Tensor,
        site_mask_logits: Tensor,
        site_query_mask: Tensor | None,
        targets: list[dict[str, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if site_logits.ndim != 3:
            raise ValueError(f"DETR site_logits must have shape [B, Q, C], got {tuple(site_logits.shape)}.")
        if site_mask_logits.ndim != 3:
            raise ValueError(f"DETR site_mask_logits must have shape [B, Q, L], got {tuple(site_mask_logits.shape)}.")
        if site_logits.shape[:2] != site_mask_logits.shape[:2]:
            raise ValueError(
                "DETR site logits/mask logits batch-query shape mismatch: "
                f"{tuple(site_logits.shape[:2])} vs {tuple(site_mask_logits.shape[:2])}."
            )
        if len(targets) != int(site_logits.size(0)):
            raise ValueError(
                f"DETR target batch size mismatch: got {len(targets)} targets for {site_logits.size(0)} predictions."
            )
        if site_query_mask is not None and site_query_mask.shape[:2] != site_logits.shape[:2]:
            raise ValueError(
                "DETR site_query_mask shape mismatch: "
                f"expected {tuple(site_logits.shape[:2])}, got {tuple(site_query_mask.shape)}."
            )

        class_logits = []
        class_targets = []
        mask_loss_total = site_logits.new_zeros(())
        dice_loss_total = site_logits.new_zeros(())
        no_object_class = site_logits.size(-1) - 1
        num_masks = self._site_target_mask_count(targets, site_logits)
        matched_mask_count = 0

        for batch_idx, target in enumerate(targets):
            target_labels = target["labels"].to(device=site_logits.device, dtype=torch.long)
            target_masks = target["pocket_masks"].to(device=site_logits.device)
            num_host = int(target_masks.size(-1))
            if site_mask_logits.size(-1) < num_host:
                raise ValueError(
                    "DETR site_mask_logits has fewer residue positions than target masks: "
                    f"{site_mask_logits.size(-1)} < {num_host}."
                )

            pred_logits = site_logits[batch_idx]
            pred_mask_logits = site_mask_logits[batch_idx, :, :num_host]
            if site_query_mask is not None:
                valid_queries = site_query_mask[batch_idx].to(device=pred_logits.device, dtype=torch.bool)
                pred_logits = pred_logits[valid_queries]
                pred_mask_logits = pred_mask_logits[valid_queries]
            if pred_logits.size(0) == 0:
                raise ValueError(f"DETR sample {batch_idx} has no valid object queries.")

            query_classes = torch.full(
                (pred_logits.size(0),),
                no_object_class,
                dtype=torch.long,
                device=pred_logits.device,
            )
            pred_idx, target_idx = self._match_site_queries(
                pred_logits=pred_logits,
                pred_mask_logits=pred_mask_logits,
                target_labels=target_labels,
                target_masks=target_masks,
            )
            if pred_idx.numel() > 0:
                query_classes[pred_idx] = target_labels[target_idx]
                sample_pred_masks = pred_mask_logits[pred_idx]
                sample_target_masks = target_masks[target_idx].to(dtype=sample_pred_masks.dtype)
                mask_loss_total = mask_loss_total + self._site_mask_loss(
                    sample_pred_masks,
                    sample_target_masks,
                    num_masks=num_masks,
                )
                dice_loss_total = dice_loss_total + self._site_dice_loss(
                    sample_pred_masks,
                    sample_target_masks,
                    num_masks=num_masks,
                )
                matched_mask_count += int(pred_idx.numel())
            class_logits.append(pred_logits)
            class_targets.append(query_classes)

        cls_weight = site_logits.new_ones((site_logits.size(-1),))
        cls_weight[no_object_class] = 0.1
        site_cls_loss = F.cross_entropy(torch.cat(class_logits, dim=0), torch.cat(class_targets, dim=0), weight=cls_weight)
        if matched_mask_count == 0:
            mask_loss_total = site_cls_loss * 0.0
            dice_loss_total = site_cls_loss * 0.0
        return site_cls_loss, mask_loss_total, dice_loss_total
