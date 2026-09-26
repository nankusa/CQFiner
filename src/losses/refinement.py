import torch
import torch.nn.functional as F
from torch import Tensor


class RefinementLossMixin:
    def _query_refiner_loss_weight(self) -> float:
        return float(getattr(self.model, "query_refiner_loss_weight", 1.0))

    def _combine_query_refiner_stage_losses(
        self, stage0_loss: Tensor, refined_loss: Tensor
    ) -> Tensor:
        weight = self._query_refiner_loss_weight()
        if weight < 0.0:
            raise ValueError(
                f"query_refiner_loss_weight must be non-negative, got {weight}."
            )
        return (stage0_loss + weight * refined_loss) / (1.0 + weight)

    def _query_refiner_aux_outputs(
        self, out: dict[str, Tensor]
    ) -> list[dict[str, Tensor]]:
        aux_outputs = out.get("query_refiner_aux_outputs")
        if not isinstance(aux_outputs, list):
            raise RuntimeError(
                "query_refiner is enabled but query_refiner_aux_outputs is missing."
            )
        expected_layers = int(getattr(self.model, "query_refiner_num_layers", 0))
        if expected_layers <= 0:
            raise ValueError(
                f"query_refiner_num_layers must be positive, got {expected_layers}."
            )
        if len(aux_outputs) != expected_layers:
            raise RuntimeError(
                "query_refiner_aux_outputs length mismatch: "
                f"expected {expected_layers}, got {len(aux_outputs)}."
            )
        for layer_idx, layer_out in enumerate(aux_outputs, start=1):
            if not isinstance(layer_out, dict):
                raise TypeError(
                    f"query_refiner_aux_outputs[{layer_idx - 1}] must be a dict, "
                    f"got {type(layer_out).__name__}."
                )
            layer_pos = layer_out.get("query_refined_pos")
            if layer_pos is None:
                raise RuntimeError(
                    f"query_refiner_aux_outputs[{layer_idx - 1}] is missing query_refined_pos."
                )
        return aux_outputs

    @staticmethod
    def _mean_query_refiner_losses(losses: list[Tensor], label: str) -> Tensor:
        if not losses:
            raise RuntimeError(
                f"{label} requires at least one query refiner layer loss."
            )
        return torch.stack(losses).mean()

    def _query_refiner_ranking_aux_losses(
        self,
        out: dict[str, Tensor],
        query_batch: Tensor,
        batch,
        target_site_ids: Tensor,
        supervise: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        layer_losses: list[Tensor] = []
        for layer_out in self._query_refiner_aux_outputs(out):
            layer_pos = layer_out["query_refined_pos"]
            layer_losses.append(
                self._query_distance_dfl_loss_for_state(
                    out=layer_out,
                    ranking_pos=layer_pos.detach(),
                    query_batch=query_batch,
                    batch=batch,
                    target_site_ids=target_site_ids,
                    supervise=supervise,
                )
            )
        return (
            layer_losses[-1],
            self._mean_query_refiner_losses(
                layer_losses, "query refiner ranking aux loss"
            ),
            layer_losses,
        )

    def _query_refiner_disp_aux_losses(
        self,
        out: dict[str, Tensor],
        target_pos: Tensor,
        query_batch: Tensor,
        target_site_ids: Tensor,
        supervise: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor], bool]:
        layer_losses: list[Tensor] = []
        layer_has_supervision: list[bool] = []
        for layer_out in self._query_refiner_aux_outputs(out):
            reference = layer_out.get("host_logits")
            if reference is None:
                raise RuntimeError(
                    "query refiner aux displacement loss requires host_logits."
                )
            layer_pos = layer_out["query_refined_pos"]
            layer_loss, has_supervision = self._query_displacement_loss_from_positions(
                reference=reference,
                pred_pos=layer_pos,
                target_pos=target_pos,
                query_batch=query_batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
            layer_losses.append(layer_loss)
            layer_has_supervision.append(has_supervision)
        return (
            layer_losses[-1],
            self._mean_query_refiner_losses(
                layer_losses, "query refiner displacement aux loss"
            ),
            layer_losses,
            any(layer_has_supervision),
        )

    def _log_query_refiner_layer_losses(
        self,
        prefix: str,
        batch_size: int,
        distance_losses: list[Tensor],
        disp_losses: list[Tensor],
        sync_dist: bool = False,
    ) -> None:
        if len(distance_losses) != len(disp_losses):
            raise RuntimeError(
                "query refiner layer loss log length mismatch: "
                f"distance={len(distance_losses)}, disp={len(disp_losses)}."
            )
        for layer_idx, (distance_loss, disp_loss) in enumerate(
            zip(distance_losses, disp_losses), start=1
        ):
            self.log(
                f"{prefix}/query_refiner_layer{layer_idx}_distance_loss",
                distance_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
            self.log(
                f"{prefix}/query_refiner_layer{layer_idx}_disp_loss",
                disp_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

    def _query_ranking_loss_for_state(
        self,
        out: dict[str, Tensor],
        ranking_pos: Tensor,
        query_batch: Tensor,
        batch,
        target_site_ids: Tensor,
        supervise: Tensor,
    ) -> Tensor:
        query_loss_site_ids = target_site_ids.clone()
        query_loss_site_ids[~supervise] = -1
        query_distance_per_query, query_distance_valid = (
            self._query_ranking_per_query_loss(
                out=out,
                ranking_pos=ranking_pos,
                query_batch=query_batch,
                batch=batch,
            )
        )
        return self._reduce_query_loss_per_site(
            query_distance_per_query,
            query_batch,
            query_loss_site_ids,
            query_distance_valid,
            include_background=True,
        )

    def _query_distance_dfl_loss_for_state(
        self,
        out: dict[str, Tensor],
        ranking_pos: Tensor,
        query_batch: Tensor,
        batch,
        target_site_ids: Tensor,
        supervise: Tensor,
    ) -> Tensor:
        if self.query_ranking_mode != "distance_distribution":
            raise ValueError(
                "Expected distance_distribution ranking mode for query distance DFL loss, "
                f"got {self.query_ranking_mode!r}."
            )
        logits = out.get("query_distance_logits")
        if logits is None:
            raise RuntimeError(
                "query distance DFL loss requires query_distance_logits."
            )
        query_loss_site_ids = target_site_ids.clone()
        query_loss_site_ids[~supervise] = -1
        query_distance_targets = self._query_distance_target_distribution(
            ranking_pos, query_batch, batch
        )
        query_distance_per_query, query_distance_valid = (
            self._query_distance_distribution_per_query_loss(
                logits,
                query_distance_targets,
            )
        )
        return self._reduce_query_loss_per_site(
            query_distance_per_query,
            query_batch,
            query_loss_site_ids,
            query_distance_valid,
            include_background=True,
        )

    def _query_displacement_loss_from_positions(
        self,
        reference: Tensor,
        pred_pos: Tensor,
        target_pos: Tensor,
        query_batch: Tensor,
        target_site_ids: Tensor,
        supervise: Tensor,
    ) -> tuple[Tensor, bool]:
        has_supervision = bool(self._loss_is_enabled("query_disp") and supervise.any())
        if not has_supervision:
            return reference.sum() * 0.0, False
        per_query = F.mse_loss(
            pred_pos[supervise], target_pos[supervise], reduction="none"
        ).mean(dim=-1)
        loss = self._reduce_query_loss_per_site(
            per_query,
            query_batch[supervise],
            target_site_ids[supervise],
            torch.ones((per_query.size(0),), dtype=torch.bool, device=per_query.device),
            include_background=False,
        )
        return loss, True

    def _query_displacement_loss_from_disp(
        self,
        reference: Tensor,
        pred_disp: Tensor,
        target_disp: Tensor,
        query_batch: Tensor,
        target_site_ids: Tensor,
        supervise: Tensor,
    ) -> tuple[Tensor, bool]:
        has_supervision = bool(self._loss_is_enabled("query_disp") and supervise.any())
        if not has_supervision:
            return reference.sum() * 0.0, False
        per_query = F.mse_loss(
            pred_disp[supervise], target_disp[supervise], reduction="none"
        ).mean(dim=-1)
        loss = self._reduce_query_loss_per_site(
            per_query,
            query_batch[supervise],
            target_site_ids[supervise],
            torch.ones((per_query.size(0),), dtype=torch.bool, device=per_query.device),
            include_background=False,
        )
        return loss, True
