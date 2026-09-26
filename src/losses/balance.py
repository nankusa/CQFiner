from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.trainer.states import TrainerFn
from torch import Tensor


class LossBalanceMixin:
    @staticmethod
    def _parse_loss_weight_spec(loss_name: str, raw_spec) -> tuple[str, float]:
        if isinstance(raw_spec, str):
            normalized = raw_spec.strip().lower()
            if normalized == "uncertainty":
                return "uncertainty", 1.0
            if normalized == "prog":
                return "prog", 1.0
            try:
                value = float(raw_spec)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported loss weight spec for {loss_name}: {raw_spec!r}"
                ) from exc
        elif isinstance(
            raw_spec, (int, float, np.integer, np.floating)
        ) and not isinstance(raw_spec, bool):
            value = float(raw_spec)
        else:
            raise TypeError(
                f"Unsupported loss weight spec type for {loss_name}: {type(raw_spec).__name__}"
            )

        if value < 0.0:
            raise ValueError(
                f"Loss weight for {loss_name} must be non-negative, got {value}."
            )
        return "fixed", value

    def _configure_loss_weight(self, loss_name: str, raw_spec) -> float:
        mode, value = self._parse_loss_weight_spec(loss_name, raw_spec)
        self._loss_weight_modes[loss_name] = mode
        self._loss_weight_values[loss_name] = value
        if mode == "uncertainty":
            setattr(
                self,
                f"{loss_name}_log_var",
                nn.Parameter(torch.zeros((), dtype=torch.float32)),
            )
            return 1.0
        if mode == "prog" and loss_name not in {"query_distance", "query_disp"}:
            raise ValueError(
                "loss weight spec 'prog' is only supported for query_distance and query_disp."
            )
        self.register_parameter(f"{loss_name}_log_var", None)
        return value

    def _loss_uses_uncertainty(self, loss_name: str) -> bool:
        return self._loss_weight_modes.get(loss_name) == "uncertainty"

    def _loss_uses_prog(self, loss_name: str) -> bool:
        return self._loss_weight_modes.get(loss_name) == "prog"

    def _loss_is_enabled(self, loss_name: str) -> bool:
        if self._loss_uses_uncertainty(loss_name) or self._loss_uses_prog(loss_name):
            return True
        return self._loss_weight_values.get(loss_name, 0.0) > 0.0

    def _prog_loss_progress(self) -> float:
        trainer = getattr(self, "trainer", None)
        trainer_state = getattr(trainer, "state", None) if trainer is not None else None
        trainer_fn = (
            getattr(trainer_state, "fn", None) if trainer_state is not None else None
        )
        estimated_steps = (
            getattr(trainer, "estimated_stepping_batches", None)
            if trainer is not None and trainer_fn == TrainerFn.FITTING
            else None
        )
        try:
            estimated_steps_value = float(estimated_steps)
        except (TypeError, ValueError):
            estimated_steps_value = 0.0
        if np.isfinite(estimated_steps_value) and estimated_steps_value > 0:
            return float(
                np.clip(float(self.global_step) / estimated_steps_value, 0.0, 1.0)
            )

        trainer_max_epochs = (
            getattr(trainer, "max_epochs", None) if trainer is not None else None
        )
        cfg_max_epochs = (self.run_config.get("trainer") or {}).get("max_epochs", None)
        max_epochs = (
            trainer_max_epochs if trainer_max_epochs is not None else cfg_max_epochs
        )
        try:
            max_epochs = int(max_epochs)
        except (TypeError, ValueError):
            max_epochs = 1
        if max_epochs <= 1:
            return 1.0
        return float(
            np.clip(float(self.current_epoch) / float(max_epochs - 1), 0.0, 1.0)
        )

    def _prog_loss_weight(self, loss_name: str, reference: Tensor) -> Tensor:
        progress = self._prog_loss_progress()
        if loss_name == "query_disp":
            weight = progress
        elif loss_name == "query_distance":
            weight = 1.0 - progress
        else:
            raise ValueError(f"Unsupported prog loss name: {loss_name}")
        return torch.tensor(weight, dtype=reference.dtype, device=reference.device)

    def _apply_configured_loss_weight(self, loss_name: str, loss: Tensor) -> Tensor:
        if not self._loss_is_enabled(loss_name):
            return loss * 0.0
        if self._loss_uses_uncertainty(loss_name):
            return self._uncertainty_weighted_loss(
                loss, getattr(self, f"{loss_name}_log_var")
            )
        if self._loss_uses_prog(loss_name):
            return self._prog_loss_weight(loss_name, loss) * loss
        return self._loss_weight_values[loss_name] * loss

    @staticmethod
    def _uncertainty_weighted_loss(loss: Tensor, log_var: Tensor) -> Tensor:
        return torch.exp(-log_var) * loss + log_var

    def _combine_site_detection_loss(
        self, site_detection_loss: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero = site_detection_loss * 0.0
        if not self._loss_is_enabled("site_detection"):
            return zero, {"site_detection_objective_loss": zero}
        total = self._apply_configured_loss_weight(
            "site_detection", site_detection_loss
        )
        return total, {"site_detection_objective_loss": total}

    def _combine_query_objective_losses(
        self,
        query_distance_loss: Tensor,
        query_disp_loss: Tensor,
        query_contrastive_loss: Tensor,
        *,
        has_query_disp_supervision: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero = query_distance_loss * 0.0
        metrics: dict[str, Tensor] = {"query_objective_loss": zero}
        total = zero
        if self._loss_is_enabled("query_distance"):
            total = total + self._apply_configured_loss_weight(
                "query_distance", query_distance_loss
            )
        if has_query_disp_supervision and self._loss_is_enabled("query_disp"):
            total = total + self._apply_configured_loss_weight(
                "query_disp", query_disp_loss
            )
        if self._loss_is_enabled("query_contrastive"):
            total = total + self._apply_configured_loss_weight(
                "query_contrastive", query_contrastive_loss
            )
        metrics["query_objective_loss"] = total
        return total, metrics

    def _current_loss_weight_metrics(
        self,
        reference: Tensor,
        *,
        has_query_disp_supervision: bool,
    ) -> dict[str, Tensor]:
        zero = reference.detach() * 0.0
        metrics: dict[str, Tensor] = {
            "site_detection_loss_weight": zero,
            "query_distance_loss_weight": zero,
            "query_disp_loss_weight": zero,
            "query_contrastive_loss_weight": zero,
            "site_detection_log_var": zero,
            "query_distance_log_var": zero,
            "query_disp_log_var": zero,
            "query_contrastive_log_var": zero,
        }
        metrics["site_detection_loss_weight"] = torch.tensor(
            0.0,
            dtype=reference.dtype,
            device=reference.device,
        )
        active_loss_names = {
            "site_detection": True,
            "query_distance": True,
            "query_disp": has_query_disp_supervision,
            "query_contrastive": True,
        }
        for loss_name, is_active in active_loss_names.items():
            if not is_active or not self._loss_is_enabled(loss_name):
                continue
            if self._loss_uses_uncertainty(loss_name):
                log_var = getattr(self, f"{loss_name}_log_var").detach()
                metrics[f"{loss_name}_loss_weight"] = torch.exp(-log_var)
                metrics[f"{loss_name}_log_var"] = log_var
            elif self._loss_uses_prog(loss_name):
                metrics[f"{loss_name}_loss_weight"] = self._prog_loss_weight(
                    loss_name, reference
                ).detach()
            else:
                metrics[f"{loss_name}_loss_weight"] = torch.tensor(
                    self._loss_weight_values[loss_name],
                    dtype=reference.dtype,
                    device=reference.device,
                )
        return metrics
