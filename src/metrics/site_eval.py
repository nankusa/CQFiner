from __future__ import annotations

from typing import Any

from torch import Tensor


class SiteEvalRecorder:
    """Collect site-level predictions and targets for AP/DCC/DCA evaluation."""

    def collect_batch(
        self,
        *,
        batch: Any,
        out: dict[str, Tensor],
        query_batch: Tensor,
        dataset_name: str,
        module: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        if module is None:
            raise RuntimeError("SiteEvalRecorder requires the Lightning module.")
        if context is None:
            raise RuntimeError("SiteEvalRecorder requires inference context.")
        if not bool(getattr(module, "rank_based_eval", False)):
            raise RuntimeError("SiteEvalRecorder requires module.rank_based_eval=true.")

        required = {"final_pos", "query_rank_scores", "site_prediction_outputs"}
        missing = required - set(context)
        if missing:
            raise RuntimeError(
                f"SiteEvalRecorder context is missing keys: {sorted(missing)}"
            )

        final_pos = context["final_pos"]
        query_rank_scores = context["query_rank_scores"]
        site_prediction_outputs = context["site_prediction_outputs"]
        if not isinstance(site_prediction_outputs, list):
            raise TypeError(
                "SiteEvalRecorder context['site_prediction_outputs'] must be a list, "
                f"got {type(site_prediction_outputs).__name__}."
            )
        if query_batch.shape != (final_pos.size(0),):
            raise ValueError(
                "SiteEvalRecorder query_batch shape mismatch: "
                f"expected {(final_pos.size(0),)}, got {tuple(query_batch.shape)}."
            )
        if query_rank_scores.shape != (final_pos.size(0),):
            raise ValueError(
                "SiteEvalRecorder query_rank_scores shape mismatch: "
                f"expected {(final_pos.size(0),)}, got {tuple(query_rank_scores.shape)}."
            )

        module._collect_site_eval_from_prediction_outputs(
            batch=batch,
            dataset_name=dataset_name,
            site_prediction_outputs=site_prediction_outputs,
        )
        module._collect_test_query_predictions(
            batch=batch,
            dataset_name=dataset_name,
            final_pos=final_pos,
            query_batch=query_batch,
            query_rank_scores=query_rank_scores,
        )
