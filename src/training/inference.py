import time
from typing import Any
import numpy as np
import torch
from torch import Tensor
from .detr import DetrInferenceMixin


class InferenceMixin(DetrInferenceMixin):
    def _notify_eval_feature_recorders(
        self,
        *,
        batch,
        out: dict[str, Tensor],
        query_batch: Tensor,
        dataset_name: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        recorders = getattr(self, "eval_feature_recorders", None)
        if not recorders:
            return
        for recorder in recorders:
            recorder.collect_batch(
                batch=batch,
                out=out,
                query_batch=query_batch,
                dataset_name=dataset_name,
                module=self,
                context=context,
            )

    def _profile_eval_stage(self, stage: str, dataset_name: str | None, batch, fn):
        recorder = getattr(self, "eval_stage_timing_recorder", None)
        if recorder is None:
            return fn()
        if dataset_name is None:
            raise RuntimeError("Eval stage timing requires dataset_name.")
        device = self.device
        if device.type != "cuda":
            raise RuntimeError(
                f"Eval stage timing requires a CUDA device, got {device}."
            )
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize(device)
        recorder.record(
            dataset_name=dataset_name,
            stage=stage,
            seconds=time.perf_counter() - start,
            num_graphs=int(getattr(batch, "num_graphs", 1)),
        )
        return output

    def _inference_outputs(
        self, batch, dataset_name: str | None = None
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, dict[str, Any]]:
        def setup_queries():
            query_pos_0, query_batch, _ = self._sample_query_positions(
                batch, training=False
            )
            query_x = self._build_query_features(batch, query_batch=query_batch)
            zero_t = self._constant_query_times(query_pos_0, 0.0)
            return (query_pos_0, query_batch, query_x, zero_t)

        query_pos_0, query_batch, query_x, zero_t = self._profile_eval_stage(
            "query_setup", dataset_name, batch, setup_queries
        )
        if self._query_refiner_enabled():
            out_init = self._profile_eval_stage(
                "forward_stage0",
                dataset_name,
                batch,
                lambda: self._forward(
                    batch,
                    query_pos=query_pos_0,
                    query_x=query_x,
                    query_batch=query_batch,
                    t_query=zero_t,
                ),
            )
            stage0_pos, query_disp_pred = self._profile_eval_stage(
                "displacement",
                dataset_name,
                batch,
                lambda: self._apply_query_displacement(
                    query_pos_0, out_init["query_disp"]
                ),
            )
            final_pos, out = self._profile_eval_stage(
                "refine_head",
                dataset_name,
                batch,
                lambda: self._refine_query_outputs(
                    out=out_init,
                    host_pos=batch.pos,
                    host_batch=batch.batch,
                    query_pos=stage0_pos.detach(),
                    query_batch=query_batch,
                    return_intermediate=False,
                ),
            )
        else:
            out_init = self._profile_eval_stage(
                "forward_stage0",
                dataset_name,
                batch,
                lambda: self._forward(
                    batch,
                    query_pos=query_pos_0,
                    query_x=query_x,
                    query_batch=query_batch,
                    t_query=zero_t,
                ),
            )
            stage0_pos, query_disp_pred = self._profile_eval_stage(
                "displacement",
                dataset_name,
                batch,
                lambda: self._apply_query_displacement(
                    query_pos_0, out_init["query_disp"]
                ),
            )
            if self.eval_forward_passes == 1:
                final_pos = (
                    stage0_pos
                    if self.eval_use_stage0_offset_without_second_forward
                    else query_pos_0
                )
                out = out_init
            elif self.eval_forward_passes == 2:
                final_pos = stage0_pos
                out = self._profile_eval_stage(
                    "forward_stage1",
                    dataset_name,
                    batch,
                    lambda: self._forward(
                        batch,
                        query_pos=final_pos,
                        query_x=query_x,
                        query_batch=query_batch,
                        t_query=zero_t,
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported eval_forward_passes: {self.eval_forward_passes}"
                )
        query_rank_scores = self._profile_eval_stage(
            "rank_scores",
            dataset_name,
            batch,
            lambda: self._query_rank_scores_from_out(out),
        )
        if query_rank_scores.shape != (final_pos.size(0),):
            raise ValueError(
                f"Inference query rank score shape mismatch: expected {(final_pos.size(0),)}, got {tuple(query_rank_scores.shape)}."
            )
        context = {
            "query_pos_0": query_pos_0,
            "stage0_pos": stage0_pos,
            "final_pos": final_pos,
            "out_init": out_init,
            "query_disp_pred": query_disp_pred,
            "query_rank_scores": query_rank_scores,
        }
        return (final_pos, query_batch, out, query_rank_scores, context)

    def _predict_site_outputs(
        self,
        batch,
        dataset_name: str,
        out: dict[str, Tensor],
        final_pos: Tensor,
        query_batch: Tensor,
        query_rank_scores: Tensor,
    ) -> list[dict[str, Any]]:
        if self._site_detector_loss_type() == "detr":
            return self._predict_detr_site_outputs(batch, dataset_name, out)
        return self._predict_vn_dot_site_outputs(
            batch=batch,
            dataset_name=dataset_name,
            out=out,
            query_pos=final_pos,
            query_batch=query_batch,
            query_rank_scores=query_rank_scores,
        )

    def _predict_vn_dot_site_outputs(
        self,
        batch,
        dataset_name: str,
        out: dict[str, Tensor],
        query_pos: Tensor,
        query_batch: Tensor,
        query_rank_scores: Tensor,
    ) -> list[dict[str, Any]]:
        host_atom_residue_index = getattr(batch, "host_atom_residue_id", None)
        if (
            self._uses_surface_atom_eval(dataset_name)
            and host_atom_residue_index is None
        ):
            raise RuntimeError(
                "Surface-atom VN-dot prediction requires batch.host_atom_residue_id."
            )
        if query_pos.ndim != 2 or query_pos.size(-1) != 3:
            raise ValueError(
                f"VN-dot prediction expects query_pos [N, 3], got {tuple(query_pos.shape)}."
            )
        if query_batch.shape != (query_pos.size(0),):
            raise ValueError(
                f"VN-dot prediction query_batch shape mismatch: expected {(query_pos.size(0),)}, got {tuple(query_batch.shape)}."
            )
        if query_rank_scores.shape != (query_pos.size(0),):
            raise ValueError(
                f"VN-dot prediction query_rank_scores shape mismatch: expected {(query_pos.size(0),)}, got {tuple(query_rank_scores.shape)}."
            )
        host_scalar = out.get("site_host_scalar", out.get("host_scalar"))
        query_scalar = out.get("query_scalar")
        if host_scalar is None or query_scalar is None:
            raise RuntimeError(
                "VN-dot prediction requires host_scalar and query_scalar in model outputs."
            )
        site_detector = getattr(self.model, "site_detector", None)
        if site_detector is None or not hasattr(site_detector, "sample_mask_logits"):
            raise RuntimeError(
                "VN-dot prediction requires VNDirectSiteMaskHead.sample_mask_logits."
            )
        site_query_mask = out.get("site_query_mask")
        site_sample_ids = out.get("site_sample_ids", batch.batch.unique(sorted=True))
        if site_sample_ids.ndim != 1:
            raise ValueError(
                f"VN-dot prediction expects 1-D site_sample_ids, got {tuple(site_sample_ids.shape)}."
            )
        if site_query_mask is not None and site_query_mask.size(
            0
        ) < site_sample_ids.size(0):
            raise ValueError(
                f"VN-dot prediction site_query_mask has fewer samples than site_sample_ids: {site_query_mask.size(0)} < {site_sample_ids.size(0)}."
            )
        sample_ids = self._batch_sample_ids(batch)
        outputs: list[dict[str, Any]] = []
        for out_idx, sample_idx in enumerate(site_sample_ids.tolist()):
            sample_idx = int(sample_idx)
            host_sel = batch.batch == sample_idx
            query_sel = query_batch == sample_idx
            if not bool(host_sel.any().item()):
                raise ValueError(
                    f"VN-dot prediction sample {sample_idx} has no host nodes."
                )
            if not bool(query_sel.any().item()):
                raise ValueError(
                    f"VN-dot prediction sample {sample_idx} has no query nodes."
                )
            query_indices = query_sel.nonzero(as_tuple=False).view(-1)
            num_host = int(host_sel.sum().item())
            num_pred = int(query_indices.numel())
            if site_query_mask is not None:
                num_pred = min(num_pred, int(site_query_mask.size(1)))
            query_indices = query_indices[:num_pred]
            valid_queries = torch.ones(
                (num_pred,), dtype=torch.bool, device=query_pos.device
            )
            if site_query_mask is not None:
                valid_queries &= site_query_mask[out_idx, :num_pred].to(
                    device=query_pos.device, dtype=torch.bool
                )
            if self.site_mask_query_score_threshold > 0:
                valid_queries &= (
                    query_rank_scores[query_indices]
                    >= self.site_mask_query_score_threshold
                )
            selected_query_indices = query_indices[valid_queries]
            sample_query_pos = query_pos[selected_query_indices]
            sample_query_scores = query_rank_scores[selected_query_indices]
            if selected_query_indices.numel() == 0:
                sample_mask_logits = host_scalar.new_zeros((0, num_host))
                sample_affinity_logits = None
            else:
                sample_mask_logits = site_detector.sample_mask_logits(
                    host_scalar=host_scalar[host_sel],
                    query_scalar=query_scalar[selected_query_indices],
                    host_pos=batch.pos[host_sel].to(device=host_scalar.device),
                    query_pos=sample_query_pos.to(device=host_scalar.device),
                )
                sample_affinity_logits = None
                if self.site_mask_grouping == "affinity":
                    if not hasattr(site_detector, "query_affinity_logits"):
                        raise RuntimeError(
                            "VN-dot affinity grouping requires query_affinity_logits."
                        )
                    sample_affinity_logits = site_detector.query_affinity_logits(
                        query_scalar=query_scalar[selected_query_indices],
                        query_pos=sample_query_pos.to(device=host_scalar.device),
                    )
            prediction = self._build_vn_dot_site_prediction(
                query_pos=sample_query_pos,
                query_scores=sample_query_scores,
                mask_logits=sample_mask_logits,
                host_pos=batch.pos[host_sel],
                affinity_logits=sample_affinity_logits,
                affinity_threshold=self.site_mask_affinity_threshold,
            )
            atom_residue_indices = None
            if host_atom_residue_index is not None:
                atom_residue_indices = (
                    host_atom_residue_index[host_sel]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )
            outputs.append(
                {
                    "dataset_name": dataset_name,
                    "sample_idx": sample_idx,
                    "sample_id": sample_ids[sample_idx]
                    if sample_idx < len(sample_ids)
                    else str(sample_idx),
                    "atom_residue_indices": atom_residue_indices,
                    "prediction": prediction,
                }
            )
        if not outputs:
            raise RuntimeError(
                f"VN-dot prediction produced no outputs for dataset {dataset_name}."
            )
        self._test_site_prediction_outputs[dataset_name].extend(outputs)
        return outputs
