import numpy as np
from torch import Tensor

from src.utils.metrics import (
    extract_ligand_groups,
)


class TestMixin:
    def _collect_test_query_predictions(
        self,
        batch,
        dataset_name: str,
        final_pos: Tensor,
        query_batch: Tensor,
        query_rank_scores: Tensor,
    ) -> None:
        sample_ids = (
            batch.sample_id if isinstance(batch.sample_id, list) else [batch.sample_id]
        )
        for sample_idx, _sample_id in enumerate(sample_ids):
            query_sel = query_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx
            if query_sel.any():
                sample_scores = (
                    query_rank_scores[query_sel].detach().float().cpu().numpy()
                )
                coords_np = (
                    final_pos[query_sel]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                pred_centers, pred_scores = self._apply_query_nms(
                    coords_np,
                    sample_scores,
                )
            else:
                pred_centers = np.zeros((0, 3), dtype=np.float32)
                pred_scores = np.zeros((0,), dtype=np.float32)

            ligands = extract_ligand_groups(
                ligand_pos=batch.ligand_pos[ligand_sel],
                ligand_ids=batch.ligand_id[ligand_sel],
            )
            self._test_pred_centers[dataset_name].append(pred_centers)
            self._test_pred_scores[dataset_name].append(pred_scores)
            self._test_ligands[dataset_name].append(ligands)

    def _prediction_only_test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        dataset_name = self._dataset_name_from_dataloader_idx(dataloader_idx)
        self._ensure_site_test_bucket(dataset_name)
        final_pos, query_batch, out, query_rank_scores, context = (
            self._inference_outputs(
                batch,
                dataset_name=dataset_name,
            )
        )
        site_prediction_outputs = self._profile_eval_stage(
            "site_prediction",
            dataset_name,
            batch,
            lambda: self._predict_site_outputs(
                batch=batch,
                dataset_name=dataset_name,
                out=out,
                final_pos=final_pos,
                query_batch=query_batch,
                query_rank_scores=query_rank_scores,
            ),
        )
        context["site_prediction_outputs"] = site_prediction_outputs
        self._notify_eval_feature_recorders(
            batch=batch,
            out=out,
            query_batch=query_batch,
            dataset_name=dataset_name,
            context=context,
        )

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if bool(getattr(self, "eval_pocket_minimal", False)):
            return self._prediction_only_test_step(
                batch, batch_idx, dataloader_idx=dataloader_idx
            )

        dataset_name = self._dataset_name_from_dataloader_idx(dataloader_idx)
        self._ensure_site_test_bucket(dataset_name)

        batch_size = int(getattr(batch, "num_graphs", 1))

        def setup_queries():
            query_pos, query_ids, _ = self._sample_query_positions(
                batch, training=False
            )
            query_features = self._build_query_features(batch, query_batch=query_ids)
            return query_pos, query_ids, query_features

        query_pos_0, query_batch, query_x = self._profile_eval_stage(
            "query_setup",
            dataset_name,
            batch,
            setup_queries,
        )
        out_init = None
        stage0_pos = None
        query_disp_pred = None
        if self._query_refiner_enabled():
            t_query = self._constant_query_times(query_pos_0, 0.0)
            out_init = self._profile_eval_stage(
                "forward_stage0",
                dataset_name,
                batch,
                lambda: self._forward(
                    batch,
                    query_pos=query_pos_0,
                    query_x=query_x,
                    query_batch=query_batch,
                    t_query=t_query,
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
                ),
            )
        else:
            final_pos, out = self._profile_eval_stage(
                "query_eval_state",
                dataset_name,
                batch,
                lambda: self._predict_query_eval_state(
                    batch=batch,
                    query_pos_0=query_pos_0,
                    query_x=query_x,
                    query_batch=query_batch,
                ),
            )

        target_pos, target_site_ids, supervise = self._profile_eval_stage(
            "target_setup",
            dataset_name,
            batch,
            lambda: self._build_query_position_targets(batch, query_pos_0, query_batch),
        )
        query_distance_stage0_loss = None
        query_disp_stage0_loss = None
        query_distance_refiner_aux_loss = None
        query_disp_refiner_aux_loss = None
        if self._query_refiner_enabled():
            if out_init is None or stage0_pos is None or query_disp_pred is None:
                raise RuntimeError("query refiner test path requires stage-0 outputs.")
            query_distance_stage0_loss = self._query_distance_dfl_loss_for_state(
                out=out_init,
                ranking_pos=query_pos_0.detach(),
                query_batch=query_batch,
                batch=batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
        if self._query_refiner_enabled():
            query_distance_refined_loss, query_distance_refiner_aux_loss, _ = (
                self._query_refiner_ranking_aux_losses(
                    out=out,
                    query_batch=query_batch,
                    batch=batch,
                    target_site_ids=target_site_ids,
                    supervise=supervise,
                )
            )
            if query_distance_stage0_loss is None:
                raise RuntimeError(
                    "query refiner test path did not compute stage-0 ranking loss."
                )
            query_distance_loss = self._combine_query_refiner_stage_losses(
                query_distance_stage0_loss,
                query_distance_refiner_aux_loss,
            )
        else:
            query_distance_loss = self._query_ranking_loss_for_state(
                out=out,
                ranking_pos=final_pos,
                query_batch=query_batch,
                batch=batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
            query_distance_refined_loss = query_distance_loss
        site_losses = self._profile_eval_stage(
            "site_losses",
            dataset_name,
            batch,
            lambda: self._site_detection_losses(
                out,
                batch,
                query_batch=query_batch,
                query_site_ids=target_site_ids,
                query_supervise=supervise,
                query_pos=final_pos,
            ),
        )
        site_detection_loss = site_losses["site_detection_loss"]
        query_rank_scores = self._profile_eval_stage(
            "rank_scores",
            dataset_name,
            batch,
            lambda: self._query_rank_scores_from_out(out),
        )
        self._notify_eval_feature_recorders(
            batch=batch,
            out=out,
            query_batch=query_batch,
            dataset_name=dataset_name,
        )
        if self._query_refiner_enabled():
            query_disp_target = self._build_query_displacement_targets(
                query_pos_0=query_pos_0,
                target_pos=target_pos,
            )
            query_disp_stage0_loss, has_stage0_disp_supervision = (
                self._query_displacement_loss_from_disp(
                    reference=out_init["host_logits"],
                    pred_disp=query_disp_pred,
                    target_disp=query_disp_target,
                    query_batch=query_batch,
                    target_site_ids=target_site_ids,
                    supervise=supervise,
                )
            )
            (
                query_disp_refined_loss,
                query_disp_refiner_aux_loss,
                _,
                has_refined_disp_supervision,
            ) = self._query_refiner_disp_aux_losses(
                out=out,
                target_pos=target_pos,
                query_batch=query_batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
            query_disp_loss = self._combine_query_refiner_stage_losses(
                query_disp_stage0_loss,
                query_disp_refiner_aux_loss,
            )
            has_query_disp_supervision = (
                has_stage0_disp_supervision or has_refined_disp_supervision
            )
        else:
            query_disp_loss, has_query_disp_supervision = (
                self._query_displacement_loss_from_positions(
                    reference=out["host_logits"],
                    pred_pos=final_pos,
                    target_pos=target_pos,
                    query_batch=query_batch,
                    target_site_ids=target_site_ids,
                    supervise=supervise,
                )
            )
        if self._loss_is_enabled("query_contrastive"):
            query_contrastive_loss = self._query_site_contrastive_loss(
                embeddings=out["query_site_embed"],
                query_batch=query_batch,
                site_ids=target_site_ids,
                supervise=supervise,
            )
        else:
            query_contrastive_loss = out["host_logits"].sum() * 0.0
        site_detection_objective_loss, site_balance_metrics = (
            self._combine_site_detection_loss(site_detection_loss)
        )
        query_objective_loss, query_balance_metrics = (
            self._combine_query_objective_losses(
                query_distance_loss=query_distance_loss,
                query_disp_loss=query_disp_loss,
                query_contrastive_loss=query_contrastive_loss,
                has_query_disp_supervision=has_query_disp_supervision,
            )
        )
        loss_weight_metrics = self._current_loss_weight_metrics(
            query_disp_loss,
            has_query_disp_supervision=has_query_disp_supervision,
        )
        total = site_detection_objective_loss + query_objective_loss
        self._test_loss_rows[dataset_name].append(
            {
                "batch_size": batch_size,
                "loss": float(total.detach().cpu().item()),
                "site_detection_loss": float(site_detection_loss.detach().cpu().item()),
                "site_cls_loss": float(
                    site_losses["site_cls_loss"].detach().cpu().item()
                ),
                "site_mask_loss": float(
                    site_losses["site_mask_loss"].detach().cpu().item()
                ),
                "site_dice_loss": float(
                    site_losses["site_dice_loss"].detach().cpu().item()
                ),
                "site_contrastive_loss": float(
                    site_losses["site_contrastive_loss"].detach().cpu().item()
                ),
                "site_affinity_loss": float(
                    site_losses["site_affinity_loss"].detach().cpu().item()
                ),
                "query_distance_loss": float(query_distance_loss.detach().cpu().item()),
                "query_confidence_loss": float(
                    query_distance_loss.detach().cpu().item()
                )
                if self.query_ranking_mode == "confidence"
                else float("nan"),
                "query_disp_loss": float(query_disp_loss.detach().cpu().item()),
                "query_stage0_distance_loss": float(
                    query_distance_stage0_loss.detach().cpu().item()
                )
                if query_distance_stage0_loss is not None
                else float("nan"),
                "query_refined_distance_loss": float(
                    query_distance_refined_loss.detach().cpu().item()
                )
                if self._query_refiner_enabled()
                else float("nan"),
                "query_refiner_aux_distance_loss": float(
                    query_distance_refiner_aux_loss.detach().cpu().item()
                )
                if query_distance_refiner_aux_loss is not None
                else float("nan"),
                "query_stage0_disp_loss": float(
                    query_disp_stage0_loss.detach().cpu().item()
                )
                if query_disp_stage0_loss is not None
                else float("nan"),
                "query_refined_disp_loss": float(
                    query_disp_refined_loss.detach().cpu().item()
                )
                if self._query_refiner_enabled()
                else float("nan"),
                "query_refiner_aux_disp_loss": float(
                    query_disp_refiner_aux_loss.detach().cpu().item()
                )
                if query_disp_refiner_aux_loss is not None
                else float("nan"),
                "query_contrastive_loss": float(
                    query_contrastive_loss.detach().cpu().item()
                ),
                "site_detection_objective_loss": float(
                    site_detection_objective_loss.detach().cpu().item()
                ),
                "query_objective_loss": float(
                    query_objective_loss.detach().cpu().item()
                ),
                **{
                    key: float(value.detach().cpu().item())
                    for key, value in site_balance_metrics.items()
                    if key != "site_detection_objective_loss"
                },
                **{
                    key: float(value.detach().cpu().item())
                    for key, value in query_balance_metrics.items()
                    if key != "query_objective_loss"
                },
                **{
                    key: float(value.detach().cpu().item())
                    for key, value in loss_weight_metrics.items()
                },
            }
        )

        if self.rank_based_eval:
            self._collect_site_detection_eval(
                batch=batch,
                out=out,
                dataset_name=dataset_name,
                query_pos=final_pos,
                query_batch=query_batch,
                query_rank_scores=query_rank_scores,
            )
            sample_ids = (
                batch.sample_id
                if isinstance(batch.sample_id, list)
                else [batch.sample_id]
            )
            for sample_idx, _sample_id in enumerate(sample_ids):
                query_sel = query_batch == sample_idx
                ligand_sel = batch.ligand_pos_batch == sample_idx
                if query_sel.any():
                    sample_scores = (
                        query_rank_scores[query_sel].detach().float().cpu().numpy()
                    )
                    coords_np = (
                        final_pos[query_sel]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    pred_centers, pred_scores = self._apply_query_nms(
                        coords_np,
                        sample_scores,
                    )
                else:
                    pred_centers = np.zeros((0, 3), dtype=np.float32)
                    pred_scores = np.zeros((0,), dtype=np.float32)

                ligands = extract_ligand_groups(
                    ligand_pos=batch.ligand_pos[ligand_sel],
                    ligand_ids=batch.ligand_id[ligand_sel],
                )
                self._test_pred_centers[dataset_name].append(pred_centers)
                self._test_pred_scores[dataset_name].append(pred_scores)
                self._test_ligands[dataset_name].append(ligands)
