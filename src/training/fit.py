from torch import Tensor


class FitMixin:
    def training_step(self, batch, batch_idx):
        batch_size = int(getattr(batch, "num_graphs", 1))
        query_pos_0, query_batch, _ = self._sample_query_positions(batch, training=True)
        query_x = self._build_query_features(batch, query_batch=query_batch)
        target_pos, target_site_ids, supervise = self._build_query_position_targets(
            batch, query_pos_0, query_batch
        )
        t_query = self._constant_query_times(query_pos_0, 0.0)
        query_disp_target = self._build_query_displacement_targets(
            query_pos_0=query_pos_0, target_pos=target_pos
        )
        out_init = self._forward(
            batch,
            query_pos=query_pos_0,
            query_x=query_x,
            query_batch=query_batch,
            t_query=t_query,
        )
        stage0_pos, query_disp_pred = self._apply_query_displacement(
            query_pos_0, out_init["query_disp"]
        )
        query_distance_stage0_loss = None
        query_disp_stage0_loss = None
        query_distance_refiner_aux_loss = None
        query_disp_refiner_aux_loss = None
        query_distance_layer_losses: list[Tensor] = []
        query_disp_layer_losses: list[Tensor] = []
        if self._query_refiner_enabled():
            out = self._refine_query_outputs(
                out=out_init,
                host_pos=batch.pos,
                host_batch=batch.batch,
                query_pos=stage0_pos.detach(),
                query_batch=query_batch,
            )[1]
            query_loss_pos = out["query_refined_pos"]
            query_distance_stage0_loss = self._query_distance_dfl_loss_for_state(
                out=out_init,
                ranking_pos=query_pos_0.detach(),
                query_batch=query_batch,
                batch=batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
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
        elif self.train_forward_passes == 2:
            out = self._forward(
                batch,
                query_pos=stage0_pos,
                query_x=query_x,
                query_batch=query_batch,
                t_query=t_query,
            )
            query_loss_pos = stage0_pos
        else:
            out = out_init
            query_loss_pos = query_pos_0
        site_query_pos = query_loss_pos
        if self._query_refiner_enabled():
            (
                query_distance_refined_loss,
                query_distance_refiner_aux_loss,
                query_distance_layer_losses,
            ) = self._query_refiner_ranking_aux_losses(
                out=out,
                query_batch=query_batch,
                batch=batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
            if query_distance_stage0_loss is None:
                raise RuntimeError(
                    "query refiner path did not compute stage-0 ranking loss."
                )
            query_distance_loss = self._combine_query_refiner_stage_losses(
                query_distance_stage0_loss, query_distance_refiner_aux_loss
            )
        else:
            query_distance_loss = self._query_ranking_loss_for_state(
                out=out,
                ranking_pos=query_loss_pos,
                query_batch=query_batch,
                batch=batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
            query_distance_refined_loss = query_distance_loss
        site_losses = self._site_detection_losses(
            out,
            batch,
            query_batch=query_batch,
            query_site_ids=target_site_ids,
            query_supervise=supervise,
            query_pos=site_query_pos,
        )
        site_detection_loss = site_losses["site_detection_loss"]
        if self._query_refiner_enabled():
            if query_disp_stage0_loss is None:
                raise RuntimeError(
                    "query refiner path did not compute stage-0 displacement loss."
                )
            (
                query_disp_refined_loss,
                query_disp_refiner_aux_loss,
                query_disp_layer_losses,
                has_refined_disp_supervision,
            ) = self._query_refiner_disp_aux_losses(
                out=out,
                target_pos=target_pos,
                query_batch=query_batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
            query_disp_loss = self._combine_query_refiner_stage_losses(
                query_disp_stage0_loss, query_disp_refiner_aux_loss
            )
            has_query_disp_supervision = (
                has_stage0_disp_supervision or has_refined_disp_supervision
            )
        else:
            query_disp_loss, has_query_disp_supervision = (
                self._query_displacement_loss_from_disp(
                    reference=out["host_logits"],
                    pred_disp=query_disp_pred,
                    target_disp=query_disp_target,
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
            query_disp_loss, has_query_disp_supervision=has_query_disp_supervision
        )
        total = site_detection_objective_loss + query_objective_loss
        self.log("train/loss", total, prog_bar=True, batch_size=batch_size)
        self.log(
            "train/site_detection_loss", site_detection_loss, batch_size=batch_size
        )
        self.log(
            "train/site_cls_loss", site_losses["site_cls_loss"], batch_size=batch_size
        )
        self.log(
            "train/site_mask_loss", site_losses["site_mask_loss"], batch_size=batch_size
        )
        self.log(
            "train/site_dice_loss", site_losses["site_dice_loss"], batch_size=batch_size
        )
        self.log(
            "train/site_contrastive_loss",
            site_losses["site_contrastive_loss"],
            batch_size=batch_size,
        )
        self.log(
            "train/site_affinity_loss",
            site_losses["site_affinity_loss"],
            batch_size=batch_size,
        )
        self.log(
            "train/query_distance_loss", query_distance_loss, batch_size=batch_size
        )
        if self.query_ranking_mode == "confidence":
            self.log(
                "train/query_confidence_loss",
                query_distance_loss,
                batch_size=batch_size,
            )
        self.log("train/query_disp_loss", query_disp_loss, batch_size=batch_size)
        self.log(
            "train/query_contrastive_loss",
            query_contrastive_loss,
            batch_size=batch_size,
        )
        if self._query_refiner_enabled():
            if query_distance_stage0_loss is None or query_disp_stage0_loss is None:
                raise RuntimeError(
                    "query refiner training logs require stage-0 query losses."
                )
            self.log(
                "train/query_stage0_distance_loss",
                query_distance_stage0_loss,
                batch_size=batch_size,
            )
            self.log(
                "train/query_refined_distance_loss",
                query_distance_refined_loss,
                batch_size=batch_size,
            )
            if (
                query_distance_refiner_aux_loss is None
                or query_disp_refiner_aux_loss is None
            ):
                raise RuntimeError("query refiner training logs require aux losses.")
            self.log(
                "train/query_refiner_aux_distance_loss",
                query_distance_refiner_aux_loss,
                batch_size=batch_size,
            )
            self.log(
                "train/query_stage0_disp_loss",
                query_disp_stage0_loss,
                batch_size=batch_size,
            )
            self.log(
                "train/query_refined_disp_loss",
                query_disp_refined_loss,
                batch_size=batch_size,
            )
            self.log(
                "train/query_refiner_aux_disp_loss",
                query_disp_refiner_aux_loss,
                batch_size=batch_size,
            )
            self._log_query_refiner_layer_losses(
                "train",
                batch_size=batch_size,
                distance_losses=query_distance_layer_losses,
                disp_losses=query_disp_layer_losses,
            )
        self.log(
            "train/site_detection_objective_loss",
            site_detection_objective_loss,
            batch_size=batch_size,
        )
        self.log(
            "train/query_objective_loss", query_objective_loss, batch_size=batch_size
        )
        for name, value in site_balance_metrics.items():
            if name == "site_detection_objective_loss":
                continue
            self.log(f"train/{name}", value, batch_size=batch_size)
        for name, value in query_balance_metrics.items():
            if name == "query_objective_loss":
                continue
            self.log(f"train/{name}", value, batch_size=batch_size)
        for name, value in loss_weight_metrics.items():
            self.log(f"train/{name}", value, batch_size=batch_size)
        return total

    def validation_step(self, batch, batch_idx):
        batch_size = int(getattr(batch, "num_graphs", 1))
        dataset_name = str(self.data_cfg["dataset_name"])

        def setup_queries():
            query_pos, query_ids, _ = self._sample_query_positions(
                batch, training=False
            )
            query_features = self._build_query_features(batch, query_batch=query_ids)
            return (query_pos, query_ids, query_features)

        query_pos_0, query_batch, query_x = self._profile_eval_stage(
            "query_setup", dataset_name, batch, setup_queries
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
        query_distance_layer_losses: list[Tensor] = []
        query_disp_layer_losses: list[Tensor] = []
        if self._query_refiner_enabled():
            if out_init is None or stage0_pos is None or query_disp_pred is None:
                raise RuntimeError(
                    "query refiner validation path requires stage-0 outputs."
                )
            query_distance_stage0_loss = self._query_distance_dfl_loss_for_state(
                out=out_init,
                ranking_pos=query_pos_0.detach(),
                query_batch=query_batch,
                batch=batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
        if self._query_refiner_enabled():
            (
                query_distance_refined_loss,
                query_distance_refiner_aux_loss,
                query_distance_layer_losses,
            ) = self._query_refiner_ranking_aux_losses(
                out=out,
                query_batch=query_batch,
                batch=batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
            if query_distance_stage0_loss is None:
                raise RuntimeError(
                    "query refiner validation path did not compute stage-0 ranking loss."
                )
            query_distance_loss = self._combine_query_refiner_stage_losses(
                query_distance_stage0_loss, query_distance_refiner_aux_loss
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
        if self._query_refiner_enabled():
            query_disp_target = self._build_query_displacement_targets(
                query_pos_0=query_pos_0, target_pos=target_pos
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
                query_disp_layer_losses,
                has_refined_disp_supervision,
            ) = self._query_refiner_disp_aux_losses(
                out=out,
                target_pos=target_pos,
                query_batch=query_batch,
                target_site_ids=target_site_ids,
                supervise=supervise,
            )
            query_disp_loss = self._combine_query_refiner_stage_losses(
                query_disp_stage0_loss, query_disp_refiner_aux_loss
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
            query_disp_loss, has_query_disp_supervision=has_query_disp_supervision
        )
        total = site_detection_objective_loss + query_objective_loss
        self.log(
            "val/loss", total, prog_bar=False, batch_size=batch_size, sync_dist=True
        )
        self.log(
            "val/site_detection_loss",
            site_detection_loss,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/site_cls_loss",
            site_losses["site_cls_loss"],
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/site_mask_loss",
            site_losses["site_mask_loss"],
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/site_dice_loss",
            site_losses["site_dice_loss"],
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/site_contrastive_loss",
            site_losses["site_contrastive_loss"],
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/site_affinity_loss",
            site_losses["site_affinity_loss"],
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/query_distance_loss",
            query_distance_loss,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        if self.query_ranking_mode == "confidence":
            self.log(
                "val/query_confidence_loss",
                query_distance_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
        self.log(
            "val/query_disp_loss",
            query_disp_loss,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/query_contrastive_loss",
            query_contrastive_loss,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        if self._query_refiner_enabled():
            if query_distance_stage0_loss is None or query_disp_stage0_loss is None:
                raise RuntimeError(
                    "query refiner validation logs require stage-0 query losses."
                )
            self.log(
                "val/query_stage0_distance_loss",
                query_distance_stage0_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "val/query_refined_distance_loss",
                query_distance_refined_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            if (
                query_distance_refiner_aux_loss is None
                or query_disp_refiner_aux_loss is None
            ):
                raise RuntimeError("query refiner validation logs require aux losses.")
            self.log(
                "val/query_refiner_aux_distance_loss",
                query_distance_refiner_aux_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "val/query_stage0_disp_loss",
                query_disp_stage0_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "val/query_refined_disp_loss",
                query_disp_refined_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "val/query_refiner_aux_disp_loss",
                query_disp_refiner_aux_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            self._log_query_refiner_layer_losses(
                "val",
                batch_size=batch_size,
                distance_losses=query_distance_layer_losses,
                disp_losses=query_disp_layer_losses,
                sync_dist=True,
            )
        self.log(
            "val/site_detection_objective_loss",
            site_detection_objective_loss,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/query_objective_loss",
            query_objective_loss,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        for name, value in site_balance_metrics.items():
            if name == "site_detection_objective_loss":
                continue
            self.log(
                f"val/{name}",
                value,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
        for name, value in query_balance_metrics.items():
            if name == "query_objective_loss":
                continue
            self.log(
                f"val/{name}",
                value,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
        for name, value in loss_weight_metrics.items():
            self.log(
                f"val/{name}",
                value,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
        self._collect_site_detection_eval(
            batch=batch,
            out=out,
            query_pos=final_pos,
            query_batch=query_batch,
            query_rank_scores=query_rank_scores,
        )
        self._collect_query_distance_eval(
            batch=batch,
            query_pos=final_pos,
            query_batch=query_batch,
            query_rank_scores=query_rank_scores,
        )
