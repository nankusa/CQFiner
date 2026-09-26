import numpy as np
import torch

from src.utils.metrics import (
    calc_dca_dcc_metrics,
    evaluate_mask_ap,
    success_rate_from_distances,
)


class EpochMetricsMixin:
    def on_validation_epoch_end(self):
        query_metric_rows = self._gather_objects(self._val_query_metric_rows)
        predictions = self._gather_objects(self._val_site_predictions)
        targets = self._gather_objects(self._val_site_targets)
        pred_centers = self._gather_objects(self._val_pred_centers)
        pred_scores = self._gather_objects(self._val_pred_scores)
        ligands = self._gather_objects(self._val_ligands)
        host_pred_centers = self._gather_objects(self._val_host_pred_centers)
        host_pred_scores = self._gather_objects(self._val_host_pred_scores)
        host_ligands = self._gather_objects(self._val_host_ligands)

        self._val_query_metric_rows.clear()
        self._val_site_predictions.clear()
        self._val_site_targets.clear()
        self._val_pred_centers.clear()
        self._val_pred_scores.clear()
        self._val_ligands.clear()
        self._val_host_pred_centers.clear()
        self._val_host_pred_scores.clear()
        self._val_host_ligands.clear()

        epoch_metrics: dict[str, float] = {}
        if self.rank_based_eval and predictions:
            ap_iou_03 = evaluate_mask_ap(predictions, targets, iou_thr=0.3)
            ap_iou_05 = evaluate_mask_ap(predictions, targets, iou_thr=0.5)

            epoch_metrics["val/ap_iou_0.3"] = ap_iou_03
            epoch_metrics["val/ap_iou_0.5"] = ap_iou_05
            epoch_metrics["val/site_ap_iou_0.3"] = ap_iou_03
            epoch_metrics["val/site_ap_iou_0.5"] = ap_iou_05

        if self.rank_based_eval and pred_centers:
            dcc_topn, dca_topn = calc_dca_dcc_metrics(
                pred_centers_list=pred_centers,
                pred_scores_list=pred_scores,
                ligands_list=ligands,
                top_n_plus=0,
            )
            dcc_topn_plus_2, dca_topn_plus_2 = calc_dca_dcc_metrics(
                pred_centers_list=pred_centers,
                pred_scores_list=pred_scores,
                ligands_list=ligands,
                top_n_plus=2,
            )

            epoch_metrics["val/query_dcc_topn"] = success_rate_from_distances(
                dcc_topn,
                threshold=self.val_dcc.threshold,
            )
            epoch_metrics["val/query_dca_topn"] = success_rate_from_distances(
                dca_topn,
                threshold=self.val_dca.threshold,
            )
            epoch_metrics["val/query_dcc_topn_plus_2"] = success_rate_from_distances(
                dcc_topn_plus_2,
                threshold=self.val_dcc.threshold,
            )
            epoch_metrics["val/query_dca_topn_plus_2"] = success_rate_from_distances(
                dca_topn_plus_2,
                threshold=self.val_dca.threshold,
            )
        elif self.rank_based_eval:
            epoch_metrics["val/query_dcc_topn"] = 0.0
            epoch_metrics["val/query_dca_topn"] = 0.0
            epoch_metrics["val/query_dcc_topn_plus_2"] = 0.0
            epoch_metrics["val/query_dca_topn_plus_2"] = 0.0

        if self.rank_based_eval and host_pred_centers:
            host_dcc_topn, host_dca_topn = calc_dca_dcc_metrics(
                pred_centers_list=host_pred_centers,
                pred_scores_list=host_pred_scores,
                ligands_list=host_ligands,
                top_n_plus=0,
            )
            host_dcc_topn_plus_2, host_dca_topn_plus_2 = calc_dca_dcc_metrics(
                pred_centers_list=host_pred_centers,
                pred_scores_list=host_pred_scores,
                ligands_list=host_ligands,
                top_n_plus=2,
            )

            epoch_metrics["val/site_dcc_topn"] = success_rate_from_distances(
                host_dcc_topn,
                threshold=self.val_dcc.threshold,
            )
            epoch_metrics["val/site_dca_topn"] = success_rate_from_distances(
                host_dca_topn,
                threshold=self.val_dca.threshold,
            )
            epoch_metrics["val/site_dcc_topn_plus_2"] = success_rate_from_distances(
                host_dcc_topn_plus_2,
                threshold=self.val_dcc.threshold,
            )
            epoch_metrics["val/site_dca_topn_plus_2"] = success_rate_from_distances(
                host_dca_topn_plus_2,
                threshold=self.val_dca.threshold,
            )

        self._log_epoch_metrics(epoch_metrics)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        dataset_names = sorted(self._test_loss_rows)
        collect_epoch_metrics = not bool(
            getattr(self, "eval_pocket_minimal", False)
        ) or bool(getattr(self, "eval_collect_epoch_metrics", False))
        for dataset_name in dataset_names:
            loss_rows = self._gather_objects(self._test_loss_rows.get(dataset_name, []))
            epoch_metrics: dict[str, float] = {}
            if collect_epoch_metrics:
                query_metric_rows = self._gather_objects(
                    self._test_query_metric_rows.get(dataset_name, [])
                )
                predictions = self._gather_objects(
                    self._test_site_predictions.get(dataset_name, [])
                )
                targets = self._gather_objects(
                    self._test_site_targets.get(dataset_name, [])
                )
                pred_centers = self._gather_objects(
                    self._test_pred_centers.get(dataset_name, [])
                )
                pred_scores = self._gather_objects(
                    self._test_pred_scores.get(dataset_name, [])
                )
                ligands = self._gather_objects(self._test_ligands.get(dataset_name, []))
                host_pred_centers = self._gather_objects(
                    self._test_host_pred_centers.get(dataset_name, [])
                )
                host_pred_scores = self._gather_objects(
                    self._test_host_pred_scores.get(dataset_name, [])
                )
                host_ligands = self._gather_objects(
                    self._test_host_ligands.get(dataset_name, [])
                )
                epoch_metrics = self._build_site_epoch_metrics(
                    query_metric_rows=query_metric_rows,
                    predictions=predictions,
                    targets=targets,
                    pred_centers=pred_centers,
                    pred_scores=pred_scores,
                    ligands=ligands,
                    host_pred_centers=host_pred_centers,
                    host_pred_scores=host_pred_scores,
                    host_ligands=host_ligands,
                    prefix=f"test/{dataset_name}",
                )
            if loss_rows:
                for key in (
                    "loss",
                    "site_detection_loss",
                    "site_cls_loss",
                    "site_mask_loss",
                    "site_dice_loss",
                    "site_contrastive_loss",
                    "site_affinity_loss",
                    "site_detection_objective_loss",
                    "query_distance_loss",
                    "query_confidence_loss",
                    "query_disp_loss",
                    "query_stage0_distance_loss",
                    "query_refined_distance_loss",
                    "query_refiner_aux_distance_loss",
                    "query_stage0_disp_loss",
                    "query_refined_disp_loss",
                    "query_refiner_aux_disp_loss",
                    "query_contrastive_loss",
                    "query_objective_loss",
                    "site_detection_loss_weight",
                    "site_detection_log_var",
                    "query_distance_loss_weight",
                    "query_disp_loss_weight",
                    "query_contrastive_loss_weight",
                    "query_distance_log_var",
                    "query_disp_log_var",
                    "query_contrastive_log_var",
                ):
                    value = self._weighted_row_mean(loss_rows, key)
                    if not np.isnan(value):
                        epoch_metrics[f"test/{dataset_name}/{key}"] = value
            self._log_epoch_metrics(epoch_metrics)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
