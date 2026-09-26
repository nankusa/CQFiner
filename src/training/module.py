from typing import Any, Dict
import math
import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from src.optimize import build_optimizer
from src.utils.metrics import DCA, DCC
from src.training.eval_collect import EvalCollectionMixin
from src.training.eval_postprocess import EvalPostprocessMixin
from src.losses.balance import LossBalanceMixin
from src.training.query import QuerySamplingMixin
from src.losses.query import QueryLossMixin
from src.training.runtime_graph import RuntimeGraphMixin
from src.losses.mask import SiteLossMixin
from src.losses.refinement import RefinementLossMixin
from .fit import FitMixin
from .inference import InferenceMixin
from .testing import TestMixin
from .epoch_metrics import EpochMetricsMixin


class SiteDisplacementModule(
    RefinementLossMixin,
    FitMixin,
    InferenceMixin,
    TestMixin,
    EpochMetricsMixin,
    EvalCollectionMixin,
    RuntimeGraphMixin,
    EvalPostprocessMixin,
    SiteLossMixin,
    QueryLossMixin,
    QuerySamplingMixin,
    LossBalanceMixin,
    pl.LightningModule,
):
    @staticmethod
    def _freeze_module_params(module: nn.Module | None) -> None:
        if module is not None:
            module.requires_grad_(False)

    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: Dict,
        run_config: Dict | None = None,
        num_query_nodes: int = 32,
        eval_num_query_nodes: int | None = None,
        host_host_cutoff: float = 12.0,
        host_query_cutoff: float = 12.0,
        graph_max_neighbors: int = 256,
        rank_based_eval: bool = False,
        loss_weights: Dict | None = None,
        query_loss_balance_mode: str = "uncertainty",
        query_ranking_mode: str = "distance_distribution",
        confidence_gamma: float = 4.0,
        confidence_c0: float = 0.001,
        confidence_positive_weight: float = 1.0,
        confidence_negative_weight: float = 1.0,
        metric_threshold: float = 4.0,
        query_distance_bin_size: float = 0.5,
        query_contrastive_temperature: float = 0.1,
        query_surface_min_offset: float = 1.5,
        query_surface_max_offset: float = 4.0,
        query_surface_tangent_jitter: float = 0.5,
        query_surface_normal_neighbors: int = 16,
        query_eval_nms_radius: float = 6.0,
        site_mask_nms_radius: float | None = None,
        site_mask_query_score_threshold: float = 0.5,
        site_mask_score_aggregation: str = "mean",
        site_mask_threshold: float = 0.5,
        site_mask_grouping: str = "nms",
        site_mask_cluster_threshold: float | None = None,
        site_mask_affinity_threshold: float = 0.45,
        site_mask_affinity_loss_weight: float = 0.0,
        site_mask_affinity_max_pairs: int = 4096,
        query_disp_supervision_cutoff: float = 10.0,
        interaction_host_to_query: bool = True,
        interaction_query_to_host: bool = False,
        interaction_query_to_query: bool = False,
        interaction_graph_mode: str = "cutoff",
        train_forward_passes: int = 1,
        eval_forward_passes: int | None = None,
        eval_use_stage0_offset_without_second_forward: bool = False,
        eval_query_position_cache: bool = False,
        eval_query_sampling_mode: str = "fps",
        runtime_host_host_filter: dict[str, Any] | None = None,
        query_sampling: str = "surface",
        query_volume_radius: float = 10.0,
        query_sampling_seed: int = 42,
        query_nms_score_aggregation: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.run_config = run_config or {}
        self.data_cfg = dict(self.run_config.get("data") or {})
        self.test_data_cfgs = [
            dict(cfg)
            for cfg in self.data_cfg.get("test_datasets", [])
            if isinstance(cfg, dict)
        ]
        self.num_query_nodes = num_query_nodes
        self.eval_num_query_nodes = (
            int(eval_num_query_nodes)
            if eval_num_query_nodes is not None
            else self.num_query_nodes
        )
        self.host_host_cutoff = float(host_host_cutoff)
        self.host_query_cutoff = float(host_query_cutoff)
        self.graph_max_neighbors = graph_max_neighbors
        if type(graph_max_neighbors) is not int or graph_max_neighbors <= 0:
            raise ValueError("graph_max_neighbors must be a positive integer.")
        self.runtime_host_host_filter = self._normalize_runtime_host_host_filter(
            runtime_host_host_filter
        )
        self.rank_based_eval = rank_based_eval
        lw = loss_weights or {"query_disp": 1.0, "site_detection": 1.0}
        self.query_loss_balance_mode = str(query_loss_balance_mode).lower()
        if self.query_loss_balance_mode not in {"uncertainty", "manual", "prog"}:
            raise ValueError(
                f"Unsupported query_loss_balance_mode: {query_loss_balance_mode}"
            )
        self.query_ranking_mode = str(query_ranking_mode).lower()
        if self.query_ranking_mode in {
            "distance",
            "distance_distribution",
            "distribution",
            "dfl",
        }:
            self.query_ranking_mode = "distance_distribution"
        elif self.query_ranking_mode in {"confidence", "vnegnn", "scalar_confidence"}:
            self.query_ranking_mode = "confidence"
        else:
            raise ValueError(f"Unsupported query_ranking_mode: {query_ranking_mode}")
        self.confidence_gamma = float(confidence_gamma)
        self.confidence_c0 = float(confidence_c0)
        self.confidence_positive_weight = float(confidence_positive_weight)
        self.confidence_negative_weight = float(confidence_negative_weight)
        self._loss_weight_modes: dict[str, str] = {}
        self._loss_weight_values: dict[str, float] = {}
        base_class_spec = lw.get("site_detection", 1.0)
        self.w_site_detection = self._configure_loss_weight(
            "site_detection", base_class_spec
        )
        self.w_query_distance = self._configure_loss_weight(
            "query_distance", lw.get("query_distance", base_class_spec)
        )
        self.w_query_disp = self._configure_loss_weight(
            "query_disp", lw.get("query_disp", 1.0)
        )
        self.w_query_contrastive = self._configure_loss_weight(
            "query_contrastive", lw.get("query_contrastive", 0.0)
        )
        self.query_distance_num_bins = int(
            max(getattr(self.model, "query_distance_num_bins", 25), 2)
        )
        self.query_distance_bin_size = float(max(query_distance_bin_size, 0.0001))
        self.query_distance_max = self.query_distance_bin_size * float(
            self.query_distance_num_bins - 1
        )
        self.query_contrastive_temperature = float(
            max(query_contrastive_temperature, 0.0001)
        )
        self.query_surface_min_offset = float(query_surface_min_offset)
        self.query_sampling = str(query_sampling)
        if self.query_sampling not in {"surface", "atom_volume"}:
            raise ValueError(f"Unsupported query sampling: {query_sampling}")
        self.query_volume_radius = float(query_volume_radius)
        if not math.isfinite(self.query_volume_radius) or self.query_volume_radius <= 0:
            raise ValueError("Query volume radius must be finite and positive.")
        if type(query_sampling_seed) is not int or query_sampling_seed < 0:
            raise ValueError("Query sampling seed must be a nonnegative integer.")
        self.query_sampling_seed = query_sampling_seed
        self.query_nms_score_aggregation = (
            ("mean" if self.query_ranking_mode == "confidence" else "square")
            if query_nms_score_aggregation is None else str(query_nms_score_aggregation)
        )
        if self.query_nms_score_aggregation not in {"max", "mean", "square"}:
            raise ValueError(f"Unsupported query NMS aggregation: {query_nms_score_aggregation}")
        self.query_surface_max_offset = float(
            max(query_surface_max_offset, query_surface_min_offset)
        )
        self.query_surface_tangent_jitter = float(query_surface_tangent_jitter)
        self.query_surface_normal_neighbors = int(
            max(query_surface_normal_neighbors, 3)
        )
        self.query_eval_nms_radius = float(max(query_eval_nms_radius, 0.0))
        resolved_site_mask_nms_radius = (
            self.query_eval_nms_radius
            if site_mask_nms_radius is None
            else site_mask_nms_radius
        )
        self.site_mask_nms_radius = float(max(resolved_site_mask_nms_radius, 0.0))
        self.site_mask_query_score_threshold = float(
            max(site_mask_query_score_threshold, 0.0)
        )
        self.site_mask_score_aggregation = str(site_mask_score_aggregation).lower()
        if self.site_mask_score_aggregation not in {
            "mean",
            "max",
            "sum",
            "square",
            "top1",
        }:
            raise ValueError(
                f"Unsupported site_mask_score_aggregation: {site_mask_score_aggregation}"
            )
        self.site_mask_threshold = float(site_mask_threshold)
        self.site_mask_grouping = str(site_mask_grouping).lower()
        if self.site_mask_grouping not in {"nms", "affinity", "cluster"}:
            raise ValueError(f"Unsupported site_mask_grouping: {site_mask_grouping}")
        resolved_cluster_threshold = (
            self.site_mask_nms_radius
            if site_mask_cluster_threshold is None
            else site_mask_cluster_threshold
        )
        self.site_mask_cluster_threshold = float(resolved_cluster_threshold)
        if self.site_mask_cluster_threshold <= 0.0:
            raise ValueError(
                f"site_mask_cluster_threshold must be positive, got {self.site_mask_cluster_threshold}."
            )
        self.site_mask_affinity_threshold = float(site_mask_affinity_threshold)
        if not 0.0 < self.site_mask_affinity_threshold < 1.0:
            raise ValueError(
                f"site_mask_affinity_threshold must be strictly between 0 and 1, got {self.site_mask_affinity_threshold}."
            )
        self.site_mask_affinity_loss_weight = float(site_mask_affinity_loss_weight)
        if self.site_mask_affinity_loss_weight < 0.0:
            raise ValueError(
                f"site_mask_affinity_loss_weight must be non-negative, got {self.site_mask_affinity_loss_weight}."
            )
        self.site_mask_affinity_max_pairs = int(site_mask_affinity_max_pairs)
        if self.site_mask_affinity_max_pairs <= 0:
            raise ValueError(
                f"site_mask_affinity_max_pairs must be positive, got {self.site_mask_affinity_max_pairs}."
            )
        if (
            self.site_mask_grouping in {"affinity", "cluster"}
            and self._site_detector_loss_type() != "per_vn_mask"
        ):
            raise ValueError(
                "site_mask_grouping=affinity/cluster is only supported for VN-dot residue-mask head."
            )
        site_detector = getattr(self.model, "site_detector", None)
        if (
            self.site_mask_grouping == "affinity"
            or self.site_mask_affinity_loss_weight > 0.0
        ) and (not hasattr(site_detector, "query_affinity_logits")):
            raise ValueError(
                "VN-dot affinity grouping/loss requires site detector query_affinity_logits."
            )
        flow_run_cfg = self.run_config.get("flow") or {}
        raw_site_dcc_cfg = flow_run_cfg.get("site_dcc_grasp") or {}
        site_dcc_cfg = (
            dict(raw_site_dcc_cfg) if isinstance(raw_site_dcc_cfg, dict) else {}
        )
        self.site_dcc_prob_threshold = float(
            site_dcc_cfg.get(
                "prob_threshold", flow_run_cfg.get("site_dcc_prob_threshold", 0.3)
            )
        )
        self.site_dcc_cluster_eps = float(
            site_dcc_cfg.get(
                "cluster_eps", flow_run_cfg.get("site_dcc_cluster_eps", 15.0)
            )
        )
        self.site_dcc_cluster_method = str(
            site_dcc_cfg.get(
                "cluster_method", flow_run_cfg.get("site_dcc_cluster_method", "average")
            )
        ).lower()
        if self.site_dcc_cluster_method not in {
            "average",
            "single",
            "complete",
            "dbscan",
            "meanshift",
        }:
            raise ValueError(
                f"Unsupported site_dcc_cluster_method: {self.site_dcc_cluster_method}"
            )
        self.site_dcc_cluster_min_samples = int(
            site_dcc_cfg.get(
                "cluster_min_samples",
                flow_run_cfg.get("site_dcc_cluster_min_samples", 5),
            )
        )
        self.site_dcc_cluster_quantile = float(
            site_dcc_cfg.get(
                "cluster_quantile", flow_run_cfg.get("site_dcc_cluster_quantile", 0.3)
            )
        )
        self.site_dcc_score_aggregation = str(
            site_dcc_cfg.get(
                "score_aggregation",
                flow_run_cfg.get("site_dcc_score_aggregation", "square"),
            )
        ).lower()
        if self.site_dcc_score_aggregation not in {"mean", "sum", "square"}:
            raise ValueError(
                f"Unsupported site_dcc_score_aggregation: {self.site_dcc_score_aggregation}"
            )
        self.site_dcc_centroid_type = str(
            site_dcc_cfg.get(
                "centroid_type", flow_run_cfg.get("site_dcc_centroid_type", "hull")
            )
        ).lower()
        if self.site_dcc_centroid_type not in {"hull", "prob", "square", "centroid"}:
            raise ValueError(
                f"Unsupported site_dcc_centroid_type: {self.site_dcc_centroid_type}"
            )
        self.site_dcc_host_prob_aggregation = str(
            site_dcc_cfg.get(
                "host_prob_aggregation",
                flow_run_cfg.get("site_dcc_host_prob_aggregation", "score_max"),
            )
        ).lower()
        if self.site_dcc_host_prob_aggregation not in {
            "score_max",
            "max",
            "score_mean",
            "mean",
        }:
            raise ValueError(
                f"Unsupported site_dcc_host_prob_aggregation: {self.site_dcc_host_prob_aggregation}"
            )
        self.query_disp_supervision_cutoff = float(query_disp_supervision_cutoff)
        self.use_host_to_query_edges = bool(interaction_host_to_query)
        self.use_query_to_host_edges = bool(interaction_query_to_host)
        self.use_query_to_query_edges = bool(interaction_query_to_query)
        if self.use_query_to_query_edges:
            raise ValueError("SurfQNet does not use query-query edges.")
        if (
            self.model.query_refiner_enabled
            and self.model.query_refiner_query_query_edges
        ):
            raise ValueError(
                "Online graphs require query_refiner.query_query_edges=false."
            )
        self.interaction_graph_mode = str(interaction_graph_mode).lower()
        if self.interaction_graph_mode not in {"cutoff", "full"}:
            raise ValueError(
                f"Unsupported interaction_graph_mode: {interaction_graph_mode}"
            )
        self.train_forward_passes = int(train_forward_passes)
        if self.train_forward_passes not in {1, 2}:
            raise ValueError(
                f"train_forward_passes must be 1 or 2, got {train_forward_passes}"
            )
        if not self._loss_is_enabled("query_disp") and self.train_forward_passes != 1:
            raise ValueError(
                "query_disp loss is disabled, so training must use train_forward_passes=1."
            )
        if (
            bool(getattr(self.model, "query_refiner_enabled", False))
            and self.train_forward_passes != 1
        ):
            raise ValueError(
                "query_refiner_enabled=true requires train_forward_passes=1."
            )
        if (
            bool(getattr(self.model, "query_refiner_enabled", False))
            and self.query_ranking_mode != "distance_distribution"
        ):
            raise ValueError(
                f"query_refiner aux ranking supervision is defined as the distance DFL loss; got query_ranking_mode={self.query_ranking_mode!r}."
            )
        self.eval_forward_passes = int(
            eval_forward_passes if eval_forward_passes is not None else 2
        )
        if self.eval_forward_passes not in {1, 2}:
            raise ValueError(
                f"eval_forward_passes must be 1 or 2, got {eval_forward_passes}"
            )
        self.eval_use_stage0_offset_without_second_forward = bool(
            eval_use_stage0_offset_without_second_forward
        )
        if (
            self.eval_use_stage0_offset_without_second_forward
            and self.eval_forward_passes != 1
        ):
            raise ValueError(
                "eval_use_stage0_offset_without_second_forward=true requires eval_forward_passes=1."
            )
        if not self._loss_is_enabled("query_disp") and self.eval_forward_passes != 1:
            raise ValueError(
                "query_disp loss is disabled, so evaluation must use a single forward pass. Set eval.query_inference.mode=single for no-offset ablations."
            )
        self.eval_query_position_cache = bool(eval_query_position_cache)
        self.eval_query_sampling_mode = str(eval_query_sampling_mode).lower()
        if self.eval_query_sampling_mode not in {"fps", "random", "curvature", "atom_volume"}:
            raise ValueError(
                f"Unsupported eval_query_sampling_mode: {eval_query_sampling_mode!r}."
            )
        if (self.query_sampling == "atom_volume") != (self.eval_query_sampling_mode == "atom_volume"):
            raise ValueError("Atom-volume sampling must be used consistently for training and evaluation.")
        self._eval_query_position_cache: dict[tuple, Tensor] = {}
        self.shared_query_embed = nn.Parameter(torch.zeros(1, self.model.input_dim))
        nn.init.normal_(self.shared_query_embed, mean=0.0, std=0.02)
        if hasattr(self.model, "host_cls_head"):
            self._freeze_module_params(getattr(self.model, "host_cls_head", None))
        if hasattr(self.model, "host_final_cls_head"):
            self._freeze_module_params(getattr(self.model, "host_final_cls_head", None))
        if hasattr(self.model, "host_encoder") and hasattr(
            self.model.host_encoder, "recon"
        ):
            self.model.host_encoder.recon.requires_grad_(False)
        if hasattr(self.model, "host_encoder"):
            self._freeze_module_params(
                getattr(self.model.host_encoder, "decoder", None)
            )
            self._freeze_module_params(
                getattr(self.model.host_encoder, "message_layers", None)
            )
        if not self._loss_is_enabled("site_detection"):
            self._freeze_module_params(getattr(self.model, "site_detector", None))
            self._freeze_module_params(
                getattr(self.model, "site_contrastive_head", None)
            )
        if not self._loss_is_enabled("query_distance"):
            self._freeze_module_params(getattr(self.model, "query_distance_head", None))
            self._freeze_module_params(
                getattr(self.model, "query_confidence_head", None)
            )
        elif self.query_ranking_mode == "confidence":
            self._freeze_module_params(getattr(self.model, "query_distance_head", None))
        else:
            self._freeze_module_params(
                getattr(self.model, "query_confidence_head", None)
            )
        if not self._loss_is_enabled("query_contrastive"):
            self._freeze_module_params(
                getattr(
                    self.model,
                    "query_site_embed_head",
                    getattr(self.model, "mask_embed_head", None),
                )
            )
        if not self._loss_is_enabled("query_disp"):
            self._freeze_module_params(getattr(self.model, "displacement_head", None))
        self.val_dcc = DCC(threshold=metric_threshold)
        self.val_dca = DCA(threshold=metric_threshold)
        self._val_site_predictions = []
        self._val_site_targets = []
        self._val_pred_centers = []
        self._val_pred_scores = []
        self._val_ligands = []
        self._val_host_pred_centers: list[np.ndarray] = []
        self._val_host_pred_scores: list[np.ndarray] = []
        self._val_host_ligands: list[list[np.ndarray]] = []
        self._val_query_metric_rows: list[dict] = []
        self._test_loss_rows: dict[str, list[dict]] = {}
        self._test_query_metric_rows: dict[str, list[dict]] = {}
        self._test_site_predictions: dict[str, list[dict]] = {}
        self._test_site_targets: dict[str, list[dict]] = {}
        self._test_pred_centers: dict[str, list[np.ndarray]] = {}
        self._test_pred_scores: dict[str, list[np.ndarray]] = {}
        self._test_ligands: dict[str, list[list[np.ndarray]]] = {}
        self._test_host_pred_centers: dict[str, list[np.ndarray]] = {}
        self._test_host_pred_scores: dict[str, list[np.ndarray]] = {}
        self._test_host_ligands: dict[str, list[list[np.ndarray]]] = {}
        self._test_query_dcc_metrics: dict[str, DCC] = {}
        self._test_query_dca_metrics: dict[str, DCA] = {}

    def configure_optimizers(self):
        return build_optimizer(self, self.optimizer_cfg)
