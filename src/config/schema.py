"""Strict schema and cross-stage invariants for SurfQNet ablations."""

import math

SCHEMA = {
    "experiment": {"name", "group", "description", "tags", "seed"},
    "runtime": {
        "accelerator",
        "devices",
        "strategy",
        "precision",
        "max_epochs",
        "output_dir",
        "log_every_n_steps",
        "test_every_n_epoch",
        "accumulate_grad_batches",
        "monitor_gpu_memory",
        "gpu_memory_monitor",
        "checkpoint",
        "gradient_clip_val",
        "gradient_clip_algorithm",
    },
    "data": {
        "root",
        "dataset",
        "split",
        "input_dir",
        "target",
        "protein_graph",
        "batch_size",
        "num_workers",
        "pin_memory",
        "test_sets",
    },
    "features": {"host_feature", "embedding_modalities", "input_dim", "plm"},
    "features.plm": {"enabled", "model", "frozen"},
    "graph": {"host_host_cutoff", "host_query_cutoff", "max_neighbors", "interaction"},
    "graph.interaction": {"mode", "host_to_query", "query_to_host", "query_to_query"},
    "model": {"backbone", "query", "query_ranking", "residue_mask", "query_refiner", "detr"},
    "model.detr": {
        "num_queries", "num_encoder_layers", "num_layers", "num_heads",
        "dim_feedforward", "dropout", "use_vn_context", "query_source", "aux_loss",
    },
    "model.backbone": {
        "name",
        "hidden_dim",
        "num_layers",
        "interaction_num_layers",
        "scalar_vector_output_head",
        "scalar_vector_output_activation",
        "scalar_vector_output_residual",
        "num_atom_types",
        "num_residue_types",
        "visnet",
        "egnn",
    },
    "model.backbone.visnet": {
        "n_rbf",
        "num_heads",
        "lmax",
        "trainable_rbf",
        "vecnorm_type",
        "trainable_vecnorm",
        "node_aggr",
        "activation_checkpoint",
    },
    "model.backbone.egnn": {
        "node_aggr",
        "dropout",
        "norm_feats",
        "norm_coords",
        "norm_coors_scale_init",
    },
    "model.query": {
        "sampling",
        "volume_radius",
        "sampling_seed",
        "num_nodes",
        "eval_num_nodes",
        "offset_prediction",
        "surface_min_offset",
        "surface_max_offset",
        "surface_tangent_jitter",
        "surface_normal_neighbors",
        "displacement_supervision_cutoff",
        "train_forward_passes",
        "displacement_head_aggregation",
    },
    "model.query_ranking": {
        "type",
        "distance_bin_size",
        "distance_num_bins",
        "nms_radius",
    },
    "model.residue_mask": {
        "head",
        "dropout",
        "distance_gate",
        "distance_cutoff",
        "adaptive_rbf_min_radius",
        "adaptive_rbf_max_radius",
        "adaptive_rbf_init_radius",
        "mask_threshold",
        "nms_radius",
        "grouping",
        "query_score_threshold",
        "score_aggregation",
        "projection_dim",
    },
    "model.query_refiner": {
        "enabled",
        "num_layers",
        "cutoff",
        "max_neighbors",
        "dropout",
        "loss_weight",
    },
    "loss": {
        "balance",
        "site_detection",
        "query_distance",
        "query_disp",
        "query_contrastive",
    },
    "eval": {"metric_threshold", "rank_based", "query_inference", "site_dcc"},
    "eval.query_inference": {
        "score_aggregation",
        "mode",
        "num_query_nodes",
        "nms_radius",
        "cache_positions",
        "use_offset_for_center",
    },
    "eval.site_dcc": {
        "prob_threshold",
        "cluster_eps",
        "cluster_method",
        "cluster_min_samples",
        "cluster_quantile",
        "score_aggregation",
        "centroid_type",
        "host_prob_aggregation",
    },
    "optimizer": {
        "name",
        "lr",
        "weight_decay",
        "betas",
        "momentum",
        "patterns",
        "scheduler",
    },
    "optimizer.scheduler": {
        "name",
        "monitor",
        "mode",
        "factor",
        "patience",
        "min_lr",
        "t_max",
        "eta_min",
    },
}


def _known_keys(mapping, path=""):
    if not isinstance(mapping, dict):
        raise TypeError(f"{path or 'config'} must be a mapping.")
    allowed = SCHEMA[path] if path else {key for key in SCHEMA if "." not in key}
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown settings in {path or 'config'}: {sorted(unknown)}")
    for key, value in mapping.items():
        child = f"{path}.{key}" if path else key
        if child in SCHEMA:
            _known_keys(value, child)


def validate_config(cfg):
    _known_keys(cfg)
    required = {key for key in SCHEMA if "." not in key}
    if required - set(cfg):
        raise ValueError(f"Missing config sections: {sorted(required - set(cfg))}")
    graph, model, runtime = cfg["graph"], cfg["model"], cfg["runtime"]
    interaction = graph["interaction"]
    if cfg["data"]["protein_graph"] != "residue":
        raise ValueError("The production SurfQNet uses residue nodes.")
    if cfg["features"]["host_feature"] != "esm":
        raise ValueError("The production SurfQNet uses ESM features.")
    if model["backbone"]["name"] != "visnet":
        raise ValueError(
            "The production backbone is ViSNet; other backbones live under baseline/."
        )
    if (
        model["residue_mask"]["head"] not in {"vn_dot", "detr"}
        or model["residue_mask"]["grouping"] != "nms"
    ):
        raise ValueError(
            "SurfQNet supports vn_dot or detr residue mask heads."
        )
    if model["residue_mask"]["head"] == "detr":
        detr = model["detr"]
        if set(detr) != SCHEMA["model.detr"]:
            raise ValueError("DETR requires all encoder, decoder and query settings explicitly.")
        if detr["query_source"] != "learned" or detr["use_vn_context"] is not False:
            raise ValueError("The restored DETR ablation uses learned mask queries without VN context.")
        if model["residue_mask"]["distance_gate"] is not False:
            raise ValueError("DETR replaces the gated head; distance_gate must be false.")
    elif "detr" in model:
        raise ValueError("model.detr is only valid with residue_mask.head=detr.")
    if model["query_ranking"]["type"] != "distance_distribution":
        raise ValueError("SurfQNet confidence uses the DFL distance distribution.")
    if interaction["mode"] not in {"cutoff", "full"}:
        raise ValueError("graph.interaction.mode must be cutoff or full.")
    for key in ("host_to_query", "query_to_host", "query_to_query"):
        if type(interaction[key]) is not bool:
            raise TypeError(f"graph.interaction.{key} must be boolean.")
    if not interaction["host_to_query"] or interaction["query_to_query"]:
        raise ValueError(
            "Residue-to-query edges are required; query-query edges are not supported."
        )
    for path, value in [
        ("refiner.enabled", model["query_refiner"]["enabled"]),
        ("mask.distance_gate", model["residue_mask"]["distance_gate"]),
    ]:
        if type(value) is not bool:
            raise TypeError(f"{path} must be boolean.")
    if model["query"]["train_forward_passes"] != 1:
        raise ValueError(
            "Training uses one ViSNet forward, with optional EGNN refinement."
        )
    if cfg["eval"]["query_inference"]["mode"] != "rollout":
        raise ValueError(
            "Use rollout: refiner when enabled, otherwise two ViSNet forwards."
        )
    if model["query"]["offset_prediction"] is not True:
        raise ValueError(
            "Both complete and no-refine variants predict the initial offset."
        )
    if (
        model["backbone"]["visnet"]["node_aggr"] != "sum"
        or model["backbone"]["egnn"]["node_aggr"] != "mean"
    ):
        raise ValueError(
            "Ablations fix ViSNet aggregation to sum and EGNN aggregation to mean."
        )
    if model["query"]["displacement_head_aggregation"] != "mean":
        raise ValueError("Ablations fix the initial displacement aggregation to mean.")
    if model["backbone"]["scalar_vector_output_head"] != "none":
        raise ValueError("SurfQNet uses the existing ViSNet scalar outputs directly.")
    ints = {
        "graph.max_neighbors": graph["max_neighbors"],
        "batch_size": cfg["data"]["batch_size"],
        "max_epochs": runtime["max_epochs"],
        "num_queries": model["query"]["num_nodes"],
        "eval_queries": cfg["eval"]["query_inference"]["num_query_nodes"],
        "refine_layers": model["query_refiner"]["num_layers"],
        "refine_neighbors": model["query_refiner"]["max_neighbors"],
        "accumulate_grad_batches": runtime.get("accumulate_grad_batches", 1),
    }
    for name, value in ints.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if type(cfg["data"]["num_workers"]) is not int or cfg["data"]["num_workers"] < 0:
        raise ValueError("data.num_workers must be a nonnegative integer.")
    numbers = {
        "host_host_cutoff": graph["host_host_cutoff"],
        "host_query_cutoff": graph["host_query_cutoff"],
        "refine_cutoff": model["query_refiner"]["cutoff"],
        "supervision_cutoff": model["query"]["displacement_supervision_cutoff"],
        "lr": cfg["optimizer"]["lr"],
    }
    for name, value in numbers.items():
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be finite and positive.")
    if interaction["mode"] == "full":
        if model["query"]["displacement_supervision_cutoff"] < 10000:
            raise ValueError(
                "Full graphs require supervision cutoff >=10000 Å to supervise all queries."
            )
    if (
        model["query"]["num_nodes"] != model["query"]["eval_num_nodes"]
        or model["query"]["num_nodes"]
        != cfg["eval"]["query_inference"]["num_query_nodes"]
    ):
        raise ValueError(
            "Use the same query count for these training/evaluation ablations."
        )
    query = model["query"]
    sampling = query.get("sampling", "surface")
    if sampling not in {"surface", "atom_volume"}:
        raise ValueError(f"Unsupported model.query.sampling: {sampling}")
    if "volume_radius" in query and (
        isinstance(query["volume_radius"], bool)
        or not math.isfinite(float(query["volume_radius"]))
        or float(query["volume_radius"]) <= 0
    ):
        raise ValueError("model.query.volume_radius must be finite and positive.")
    if "sampling_seed" in query and (type(query["sampling_seed"]) is not int or query["sampling_seed"] < 0):
        raise ValueError("model.query.sampling_seed must be a nonnegative integer.")
    if sampling == "atom_volume" and cfg["eval"]["query_inference"].get("cache_positions", False):
        raise ValueError("Atom-volume sampling uses deterministic seeds, not surface-position caching.")
    aggregation = cfg["eval"]["query_inference"].get("score_aggregation")
    if aggregation is not None and aggregation not in {"max", "mean", "square"}:
        raise ValueError(f"Unsupported query NMS score aggregation: {aggregation}")
