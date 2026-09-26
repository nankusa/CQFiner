from copy import deepcopy
from typing import Any


def _normalize_runtime_host_host_filter(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {
            "enabled": False,
            "min_host_nodes": 0,
            "cutoff": None,
            "max_neighbors": None,
        }
    if not isinstance(value, dict):
        raise TypeError(
            f"graph.runtime_host_host_filter must be a mapping, got {type(value)!r}."
        )
    enabled = bool(value.get("enabled", False))
    min_host_nodes = int(value.get("min_host_nodes", 0))
    if min_host_nodes < 0:
        raise ValueError(
            f"graph.runtime_host_host_filter.min_host_nodes must be non-negative, got {min_host_nodes}."
        )
    cutoff = value.get("cutoff")
    cutoff = None if cutoff is None else float(cutoff)
    if cutoff is not None and cutoff <= 0.0:
        raise ValueError(
            f"graph.runtime_host_host_filter.cutoff must be positive, got {cutoff}."
        )
    max_neighbors = value.get("max_neighbors")
    max_neighbors = None if max_neighbors is None else int(max_neighbors)
    if max_neighbors is not None and max_neighbors <= 0:
        raise ValueError(
            f"graph.runtime_host_host_filter.max_neighbors must be positive, got {max_neighbors}."
        )
    if enabled and cutoff is None and (max_neighbors is None):
        raise ValueError(
            "graph.runtime_host_host_filter.enabled=true requires cutoff and/or max_neighbors."
        )
    return {
        "enabled": enabled,
        "min_host_nodes": min_host_nodes,
        "cutoff": cutoff,
        "max_neighbors": max_neighbors,
    }


def _semantic_to_train_config(cfg: dict[str, Any]) -> dict[str, Any]:
    experiment = cfg["experiment"]
    runtime = cfg["runtime"]
    data = cfg["data"]
    features = cfg["features"]
    graph = cfg["graph"]
    model = cfg["model"]
    loss = cfg["loss"]
    eval_cfg = cfg["eval"]
    optimizer = cfg["optimizer"]
    backbone = model["backbone"]
    query = model["query"]
    ranking = model["query_ranking"]
    query_refiner = model.get("query_refiner", {}) or {}
    mask = model["residue_mask"]
    inference = eval_cfg["query_inference"]
    interaction = graph["interaction"]
    runtime_host_host_filter = _normalize_runtime_host_host_filter(
        graph.get("runtime_host_host_filter")
    )
    output_dir = str(runtime["output_dir"])
    trainer = {
        "accelerator": runtime["accelerator"],
        "devices": runtime["devices"],
        "max_epochs": runtime["max_epochs"],
        "test_every_n_epoch": runtime.get("test_every_n_epoch", 0),
        "log_every_n_steps": runtime.get("log_every_n_steps", 10),
        "strategy": runtime["strategy"],
        "precision": runtime["precision"],
        "logger": {
            "save_dir": ".",
            "name": output_dir,
            "version": None,
            "default_hp_metric": False,
        },
    }
    for key in (
        "monitor_gpu_memory",
        "gpu_memory_monitor",
        "checkpoint",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "accumulate_grad_batches",
    ):
        if key in runtime:
            trainer[key] = deepcopy(runtime[key])
    test_datasets = []
    for item in data.get("test_sets", []):
        if not isinstance(item, dict):
            raise TypeError("data.test_sets entries must be mappings")
        test_datasets.append(
            {
                "name": item["name"],
                "dataset_name": item["dataset"],
                "split_tag": item["split"],
            }
        )
    host_feature = str(features["host_feature"]).lower()
    if host_feature not in {"esm", "zeros"}:
        raise ValueError(
            f"features.host_feature must be 'esm' or 'zeros', got {host_feature!r}."
        )
    host_feature_dim = int(features["input_dim"])
    data_cfg = {
        "root": data["root"],
        "input_dir": data.get("input_dir", "protein_ligand"),
        "target_modal": data.get("target", "pocket"),
        "surface_atom_sasa_filter": deepcopy(data.get("surface_atom_sasa_filter")),
        "surface_atom_downsample": deepcopy(data.get("surface_atom_downsample")),
        "dataset_name": data["dataset"],
        "split_tag": data["split"],
        "embedding_modalities": deepcopy(features["embedding_modalities"]),
        "host_feature_mode": features["host_feature"],
        "host_node_mode": data["protein_graph"],
        "host_feature_dim": host_feature_dim,
        "num_query_nodes": query["num_nodes"],
        "batch_size": data["batch_size"],
        "num_workers": data["num_workers"],
        "pin_memory": data.get("pin_memory", True),
        "test_datasets": test_datasets,
    }
    graph_cfg = {
        "host_host_cutoff": graph["host_host_cutoff"],
        "host_query_cutoff": graph["host_query_cutoff"],
        "max_neighbors": graph["max_neighbors"],
    }
    model_cfg = {
        "gnn_type": backbone["name"],
        "cutoff": max(
            float(graph["host_host_cutoff"]), float(graph["host_query_cutoff"])
        ),
        "input_dim": features["input_dim"],
        "hidden_dim": backbone["hidden_dim"],
        "scalar_vector_output_head": backbone["scalar_vector_output_head"],
        "scalar_vector_output_activation": backbone["scalar_vector_output_activation"],
        "scalar_vector_output_residual": backbone["scalar_vector_output_residual"],
        "num_layers": backbone["num_layers"],
        "interaction_num_layers": backbone.get(
            "interaction_num_layers", backbone["num_layers"]
        ),
        "num_atom_types": backbone.get("num_atom_types", 5),
        "num_residue_types": backbone.get("num_residue_types", 21),
        "query_distance_num_bins": ranking.get("distance_num_bins", 49),
        "site_detector_type": mask["head"],
        "site_decoder_dropout": mask["dropout"],
        "site_mask_projection_dim": mask.get("projection_dim"),
        "site_mask_distance_cutoff": mask["distance_cutoff"],
        "site_mask_use_distance_gate": mask["distance_gate"],
        "site_mask_adaptive_rbf_min_radius": mask["adaptive_rbf_min_radius"],
        "site_mask_adaptive_rbf_max_radius": mask["adaptive_rbf_max_radius"],
        "site_mask_adaptive_rbf_init_radius": mask["adaptive_rbf_init_radius"],
        "query_to_host_aggregation": interaction.get(
            "query_to_host_aggregation", "sum"
        ),
        "query_to_host_coord_update": interaction.get(
            "query_to_host_coord_update", False
        ),
        "displacement_head_aggregation": query.get(
            "displacement_head_aggregation", "mean"
        ),
        "visnet": deepcopy(backbone.get("visnet", {})),
        "egnn": deepcopy(backbone.get("egnn", {})),
    }
    if mask["head"] == "detr":
        model_cfg["site_detr_config"] = deepcopy(model["detr"])
    if bool(query_refiner.get("enabled", False)):
        model_cfg.update(
            {
                "query_refiner_enabled": True,
                "query_refiner_bidirectional": interaction["query_to_host"],
                "query_refiner_num_layers": query_refiner.get("num_layers", 2),
                "query_refiner_graph_mode": interaction["mode"],
                "query_refiner_cutoff": query_refiner.get("cutoff", 8.0),
                "query_refiner_max_neighbors": query_refiner.get("max_neighbors", 32),
                "query_refiner_query_query_edges": False,
                "query_refiner_dropout": query_refiner.get(
                    "dropout", backbone.get("egnn", {}).get("dropout", mask["dropout"])
                ),
                "query_refiner_loss_weight": query_refiner.get("loss_weight", 1.0),
            }
        )
    inference_mode = str(inference["mode"]).lower()
    eval_forward_passes = 1 if inference_mode == "single" else 2
    flow_cfg = {
        "query_loss_balance_mode": _normalize_loss_spec(loss["balance"]),
        "train_forward_passes": query["train_forward_passes"],
        "eval_forward_passes": eval_forward_passes,
        "eval_query_position_cache": inference.get("cache_positions", False),
        "eval_query_sampling_mode": "atom_volume" if query.get("sampling", "surface") == "atom_volume" else inference.get("sampling", "fps"),
        "query_sampling": query.get("sampling", "surface"),
        "query_volume_radius": query.get("volume_radius", 10.0),
        "query_sampling_seed": query.get("sampling_seed", cfg["experiment"]["seed"]),
        "query_nms_score_aggregation": inference.get("score_aggregation"),
        "eval_num_query_nodes": inference.get(
            "num_query_nodes", query.get("eval_num_nodes", query["num_nodes"])
        ),
        "interaction_host_to_query": interaction["host_to_query"],
        "interaction_query_to_host": interaction["query_to_host"],
        "interaction_query_to_query": interaction["query_to_query"],
        "interaction_graph_mode": interaction.get("mode", "cutoff"),
        "interaction_query_to_host_aggregation": interaction.get(
            "query_to_host_aggregation", "sum"
        ),
        "interaction_query_to_host_coord_update": interaction.get(
            "query_to_host_coord_update", False
        ),
        "runtime_host_host_filter": runtime_host_host_filter,
        "rank_based_eval": eval_cfg["rank_based"],
        "metric_threshold": eval_cfg["metric_threshold"],
        "query_ranking_mode": ranking["type"],
        "query_eval_nms_radius": inference.get("nms_radius", ranking["nms_radius"]),
        "site_mask_nms_radius": mask["nms_radius"],
        "site_mask_grouping": mask.get("grouping", "nms"),
        "site_mask_cluster_threshold": mask.get(
            "cluster_threshold", mask["nms_radius"]
        ),
        "site_mask_affinity_threshold": mask.get("affinity_threshold", 0.45),
        "site_mask_affinity_loss_weight": mask.get("affinity_loss_weight", 0.0),
        "site_mask_affinity_max_pairs": mask.get("affinity_max_pairs", 4096),
        "site_mask_query_score_threshold": mask["query_score_threshold"],
        "site_mask_score_aggregation": mask["score_aggregation"],
        "site_mask_threshold": mask["mask_threshold"],
        "query_distance_bin_size": ranking["distance_bin_size"],
        "query_contrastive_temperature": loss.get("query_contrastive_temperature", 0.1),
        "query_disp_supervision_cutoff": query["displacement_supervision_cutoff"],
        "query_surface_min_offset": query["surface_min_offset"],
        "query_surface_max_offset": query["surface_max_offset"],
        "query_surface_tangent_jitter": query["surface_tangent_jitter"],
        "query_surface_normal_neighbors": query["surface_normal_neighbors"],
    }
    confidence = loss.get("confidence", {})
    flow_cfg.update(
        {
            "confidence_gamma": confidence.get("gamma", 4.0),
            "confidence_c0": confidence.get("c0", 0.001),
            "confidence_positive_weight": confidence.get("positive_weight", 1.0),
            "confidence_negative_weight": confidence.get("negative_weight", 1.0),
        }
    )
    if "site_dcc" in eval_cfg:
        flow_cfg["site_dcc_grasp"] = deepcopy(eval_cfg["site_dcc"])
    loss_weights = {
        "site_detection": loss["site_detection"],
        "query_disp": _normalize_loss_spec(loss["query_disp"]),
        "query_distance": _normalize_loss_spec(loss["query_distance"]),
        "query_contrastive": loss["query_contrastive"],
    }
    feature_build = deepcopy(cfg.get("feature_build", {}))
    feature_build.setdefault("input_dir", data.get("input_dir", "protein_ligand"))
    feature_build.setdefault(
        "embedding_modal",
        features["embedding_modalities"][0]
        if features["embedding_modalities"]
        else None,
    )
    feature_build.setdefault("target_modal", data.get("target", "pocket"))
    return {
        "seed": experiment.get("seed"),
        "trainer": trainer,
        "data": data_cfg,
        "graph": graph_cfg,
        "model": model_cfg,
        "flow": flow_cfg,
        "loss_weights": loss_weights,
        "optimizer": deepcopy(optimizer),
        "feature_build": feature_build,
    }


def _normalize_loss_spec(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"progressive", "prog"}:
            return "prog"
        return normalized
    return value
