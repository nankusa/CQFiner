"""The production residue SurfQNet architecture."""

import inspect

from .config import resolve_model_config
from .surfqnet import SurfQNet


def build_site_model(cfg):
    config = resolve_model_config(cfg)
    unknown = set(config) - set(inspect.signature(SurfQNet).parameters)
    if unknown:
        raise ValueError(f"Unsupported SurfQNet model settings: {sorted(unknown)}")
    return SurfQNet(**config)


def summarize_site_model(model):
    return {
        "model_class": type(model).__name__,
        "gnn_type": model.gnn_type,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "site_detector_type": model.site_detector_type,
        "query_refiner_enabled": model.query_refiner_enabled,
        "query_refiner_class": type(model.query_refiner).__name__
        if model.query_refiner_enabled
        else None,
        "visnet": {
            "num_layers": model.interaction_num_layers,
            "node_aggr": model.interaction_encoder.node_aggr,
        },
        "query_refiner": {
            "num_layers": model.query_refiner_num_layers,
            "graph_mode": model.query_refiner_graph_mode,
            "cutoff": model.query_refiner_cutoff,
            "max_neighbors": model.query_refiner_max_neighbors,
            "bidirectional": model.query_refiner_bidirectional,
            "fixed_host_coordinates": True,
        },
        "distance_gate": model.site_mask_use_distance_gate,
    }


__all__ = [
    "SurfQNet",
    "build_site_model",
    "resolve_model_config",
    "summarize_site_model",
]
