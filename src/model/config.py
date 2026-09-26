"""Normalize explicit ViSNet and refiner parameters into model constructor fields."""

from copy import deepcopy


def resolve_model_config(cfg):
    config = deepcopy(cfg)
    if config.get("gnn_type", "visnet") != "visnet":
        raise ValueError(
            "This release supports the CQFiner ViSNet architecture."
        )
    config["gnn_type"] = "visnet"
    for name in ("visnet", "egnn"):
        values = config.pop(name, {})
        for key, value in values.items():
            target = key
            if name == "egnn":
                target = f"egnn_{key}"
            elif key in {"node_aggr", "activation_checkpoint"}:
                target = f"visnet_{key}"
            if target in config and config[target] != value:
                raise ValueError(f"Conflicting model setting: {target}")
            config[target] = value
    for key, expected in INACTIVE_METADATA.items():
        if key in config:
            value = config.pop(key)
            if value != expected:
                raise ValueError(f"Unsupported inactive model metadata {key}={value!r}")
    return config


# Obsolete non-network metadata present in existing formal checkpoints.
# Non-default values are rejected; no alternative architecture is substituted.
INACTIVE_METADATA = {
    "gotennet_attn_dropout": 0.0,
    "gotennet_layernorm": True,
    "gotennet_tensor_norm": True,
    "gotennet_scale_edge": False,
    "query_confidence_dropout": 0.1,
    "egnn_dropout": 0.1,
    "site_num_queries": 32,
    "site_decoder_num_encoder_layers": 6,
    "site_decoder_num_layers": 6,
    "site_decoder_num_heads": 8,
    "site_decoder_dim_feedforward": 1024,
    "site_decoder_use_vn_context": False,
    "site_decoder_query_source": "learned",
    "site_decoder_aux_loss": True,
    "site_mask_encoder_num_layers": 6,
    "site_mask_encoder_num_heads": 8,
    "site_mask_encoder_dim_feedforward": 1024,
    "site_mask_encoder_dropout": 0.1,
    "site_contrastive_projection_dim": None,
    "scalar_vector_output_activation": "silu",
    "scalar_vector_output_residual": True,
}
