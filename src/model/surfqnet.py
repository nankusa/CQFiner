from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
from .gnn.visnet import ViSNetEncoder
from .layers import Dense
from .heads.displacement import ViSNetDisplacementHead
from .heads.query_refiner import QueryEGNNRefiner
from .heads.site_mask import VNDirectSiteMaskHead
from .heads.detr import SiteDETRHead
from .gnn.query_host_attn import normalize_query_to_host_aggregation


class SurfQNet(nn.Module):
    """Equivariant GNN backbone with host/query interaction nodes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
        interaction_num_layers: int | None = None,
        cutoff: float = 12.0,
        n_rbf: int = 64,
        gnn_type: str = "visnet",
        use_time_conditioning: bool = False,
        num_heads: int = 8,
        lmax: int = 1,
        trainable_rbf: bool = False,
        vecnorm_type: str | None = "max_min",
        trainable_vecnorm: bool = False,
        visnet_activation_checkpoint: bool = False,
        num_atom_types: int = 5,
        num_residue_types: int = 21,
        query_distance_num_bins: int = 25,
        query_ranking_head: str = "distance_distribution",
        visnet_node_aggr: str = "sum",
        egnn_node_aggr: str = "mean",
        egnn_norm_feats: bool = False,
        egnn_norm_coords: bool = True,
        egnn_norm_coors_scale_init: float = 0.01,
        egnn_initialization_gain: float = 1.0,
        egnn_degree_norm: str = "none",
        egnn_degree_norm_scope: str = "global",
        egnn_pairnorm: str = "none",
        egnn_pairnorm_scale: float = 1.0,
        egnn_pairnorm_scope: str = "all",
        query_to_host_aggregation: str = "sum",
        query_to_host_coord_update: bool = False,
        displacement_head_aggregation: str = "mean",
        site_detector_type: str = "vn_dot",
        site_detr_config: dict | None = None,
        site_decoder_dropout: float = 0.1,
        site_mask_projection_dim: int | None = None,
        site_mask_distance_cutoff: float = 30.0,
        site_mask_use_distance_gate: bool = True,
        site_mask_rbf_mode: str = "adaptive",
        site_mask_adaptive_rbf_min_radius: float = 2.0,
        site_mask_adaptive_rbf_max_radius: float = 30.0,
        site_mask_adaptive_rbf_init_radius: float = 12.0,
        site_mask_use_affinity: bool = False,
        site_mask_residue_encoder: str = "none",
        query_refiner_enabled: bool = False,
        query_refiner_num_layers: int = 2,
        query_refiner_graph_mode: str = "cutoff",
        query_refiner_cutoff: float = 8.0,
        query_refiner_max_neighbors: int = 32,
        query_refiner_dropout: float = 0.1,
        query_refiner_loss_weight: float = 1.0,
        query_refiner_query_query_edges: bool = False,
        query_refiner_bidirectional: bool = False,
        scalar_vector_output_head: str = "none",
    ):
        super().__init__()
        if (
            gnn_type != "visnet"
            or site_detector_type not in {"vn_dot", "detr"}
            or query_ranking_head != "distance_distribution"
        ):
            raise ValueError(
                "SurfQNet requires ViSNet, a vn_dot or detr mask head, and DFL ranking."
            )
        if (
            use_time_conditioning
            or query_to_host_aggregation != "sum"
            or query_to_host_coord_update
        ):
            raise ValueError(
                "SurfQNet uses fixed protein coordinates and ordinary message aggregation."
            )
        if (
            scalar_vector_output_head != "none"
            or site_mask_use_affinity
            or site_mask_residue_encoder != "none"
        ):
            raise ValueError(
                "Unsupported experimental output or mask encoder; use the baseline implementation."
            )
        self.gnn_type = gnn_type.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_time_conditioning = bool(use_time_conditioning)
        self.num_atom_types = num_atom_types
        self.num_residue_types = num_residue_types
        self.query_distance_num_bins = int(max(query_distance_num_bins, 2))
        self.query_ranking_head = str(query_ranking_head).lower()
        self.query_ranking_head = "distance_distribution"
        self.site_mask_distance_cutoff = float(site_mask_distance_cutoff)
        if self.site_mask_distance_cutoff <= 0.0:
            raise ValueError(
                f"site_mask_distance_cutoff must be positive, got {self.site_mask_distance_cutoff}."
            )
        self.site_mask_use_distance_gate = bool(site_mask_use_distance_gate)
        self.site_mask_rbf_mode = str(site_mask_rbf_mode).lower()
        self.site_mask_adaptive_rbf_min_radius = float(
            site_mask_adaptive_rbf_min_radius
        )
        self.site_mask_adaptive_rbf_max_radius = float(
            site_mask_adaptive_rbf_max_radius
        )
        self.site_mask_adaptive_rbf_init_radius = float(
            site_mask_adaptive_rbf_init_radius
        )
        self.site_mask_use_affinity = bool(site_mask_use_affinity)
        self.site_mask_residue_encoder = str(site_mask_residue_encoder).lower()
        self.query_refiner_enabled = bool(query_refiner_enabled)
        self.query_refiner_num_layers = int(query_refiner_num_layers)
        self.query_refiner_graph_mode = str(query_refiner_graph_mode).lower()
        self.query_refiner_cutoff = float(query_refiner_cutoff)
        self.query_refiner_max_neighbors = int(query_refiner_max_neighbors)
        self.query_refiner_dropout = float(query_refiner_dropout)
        self.query_refiner_loss_weight = float(query_refiner_loss_weight)
        self.query_refiner_query_query_edges = bool(query_refiner_query_query_edges)
        self.query_refiner_bidirectional = bool(query_refiner_bidirectional)
        self.query_to_host_aggregation = normalize_query_to_host_aggregation(
            query_to_host_aggregation
        )
        self.query_to_host_coord_update = bool(query_to_host_coord_update)
        self.displacement_head_aggregation = str(displacement_head_aggregation).lower()
        if self.displacement_head_aggregation not in {"sum", "mean"}:
            raise ValueError(
                f"displacement_head_aggregation must be 'sum' or 'mean', got {displacement_head_aggregation!r}."
            )
        self.interaction_num_layers = int(
            interaction_num_layers if interaction_num_layers is not None else num_layers
        )
        self.cutoff = float(cutoff)
        self.atom_type_embed = nn.Embedding(num_atom_types, input_dim)
        self.residue_type_embed = nn.Embedding(num_residue_types, hidden_dim)
        nn.init.normal_(self.atom_type_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.residue_type_embed.weight, mean=0.0, std=0.02)
        self.input_proj = Dense(input_dim, hidden_dim, activation=nn.SiLU())
        self.interaction_time_mlp = None
        self.interaction_encoder = ViSNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=self.interaction_num_layers,
            cutoff=self.cutoff,
            n_rbf=n_rbf,
            num_heads=num_heads,
            lmax=lmax,
            trainable_rbf=trainable_rbf,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            node_aggr=visnet_node_aggr,
            query_to_host_aggregation=self.query_to_host_aggregation,
            activation_checkpoint=visnet_activation_checkpoint,
        )
        self.scalar_vector_output_head_mode = str(scalar_vector_output_head).lower()
        self.scalar_vector_output_head_mode = "none"
        self.scalar_vector_output_head = None
        self.host_cls_head = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()), Dense(hidden_dim, 2)
        )
        self.host_final_cls_head = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()), Dense(hidden_dim, 2)
        )
        self.query_distance_head = None
        self.query_confidence_head = None
        self.query_distance_head = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()),
            Dense(hidden_dim, self.query_distance_num_bins),
        )
        self.mask_embed_head = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activation=nn.SiLU()),
            Dense(hidden_dim, hidden_dim),
        )
        self.displacement_head = ViSNetDisplacementHead(
            hidden_dim=hidden_dim, aggregation=self.displacement_head_aggregation
        )
        self.site_detector_type = str(site_detector_type).lower()
        if self.query_refiner_enabled:
            if self.query_refiner_loss_weight < 0.0:
                raise ValueError(
                    f"query_refiner_loss_weight must be non-negative, got {self.query_refiner_loss_weight}."
                )
            if self.query_refiner_num_layers <= 0:
                raise ValueError(
                    f"query_refiner_num_layers must be positive, got {self.query_refiner_num_layers}."
                )
            if self.query_refiner_graph_mode not in {"cutoff", "full"}:
                raise ValueError(
                    f"query_refiner_graph_mode must be cutoff or full, got {self.query_refiner_graph_mode!r}."
                )
            if (
                self.query_refiner_graph_mode == "cutoff"
                and self.query_refiner_cutoff <= 0.0
            ):
                raise ValueError(
                    f"query_refiner_cutoff must be positive, got {self.query_refiner_cutoff}."
                )
            if self.query_refiner_max_neighbors <= 0:
                raise ValueError(
                    f"query_refiner_max_neighbors must be positive, got {self.query_refiner_max_neighbors}."
                )
        self.site_contrastive_head = None
        self.site_detector = None
        self.query_refiner = None
        if self.site_detector_type == "detr":
            if site_detr_config is None or site_mask_use_distance_gate:
                raise ValueError(
                    "DETR requires an explicit decoder configuration and distance_gate=false."
                )
            self.site_detector = SiteDETRHead(hidden_dim=hidden_dim, **site_detr_config)
        else:
            if site_detr_config is not None:
                raise ValueError("DETR configuration cannot be used with the vn_dot head.")
            self.site_detector = VNDirectSiteMaskHead(
                hidden_dim=hidden_dim,
                projection_dim=site_mask_projection_dim,
                dropout=site_decoder_dropout,
                distance_num_rbf=n_rbf,
                distance_cutoff=self.site_mask_distance_cutoff,
                use_distance_gate=self.site_mask_use_distance_gate,
                distance_rbf_mode=self.site_mask_rbf_mode,
                adaptive_rbf_min_radius=self.site_mask_adaptive_rbf_min_radius,
                adaptive_rbf_max_radius=self.site_mask_adaptive_rbf_max_radius,
                adaptive_rbf_init_radius=self.site_mask_adaptive_rbf_init_radius,
                use_affinity=self.site_mask_use_affinity,
                residue_encoder=self.site_mask_residue_encoder,
            )
        if self.query_refiner_enabled:
            self.query_refiner = QueryEGNNRefiner(
                hidden_dim=hidden_dim,
                num_layers=self.query_refiner_num_layers,
                graph_mode=self.query_refiner_graph_mode,
                cutoff=self.query_refiner_cutoff,
                max_neighbors=self.query_refiner_max_neighbors,
                dropout=self.query_refiner_dropout,
                query_query_edges=self.query_refiner_query_query_edges,
                bidirectional=self.query_refiner_bidirectional,
                node_aggr=egnn_node_aggr,
                norm_feats=egnn_norm_feats,
                norm_coords=egnn_norm_coords,
                norm_coors_scale_init=egnn_norm_coors_scale_init,
                initialization_gain=egnn_initialization_gain,
                degree_norm=egnn_degree_norm,
                degree_norm_scope=egnn_degree_norm_scope,
                pairnorm=egnn_pairnorm,
                pairnorm_scale=egnn_pairnorm_scale,
                pairnorm_scope=egnn_pairnorm_scope,
            )

    @property
    def query_site_embed_head(self):
        return self.mask_embed_head

    def _apply_scalar_vector_output_head(self, q: Tensor, v: Tensor | None) -> Tensor:
        if self.scalar_vector_output_head is None:
            return q
        return self.scalar_vector_output_head(q, v)

    def _add_atom_type_embedding(self, x: Tensor, atom_type_id: Tensor) -> Tensor:
        if self.atom_type_embed is None:
            return x
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected x feature dim {self.input_dim}, got {x.size(-1)}"
            )
        if bool(((atom_type_id < 0) | (atom_type_id >= self.num_atom_types)).any()):
            raise ValueError("Atom type index is outside the embedding vocabulary.")
        return x + self.atom_type_embed(atom_type_id)

    def _add_residue_type_embedding(self, q: Tensor, residue_type_id: Tensor) -> Tensor:
        if self.residue_type_embed is None:
            return q
        if bool(
            ((residue_type_id < 0) | (residue_type_id >= self.num_residue_types)).any()
        ):
            raise ValueError("Residue type index is outside the embedding vocabulary.")
        return q + self.residue_type_embed(residue_type_id)

    def _encode_host_inputs(
        self, x: Tensor, atom_type_id: Tensor, residue_type_id: Tensor
    ) -> Tensor:
        x = self._add_atom_type_embedding(x, atom_type_id=atom_type_id)
        q = self.input_proj(x)
        return self._add_residue_type_embedding(q, residue_type_id=residue_type_id)

    def forward(
        self,
        host_x: Tensor,
        query_x: Tensor,
        host_atom_type_id: Tensor,
        host_residue_type_id: Tensor,
        host_pos: Tensor,
        query_pos: Tensor,
        interaction_edge_index: Tensor,
        t_query: Tensor,
        interaction_edge_type: Tensor | None = None,
        host_batch: Tensor | None = None,
        query_batch: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        num_host = host_x.size(0)
        num_query = query_x.size(0)
        host_q0 = self._encode_host_inputs(
            x=host_x,
            atom_type_id=host_atom_type_id,
            residue_type_id=host_residue_type_id,
        )
        query_q0 = self.input_proj(query_x)
        interaction_q = torch.cat([host_q0, query_q0], dim=0)
        interaction_pos = torch.cat([host_pos, query_pos], dim=0)
        interaction_node_type = torch.cat(
            [
                torch.zeros(num_host, device=interaction_q.device, dtype=torch.long),
                torch.ones(num_query, device=interaction_q.device, dtype=torch.long),
            ],
            dim=0,
        )
        interaction_t = None
        interaction_mu = interaction_pos.new_zeros(
            (num_host + num_query, 3, self.hidden_dim)
        )
        edge_state = None
        if interaction_edge_index.numel() > 0:
            interaction_q_input = interaction_q
            if self.interaction_time_mlp is not None:
                interaction_q_input = self.interaction_time_mlp(
                    torch.cat([interaction_q, interaction_t], dim=-1)
                )
            interaction_q, interaction_mu, edge_state = self.interaction_encoder(
                pos=interaction_pos,
                edge_index=interaction_edge_index,
                q=interaction_q_input,
                edge_type=interaction_edge_type,
            )
        interaction_q = self._apply_scalar_vector_output_head(
            interaction_q, interaction_mu
        )
        host_q = interaction_q[:num_host]
        query_start = num_host
        query_q = interaction_q[query_start:]
        query_mu = interaction_mu[query_start:]
        host_logits = self.host_final_cls_head(host_q)
        query_displacement = query_pos.new_zeros((num_query, 3))
        if (
            self.displacement_head is not None
            and num_query > 0
            and (interaction_edge_index.numel() > 0)
            and (edge_state is not None)
        ):
            interaction_displacement = self.displacement_head(
                edge_attr=edge_state["edge_attr"],
                edge_vec=edge_state["edge_vec"],
                edge_index=interaction_edge_index,
                num_nodes=interaction_q.size(0),
                edge_gate=None,
            )
            query_displacement = interaction_displacement[query_start:]
        query_site_embed = self.query_site_embed_head(query_q)
        outputs = {
            "host_scalar": host_q,
            "query_scalar": query_q,
            "query_vector": query_mu,
            "host_stage_logits": host_logits,
            "host_final_logits": host_logits,
            "query_site_embed": query_site_embed,
            "mask_embed": query_site_embed,
            "query_disp": query_displacement,
        }
        if self.query_distance_head is not None:
            outputs["query_distance_logits"] = self.query_distance_head(query_q)
        if self.query_confidence_head is not None:
            outputs["query_conf_logits"] = self.query_confidence_head(query_q).squeeze(
                -1
            )
        if not (self.query_refiner_enabled and self.site_detector_type == "detr"):
            outputs.update(
                self.site_detector(
                    host_scalar=host_q,
                    host_batch=host_batch,
                    query_scalar=query_q,
                    query_batch=query_batch,
                    host_pos=host_pos,
                    query_pos=query_pos,
                )
            )
        if self.site_contrastive_head is not None:
            outputs["site_contrastive_embed"] = self.site_contrastive_head(host_q)
        return outputs

    def refine_queries(
        self,
        out: Dict[str, Tensor],
        host_pos: Tensor,
        host_batch: Tensor,
        query_pos: Tensor,
        query_batch: Tensor,
        return_intermediate: bool = True,
    ) -> Dict[str, Tensor]:
        if self.query_refiner is None:
            raise RuntimeError(
                "SiteGNNModel.refine_queries requires query_refiner to be enabled."
            )
        host_scalar = out.get("host_scalar")
        if host_scalar is None:
            raise RuntimeError(
                "query_refiner requires host_scalar in stage-0 model outputs."
            )
        query_scalar = out.get("query_scalar")
        if query_scalar is None:
            raise RuntimeError(
                "query_refiner requires query_scalar in stage-0 model outputs."
            )
        refiner_outputs = self.query_refiner(
            host_scalar=host_scalar,
            host_pos=host_pos,
            host_batch=host_batch,
            query_scalar=query_scalar,
            query_pos=query_pos,
            query_batch=query_batch,
            return_intermediate=return_intermediate,
        )
        state = refiner_outputs.final
        refined_query_scalar, refined_query_pos = (state.query_scalar, state.query_pos)
        intermediate_states = refiner_outputs.layers
        if (
            return_intermediate
            and len(intermediate_states) != self.query_refiner_num_layers
        ):
            raise RuntimeError("Refiner intermediate layer count mismatch.")
        query_site_embed = self.query_site_embed_head(refined_query_scalar)
        refined = dict(out)
        refined["host_scalar"] = state.host_scalar
        refined["site_host_scalar"] = state.host_scalar
        refined["query_scalar"] = refined_query_scalar
        refined["query_refined_pos"] = refined_query_pos
        refined["query_site_embed"] = query_site_embed
        refined["mask_embed"] = query_site_embed
        if self.query_distance_head is not None:
            refined["query_distance_logits"] = self.query_distance_head(
                refined_query_scalar
            )
        if self.query_confidence_head is not None:
            refined["query_conf_logits"] = self.query_confidence_head(
                refined_query_scalar
            ).squeeze(-1)
        if return_intermediate:
            reference_logits = out.get("host_logits", out.get("host_final_logits"))
            if reference_logits is None:
                raise RuntimeError(
                    "query_refiner aux outputs require host_logits or host_final_logits in model outputs."
                )
            aux_outputs: list[Dict[str, Tensor]] = []
            for layer_state in intermediate_states:
                layer_scalar, layer_pos = (
                    layer_state.query_scalar,
                    layer_state.query_pos,
                )
                layer_embed = self.query_site_embed_head(layer_scalar)
                layer_out: Dict[str, Tensor] = {
                    "host_logits": reference_logits,
                    "query_scalar": layer_scalar,
                    "query_refined_pos": layer_pos,
                    "query_site_embed": layer_embed,
                    "mask_embed": layer_embed,
                }
                if self.query_distance_head is not None:
                    layer_out["query_distance_logits"] = self.query_distance_head(
                        layer_scalar
                    )
                if self.query_confidence_head is not None:
                    layer_out["query_conf_logits"] = self.query_confidence_head(
                        layer_scalar
                    ).squeeze(-1)
                aux_outputs.append(layer_out)
            refined["query_refiner_aux_outputs"] = aux_outputs
        if self.site_detector_type == "detr":
            refined.update(
                self.site_detector(
                    host_scalar=state.host_scalar,
                    host_batch=host_batch,
                    query_scalar=refined_query_scalar,
                    query_batch=query_batch,
                    host_pos=host_pos,
                    query_pos=refined_query_pos,
                )
            )
        return refined
