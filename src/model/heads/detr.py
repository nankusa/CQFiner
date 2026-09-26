from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


def get_index_embedding(indices: Tensor, embed_size: int, max_len: int = 2056) -> Tensor:
    """UniSite sine/cosine residue-index positional embedding."""
    if embed_size % 2 != 0:
        raise ValueError(f"UniSite index embedding requires an even embed_size, got {embed_size}.")
    half = embed_size // 2
    k = torch.arange(half, device=indices.device, dtype=torch.float32)
    indices = indices.to(device=indices.device, dtype=torch.float32)
    scale = max_len ** (2 * k / float(embed_size))
    sin = torch.sin(indices[..., None] * math.pi / scale)
    cos = torch.cos(indices[..., None] * math.pi / scale)
    return torch.cat([sin, cos], dim=-1)


class UniSiteTransformer(nn.Module):
    """Transformer used by UniSite's sequence-only detector."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = True,
    ) -> None:
        super().__init__()
        encoder_layer = UniSiteTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = UniSiteTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = UniSiteTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = UniSiteTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self._reset_parameters()
        self.d_model = int(d_model)
        self.nhead = int(nhead)

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        src: Tensor,
        mask: Tensor,
        pos_embed: Tensor | None = None,
        query_embed: Tensor | None = None,
        query_target: Tensor | None = None,
        query_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if src.ndim != 3:
            raise ValueError(f"UniSiteTransformer expects src [B, L, C], got shape {tuple(src.shape)}.")
        if mask.shape != src.shape[:2]:
            raise ValueError(f"UniSiteTransformer mask shape {tuple(mask.shape)} does not match src {tuple(src.shape[:2])}.")
        batch_size = int(src.size(0))
        src = src.permute(1, 0, 2)
        if pos_embed is not None:
            if pos_embed.shape != (batch_size, src.size(0), src.size(-1)):
                raise ValueError(
                    "UniSiteTransformer positional embedding shape mismatch: "
                    f"expected {(batch_size, src.size(0), src.size(-1))}, got {tuple(pos_embed.shape)}."
            )
            pos_embed = pos_embed.permute(1, 0, 2)

        if query_target is None:
            if query_embed is None:
                raise ValueError("UniSiteTransformer learned-query mode requires query_embed.")
            if query_embed.ndim != 2 or query_embed.size(-1) != src.size(-1):
                raise ValueError(
                    "UniSiteTransformer learned query_embed must have shape [Q, C], "
                    f"got {tuple(query_embed.shape)} for C={src.size(-1)}."
                )
            query_pos = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
            target = torch.zeros_like(query_pos)
            target_key_padding_mask = None
        else:
            if query_target.ndim != 3:
                raise ValueError(
                    "UniSiteTransformer query_target mode expects query_target [B, Q, C], "
                    f"got {tuple(query_target.shape)}."
                )
            if query_target.shape[0] != batch_size or query_target.size(-1) != src.size(-1):
                raise ValueError(
                    "UniSiteTransformer query_target shape mismatch: "
                    f"expected [B={batch_size}, Q, C={src.size(-1)}], got {tuple(query_target.shape)}."
                )
            if query_target.device != src.device:
                raise ValueError(
                    f"UniSiteTransformer query_target device mismatch: expected {src.device}, got {query_target.device}."
                )
            if query_mask is None:
                raise ValueError("UniSiteTransformer query_target mode requires query_mask.")
            if query_mask.shape != query_target.shape[:2]:
                raise ValueError(
                    "UniSiteTransformer query_mask shape mismatch: "
                    f"expected {tuple(query_target.shape[:2])}, got {tuple(query_mask.shape)}."
                )
            if query_mask.dtype != torch.bool:
                raise TypeError(f"UniSiteTransformer query_mask must be bool, got {query_mask.dtype}.")
            target = query_target.permute(1, 0, 2)
            target_key_padding_mask = query_mask
            if query_embed is None:
                query_pos = None
            elif query_embed.ndim == 2:
                if query_embed.shape != (query_target.size(1), src.size(-1)):
                    raise ValueError(
                        "UniSiteTransformer query_embed shape mismatch for query_target mode: "
                        f"expected {(query_target.size(1), src.size(-1))}, got {tuple(query_embed.shape)}."
                    )
                query_pos = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
            elif query_embed.ndim == 3:
                if query_embed.shape != query_target.shape:
                    raise ValueError(
                        "UniSiteTransformer query_embed shape mismatch for query_target mode: "
                        f"expected {tuple(query_target.shape)}, got {tuple(query_embed.shape)}."
                    )
                query_pos = query_embed.permute(1, 0, 2)
            else:
                raise ValueError(
                    "UniSiteTransformer query_embed must be [Q, C] or [B, Q, C] in query_target mode, "
                    f"got {tuple(query_embed.shape)}."
                )

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            target,
            memory,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_pos,
        )
        return hs.transpose(1, 2), memory.permute(1, 0, 2)


class UniSiteTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = int(num_layers)
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class UniSiteTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = int(num_layers)
        self.norm = norm
        self.return_intermediate = bool(return_intermediate)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        output = tgt
        intermediate: list[Tensor] = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                if self.norm is None:
                    raise ValueError("UniSite intermediate decoder outputs require a decoder norm.")
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class UniSiteTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = bool(normalize_before)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(src, pos)
        src2 = self.self_attn(query, key, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        src2 = self.norm1(src)
        query = key = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(query, key, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class UniSiteTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = bool(normalize_before)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(query, key, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        tgt2 = self.norm1(tgt)
        query = key = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(query, key, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
        )


class UniSiteMLP(nn.Module):
    """UniSite MLP block used by the class and mask heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        hidden_dims = [hidden_dim] * (self.num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if idx < self.num_layers - 1 else layer(x)
        return x


class SiteDETRHead(nn.Module):
    """UniSite-style DETR residue-mask detector over SiteFlow residue features."""

    loss_type = "detr"

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_encoder_layers: int = 6,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int | None = 1024,
        dropout: float = 0.1,
        use_vn_context: bool = False,
        query_source: str = "learned",
        aux_loss: bool = True,
        index_max_len: int = 2056,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dim_feedforward = int(dim_feedforward if dim_feedforward is not None else 1024)
        self.dropout = float(dropout)
        self.aux_loss = bool(aux_loss)
        self.index_max_len = int(index_max_len)
        query_source = str(query_source).lower()
        if query_source == "learnable":
            query_source = "learned"
        if query_source in {"gnn", "query", "query_node", "query_nodes"}:
            query_source = "vn"
        self.query_source = query_source
        self.use_vn_context = bool(use_vn_context)
        if query_source not in {"learned", "vn"}:
            raise ValueError(f"Unsupported UniSite-style DETR query_source: {query_source!r}.")
        if query_source == "learned" and self.use_vn_context:
            raise ValueError("UniSite-style DETR learned-query mode must set use_vn_context=false.")
        if query_source == "vn" and not self.use_vn_context:
            raise ValueError("UniSite-style DETR VN-query mode must set use_vn_context=true.")
        if self.num_queries <= 0:
            raise ValueError(f"UniSite-style DETR head requires num_queries > 0, got {self.num_queries}.")
        if self.num_encoder_layers <= 0:
            raise ValueError(
                f"UniSite-style DETR head requires num_encoder_layers > 0, got {self.num_encoder_layers}."
            )
        if self.num_decoder_layers <= 0:
            raise ValueError(f"UniSite-style DETR head requires num_layers > 0, got {self.num_decoder_layers}.")
        if self.num_heads <= 0:
            raise ValueError(f"UniSite-style DETR head requires num_heads > 0, got {self.num_heads}.")
        if self.dim_feedforward <= 0:
            raise ValueError(
                f"UniSite-style DETR head requires dim_feedforward > 0, got {self.dim_feedforward}."
            )
        if self.hidden_dim % 2 != 0:
            raise ValueError(f"UniSite-style DETR requires even hidden_dim for index embeddings, got {self.hidden_dim}.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads for UniSite DETR: {self.hidden_dim} vs {self.num_heads}."
            )

        self.transformer = UniSiteTransformer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="relu",
            return_intermediate_dec=True,
        )
        self.query_embed = None
        if self.query_source == "learned":
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.class_head = UniSiteMLP(self.hidden_dim, self.hidden_dim, 2, 2)
        self.mask_embed_head = UniSiteMLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)

    def _build_vn_decoder_target(
        self,
        query_scalar: Tensor,
        query_batch: Tensor,
        sample_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if query_scalar.ndim != 2 or query_scalar.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Expected query_scalar [N, {self.hidden_dim}] for VN-query DETR, got shape {tuple(query_scalar.shape)}."
            )
        if query_batch.shape != (query_scalar.size(0),):
            raise ValueError(
                "VN-query DETR query_batch shape mismatch: "
                f"expected {(query_scalar.size(0),)}, got {tuple(query_batch.shape)}."
            )
        sample_id_list = [int(sample_id) for sample_id in sample_ids.tolist()]
        host_sample_ids = set(sample_id_list)
        query_sample_ids = {int(sample_id) for sample_id in query_batch.unique(sorted=True).tolist()}
        missing = sorted(host_sample_ids - query_sample_ids)
        extra = sorted(query_sample_ids - host_sample_ids)
        if missing or extra:
            raise ValueError(
                "VN-query DETR host/query sample id mismatch: "
                f"missing query samples={missing}, extra query samples={extra}."
            )

        query_target = query_scalar.new_zeros((len(sample_id_list), self.num_queries, self.hidden_dim))
        query_padding_mask = torch.ones(
            (len(sample_id_list), self.num_queries),
            dtype=torch.bool,
            device=query_scalar.device,
        )
        site_query_mask = torch.zeros_like(query_padding_mask)
        for out_idx, sample_id in enumerate(sample_id_list):
            query_sel = query_batch == sample_id
            sample_queries = query_scalar[query_sel]
            num_query = int(sample_queries.size(0))
            if num_query != self.num_queries:
                raise ValueError(
                    "VN-query DETR requires exactly num_queries query nodes per sample: "
                    f"sample={sample_id}, expected={self.num_queries}, got={num_query}."
                )
            query_target[out_idx] = sample_queries
            query_padding_mask[out_idx, :num_query] = False
            site_query_mask[out_idx, :num_query] = True
        return query_target, query_padding_mask, site_query_mask

    def forward(
        self,
        host_scalar: Tensor,
        host_batch: Tensor | None,
        query_scalar: Tensor | None = None,
        query_batch: Tensor | None = None,
        host_pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Dict[str, Tensor | list[dict[str, Tensor]]]:
        del host_pos, query_pos
        if host_batch is None:
            raise ValueError("UniSite-style DETR head requires host_batch to build per-protein residue memory.")
        if host_scalar.ndim != 2 or host_scalar.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Expected host_scalar [N, {self.hidden_dim}], got shape {tuple(host_scalar.shape)}."
            )

        sample_ids = host_batch.unique(sorted=True)
        batch_size = int(sample_ids.numel())
        if batch_size <= 0:
            raise ValueError("UniSite-style DETR head received an empty host batch.")

        host_counts = [int((host_batch == sample_id).sum().item()) for sample_id in sample_ids.tolist()]
        max_host = max(host_counts)
        if max_host <= 0:
            raise ValueError("UniSite-style DETR head received no residue nodes.")

        memory_input = host_scalar.new_zeros((batch_size, max_host, self.hidden_dim))
        attn_mask = torch.ones((batch_size, max_host), dtype=torch.bool, device=host_scalar.device)
        seq_idx = torch.zeros((batch_size, max_host), dtype=torch.long, device=host_scalar.device)
        site_host_mask = torch.zeros((batch_size, max_host), dtype=torch.bool, device=host_scalar.device)
        for out_idx, sample_id in enumerate(sample_ids.tolist()):
            host_sel = host_batch == sample_id
            sample_memory = host_scalar[host_sel]
            num_host = int(sample_memory.size(0))
            if num_host <= 0:
                raise ValueError(f"Sample {sample_id} has no residue nodes for UniSite-style DETR.")
            memory_input[out_idx, :num_host] = sample_memory
            attn_mask[out_idx, :num_host] = False
            seq_idx[out_idx, :num_host] = torch.arange(
                1,
                num_host + 1,
                dtype=torch.long,
                device=host_scalar.device,
            )
            site_host_mask[out_idx, :num_host] = True

        pos_embed = get_index_embedding(seq_idx, embed_size=self.hidden_dim, max_len=self.index_max_len).to(dtype=host_scalar.dtype)
        if self.query_source == "learned":
            if self.query_embed is None:
                raise RuntimeError("UniSite-style DETR learned-query mode has no query_embed module.")
            hs, node_embed = self.transformer(
                memory_input,
                attn_mask,
                pos_embed=pos_embed,
                query_embed=self.query_embed.weight,
            )
            site_query_mask = torch.ones((batch_size, self.num_queries), dtype=torch.bool, device=host_scalar.device)
        else:
            if query_scalar is None or query_batch is None:
                raise ValueError("UniSite-style DETR VN-query mode requires query_scalar and query_batch.")
            query_target, query_padding_mask, site_query_mask = self._build_vn_decoder_target(
                query_scalar=query_scalar,
                query_batch=query_batch,
                sample_ids=sample_ids,
            )
            hs, node_embed = self.transformer(
                memory_input,
                attn_mask,
                pos_embed=pos_embed,
                query_target=query_target,
                query_mask=query_padding_mask,
            )
        output_class = self.class_head(hs)
        mask_embed = self.mask_embed_head(hs)
        output_mask = torch.einsum("dbqc,blc->dbql", mask_embed, node_embed)

        outputs: Dict[str, Tensor | list[dict[str, Tensor]]] = {
            "site_logits": output_class[-1],
            "site_mask_logits": output_mask[-1],
            "site_host_mask": site_host_mask,
            "site_query_mask": site_query_mask,
            "site_sample_ids": sample_ids,
        }
        if self.aux_loss:
            outputs["site_aux_outputs"] = [
                {"site_logits": class_logits, "site_mask_logits": mask_logits}
                for class_logits, mask_logits in zip(output_class[:-1], output_mask[:-1])
            ]
        return outputs


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(int(count))])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise ValueError(f"Unsupported UniSite transformer activation: {activation}.")
