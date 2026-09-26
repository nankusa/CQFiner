from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.utils.metrics import (
    build_query_site_predictions,
    build_target_sites,
    extract_ligand_groups,
)


class EvalCollectionMixin:
    def on_validation_epoch_start(self):
        self._val_query_metric_rows = []
        self._val_site_predictions = []
        self._val_site_targets = []
        self._val_pred_centers = []
        self._val_pred_scores = []
        self._val_ligands = []
        self._val_host_pred_centers = []
        self._val_host_pred_scores = []
        self._val_host_ligands = []

    @staticmethod
    def _build_site_detector_prediction(
        host_pos: Tensor,
        site_logits: Tensor,
        site_mask_logits: Tensor,
        mask_threshold: float = 0.5,
    ) -> dict[str, np.ndarray]:
        num_queries = int(site_logits.size(0))
        num_host = int(host_pos.size(0))
        if num_queries == 0:
            return {
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "centers": np.zeros((0, 3), dtype=np.float32),
                "pocket_masks": np.zeros((0, num_host), dtype=np.float32),
                "pocket_mask_probs": np.zeros((0, num_host), dtype=np.float32),
            }

        class_scores = site_logits.softmax(dim=-1)[:, 0]
        mask_probs = (
            site_mask_logits[:, :num_host].sigmoid()
            if num_host > 0
            else site_logits.new_zeros((num_queries, 0))
        )
        masks = (mask_probs > mask_threshold).float()
        mask_mass = masks.sum(dim=-1)
        mean_mask_prob = (masks * mask_probs).sum(dim=-1) / mask_mass.clamp(min=1.0)
        scores = class_scores * torch.where(
            mask_mass > 0, mean_mask_prob, torch.zeros_like(mean_mask_prob)
        )

        centers = host_pos.new_zeros((num_queries, 3))
        if num_host > 0:
            centers = torch.matmul(masks, host_pos) / mask_mass.clamp(
                min=1.0
            ).unsqueeze(-1)
        order = torch.argsort(scores, descending=True)
        return {
            "scores": scores[order].detach().float().cpu().numpy().astype(np.float32),
            "labels": np.zeros((num_queries,), dtype=np.int64),
            "centers": centers[order].detach().float().cpu().numpy().astype(np.float32),
            "pocket_masks": masks[order]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32),
            "pocket_mask_probs": mask_probs[order]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32),
        }

    def _site_detector_query_source(self) -> str:
        site_detector = getattr(self.model, "site_detector", None)
        raw_query_source = getattr(
            site_detector,
            "query_source",
            getattr(self.model, "site_decoder_query_source", "learned"),
        )
        query_source = str(raw_query_source).lower()
        if query_source == "learnable":
            query_source = "learned"
        if query_source in {"gnn", "query", "query_node", "query_nodes"}:
            query_source = "vn"
        if query_source not in {"learned", "vn"}:
            raise ValueError(
                f"Unsupported site detector query_source during evaluation: {query_source!r}."
            )
        return query_source

    def _collect_vn_dot_site_detection_eval(
        self,
        batch,
        out: dict[str, Tensor],
        query_pos: Tensor | None,
        query_batch: Tensor | None,
        query_rank_scores: Tensor | None,
        dataset_name: str | None = None,
    ) -> None:
        if query_pos is None or query_batch is None or query_rank_scores is None:
            return
        host_scalar = out.get("site_host_scalar", out.get("host_scalar"))
        query_scalar = out.get("query_scalar")
        if host_scalar is None or query_scalar is None:
            raise RuntimeError(
                "VN-dot site-mask evaluation requires host_scalar and query_scalar in model outputs."
            )
        site_detector = getattr(self.model, "site_detector", None)
        if site_detector is None or not hasattr(site_detector, "sample_mask_logits"):
            raise RuntimeError(
                "VN-dot site-mask evaluation requires VNDirectSiteMaskHead.sample_mask_logits."
            )

        target_mask_batch = getattr(
            batch,
            "target_mask_host_id_batch",
            torch.zeros(
                getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                ).size(0),
                dtype=torch.long,
                device=batch.x.device,
            ),
        )
        site_query_mask = out.get("site_query_mask")
        site_sample_ids = out.get("site_sample_ids", batch.batch.unique(sorted=True))
        sample_ids = self._batch_sample_ids(batch)

        for out_idx, sample_idx in enumerate(site_sample_ids.tolist()):
            if site_query_mask is not None and out_idx >= site_query_mask.size(0):
                break
            sample_idx = int(sample_idx)
            host_sel = batch.batch == sample_idx
            query_sel = query_batch == sample_idx
            if not host_sel.any():
                continue
            target_sel = batch.target_pos_batch == sample_idx
            mask_sel = target_mask_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx

            query_indices = query_sel.nonzero(as_tuple=False).view(-1)
            num_host = int(host_sel.sum().item())
            num_pred = int(query_indices.numel())
            if site_query_mask is not None:
                num_pred = min(num_pred, int(site_query_mask.size(1)))
            if num_pred == 0:
                sample_query_pos = query_pos.new_zeros((0, 3))
                sample_query_scores = query_rank_scores.new_zeros((0,))
                sample_mask_logits = host_scalar.new_zeros((0, num_host))
                sample_affinity_logits = None
            else:
                query_indices = query_indices[:num_pred]
                valid_queries = torch.ones(
                    (num_pred,), dtype=torch.bool, device=query_pos.device
                )
                if site_query_mask is not None:
                    valid_queries &= site_query_mask[out_idx, :num_pred].to(
                        device=query_pos.device, dtype=torch.bool
                    )
                if self.site_mask_query_score_threshold > 0:
                    valid_queries &= (
                        query_rank_scores[query_indices]
                        >= self.site_mask_query_score_threshold
                    )
                selected_query_indices = query_indices[valid_queries]
                sample_query_pos = query_pos[selected_query_indices]
                sample_query_scores = query_rank_scores[selected_query_indices]
                if selected_query_indices.numel() == 0:
                    sample_mask_logits = host_scalar.new_zeros((0, num_host))
                    sample_affinity_logits = None
                else:
                    sample_mask_logits = site_detector.sample_mask_logits(
                        host_scalar=host_scalar[host_sel],
                        query_scalar=query_scalar[selected_query_indices],
                        host_pos=batch.pos[host_sel].to(device=host_scalar.device),
                        query_pos=sample_query_pos.to(device=host_scalar.device),
                    )
                    sample_affinity_logits = None
                    if self.site_mask_grouping == "affinity":
                        if not hasattr(site_detector, "query_affinity_logits"):
                            raise RuntimeError(
                                "VN-dot affinity grouping requires query_affinity_logits."
                            )
                        sample_affinity_logits = site_detector.query_affinity_logits(
                            query_scalar=query_scalar[selected_query_indices],
                            query_pos=sample_query_pos.to(device=host_scalar.device),
                        )

            prediction = self._build_vn_dot_site_prediction(
                query_pos=sample_query_pos,
                query_scores=sample_query_scores,
                mask_logits=sample_mask_logits,
                host_pos=batch.pos[host_sel],
                affinity_logits=sample_affinity_logits,
                affinity_threshold=self.site_mask_affinity_threshold,
            )
            target = build_target_sites(
                num_targets=int(target_sel.sum().item()),
                num_host=num_host,
                target_mask_site_id=getattr(
                    batch,
                    "target_mask_site_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
                target_mask_host_id=getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
            )
            sample_id = (
                sample_ids[sample_idx]
                if sample_idx < len(sample_ids)
                else str(sample_idx)
            )
            host_atom_residue_index = getattr(batch, "host_atom_residue_id", None)
            selected_atom_residue_indices = None
            if host_atom_residue_index is not None:
                selected_atom_residue_indices = (
                    host_atom_residue_index[host_sel]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )
            prediction, target = self._maybe_convert_site_eval_to_residue_level(
                prediction=prediction,
                target=target,
                sample_id=sample_id,
                dataset_name=dataset_name,
                atom_residue_indices=selected_atom_residue_indices,
            )
            site_dcc_centers = prediction.get("site_centers", prediction["centers"])
            site_dcc_scores = prediction.get("site_scores", prediction["scores"])
            ligands = extract_ligand_groups(
                ligand_pos=batch.ligand_pos[ligand_sel],
                ligand_ids=batch.ligand_id[ligand_sel],
            )

            if dataset_name is None:
                self._val_site_predictions.append(prediction)
                self._val_site_targets.append(target)
                self._val_host_pred_centers.append(site_dcc_centers)
                self._val_host_pred_scores.append(site_dcc_scores)
                self._val_host_ligands.append(ligands)
            else:
                self._test_site_predictions[dataset_name].append(prediction)
                self._test_site_targets[dataset_name].append(target)
                self._test_host_pred_centers[dataset_name].append(site_dcc_centers)
                self._test_host_pred_scores[dataset_name].append(site_dcc_scores)
                self._test_host_ligands[dataset_name].append(ligands)

    def _collect_detr_vn_site_detection_eval(
        self,
        batch,
        out: dict[str, Tensor],
        query_pos: Tensor | None,
        query_batch: Tensor | None,
        query_rank_scores: Tensor | None,
        dataset_name: str | None = None,
    ) -> None:
        if query_pos is None or query_batch is None or query_rank_scores is None:
            raise RuntimeError(
                "DETR VN-query evaluation requires query_pos, query_batch, and query_rank_scores."
            )
        if query_pos.ndim != 2 or query_pos.size(-1) != 3:
            raise ValueError(
                f"DETR VN-query evaluation expects query_pos [N, 3], got {tuple(query_pos.shape)}."
            )
        if query_batch.shape != (query_pos.size(0),):
            raise ValueError(
                "DETR VN-query evaluation query_batch shape mismatch: "
                f"expected {(query_pos.size(0),)}, got {tuple(query_batch.shape)}."
            )
        if query_rank_scores.shape != (query_pos.size(0),):
            raise ValueError(
                "DETR VN-query evaluation query_rank_scores shape mismatch: "
                f"expected {(query_pos.size(0),)}, got {tuple(query_rank_scores.shape)}."
            )

        site_logits = out.get("site_logits")
        site_mask_logits = out.get("site_mask_logits")
        site_query_mask = out.get("site_query_mask")
        if site_logits is None or site_mask_logits is None:
            raise RuntimeError(
                "DETR VN-query evaluation requires site_logits and site_mask_logits in model outputs."
            )
        if site_logits.ndim != 3:
            raise ValueError(
                f"DETR VN-query evaluation expects site_logits [B, Q, C], got {tuple(site_logits.shape)}."
            )
        if site_mask_logits.ndim != 3:
            raise ValueError(
                f"DETR VN-query evaluation expects site_mask_logits [B, Q, L], got {tuple(site_mask_logits.shape)}."
            )
        if site_logits.shape[:2] != site_mask_logits.shape[:2]:
            raise ValueError(
                "DETR VN-query evaluation logits/mask logits batch-query shape mismatch: "
                f"{tuple(site_logits.shape[:2])} vs {tuple(site_mask_logits.shape[:2])}."
            )
        if (
            site_query_mask is not None
            and site_query_mask.shape[:2] != site_logits.shape[:2]
        ):
            raise ValueError(
                "DETR VN-query evaluation site_query_mask shape mismatch: "
                f"expected {tuple(site_logits.shape[:2])}, got {tuple(site_query_mask.shape)}."
            )

        target_mask_batch = getattr(
            batch,
            "target_mask_host_id_batch",
            torch.zeros(
                getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                ).size(0),
                dtype=torch.long,
                device=batch.x.device,
            ),
        )
        site_sample_ids = out.get("site_sample_ids", batch.batch.unique(sorted=True))
        if site_sample_ids.ndim != 1 or site_sample_ids.size(0) != site_logits.size(0):
            raise ValueError(
                "DETR VN-query evaluation site_sample_ids shape mismatch: "
                f"expected {(site_logits.size(0),)}, got {tuple(site_sample_ids.shape)}."
            )
        sample_ids = self._batch_sample_ids(batch)

        for out_idx, sample_idx in enumerate(site_sample_ids.tolist()):
            sample_idx = int(sample_idx)
            host_sel = batch.batch == sample_idx
            query_sel = query_batch == sample_idx
            if not host_sel.any():
                raise ValueError(
                    f"DETR VN-query evaluation sample {sample_idx} has no host residues."
                )
            if not query_sel.any():
                raise ValueError(
                    f"DETR VN-query evaluation sample {sample_idx} has no query nodes."
                )
            target_sel = batch.target_pos_batch == sample_idx
            mask_sel = target_mask_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx

            query_indices = query_sel.nonzero(as_tuple=False).view(-1)
            sample_site_logits = site_logits[out_idx]
            sample_site_mask_logits = site_mask_logits[out_idx]
            num_host = int(host_sel.sum().item())
            if sample_site_mask_logits.size(-1) < num_host:
                raise ValueError(
                    "DETR VN-query evaluation site_mask_logits has fewer residue positions than host residues: "
                    f"{sample_site_mask_logits.size(-1)} < {num_host}."
                )
            if query_indices.size(0) != sample_site_logits.size(0):
                raise ValueError(
                    "DETR VN-query evaluation query count mismatch between decoder outputs and query_batch: "
                    f"sample={sample_idx}, logits={sample_site_logits.size(0)}, query_batch={query_indices.size(0)}."
                )

            valid_queries = torch.ones(
                (sample_site_logits.size(0),),
                dtype=torch.bool,
                device=sample_site_logits.device,
            )
            if site_query_mask is not None:
                valid_queries &= site_query_mask[out_idx].to(
                    device=sample_site_logits.device, dtype=torch.bool
                )
            if self.site_mask_query_score_threshold > 0:
                valid_queries &= (
                    query_rank_scores[query_indices].to(
                        device=sample_site_logits.device
                    )
                    >= self.site_mask_query_score_threshold
                )
            selected_local_indices = valid_queries.nonzero(as_tuple=False).view(-1)
            selected_query_indices = query_indices[
                selected_local_indices.to(device=query_indices.device)
            ]
            sample_query_pos = query_pos[selected_query_indices]
            sample_query_scores = query_rank_scores[selected_query_indices]
            sample_mask_logits = sample_site_mask_logits[
                selected_local_indices, :num_host
            ]

            prediction = self._build_vn_dot_site_prediction(
                query_pos=sample_query_pos,
                query_scores=sample_query_scores,
                mask_logits=sample_mask_logits,
                host_pos=batch.pos[host_sel],
            )
            target = build_target_sites(
                num_targets=int(target_sel.sum().item()),
                num_host=num_host,
                target_mask_site_id=getattr(
                    batch,
                    "target_mask_site_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
                target_mask_host_id=getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
            )
            sample_id = (
                sample_ids[sample_idx]
                if sample_idx < len(sample_ids)
                else str(sample_idx)
            )
            host_atom_residue_index = getattr(batch, "host_atom_residue_id", None)
            selected_atom_residue_indices = None
            if host_atom_residue_index is not None:
                selected_atom_residue_indices = (
                    host_atom_residue_index[host_sel]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )
            prediction, target = self._maybe_convert_site_eval_to_residue_level(
                prediction=prediction,
                target=target,
                sample_id=sample_id,
                dataset_name=dataset_name,
                atom_residue_indices=selected_atom_residue_indices,
            )
            site_dcc_centers = prediction.get("site_centers", prediction["centers"])
            site_dcc_scores = prediction.get("site_scores", prediction["scores"])
            ligands = extract_ligand_groups(
                ligand_pos=batch.ligand_pos[ligand_sel],
                ligand_ids=batch.ligand_id[ligand_sel],
            )

            if dataset_name is None:
                self._val_site_predictions.append(prediction)
                self._val_site_targets.append(target)
                self._val_host_pred_centers.append(site_dcc_centers)
                self._val_host_pred_scores.append(site_dcc_scores)
                self._val_host_ligands.append(ligands)
            else:
                self._test_site_predictions[dataset_name].append(prediction)
                self._test_site_targets[dataset_name].append(target)
                self._test_host_pred_centers[dataset_name].append(site_dcc_centers)
                self._test_host_pred_scores[dataset_name].append(site_dcc_scores)
                self._test_host_ligands[dataset_name].append(ligands)

    def _collect_site_detection_eval(
        self,
        batch,
        out: dict[str, Tensor],
        dataset_name: str | None = None,
        query_pos: Tensor | None = None,
        query_batch: Tensor | None = None,
        query_rank_scores: Tensor | None = None,
    ) -> None:
        if not self.rank_based_eval:
            return
        if self._site_detector_loss_type() == "per_vn_mask":
            self._collect_vn_dot_site_detection_eval(
                batch=batch,
                out=out,
                query_pos=query_pos,
                query_batch=query_batch,
                query_rank_scores=query_rank_scores,
                dataset_name=dataset_name,
            )
            return
        site_logits = out.get("site_logits")
        site_mask_logits = out.get("site_mask_logits")
        site_query_mask = out.get("site_query_mask")
        if site_logits is None or site_mask_logits is None:
            return
        if self._site_detector_query_source() == "vn":
            self._collect_detr_vn_site_detection_eval(
                batch=batch,
                out=out,
                query_pos=query_pos,
                query_batch=query_batch,
                query_rank_scores=query_rank_scores,
                dataset_name=dataset_name,
            )
            return

        target_mask_batch = getattr(
            batch,
            "target_mask_host_id_batch",
            torch.zeros(
                getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                ).size(0),
                dtype=torch.long,
                device=batch.x.device,
            ),
        )
        site_sample_ids = out.get("site_sample_ids", batch.batch.unique(sorted=True))
        sample_ids = self._batch_sample_ids(batch)

        for out_idx, sample_idx in enumerate(site_sample_ids.tolist()):
            if out_idx >= site_logits.size(0):
                break
            sample_idx = int(sample_idx)
            host_sel = batch.batch == sample_idx
            if not host_sel.any():
                continue
            target_sel = batch.target_pos_batch == sample_idx
            mask_sel = target_mask_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx

            sample_site_logits = site_logits[out_idx]
            sample_site_mask_logits = site_mask_logits[out_idx]
            if site_query_mask is not None:
                valid_queries = site_query_mask[out_idx].to(
                    device=sample_site_logits.device, dtype=torch.bool
                )
                valid_queries = valid_queries[: sample_site_logits.size(0)]
                sample_site_logits = sample_site_logits[valid_queries]
                sample_site_mask_logits = sample_site_mask_logits[valid_queries]

            prediction = self._build_site_detector_prediction(
                host_pos=batch.pos[host_sel],
                site_logits=sample_site_logits,
                site_mask_logits=sample_site_mask_logits,
                mask_threshold=self.site_mask_threshold,
            )
            target = build_target_sites(
                num_targets=int(target_sel.sum().item()),
                num_host=int(host_sel.sum().item()),
                target_mask_site_id=getattr(
                    batch,
                    "target_mask_site_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
                target_mask_host_id=getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
            )
            sample_id = (
                sample_ids[sample_idx]
                if sample_idx < len(sample_ids)
                else str(sample_idx)
            )
            host_atom_residue_index = getattr(batch, "host_atom_residue_id", None)
            selected_atom_residue_indices = None
            if host_atom_residue_index is not None:
                selected_atom_residue_indices = (
                    host_atom_residue_index[host_sel]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )
            prediction, target = self._maybe_convert_site_eval_to_residue_level(
                prediction=prediction,
                target=target,
                sample_id=sample_id,
                dataset_name=dataset_name,
                atom_residue_indices=selected_atom_residue_indices,
            )
            ligands = extract_ligand_groups(
                ligand_pos=batch.ligand_pos[ligand_sel],
                ligand_ids=batch.ligand_id[ligand_sel],
            )

            if dataset_name is None:
                self._val_site_predictions.append(prediction)
                self._val_site_targets.append(target)
                self._val_host_pred_centers.append(prediction["centers"])
                self._val_host_pred_scores.append(prediction["scores"])
                self._val_host_ligands.append(ligands)
            else:
                self._test_site_predictions[dataset_name].append(prediction)
                self._test_site_targets[dataset_name].append(target)
                self._test_host_pred_centers[dataset_name].append(prediction["centers"])
                self._test_host_pred_scores[dataset_name].append(prediction["scores"])
                self._test_host_ligands[dataset_name].append(ligands)

    def _collect_query_distance_eval(
        self,
        batch,
        query_pos: Tensor,
        query_batch: Tensor,
        query_rank_scores: Tensor,
    ) -> None:
        if not self.rank_based_eval:
            return

        for sample_idx in batch.batch.unique(sorted=True).tolist():
            query_sel = query_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx

            if query_sel.any():
                sample_scores = (
                    query_rank_scores[query_sel].detach().float().cpu().numpy()
                )
                coords_np = (
                    query_pos[query_sel]
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
            self._val_pred_centers.append(pred_centers)
            self._val_pred_scores.append(pred_scores)
            self._val_ligands.append(ligands)

    def _collect_unisite_eval(
        self,
        batch,
        final_pos: Tensor,
        query_batch: Tensor,
        host_scalar: Tensor,
        query_site_embed: Tensor,
        query_scores: Tensor,
    ):
        if not self.rank_based_eval:
            return
        host_batch = batch.batch
        target_mask_batch = getattr(
            batch,
            "target_mask_host_id_batch",
            torch.zeros(
                getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                ).size(0),
                dtype=torch.long,
                device=batch.x.device,
            ),
        )

        for sample_idx in batch.batch.unique(sorted=True):
            host_sel = host_batch == sample_idx
            query_sel = query_batch == sample_idx
            target_sel = batch.target_pos_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx
            mask_sel = target_mask_batch == sample_idx

            sample_mask_logits = torch.matmul(
                query_site_embed[query_sel], host_scalar[host_sel].transpose(0, 1)
            )
            prediction = build_query_site_predictions(
                query_pos=final_pos[query_sel],
                query_conf=query_scores[query_sel],
                mask_logits=sample_mask_logits,
            )
            target = build_target_sites(
                num_targets=int(target_sel.sum().item()),
                num_host=int(host_sel.sum().item()),
                target_mask_site_id=getattr(
                    batch,
                    "target_mask_site_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
                target_mask_host_id=getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
            )
            ligands = extract_ligand_groups(
                ligand_pos=batch.ligand_pos[ligand_sel],
                ligand_ids=batch.ligand_id[ligand_sel],
            )

            self._val_site_predictions.append(prediction)
            self._val_site_targets.append(target)
            self._val_pred_centers.append(prediction["centers"])
            self._val_pred_scores.append(prediction["scores"])
            self._val_ligands.append(ligands)

    def on_test_epoch_start(self):
        self._test_loss_rows = {}
        self._test_query_metric_rows = {}
        self._test_site_prediction_outputs = {}
        self._test_site_predictions = {}
        self._test_site_targets = {}
        self._test_pred_centers = {}
        self._test_pred_scores = {}
        self._test_ligands = {}
        self._test_host_pred_centers = {}
        self._test_host_pred_scores = {}
        self._test_host_ligands = {}
        self._test_query_dcc_metrics = {}
        self._test_query_dca_metrics = {}

    def _collect_site_eval_from_prediction_outputs(
        self,
        batch,
        dataset_name: str,
        site_prediction_outputs: list[dict[str, Any]],
    ) -> None:
        target_mask_batch = getattr(
            batch,
            "target_mask_host_id_batch",
            torch.zeros(
                getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                ).size(0),
                dtype=torch.long,
                device=batch.x.device,
            ),
        )
        sample_ids = self._batch_sample_ids(batch)

        for item_idx, item in enumerate(site_prediction_outputs):
            if not isinstance(item, dict):
                raise TypeError(
                    f"site_prediction_outputs[{item_idx}] must be a dict, got {type(item).__name__}."
                )
            if "sample_idx" not in item or "prediction" not in item:
                raise RuntimeError(
                    f"site_prediction_outputs[{item_idx}] requires sample_idx and prediction."
                )
            sample_idx = int(item["sample_idx"])
            prediction = item["prediction"]
            if not isinstance(prediction, dict):
                raise TypeError(
                    f"site_prediction_outputs[{item_idx}]['prediction'] must be a dict, "
                    f"got {type(prediction).__name__}."
                )

            host_sel = batch.batch == sample_idx
            if not bool(host_sel.any().item()):
                raise ValueError(
                    f"Site eval prediction output references sample {sample_idx} with no host nodes."
                )
            target_sel = batch.target_pos_batch == sample_idx
            mask_sel = target_mask_batch == sample_idx
            ligand_sel = batch.ligand_pos_batch == sample_idx

            target = build_target_sites(
                num_targets=int(target_sel.sum().item()),
                num_host=int(host_sel.sum().item()),
                target_mask_site_id=getattr(
                    batch,
                    "target_mask_site_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
                target_mask_host_id=getattr(
                    batch,
                    "target_mask_host_id",
                    torch.empty(0, dtype=torch.long, device=batch.x.device),
                )[mask_sel],
            )
            sample_id = str(
                item.get(
                    "sample_id",
                    sample_ids[sample_idx]
                    if sample_idx < len(sample_ids)
                    else sample_idx,
                )
            )
            atom_residue_indices = item.get("atom_residue_indices")
            if atom_residue_indices is not None:
                atom_residue_indices = np.asarray(atom_residue_indices, dtype=np.int64)
            prediction, target = self._maybe_convert_site_eval_to_residue_level(
                prediction=prediction,
                target=target,
                sample_id=sample_id,
                dataset_name=dataset_name,
                atom_residue_indices=atom_residue_indices,
            )
            site_dcc_centers = prediction.get("site_centers", prediction["centers"])
            site_dcc_scores = prediction.get("site_scores", prediction["scores"])
            ligands = extract_ligand_groups(
                ligand_pos=batch.ligand_pos[ligand_sel],
                ligand_ids=batch.ligand_id[ligand_sel],
            )

            self._test_site_predictions[dataset_name].append(prediction)
            self._test_site_targets[dataset_name].append(target)
            self._test_host_pred_centers[dataset_name].append(site_dcc_centers)
            self._test_host_pred_scores[dataset_name].append(site_dcc_scores)
            self._test_host_ligands[dataset_name].append(ligands)
