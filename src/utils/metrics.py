import numpy as np
import torch
from torch_geometric.nn.pool import knn
from torchmetrics import Metric


def _unique_ligand_index(ligand_ids: torch.Tensor, batch_ids: torch.Tensor):
    change = ligand_ids[1:] != ligand_ids[:-1]
    change = torch.cat(
        [
            torch.tensor([0], dtype=torch.bool, device=ligand_ids.device),
            change,
        ]
    )
    unique = torch.cumsum(change, dim=0) + batch_ids
    change = unique[1:] != unique[:-1]
    change = torch.cat(
        [
            torch.tensor([0], dtype=torch.bool, device=ligand_ids.device),
            change,
        ]
    )
    return torch.cumsum(change, dim=0)


class DCC(Metric):
    def __init__(self, threshold: float = 4.0):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred_pos: torch.Tensor,
        target_pos: torch.Tensor,
        pred_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ):
        if target_pos.numel() == 0:
            return
        if pred_pos.numel() == 0:
            self.total += target_pos.size(0)
            return
        assign = knn(
            x=pred_pos.float(),
            y=target_pos.float(),
            batch_x=pred_batch,
            batch_y=target_batch,
            k=1,
        )
        dist = torch.norm(target_pos[assign[0]] - pred_pos[assign[1]], dim=-1)
        self.correct += (dist <= self.threshold).sum()
        self.total += target_pos.size(0)

    def compute(self):
        return self.correct.float() / self.total.clamp(min=1)


class DCA(Metric):
    def __init__(self, threshold: float = 4.0):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred_pos: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_ids: torch.Tensor,
        pred_batch: torch.Tensor,
        ligand_batch: torch.Tensor,
    ):
        unique_lig = _unique_ligand_index(ligand_ids, ligand_batch)
        for b in ligand_batch.unique():
            sample_ligs = unique_lig[ligand_batch == b].unique()
            sample_pred = pred_pos[pred_batch == b]
            for lig in sample_ligs:
                lp = ligand_pos[unique_lig == lig]
                dist = torch.norm(lp[:, None] - sample_pred[None], dim=-1)
                self.correct += (dist <= self.threshold).any()
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total.clamp(min=1)


def dense_site_masks(
    num_targets: int,
    num_host: int,
    target_mask_site_id: torch.Tensor,
    target_mask_host_id: torch.Tensor,
) -> np.ndarray:
    masks = np.zeros((num_targets, num_host), dtype=np.float32)
    if num_targets == 0 or num_host == 0 or target_mask_site_id.numel() == 0:
        return masks

    site_ids = target_mask_site_id.detach().cpu().numpy()
    host_ids = target_mask_host_id.detach().cpu().numpy()
    masks[site_ids, host_ids] = 1.0
    return masks


def build_query_site_predictions(
    query_pos: torch.Tensor,
    query_conf: torch.Tensor,
    mask_logits: torch.Tensor,
    mask_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    num_queries = query_pos.size(0)
    num_host = mask_logits.size(1) if mask_logits.ndim == 2 else 0

    if num_queries == 0:
        return {
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "centers": np.zeros((0, 3), dtype=np.float32),
            "pocket_masks": np.zeros((0, num_host), dtype=np.float32),
            "pocket_mask_probs": np.zeros((0, num_host), dtype=np.float32),
        }

    scores = torch.sigmoid(query_conf.reshape(-1))
    masks = torch.zeros(
        (num_queries, num_host), dtype=torch.float32, device=query_pos.device
    )
    mask_probs = torch.zeros(
        (num_queries, num_host), dtype=torch.float32, device=query_pos.device
    )

    if num_host > 0 and mask_logits.numel() > 0:
        mask_probs = torch.sigmoid(mask_logits)
        masks = (mask_probs > mask_threshold).float()
        mean_mask_prob = (masks * mask_probs).sum(dim=-1) / masks.sum(dim=-1).clamp(
            min=1.0
        )
        scores = scores * torch.where(
            masks.sum(dim=-1) > 0, mean_mask_prob, torch.zeros_like(mean_mask_prob)
        )
    else:
        scores = torch.zeros_like(scores)

    order = torch.argsort(scores, descending=True)
    return {
        "scores": scores[order].detach().float().cpu().numpy().astype(np.float32),
        "labels": np.zeros((num_queries,), dtype=np.int64),
        "centers": query_pos[order].detach().float().cpu().numpy().astype(np.float32),
        "pocket_masks": masks[order].detach().float().cpu().numpy().astype(np.float32),
        "pocket_mask_probs": mask_probs[order]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def build_target_sites(
    num_targets: int,
    num_host: int,
    target_mask_site_id: torch.Tensor,
    target_mask_host_id: torch.Tensor,
) -> dict[str, np.ndarray]:
    return {
        "labels": np.zeros((num_targets,), dtype=np.int64),
        "pocket_masks": dense_site_masks(
            num_targets=num_targets,
            num_host=num_host,
            target_mask_site_id=target_mask_site_id,
            target_mask_host_id=target_mask_host_id,
        ),
        "res_mask": np.ones((num_host,), dtype=bool),
    }


def extract_ligand_groups(
    ligand_pos: torch.Tensor, ligand_ids: torch.Tensor
) -> list[np.ndarray]:
    if ligand_pos.numel() == 0:
        return []

    groups = []
    for ligand_id in torch.unique(ligand_ids, sorted=True):
        coords = ligand_pos[ligand_ids == ligand_id]
        groups.append(coords.detach().float().cpu().numpy().astype(np.float32))
    return groups


def calc_dca_dcc_metrics(
    pred_centers_list,
    pred_scores_list,
    ligands_list,
    top_n_plus: int = 0,
):
    dca_results = []
    dcc_results = []
    for pred_centers, pred_scores, ligands in zip(
        pred_centers_list, pred_scores_list, ligands_list
    ):
        pred_centers = np.asarray(pred_centers, dtype=np.float32)
        pred_scores = np.asarray(pred_scores, dtype=np.float32)
        num_preds = len(pred_centers)
        num_ligs = len(ligands)

        if num_ligs == 0:
            dca_results.append(np.zeros((0,), dtype=np.float32))
            dcc_results.append(np.zeros((0,), dtype=np.float32))
            continue
        if num_preds == 0:
            dca_results.append(np.full(num_ligs, np.inf, dtype=np.float32))
            dcc_results.append(np.full(num_ligs, np.inf, dtype=np.float32))
            continue

        sorted_ids = np.argsort(pred_scores)[::-1]
        sorted_ids = sorted_ids[: num_ligs + top_n_plus]
        pred_centers = pred_centers[sorted_ids]

        dcc_matrix = np.zeros((num_ligs, len(pred_centers)), dtype=np.float32)
        dca_matrix = np.zeros((num_ligs, len(pred_centers)), dtype=np.float32)
        for lig_idx, ligand_coords in enumerate(ligands):
            ligand_coords = np.asarray(ligand_coords, dtype=np.float32)
            ligand_center = ligand_coords.mean(axis=0)
            dcc_matrix[lig_idx] = np.linalg.norm(pred_centers - ligand_center, axis=1)
            distances = np.linalg.norm(
                pred_centers[:, None, :] - ligand_coords[None, :, :], axis=-1
            )
            dca_matrix[lig_idx] = distances.min(axis=1)

        dcc_results.append(dcc_matrix.min(axis=1))
        dca_results.append(dca_matrix.min(axis=1))
    return dcc_results, dca_results


def success_rate_from_distances(distances_list, threshold: float = 4.0) -> float:
    non_empty = [
        np.asarray(distances, dtype=np.float32)
        for distances in distances_list
        if len(distances) > 0
    ]
    if len(non_empty) == 0:
        return 0.0
    merged = np.concatenate(non_empty)
    if merged.size == 0:
        return 0.0
    return float(np.mean(merged < threshold))


def mask_overlaps(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    masks1 = masks1.astype(bool)
    masks2 = masks2.astype(bool)
    intersection = masks1[:, None, :] & masks2[None, :, :]
    union = masks1[:, None, :] | masks2[None, :, :]
    return intersection.sum(axis=-1) / (union.sum(axis=-1) + 1e-6)


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0 or precisions.size == 0:
        return 0.0

    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]
    zeros = np.zeros((1, 1), dtype=recalls.dtype)
    ones = np.ones((1, 1), dtype=recalls.dtype)
    mrec = np.hstack((zeros, recalls, ones))
    mpre = np.hstack((zeros, precisions, zeros))
    for i in range(mpre.shape[1] - 1, 0, -1):
        mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
    ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
    return float(np.sum((mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1]))


def tpfp_mask(
    pred_masks: np.ndarray,
    pred_scores: np.ndarray,
    gt_masks: np.ndarray,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_dets = pred_masks.shape[0]
    num_gts = gt_masks.shape[0]
    tp = np.zeros((num_dets,), dtype=np.float32)
    fp = np.zeros((num_dets,), dtype=np.float32)

    if num_gts == 0:
        fp[...] = 1
        return tp, fp
    if num_dets == 0:
        return tp, fp

    ious = mask_overlaps(pred_masks, gt_masks)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-pred_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)
    for idx in sort_inds:
        if ious_max[idx] >= iou_thr:
            matched_gt = ious_argmax[idx]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[idx] = 1
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1

    return tp, fp


def evaluate_mask_ap(predictions, targets, iou_thr: float) -> float:
    if len(predictions) == 0:
        return 0.0

    all_scores = []
    all_tp = []
    all_fp = []
    num_gts = 0

    for prediction, target in zip(predictions, targets):
        pred_masks = np.asarray(prediction["pocket_masks"], dtype=np.float32)
        pred_scores = np.asarray(prediction["scores"], dtype=np.float32)
        pred_labels = np.asarray(
            prediction.get("labels", np.zeros((pred_masks.shape[0],), dtype=np.int64)),
            dtype=np.int64,
        )
        gt_masks = np.asarray(target["pocket_masks"], dtype=np.float32)
        gt_labels = np.asarray(
            target.get("labels", np.zeros((gt_masks.shape[0],), dtype=np.int64)),
            dtype=np.int64,
        )
        res_mask = np.asarray(
            target.get("res_mask", np.ones((gt_masks.shape[1],), dtype=bool)),
            dtype=bool,
        )

        pred_class = pred_labels == 0
        gt_class = gt_labels == 0
        pred_masks = pred_masks[pred_class]
        pred_scores = pred_scores[pred_class]
        gt_masks = gt_masks[gt_class]

        if pred_masks.size > 0:
            pred_masks = pred_masks[:, res_mask]
        if gt_masks.size > 0:
            gt_masks = gt_masks[:, res_mask]

        tp, fp = tpfp_mask(pred_masks, pred_scores, gt_masks, iou_thr=iou_thr)
        all_scores.append(pred_scores)
        all_tp.append(tp)
        all_fp.append(fp)
        num_gts += gt_masks.shape[0]

    if num_gts == 0:
        return 0.0

    non_empty_scores = [scores for scores in all_scores if scores.size > 0]
    if len(non_empty_scores) == 0:
        return 0.0

    scores = np.concatenate(non_empty_scores)
    order = np.argsort(-scores)
    tp = np.concatenate(all_tp)[order]
    fp = np.concatenate(all_fp)[order]
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    eps = np.finfo(np.float32).eps
    recalls = tp / max(num_gts, eps)
    precisions = tp / np.maximum(tp + fp, eps)
    return average_precision(recalls, precisions)
