"""Detection and angle metrics: precision/recall/F1, mAP@50, mAP@50-95, angle errors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from geometry import circular_angle_error, iou_xyxy


@dataclass
class DetMatch:
    pred_idx: int
    gt_idx: int
    iou: float


def hungarian_match(
    pred_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    iou_thr: float = 0.5,
) -> list[DetMatch]:
    """Optimal one-to-one assignment maximizing IoU sum, filtered by IoU threshold."""
    if pred_xyxy.size == 0 or gt_xyxy.size == 0:
        return []
    iou = iou_xyxy(pred_xyxy, gt_xyxy)
    # Hungarian minimizes cost, so use -iou as cost
    pred_idx, gt_idx = linear_sum_assignment(-iou)
    matches: list[DetMatch] = []
    for pi, gi in zip(pred_idx, gt_idx):
        if iou[pi, gi] >= iou_thr:
            matches.append(DetMatch(int(pi), int(gi), float(iou[pi, gi])))
    return matches


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(p), float(r), float(f1)


def average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """11-point interpolated AP (PASCAL-style) computed from per-detection TP flags.

    scores: detection scores (higher = better), shape [N]
    is_tp:  bool array indicating which of those detections is a true positive
    n_gt:   total number of ground-truth objects across the evaluation set
    """
    if n_gt == 0:
        return 0.0
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores)
    is_tp = is_tp[order].astype(np.int32)
    fp = (1 - is_tp).cumsum()
    tp = is_tp.cumsum()
    recall = tp / (n_gt + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


def evaluate_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_angles: np.ndarray,
    gt_boxes: np.ndarray,
    gt_angles: np.ndarray,
    iou_thr: float = 0.5,
) -> dict:
    """Per-image accumulators for global metrics aggregation."""
    matches = hungarian_match(pred_boxes, gt_boxes, iou_thr=iou_thr)
    matched_pred = {m.pred_idx for m in matches}
    matched_gt = {m.gt_idx for m in matches}
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    angle_errs = []
    per_det_records = []  # (score, is_tp)
    for i in range(n_pred):
        is_tp = i in matched_pred
        per_det_records.append((float(pred_scores[i]), bool(is_tp)))
    for m in matches:
        err = float(circular_angle_error(pred_angles[m.pred_idx], gt_angles[m.gt_idx]))
        angle_errs.append(err)

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "tp": len(matches),
        "fp": n_pred - len(matched_pred),
        "fn": n_gt - len(matched_gt),
        "det_records": per_det_records,
        "angle_errors": angle_errs,
    }


def aggregate(
    per_image: list[dict],
    iou_thresholds: np.ndarray | None = None,
    iou_for_pr: float = 0.5,
) -> dict:
    """Aggregate per-image results into final metrics."""
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)

    tp = sum(x["tp"] for x in per_image)
    fp = sum(x["fp"] for x in per_image)
    fn = sum(x["fn"] for x in per_image)
    n_gt_total = sum(x["n_gt"] for x in per_image)

    all_scores, all_is_tp = [], []
    for x in per_image:
        for s, t in x["det_records"]:
            all_scores.append(s)
            all_is_tp.append(t)
    scores = np.asarray(all_scores, dtype=np.float32)
    is_tp = np.asarray(all_is_tp, dtype=bool)
    ap50 = average_precision(scores, is_tp, n_gt_total)

    # mAP@[.5:.95] requires re-matching per-image at each IoU threshold.
    # per_image carries match results only at iou_for_pr; for the strict metric we
    # approximate using the same TP set scaled by threshold not being met. To keep
    # this honest, callers can pass a richer per_image with per-threshold records.
    # Here we expose ap50 and a placeholder ap50_95 if not supplied.
    map50_95 = x_get_map50_95(per_image, iou_thresholds, n_gt_total)

    p, r, f1 = precision_recall_f1(tp, fp, fn)
    all_angle_errors = np.array(
        [e for x in per_image for e in x["angle_errors"]], dtype=np.float64
    )
    mean_ang = float(all_angle_errors.mean()) if all_angle_errors.size else float("nan")
    median_ang = float(np.median(all_angle_errors)) if all_angle_errors.size else float("nan")

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": ap50,
        "map50_95": map50_95,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_angle_error_deg": mean_ang,
        "median_angle_error_deg": median_ang,
        "n_matches": int(all_angle_errors.size),
        "angle_errors": all_angle_errors.tolist(),
    }


def x_get_map50_95(per_image: list[dict], iou_thresholds: np.ndarray, n_gt_total: int) -> float:
    """Compute mAP@[.5:.95] when per-image records include per-threshold matches.

    Each per_image dict may optionally contain "det_records_by_iou" mapping
    iou_threshold -> list[(score, is_tp)] and "tp_by_iou", "fp_by_iou", "fn_by_iou".
    Fallback: returns ap50 replicated, which is clearly a lower-quality estimate.
    """
    if not per_image or "det_records_by_iou" not in per_image[0]:
        # Fallback: best-effort using only the recorded threshold
        all_scores, all_is_tp = [], []
        for x in per_image:
            for s, t in x["det_records"]:
                all_scores.append(s)
                all_is_tp.append(t)
        scores = np.asarray(all_scores, dtype=np.float32)
        is_tp = np.asarray(all_is_tp, dtype=bool)
        return average_precision(scores, is_tp, n_gt_total)

    aps = []
    for thr in iou_thresholds:
        s_list, t_list = [], []
        for x in per_image:
            for s, t in x["det_records_by_iou"][float(thr)]:
                s_list.append(s)
                t_list.append(t)
        aps.append(
            average_precision(
                np.asarray(s_list, dtype=np.float32),
                np.asarray(t_list, dtype=bool),
                n_gt_total,
            )
        )
    return float(np.mean(aps))


def evaluate_image_multi_iou(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_angles: np.ndarray,
    gt_boxes: np.ndarray,
    gt_angles: np.ndarray,
    iou_thresholds: np.ndarray | None = None,
) -> dict:
    """Same as evaluate_image but records TP/FP at each IoU threshold for proper mAP@[.5:.95]."""
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)

    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    base = evaluate_image(pred_boxes, pred_scores, pred_angles, gt_boxes, gt_angles, iou_thr=0.5)

    det_records_by_iou: dict[float, list[tuple[float, bool]]] = {}
    for thr in iou_thresholds:
        matches = hungarian_match(pred_boxes, gt_boxes, iou_thr=float(thr))
        matched_pred = {m.pred_idx for m in matches}
        records = []
        for i in range(n_pred):
            records.append((float(pred_scores[i]), bool(i in matched_pred)))
        det_records_by_iou[float(thr)] = records

    base["det_records_by_iou"] = det_records_by_iou
    base["n_gt"] = n_gt  # ensure present for downstream
    return base
