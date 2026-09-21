"""
Evaluation Metrics Computation: Precision, Recall, F1-Score, and mAP@0.5.
"""

from typing import List, Dict, Any, Tuple
import torch
import numpy as np

from src.dataset.parser import CLASS_NAMES
from src.utils.box_ops import box_iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision (AP) using standard 101-point or all-points interpolation.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Ensure precision is monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # Sum (\Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_detections(
    predictions_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    iou_threshold: float = 0.5,
    num_classes: int = 6,
) -> Dict[str, Any]:
    """
    Evaluate detected bounding boxes against ground truth.
    
    Args:
        predictions_list: List of NMS detection tensors, each (K, 6): [xmin, ymin, xmax, ymax, conf, class_id]
        targets_list: List of ground-truth tensors, each (M, 5): [class_id, xmin, ymin, xmax, ymax] (in pixel coords)
        iou_threshold: IoU threshold for a true positive match (default 0.5).
        num_classes: Total defect classes.
        
    Returns:
        dict containing 'precision', 'recall', 'f1', 'map50', and 'per_class' metrics.
    """
    per_class_stats = {c: {"tp": [], "conf": [], "num_gt": 0} for c in range(num_classes)}

    for preds, targets in zip(predictions_list, targets_list):
        # Update ground truth counts
        if targets.numel() > 0:
            for t in targets:
                cls_id = int(t[0].item())
                if cls_id in per_class_stats:
                    per_class_stats[cls_id]["num_gt"] += 1

        if preds.numel() == 0:
            continue

        if targets.numel() == 0:
            # All predictions are false positives
            for p in preds:
                cls_id = int(p[5].item())
                if cls_id in per_class_stats:
                    per_class_stats[cls_id]["tp"].append(0)
                    per_class_stats[cls_id]["conf"].append(float(p[4].item()))
            continue

        # Match predictions to ground truth (sorted by confidence descending)
        if preds.shape[0] > 1:
            sort_idx = torch.argsort(preds[:, 4], descending=True)
            preds = preds[sort_idx]

        matched_gt = set()
        for p in preds:
            cls_id = int(p[5].item())
            conf = float(p[4].item())
            if cls_id not in per_class_stats:
                continue

            # Filter targets with the same class
            gt_mask = (targets[:, 0] == cls_id)
            cls_targets = targets[gt_mask]

            is_tp = 0
            if cls_targets.numel() > 0:
                p_box = p[0:4].unsqueeze(0)
                gt_boxes = cls_targets[:, 1:5]
                ious = box_iou(p_box, gt_boxes)[0]

                best_iou, best_idx = ious.max(dim=0)
                if best_iou >= iou_threshold:
                    # Find original target index
                    orig_idx = torch.where(gt_mask)[0][best_idx].item()
                    if orig_idx not in matched_gt:
                        is_tp = 1
                        matched_gt.add(orig_idx)

            per_class_stats[cls_id]["tp"].append(is_tp)
            per_class_stats[cls_id]["conf"].append(conf)

    # Compute AP per class
    aps = []
    precisions = []
    recalls = []
    per_class_metrics = {}

    for c in range(num_classes):
        c_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        stats = per_class_stats[c]
        num_gt = stats["num_gt"]

        if num_gt == 0:
            # Class has no ground truth targets in this evaluation set
            per_class_metrics[c_name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap50": 0.0,
                "num_gt": 0,
            }
            continue

        if len(stats["tp"]) == 0:
            # Class has ground truth targets, but model failed to predict any detections
            per_class_metrics[c_name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap50": 0.0,
                "num_gt": num_gt,
            }
            aps.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        # Sort by confidence descending
        tp_arr = np.array(stats["tp"])
        conf_arr = np.array(stats["conf"])
        sorted_indices = np.argsort(-conf_arr)
        tp_sorted = tp_arr[sorted_indices]

        # Cumulative TP and FP
        tp_cum = np.cumsum(tp_sorted)
        fp_cum = np.cumsum(1 - tp_sorted)

        rec = tp_cum / max(1, num_gt)
        prec = tp_cum / (tp_cum + fp_cum + 1e-7)

        ap = compute_ap(rec, prec)
        aps.append(ap)

        final_prec = float(prec[-1]) if len(prec) > 0 else 0.0
        final_rec = float(rec[-1]) if len(rec) > 0 else 0.0
        final_f1 = (
            float(2 * final_prec * final_rec / (final_prec + final_rec + 1e-7))
            if (final_prec + final_rec) > 0
            else 0.0
        )

        precisions.append(final_prec)
        recalls.append(final_rec)

        per_class_metrics[c_name] = {
            "precision": final_prec,
            "recall": final_rec,
            "f1": final_f1,
            "ap50": ap,
            "num_gt": num_gt,
        }

    mean_ap = float(np.mean(aps)) if aps else 0.0
    mean_prec = float(np.mean(precisions)) if precisions else 0.0
    mean_rec = float(np.mean(recalls)) if recalls else 0.0
    mean_f1 = (
        float(2 * mean_prec * mean_rec / (mean_prec + mean_rec + 1e-7))
        if (mean_prec + mean_rec) > 0
        else 0.0
    )

    return {
        "mAP@0.5": mean_ap,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "mean_f1": mean_f1,
        "per_class": per_class_metrics,
    }
