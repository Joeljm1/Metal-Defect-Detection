"""
Evaluation Metrics Computation: Precision, Recall, F1-Score, and mAP@0.5.
"""

from typing import Any

import numpy as np
import torch

from src.dataset.parser import CLASS_NAMES
from src.utils.box_ops import box_iou, non_max_suppression

# Shared evaluation policy (see README "Training & Evaluation Policy"):
# - NMS prefilters detections at EVAL_CONF_THRES (near zero) so the AP curve
#   spans the full confidence range.
# - Precision / Recall / F1 are reported at the fixed operating point
#   REPORT_CONF_THRES.
EVAL_CONF_THRES = 0.001
REPORT_CONF_THRES = 0.25
IOU_MATCH_THRESHOLD = 0.5


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
    predictions_list: list[torch.Tensor],
    targets_list: list[torch.Tensor],
    iou_threshold: float = 0.5,
    num_classes: int = 6,
    report_conf_thres: float = REPORT_CONF_THRES,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate detected bounding boxes against ground truth.
    
    Args:
        predictions_list: List of NMS detection tensors, each (K, 6): [xmin, ymin, xmax, ymax, conf, class_id]
        targets_list: List of ground-truth tensors, each (M, 5): [class_id, xmin, ymin, xmax, ymax] (in pixel coords)
        iou_threshold: IoU threshold for a true positive match (default 0.5).
        num_classes: Total defect classes.
        report_conf_thres: Operating confidence threshold at which precision,
            recall, and F1 are reported (AP uses the full ranked curve).
        class_names: Optional explicit list of class names. If omitted, defaults to dataset CLASS_NAMES.
        
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

    names = class_names if class_names is not None else CLASS_NAMES

    for c in range(num_classes):
        c_name = names[c] if c < len(names) else f"class_{c}"
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

        # Precision / Recall / F1 at the fixed operating confidence threshold.
        # (AP above is computed over the full confidence-ranked curve.)
        op_mask = conf_arr >= report_conf_thres
        num_det_op = int(op_mask.sum())
        tp_op = float(tp_arr[op_mask].sum())
        final_prec = tp_op / num_det_op if num_det_op > 0 else 0.0
        final_rec = tp_op / num_gt
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
    # True macro-averaged F1 (mean of per-class F1 for active ground-truth classes)
    active_f1s = [m["f1"] for m in per_class_metrics.values() if m["num_gt"] > 0]
    mean_f1 = float(np.mean(active_f1s)) if active_f1s else 0.0
    harmonic_f1 = (
        float(2 * mean_prec * mean_rec / (mean_prec + mean_rec + 1e-7))
        if (mean_prec + mean_rec) > 0
        else 0.0
    )

    return {
        "mAP@0.5": mean_ap,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "mean_f1": mean_f1,
        "harmonic_f1": harmonic_f1,
        "per_class": per_class_metrics,
    }


def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device | str,
    conf_thres: float = EVAL_CONF_THRES,
    iou_thres: float = 0.45,
    num_classes: int = 6,
    report_conf_thres: float = REPORT_CONF_THRES,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Shared evaluation loop: runs the model over a DataLoader, applies NMS, and
    computes detection metrics (full-curve mAP@0.5 plus P/R/F1 at the operating
    confidence threshold). Single source of truth for all evaluation scripts.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            decoded, _ = model(images)
            nms_preds = non_max_suppression(decoded, conf_thres=conf_thres, iou_thres=iou_thres)

            img_h, img_w = images.shape[2], images.shape[3]
            for b_idx in range(images.shape[0]):
                img_targets = targets[targets[:, 0] == b_idx]
                if img_targets.numel() > 0:
                    xc = img_targets[:, 2] * img_w
                    yc = img_targets[:, 3] * img_h
                    w = img_targets[:, 4] * img_w
                    h = img_targets[:, 5] * img_h
                    boxes_xyxy = torch.stack(
                        [img_targets[:, 1], xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=1
                    )
                else:
                    boxes_xyxy = torch.zeros((0, 5), dtype=torch.float32)

                all_preds.append(nms_preds[b_idx].cpu())
                all_targets.append(boxes_xyxy)

    return evaluate_detections(
        all_preds,
        all_targets,
        iou_threshold=IOU_MATCH_THRESHOLD,
        num_classes=num_classes,
        report_conf_thres=report_conf_thres,
        class_names=class_names,
    )
