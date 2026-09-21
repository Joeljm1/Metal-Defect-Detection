"""
Unit tests for loss functions, box operations, NMS, and evaluation metrics.
"""

import torch
import pytest

from src.utils.box_ops import box_iou, bbox_ciou, non_max_suppression
from src.evaluation.metrics import compute_ap, evaluate_detections
from src.training.loss import ComputeLoss
from src.models.detector import DefectDetector


def test_ciou_computation():
    box1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    box2 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])  # identical box

    ciou_identical = bbox_ciou(box1, box2)
    assert torch.isclose(ciou_identical, torch.tensor([1.0]), atol=1e-3)

    box3 = torch.tensor([[100.0, 100.0, 150.0, 150.0]])  # disjoint box
    ciou_disjoint = bbox_ciou(box1, box3)
    assert ciou_disjoint.item() < 0.0  # negative due to distance penalty


def test_compute_loss_forward():
    model = DefectDetector.build_model(variant="M1")
    model.train()
    loss_fn = ComputeLoss()

    images = torch.randn(2, 3, 200, 200)
    # Target: [img_idx, class_id, xc, yc, w, h]
    targets = torch.tensor([
        [0.0, 0.0, 0.5, 0.5, 0.2, 0.2],
        [1.0, 2.0, 0.3, 0.3, 0.15, 0.15],
    ], dtype=torch.float32)

    preds = model(images)
    loss, items = loss_fn(preds, targets, model)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert "loss_box" in items
    assert "loss_obj" in items
    assert "loss_cls" in items


def test_evaluate_detections_metrics():
    # Perfect detection scenario
    preds = [
        torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.95, 0.0]]),
    ]
    targets = [
        torch.tensor([[0.0, 10.0, 10.0, 50.0, 50.0]]),
    ]

    metrics = evaluate_detections(preds, targets, iou_threshold=0.5, num_classes=6)
    assert metrics["mAP@0.5"] > 0.0
    assert "per_class" in metrics


def test_map_unpredicted_class_averaging():
    # Two classes present in ground truth:
    # Class 0 has 1 perfect detection (AP=1.0)
    # Class 1 has 1 ground truth object, but 0 predictions (AP=0.0)
    # The overall mAP must be (1.0 + 0.0) / 2 = 0.5, not 1.0!
    preds = [
        torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.95, 0.0]]),
    ]
    targets = [
        torch.tensor([
            [0.0, 10.0, 10.0, 50.0, 50.0],  # class 0
            [1.0, 60.0, 60.0, 90.0, 90.0],  # class 1 (missed)
        ]),
    ]

    metrics = evaluate_detections(preds, targets, iou_threshold=0.5, num_classes=2)
    assert abs(metrics["mAP@0.5"] - 0.5) < 1e-4, f"Expected mAP ~0.5, but got {metrics['mAP@0.5']}"
    assert abs(metrics["per_class"]["crazing"]["ap50"] - 1.0) < 1e-4
    assert metrics["per_class"]["inclusion"]["ap50"] == 0.0
