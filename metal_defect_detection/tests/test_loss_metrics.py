"""
Unit tests for loss functions, box operations, NMS, and evaluation metrics.
"""

import torch

from src.evaluation.metrics import evaluate_detections
from src.models.detector import DefectDetector
from src.training.loss import ComputeLoss
from src.utils.box_ops import bbox_ciou


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


def test_objectness_pos_weight_default_is_unweighted():
    # Standard YOLOv5 operating regime: no objectness class re-weighting.
    # Values > 1 inflate recall at the cost of precision.
    assert ComputeLoss().obj_pos_weight == 1.0


def test_metrics_operating_point_precision_recall():
    # One image: 1 TP (conf 0.90) + 1 FP (conf 0.80), 1 GT total.
    preds = [
        torch.tensor([
            [10.0, 10.0, 50.0, 50.0, 0.90, 0.0],
            [100.0, 100.0, 140.0, 140.0, 0.80, 0.0],
        ]),
    ]
    targets = [
        torch.tensor([[0.0, 10.0, 10.0, 50.0, 50.0]]),
    ]

    metrics = evaluate_detections(
        preds, targets, iou_threshold=0.5, num_classes=1, report_conf_thres=0.25
    )
    per = metrics["per_class"]["crazing"]
    assert per["ap50"] > 0.99  # AP uses the full ranked curve
    assert abs(per["precision"] - 0.5) < 1e-6  # 1 TP out of 2 detections
    assert abs(per["recall"] - 1.0) < 1e-6

    metrics_strict = evaluate_detections(
        preds, targets, iou_threshold=0.5, num_classes=1, report_conf_thres=0.85
    )
    per_strict = metrics_strict["per_class"]["crazing"]
    assert abs(per_strict["precision"] - 1.0) < 1e-6  # FP filtered at the operating point


def test_evaluate_model_on_loader_shared_policy():
    # Smoke test for the single evaluation code path used by all scripts
    from torch.utils.data import DataLoader, TensorDataset

    from src.evaluation.metrics import evaluate_model_on_loader

    class StubDetector(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            decoded = torch.rand(b, 10, 11)  # [x, y, w, h, obj, cls...] in [0, 1]
            return decoded, None

    def collate(batch):
        imgs = torch.stack([item[0] for item in batch])
        targets = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.2, 0.2]])  # [img_idx, cls, xc, yc, w, h]
        return imgs, targets, ["a.jpg", "b.jpg"]

    loader = DataLoader(
        TensorDataset(torch.zeros(2, 3, 200, 200)), batch_size=2, collate_fn=collate
    )
    metrics = evaluate_model_on_loader(StubDetector(), loader, device="cpu")
    assert 0.0 <= metrics["mAP@0.5"] <= 1.0
    assert "per_class" in metrics
    assert "mean_precision" in metrics


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
