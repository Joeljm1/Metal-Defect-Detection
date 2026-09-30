"""
Unit tests for GC10-DET domain adaptation and cross-dataset evaluation (Task 6).
"""

import tempfile
from pathlib import Path
import torch
import numpy as np

from src.models.detector import DefectDetector
from src.evaluation.domain_adaptation import (
    GC10_CLASSES,
    GC10Dataset,
    DomainAdaptedDetector,
    generate_synthetic_gc10_benchmark,
    evaluate_zero_shot_domain_transfer,
    train_few_shot_adaptation,
)
from torch.utils.data import DataLoader
from src.dataset.loader import yolo_collate_fn


def test_gc10_classes_and_mapping():
    assert len(GC10_CLASSES) == 10
    assert "inclusion" in GC10_CLASSES
    assert "welding_line" in GC10_CLASSES


def test_generate_synthetic_gc10_and_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gc10_dir = generate_synthetic_gc10_benchmark(
            base_dir=tmp_dir,
            num_train_per_class=2,
            num_test_per_class=2,
        )
        assert Path(gc10_dir).exists()
        train_img = Path(gc10_dir) / "train" / "images"
        assert len(list(train_img.glob("*.jpg"))) == 20

        ds = GC10Dataset(
            image_dir=gc10_dir / "train" / "images",
            label_dir=gc10_dir / "train" / "labels",
            target_size=(200, 200),
        )
        assert len(ds) == 20
        img, targets, path = ds[0]
        assert img.shape == (3, 200, 200)
        assert isinstance(targets, torch.Tensor)
        assert targets.shape[1] == 5 if targets.numel() > 0 else True


def test_domain_adapted_detector_architecture():
    base_model = DefectDetector.build_model("M4", num_classes=6)
    adapted = DomainAdaptedDetector(base_model, num_target_classes=10, freeze_backbone=True)
    assert adapted.head.num_classes == 10

    # Verify backbone parameters are frozen
    for p in adapted.backbone.parameters():
        assert not p.requires_grad

    # Verify head parameters require grad
    for p in adapted.head.parameters():
        assert p.requires_grad

    x = torch.randn(2, 3, 200, 200)
    # In training mode, returns list of multi-scale raw outputs
    out_train = adapted(x)
    assert isinstance(out_train, list)
    assert len(out_train) == 3

    # In eval mode, returns (decoded, raw)
    adapted.eval()
    out_eval = adapted(x)
    assert isinstance(out_eval, tuple)
    decoded, raw = out_eval
    # 10 classes + 5 bbox coords = 15 channels per anchor
    assert decoded.shape[-1] == 15


def test_zero_shot_transfer_evaluation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gc10_dir = generate_synthetic_gc10_benchmark(
            base_dir=tmp_dir,
            num_train_per_class=1,
            num_test_per_class=1,
        )
        test_ds = GC10Dataset(
            image_dir=gc10_dir / "test" / "images",
            label_dir=gc10_dir / "test" / "labels",
        )
        loader = DataLoader(test_ds, batch_size=2, collate_fn=yolo_collate_fn)

        model = DefectDetector.build_model("M1", num_classes=6).eval()
        metrics = evaluate_zero_shot_domain_transfer(model, loader, device="cpu")
        assert "mAP@0.5" in metrics
        assert "mean_precision" in metrics
        assert "mean_recall" in metrics
