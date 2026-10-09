"""
Unit tests for GC10-DET domain adaptation and cross-dataset evaluation (Task 6).
"""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset.loader import yolo_collate_fn
from src.evaluation.domain_adaptation import (
    GC10_CLASSES,
    DomainAdaptedDetector,
    GC10Dataset,
    create_mock_gc10_test_fixtures,
    evaluate_zero_shot_domain_transfer,
    generate_synthetic_gc10_benchmark,
)
from src.models.detector import DefectDetector


def test_gc10_classes_and_mapping():
    assert len(GC10_CLASSES) == 10
    assert "inclusion" in GC10_CLASSES
    assert "welding_line" in GC10_CLASSES
    assert generate_synthetic_gc10_benchmark is create_mock_gc10_test_fixtures


def test_create_mock_gc10_fixtures_and_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gc10_dir = create_mock_gc10_test_fixtures(
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
        img, targets, _path = ds[0]
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
    # Both train and eval modes return (decoded, raw) consistently
    decoded_train, raw_train = adapted(x)
    assert isinstance(decoded_train, torch.Tensor)
    assert isinstance(raw_train, list)
    assert len(raw_train) == 3

    # In eval mode, returns (decoded, raw)
    adapted.eval()
    out_eval = adapted(x)
    assert isinstance(out_eval, tuple)
    decoded, _raw = out_eval
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


def test_sample_few_shot_subset():
    from src.evaluation.domain_adaptation import sample_few_shot_subset

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        img_dir = base / "images"
        lbl_dir = base / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # Create dummy images and labels for 3 classes
        for i in range(15):
            img_file = img_dir / f"img_{i:03d}.jpg"
            img_file.write_text("dummy")
            lbl_file = lbl_dir / f"img_{i:03d}.txt"
            cls_id = i % 3
            lbl_file.write_text(f"{cls_id} 0.5 0.5 0.2 0.2\n")

        subset_seed1 = sample_few_shot_subset(img_dir, lbl_dir, shots_per_class=2, seed=42)
        assert len(subset_seed1) == 6  # 3 classes * 2 shots = 6 unique images
        assert all(p.exists() for p in subset_seed1)

        # Verify deterministic reproducibility
        subset_seed1_repeat = sample_few_shot_subset(img_dir, lbl_dir, shots_per_class=2, seed=42)
        assert subset_seed1 == subset_seed1_repeat


def test_voc_xml_parsing():
    from scripts.convert_gc10 import parse_voc_xml

    with tempfile.TemporaryDirectory() as tmp_dir:
        xml_path = Path(tmp_dir) / "sample.xml"
        xml_content = """<annotation>
    <size>
        <width>2048</width>
        <height>1000</height>
        <depth>3</depth>
    </size>
    <object>
        <name>punching_hole</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>"""
        xml_path.write_text(xml_content)
        boxes = parse_voc_xml(xml_path)
        assert len(boxes) == 1
        cls_id, xc, yc, w, h = boxes[0]
        assert cls_id == 4  # punching_hole is Class 4
        assert abs(xc - (200.0 / 2048.0)) < 1e-4
        assert abs(yc - (300.0 / 1000.0)) < 1e-4
        assert abs(w - (200.0 / 2048.0)) < 1e-4
        assert abs(h - (200.0 / 1000.0)) < 1e-4

