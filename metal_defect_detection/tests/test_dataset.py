"""
Unit tests for dataset parsing, conversions, and DataLoader collation.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

from src.dataset.parser import (
    parse_yolo_label_file,
    xywh_to_xyxy,
    xyxy_to_xywh,
    CLASS_NAMES,
)
from src.dataset.loader import yolo_collate_fn


def test_box_conversions_roundtrip():
    # xc, yc, w, h in normalized [0, 1]
    original_xywh = np.array([
        [0.5, 0.5, 0.2, 0.4],
        [0.2, 0.3, 0.1, 0.1],
    ], dtype=np.float32)

    img_w, img_h = 200, 200
    boxes_xyxy = xywh_to_xyxy(original_xywh, img_w=img_w, img_h=img_h)
    recovered_xywh = xyxy_to_xywh(boxes_xyxy, img_w=img_w, img_h=img_h)

    np.testing.assert_allclose(original_xywh, recovered_xywh, atol=1e-5)


def test_parse_empty_or_missing_label(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    res = parse_yolo_label_file(empty_file)
    assert res.shape == (0, 5)

    missing_file = tmp_path / "non_existent.txt"
    res_missing = parse_yolo_label_file(missing_file)
    assert res_missing.shape == (0, 5)


def test_yolo_collate_function():
    # Simulate batch of 2 images with 1 and 2 target boxes
    img1 = torch.zeros((3, 200, 200), dtype=torch.float32)
    img2 = torch.zeros((3, 200, 200), dtype=torch.float32)

    tgt1 = torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    tgt2 = torch.tensor([
        [1.0, 0.3, 0.3, 0.1, 0.1],
        [2.0, 0.7, 0.7, 0.3, 0.3],
    ], dtype=torch.float32)

    batch = [
        (img1, tgt1, "img1.jpg"),
        (img2, tgt2, "img2.jpg"),
    ]

    images, targets, paths = yolo_collate_fn(batch)

    assert images.shape == (2, 3, 200, 200)
    assert targets.shape == (3, 6)  # 3 total boxes, each has [batch_idx, class_id, xc, yc, w, h]
    assert targets[0, 0].item() == 0  # from img 0
    assert targets[1, 0].item() == 1  # from img 1
    assert targets[2, 0].item() == 1  # from img 1
    assert len(paths) == 2
