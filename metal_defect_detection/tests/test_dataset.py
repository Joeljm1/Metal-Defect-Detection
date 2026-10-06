"""
Unit tests for dataset parsing, conversions, and DataLoader collation.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.dataset.loader import create_dataloaders, yolo_collate_fn
from src.dataset.parser import (
    load_dataset_config,
    parse_yolo_label_file,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


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


def test_create_dataloaders_three_way_split(tmp_path: Path):
    for split, n in (("train", 10), ("test", 4)):
        img_dir = tmp_path / split / "images"
        lbl_dir = tmp_path / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(
                str(img_dir / f"{split}_{i:03d}.png"),
                np.zeros((200, 200, 3), dtype=np.uint8),
            )
            (lbl_dir / f"{split}_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        tmp_path, batch_size=2, num_workers=0, val_fraction=0.2, split_seed=42
    )

    train_ds: Any = train_loader.dataset
    val_ds: Any = val_loader.dataset
    test_ds: Any = test_loader.dataset

    # 10 train images -> 8 train + 2 val; test split untouched
    assert len(train_ds) == 8
    assert len(val_ds) == 2
    assert len(test_ds) == 4

    # Train and validation partitions must be disjoint (no checkpoint-selection leak)
    train_names = {p.name for p in train_ds.image_paths}
    val_names = {p.name for p in val_ds.image_paths}
    assert train_names.isdisjoint(val_names)
    assert train_names | val_names == {f"train_{i:03d}.png" for i in range(10)}


def test_load_dataset_config(tmp_path: Path):
    # Default configs/dataset.yaml should be found
    cfg = load_dataset_config()
    assert cfg.get("num_classes") == 6
    assert "crazing" in cfg.get("classes", [])

    # Custom YAML should be dynamically parsed
    custom_cfg = tmp_path / "custom_dataset.yaml"
    custom_cfg.write_text("""
name: "Custom-Metal"
num_classes: 2
classes:
  - rust
  - dent
path: "data/custom"
""")
    loaded = load_dataset_config(custom_cfg)
    assert loaded["name"] == "Custom-Metal"
    assert loaded["num_classes"] == 2
    assert loaded["classes"] == ["rust", "dent"]

