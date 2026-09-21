"""
Unit tests for visualization utilities, verifying both NMS and YOLO box formats.
"""

import numpy as np
import torch
import pytest

from src.utils.visualization import draw_bounding_boxes


def test_draw_bounding_boxes_yolo_format():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Ground truth format: [cls, xc, yc, w, h]
    yolo_boxes = np.array([
        [0, 0.5, 0.5, 0.2, 0.2],
        [1, 0.2, 0.2, 0.1, 0.1],
    ])
    result = draw_bounding_boxes(img, yolo_boxes, box_format="yolo")
    assert result.shape == img.shape
    assert (result > 0).any()  # drawn something


def test_draw_bounding_boxes_nms_format():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # NMS format: [x1, y1, x2, y2, conf, cls]
    nms_boxes = torch.tensor([
        [20.0, 20.0, 80.0, 80.0, 0.95, 0.0],
        [100.0, 100.0, 150.0, 150.0, 0.88, 3.0],
    ])
    # Auto-detection
    result_auto = draw_bounding_boxes(img, nms_boxes, box_format="auto")
    assert result_auto.shape == img.shape
    assert (result_auto > 0).any()

    # Explicit nms format
    result_nms = draw_bounding_boxes(img, nms_boxes, box_format="nms")
    assert np.array_equal(result_auto, result_nms)
