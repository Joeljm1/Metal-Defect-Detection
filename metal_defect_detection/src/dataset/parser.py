"""
Bounding Box and Dataset Annotation Parsing Utilities.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
import numpy as np


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def parse_yolo_label_file(label_path: Path) -> torch.Tensor:
    """
    Parse a YOLO format label file into a Tensor of shape (N, 5)
    where each row is [class_id, x_center, y_center, width, height].
    
    Returns:
        torch.Tensor of shape (N, 5), dtype=float32. If file empty, returns shape (0, 5).
    """
    if not label_path.exists() or label_path.stat().st_size == 0:
        return torch.zeros((0, 5), dtype=torch.float32)

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = float(parts[0])
                xc, yc, w, h = [float(p) for p in parts[1:5]]
                # Clamp coordinates to [0, 1] range to avoid out-of-bound errors
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                boxes.append([cls_id, xc, yc, w, h])

    if len(boxes) == 0:
        return torch.zeros((0, 5), dtype=torch.float32)

    return torch.tensor(boxes, dtype=torch.float32)


def xywh_to_xyxy(boxes_xywh: np.ndarray, img_w: int = 200, img_h: int = 200) -> np.ndarray:
    """
    Convert normalized [xc, yc, w, h] to pixel coordinate [xmin, ymin, xmax, ymax].
    """
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    xc = boxes_xywh[:, 0] * img_w
    yc = boxes_xywh[:, 1] * img_h
    w = boxes_xywh[:, 2] * img_w
    h = boxes_xywh[:, 3] * img_h

    xmin = np.clip(xc - w / 2, 0, img_w)
    ymin = np.clip(yc - h / 2, 0, img_h)
    xmax = np.clip(xc + w / 2, 0, img_w)
    ymax = np.clip(yc + h / 2, 0, img_h)

    return np.column_stack([xmin, ymin, xmax, ymax])


def xyxy_to_xywh(boxes_xyxy: np.ndarray, img_w: int = 200, img_h: int = 200) -> np.ndarray:
    """
    Convert pixel coordinate [xmin, ymin, xmax, ymax] to normalized [xc, yc, w, h].
    """
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    xmin, ymin, xmax, ymax = (
        boxes_xyxy[:, 0],
        boxes_xyxy[:, 1],
        boxes_xyxy[:, 2],
        boxes_xyxy[:, 3],
    )
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    xc = (xmin + xmax) / (2 * img_w)
    yc = (ymin + ymax) / (2 * img_h)

    return np.column_stack([xc, yc, w, h])
