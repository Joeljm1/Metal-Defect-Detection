"""
Visualization Utilities for Defect Inspection and Preprocessing Stages.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.dataset.parser import CLASS_NAMES, xywh_to_xyxy


CLASS_COLORS = [
    (230, 25, 75),    # Crazing - Red
    (60, 180, 75),    # Inclusion - Green
    (255, 225, 25),   # Patches - Yellow
    (0, 130, 200),    # Pitted Surface - Blue
    (245, 130, 48),   # Rolled-in Scale - Orange
    (145, 30, 180),   # Scratches - Purple
]


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Union[np.ndarray, torch.Tensor],
    box_format: str = "auto",
    is_normalized_xywh: Optional[bool] = None,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes on image canvas.

    Supports:
    - NMS detection predictions: [x1, y1, x2, y2, conf, cls_id] or [x1, y1, x2, y2, cls_id]
    - YOLO ground truth labels: [cls_id, xc, yc, w, h] or [cls_id, xc, yc, w, h, conf]
    - Auto-detection between YOLO and NMS formats.
    """
    canvas = image.copy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()

    if len(boxes) == 0:
        return canvas

    h, w = canvas.shape[:2]

    for b in boxes:
        # Determine format if auto
        fmt = box_format
        if fmt == "auto":
            if len(b) >= 6:
                # NMS output: [x1, y1, x2, y2, conf, cls_id]
                # Check if b[4] is confidence (<= 1.0) and b[5] is integer class_id
                if b[0] > 1.0 or b[2] > 1.0 or (b[0] <= b[2] and b[4] <= 1.0 and float(b[5]).is_integer()):
                    fmt = "nms"
                elif float(b[0]).is_integer() and 0 <= b[0] < len(CLASS_NAMES) and all(0 <= v <= 1.0 for v in b[1:5]):
                    fmt = "yolo"
                else:
                    fmt = "nms"
            elif len(b) == 5:
                if is_normalized_xywh is True:
                    fmt = "yolo"
                elif float(b[0]).is_integer() and 0 <= b[0] < len(CLASS_NAMES) and all(0 <= v <= 1.0 for v in b[1:5]):
                    fmt = "yolo"
                elif float(b[4]).is_integer() and 0 <= b[4] < len(CLASS_NAMES):
                    fmt = "xyxy"
                else:
                    fmt = "yolo"

        if fmt in ("nms", "xyxy"):
            xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
            # If coordinates are normalized in [0, 1]
            if xmin <= 1.0 and ymin <= 1.0 and xmax <= 1.0 and ymax <= 1.0 and w > 1 and h > 1:
                xmin, xmax = xmin * w, xmax * w
                ymin, ymax = ymin * h, ymax * h
            box_xyxy = [xmin, ymin, xmax, ymax]
            conf = float(b[4]) if len(b) >= 6 else None
            cls_id = int(b[5]) if len(b) >= 6 else (int(b[4]) if len(b) == 5 else 0)
        else:  # YOLO format: [cls_id, xc, yc, w, h, (conf)]
            cls_id = int(b[0])
            norm = is_normalized_xywh if is_normalized_xywh is not None else (max(b[1:5]) <= 1.0)
            if norm:
                box_xyxy = xywh_to_xyxy(b[1:5].reshape(1, 4), img_w=w, img_h=h)[0]
            else:
                xc, yc, bw, bh = b[1:5]
                box_xyxy = [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2]
            conf = float(b[5]) if len(b) > 5 else None

        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        xmin, ymin, xmax, ymax = [int(round(v)) for v in box_xyxy]
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)

        if show_labels:
            cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"C{cls_id}"
            conf_str = f" {conf:.2f}" if conf is not None else ""
            label_text = f"{cls_name}{conf_str}"

            # Text background badge
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (xmin, max(0, ymin - th - 4)), (xmin + tw + 4, ymin), color, -1)
            cv2.putText(
                canvas,
                label_text,
                (xmin + 2, max(th, ymin - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas


def plot_preprocessing_comparison(
    stages: dict,
    title: str = "Preprocessing Stages Comparison",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot Raw vs Bilateral vs CLAHE vs Enhanced side-by-side with histogram analysis.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    stage_names = ["raw", "bilateral", "clahe_only", "enhanced"]
    display_titles = [
        "1. Raw Metallic Image",
        "2. Bilateral Filter (Denoised)",
        "3. CLAHE Only (Contrast Boost)",
        "4. Combined Pipeline (Enhanced)",
    ]

    for idx, (name, d_title) in enumerate(zip(stage_names, display_titles)):
        img = stages.get(name)
        if img is None:
            continue

        # Display image
        if len(img.shape) == 3:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
            axes[0, idx].imshow(rgb_img)
        else:
            axes[0, idx].imshow(img, cmap="gray")
        axes[0, idx].set_title(d_title, fontsize=11, fontweight="semibold")
        axes[0, idx].axis("off")

        # Display grayscale histogram
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        axes[1, idx].hist(gray.ravel(), bins=64, range=(0, 256), color="navy", alpha=0.75)
        axes[1, idx].set_title(f"Histogram: {name}", fontsize=10)
        axes[1, idx].set_xlim(0, 256)
        axes[1, idx].set_xlabel("Pixel Intensity")
        axes[1, idx].set_ylabel("Pixel Frequency")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
