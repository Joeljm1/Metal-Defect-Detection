"""
Unit tests for interactive inspection dashboard and plant operator pipeline (Task 7).
"""

import numpy as np
import torch
import cv2

from src.models.detector import DefectDetector
from src.deployment.dashboard import (
    load_inspection_model,
    run_defect_inspection,
    CLASS_COLORS,
)


def test_class_colors():
    assert len(CLASS_COLORS) == 6
    for c in range(6):
        assert c in CLASS_COLORS
        assert len(CLASS_COLORS[c]) == 3


def test_load_inspection_model():
    model = load_inspection_model(variant="M4", device="cpu")
    assert isinstance(model, DefectDetector)
    assert model.use_attention is True
    assert model.use_preprocessing is True


def test_run_defect_inspection_pipeline():
    model = DefectDetector.build_model("M4", num_classes=6).eval()
    dummy_img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)

    res = run_defect_inspection(
        image_rgb=dummy_img,
        model=model,
        conf_thres=0.10,
        iou_thres=0.45,
        generate_gradcam=True,
        cam_alpha=0.5,
        cam_colormap=cv2.COLORMAP_JET,
        device="cpu",
    )

    assert "raw_image" in res
    assert "preprocessed_image" in res
    assert "annotated_image" in res
    assert "gradcam_overlay" in res
    assert "telemetry" in res
    assert "detections" in res

    assert res["annotated_image"].shape == (200, 200, 3)
    assert res["gradcam_overlay"].shape == (200, 200, 3)
    assert res["telemetry"]["fps"] > 0
    assert res["telemetry"]["total_latency_ms"] > 0


def test_dashboard_report_serialization():
    import json
    import time
    telemetry = {"fps": 45.2, "total_latency_ms": 22.1}
    detections = [{
        "class_id": 0,
        "class_name": "crazing",
        "confidence": 0.85,
        "box_xyxy": [10, 10, 50, 50],
    }]
    report = json.dumps({
        "timestamp": time.time(),
        "model": "M4",
        "telemetry": telemetry,
        "detections": detections,
    })
    parsed = json.loads(report)
    assert parsed["model"] == "M4"
    assert len(parsed["detections"]) == 1
    assert parsed["timestamp"] > 0
