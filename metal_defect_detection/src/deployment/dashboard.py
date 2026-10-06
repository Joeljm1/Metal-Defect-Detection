"""
Interactive Metal Surface Defect Inspection Dashboard (Phase 6 / Task 7).

Provides modular utilities and Streamlit UI for plant operators:
1. Real-time image upload and sample defect selection.
2. Side-by-side comparison of raw metallic surface vs CLAHE/Bilateral preprocessing.
3. Multi-scale defect detection with adjustable confidence and NMS IoU sliders.
4. Interactive Grad-CAM visual saliency heatmaps with customizable opacity and colormaps.
5. Live telemetry panel measuring inference latency, FPS throughput, and defect counts.
"""

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.dataset.parser import CLASS_NAMES
from src.evaluation.gradcam import DefectGradCAM
from src.models.detector import DefectDetector
from src.utils.box_ops import non_max_suppression

logger = logging.getLogger(__name__)


# Defect class color palette (RGB)
CLASS_COLORS = {
    0: (239, 68, 68),    # crazing: red
    1: (249, 115, 22),   # inclusion: orange
    2: (234, 179, 8),    # patches: yellow
    3: (34, 197, 94),    # pitted_surface: green
    4: (59, 130, 246),   # rolled-in_scale: blue
    5: (168, 85, 247),   # scratches: purple
}


def load_inspection_model(
    variant: str = "M4",
    checkpoint_dir: Path | str = Path("checkpoints"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DefectDetector:
    """
    Loads and caches DefectDetector model with trained checkpoint weights.
    """
    checkpoint_dir = Path(checkpoint_dir)
    model = DefectDetector.build_model(variant=variant, num_classes=6)

    ckpt_map = {
        "M1": [
            checkpoint_dir / "M1_Baseline_best.pt",
            checkpoint_dir / "M1_best.pt",
        ],
        "M2": [
            checkpoint_dir / "M2_Baseline_Preprocessing_best.pt",
            checkpoint_dir / "M2_Preprocessing_best.pt",
            checkpoint_dir / "M2_best.pt",
        ],
        "M3": [
            checkpoint_dir / "M3_Baseline_Attention_best.pt",
            checkpoint_dir / "M3_Attention_best.pt",
            checkpoint_dir / "M3_best.pt",
        ],
        "M4": [
            checkpoint_dir / "M4_Proposed_Integrated_best.pt",
            checkpoint_dir / "M4_best.pt",
        ],
    }

    candidates = ckpt_map.get(variant, [checkpoint_dir / f"{variant}_best.pt"])
    loaded = False
    for candidate in candidates:
        if candidate.exists():
            ckpt_data = torch.load(candidate, map_location=device)
            model.load_state_dict(ckpt_data["model_state"])
            model.loaded_checkpoint = str(candidate)
            loaded = True
            break

    if not loaded:
        import warnings
        warnings.warn(
            f"Checkpoint not found for variant '{variant}' in {checkpoint_dir}. "
            f"Searched: {[str(c) for c in candidates]}. "
            f"Model '{variant}' will run with UNTRAINED random weights!",
            UserWarning,
            stacklevel=2,
        )
        print(f"[WARNING] Untrained model loaded: No checkpoint found for {variant}. Fallback to random weights.")
        model.loaded_checkpoint = None

    model.to(device).eval()
    return model


def run_defect_inspection(
    image_rgb: np.ndarray,
    model: DefectDetector,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    generate_gradcam: bool = True,
    cam_alpha: float = 0.55,
    cam_colormap: int = cv2.COLORMAP_JET,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    raise_on_cam_error: bool = False,
) -> dict[str, Any]:
    """
    Executes end-to-end defect inspection on an input RGB image:
    1. Preprocessing (CLAHE + Bilateral if M2/M4 + tensor transfer)
    2. Neural inference & NMS
    3. Grad-CAM visual attribution
    4. Bounding box overlay & Telemetry metrics
    """
    t_start = time.perf_counter()
    h_orig, w_orig = image_rgb.shape[:2]

    # Preprocessing (resizing, filtering, and tensor conversion)
    t_prep_start = time.perf_counter()
    image_resized = cv2.resize(image_rgb, (200, 200), interpolation=cv2.INTER_LINEAR)
    if model.use_preprocessing and model.preprocessor is not None:
        preprocessed_rgb = model.preprocessor.process(image_resized)
    else:
        preprocessed_rgb = image_resized.copy()
    tensor_in = (
        torch.from_numpy(preprocessed_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ).to(device)
    prep_latency_ms = (time.perf_counter() - t_prep_start) * 1000.0

    # Model inference
    t_infer_start = time.perf_counter()
    with torch.no_grad():
        decoded, _ = model(tensor_in)
        detections = non_max_suppression(decoded, conf_thres=conf_thres, iou_thres=iou_thres)[0]
    infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

    # Grad-CAM Visual Attribution
    gradcam_overlay = None
    gradcam_heatmap = None
    gradcam_error = None
    if generate_gradcam:
        try:
            with DefectGradCAM(model) as cam:
                # If defects detected, visualize attribution for highest-confidence class
                target_cls = int(detections[0, 5].item()) if detections.numel() > 0 else None
                gradcam_heatmap = cam.generate_cam(tensor_in, class_idx=target_cls, target_scale=(h_orig, w_orig))
                gradcam_overlay = cam.overlay_cam(image_rgb, gradcam_heatmap, alpha=cam_alpha, colormap=cam_colormap)
        except (RuntimeError, ValueError, IndexError, TypeError) as exc:
            if raise_on_cam_error:
                raise
            # Keep the dashboard usable, but make the failure visible instead of
            # silently showing the raw image as if it were a saliency overlay.
            gradcam_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Grad-CAM generation failed: %s", gradcam_error, exc_info=True)
            gradcam_overlay = image_rgb.copy()
            gradcam_heatmap = np.zeros((h_orig, w_orig), dtype=np.float32)

    # Scale detections back to original image size
    boxes_detected = []
    scale_x = w_orig / 200.0
    scale_y = h_orig / 200.0

    annotated_img = image_rgb.copy()
    if detections.numel() > 0:
        det_cpu = detections.cpu().numpy()
        for det in det_cpu:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            scaled_box = [
                int(np.clip(x1 * scale_x, 0, w_orig)),
                int(np.clip(y1 * scale_y, 0, h_orig)),
                int(np.clip(x2 * scale_x, 0, w_orig)),
                int(np.clip(y2 * scale_y, 0, h_orig)),
            ]

            boxes_detected.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": float(conf),
                "box_xyxy": scaled_box,
            })

            # Draw on annotated image
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            cv2.rectangle(annotated_img, (scaled_box[0], scaled_box[1]), (scaled_box[2], scaled_box[3]), color, 2)
            label_text = f"{cls_name} {conf * 100:.1f}%"
            cv2.putText(
                annotated_img,
                label_text,
                (scaled_box[0], max(15, scaled_box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0
    fps = 1000.0 / max(1e-3, total_latency_ms)
    infer_fps = 1000.0 / max(1e-3, infer_latency_ms)
    # Deployment-relevant latency: preprocessing + network + NMS (excludes
    # optional Grad-CAM and drawing). Real-time status is judged on this.
    pipeline_latency_ms = prep_latency_ms + infer_latency_ms
    pipeline_fps = 1000.0 / max(1e-3, pipeline_latency_ms)

    return {
        "raw_image": image_rgb,
        "preprocessed_image": preprocessed_rgb,
        "annotated_image": annotated_img,
        "gradcam_overlay": gradcam_overlay,
        "gradcam_heatmap": gradcam_heatmap,
        "gradcam_error": gradcam_error,
        "detections": boxes_detected,
        "num_defects": len(boxes_detected),
        "telemetry": {
            "total_latency_ms": round(total_latency_ms, 2),
            "prep_latency_ms": round(prep_latency_ms, 2),
            "infer_latency_ms": round(infer_latency_ms, 2),
            "pipeline_latency_ms": round(pipeline_latency_ms, 2),
            "fps": round(fps, 1),
            "infer_fps": round(infer_fps, 1),
            "pipeline_fps": round(pipeline_fps, 1),
            "device": device,
            "realtime_pass": bool(pipeline_fps >= 30.0),
            "checkpoint_loaded": getattr(model, "loaded_checkpoint", None),
            "is_trained": getattr(model, "loaded_checkpoint", None) is not None,
        },
    }
