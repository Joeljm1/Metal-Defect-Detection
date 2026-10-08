#!/usr/bin/env python3
"""
Step 3: Run Inference with Trained Baseline Weights (M1).
=========================================================
This script teaches you:
1. How to load trained PyTorch checkpoint weights (.pt).
2. How to pass a real test image into the network.
3. How Non-Maximum Suppression (NMS) filters 2,529 raw candidate boxes down to real defects.
4. Draws detected defect bounding boxes and saves to learn/output/baseline_prediction.png.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'import src...' works regardless of current working directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import torch

from src.models.detector import DefectDetector
from src.utils.box_ops import non_max_suppression

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Color palette for defect classes (BGR for OpenCV)
CLASS_COLORS = [
    (0, 165, 255),   # crazing: Orange
    (255, 0, 0),     # inclusion: Blue
    (0, 255, 255),   # patches: Yellow
    (255, 0, 255),   # pitted_surface: Magenta
    (0, 0, 255),     # rolled-in_scale: Red
    (0, 255, 0),     # scratches: Green
]

def main():
    print("=" * 70)
    print("STEP 3: RUNNING DEFECT DETECTION INFERENCE (M1 BASELINE)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Running on device: {device}")

    # 1. Instantiate M1 Baseline Model using build_model
    model = DefectDetector.build_model(
        variant="M1",
        num_classes=6,
    ).to(device)

    # 2. Load trained checkpoint weights
    ckpt_path = REPO_ROOT / "checkpoints" / "M1_Baseline_best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/M1_Baseline_best.pt")
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print("\n[2] Loading trained weights from:")
    print(f"    -> {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("    Weights loaded successfully!")

    # 3. Select a test image from unseen test split
    test_img_dir = SCRIPT_DIR / "data" / "NEU-DET" / "test" / "images"
    if not test_img_dir.exists():
        test_img_dir = REPO_ROOT / "learn" / "data" / "NEU-DET" / "test" / "images"
    test_images = sorted(test_img_dir.glob("*.jpg"))
    if not test_images:
        raise FileNotFoundError(f"No test images found in {test_img_dir}")
    
    # Pick a scratch defect image (scratches_241 or first available)
    scratch_samples = [p for p in test_images if "scratches" in p.name]
    sample_path = scratch_samples[0] if scratch_samples else test_images[0]
    print("\n[3] Testing on unseen test image:")
    print(f"    -> {sample_path.name}")

    orig_bgr = cv2.imread(str(sample_path))
    if orig_bgr is None:
        raise FileNotFoundError(f"Failed to load image from {sample_path}")
    h_orig, w_orig = orig_bgr.shape[:2]

    # Preprocess image tensor: BGR -> RGB, resize to (200, 200), normalize to [0, 1]
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (200, 200))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # Add batch dimension: [1, 3, 200, 200]

    # 4. Neural Network Forward Pass
    with torch.no_grad():
        detections, _ = model(tensor)  # detections shape: [1, 2529, 11]

    print("\n[4] Raw Neural Network Output:")
    print(f"    Total candidate boxes predicted: {detections.shape[1]}")

    # 5. Non-Maximum Suppression (NMS)
    conf_thres = 0.25   # Minimum detection confidence
    iou_thres = 0.45    # IoU overlap suppression threshold
    print("\n[5] Applying Non-Maximum Suppression (NMS):")
    print(f"    Confidence Threshold = {conf_thres}  |  NMS IoU Threshold = {iou_thres}")

    filtered_preds = non_max_suppression(detections, conf_thres=conf_thres, iou_thres=iou_thres)
    boxes = filtered_preds[0]  # First image in batch

    print(f"\n[6] Detections Found: {len(boxes)}")
    annotated = orig_bgr.copy()

    if len(boxes) > 0:
        for idx, det in enumerate(boxes, 1):
            # Box format: [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            cls_id = int(cls_id)
            cls_name = CLASS_NAMES[cls_id]

            # Scale coordinates from 200x200 to original image resolution
            rx1 = int(x1 * (w_orig / 200.0))
            ry1 = int(y1 * (h_orig / 200.0))
            rx2 = int(x2 * (w_orig / 200.0))
            ry2 = int(y2 * (h_orig / 200.0))

            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            print(f"    • Detection #{idx}: {cls_name.upper()} ({conf*100:.1f}%) at [{rx1}, {ry1}, {rx2}, {ry2}]")

            # Draw bounding box and label
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, 2)
            label_text = f"{cls_name}: {conf*100:.1f}%"
            cv2.putText(annotated, label_text, (rx1, max(ry1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    else:
        print("    No defects detected above confidence threshold.")

    # 6. Save annotated image
    out_dir = SCRIPT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "baseline_prediction.png"
    cv2.imwrite(str(out_img_path), annotated)
    print("\n[7] Saved detection visualization to:")
    print(f"    -> {out_img_path}")

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE: You now know how to run inference and NMS!")
    print("Next step: Run `python learn/04_train_one_step.py` to see the training loop.")
    print("=" * 70)

if __name__ == "__main__":
    main()
