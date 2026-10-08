#!/usr/bin/env python3
"""
Step 1: Explore the Dataset and PyTorch DataLoader.
===================================================
This script teaches you:
1. How defect images and bounding box labels are stored.
2. What YOLO-format bounding box coordinates mean: (class_id, x_center, y_center, width, height).
3. How the PyTorch DataLoader batches images into tensors.
4. Saves a visual preview of ground-truth annotations to learn/output/sample_ground_truth.png.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'import src...' works regardless of current working directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
from torch.utils.data import DataLoader

# Import our project's dataset classes
from src.dataset.loader import NEUDataset, yolo_collate_fn

# Class names in NEU-DET
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

def main():
    print("=" * 70)
    print("STEP 1: EXPLORING THE NEU-DET DATASET")
    print("=" * 70)

    data_dir = SCRIPT_DIR / "data" / "NEU-DET"
    if not data_dir.exists():
        data_dir = REPO_ROOT / "learn" / "data" / "NEU-DET"

    train_img_dir = data_dir / "train" / "images"
    train_lbl_dir = data_dir / "train" / "labels"

    # 1. Inspect a raw label file
    lbl_files = sorted(train_lbl_dir.glob("*.txt"))
    if not lbl_files:
        raise FileNotFoundError(f"No label files found in {train_lbl_dir}")
    sample_lbl_path = lbl_files[0]
    sample_img_path = train_img_dir / f"{sample_lbl_path.stem}.jpg"

    print("\n[1] Examining a sample file pair:")
    print(f"    Image: {sample_img_path}")
    print(f"    Label: {sample_lbl_path}")

    with open(sample_lbl_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print("\n[2] Raw YOLO Annotation Format:")
    print("    Format: <class_id> <x_center> <y_center> <width> <height>")
    print("    (Coordinates are normalized between 0.0 and 1.0 relative to image size)")
    for i, line in enumerate(lines, 1):
        parts = line.split()
        cls_id = int(parts[0])
        xc, yc, w, h = [float(p) for p in parts[1:]]
        print(f"    Box {i}: Class {cls_id} ({CLASS_NAMES[cls_id]}) -> Center=({xc:.3f}, {yc:.3f}), Size=({w:.3f}, {h:.3f})")

    # 2. Visualize and save ground-truth boxes
    img = cv2.imread(str(sample_img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image from {sample_img_path}")
    h_img, w_img = img.shape[:2]

    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        xc, yc, bw, bh = [float(p) for p in parts[1:]]

        # Denormalize to pixel coordinates
        x1 = int((xc - bw / 2) * w_img)
        y1 = int((yc - bh / 2) * h_img)
        x2 = int((xc + bw / 2) * w_img)
        y2 = int((yc + bh / 2) * h_img)

        # Draw green rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{CLASS_NAMES[cls_id]}"
        cv2.putText(img, label, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    out_dir = SCRIPT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "sample_ground_truth.png"
    cv2.imwrite(str(out_img_path), img)
    print("\n[3] Saved ground-truth visual verification to:")
    print(f"    -> {out_img_path}")

    # 3. Create a PyTorch Dataset and DataLoader
    print("\n[4] PyTorch DataLoader Batching:")
    dataset = NEUDataset(
        image_dir=train_img_dir,
        label_dir=train_lbl_dir,
        target_size=(200, 200),
        augment=False,
    )
    print(f"    Total training images in dataset: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=yolo_collate_fn,
    )

    # Fetch one batch
    images, targets, _paths = next(iter(loader))
    print(f"    Batch Images Tensor:  {images.shape} (Batch=4, Channels=3, H=200, W=200)")
    print(f"    Images dtype & range: {images.dtype}, [{images.min():.2f}, {images.max():.2f}]")
    print(f"    Batch Targets Tensor: {targets.shape}")
    print("    Target columns:       [image_index_in_batch, class_id, x_center, y_center, width, height]")
    print(f"    Sample target row:    {targets[0].tolist()}")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE: You now understand how data flows into PyTorch!")
    print("Next step: Run `python learn/02_inspect_baseline_model.py` to see the neural network.")
    print("=" * 70)

if __name__ == "__main__":
    main()
