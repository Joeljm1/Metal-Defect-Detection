#!/usr/bin/env python3
"""
Tutorial 00: Writing the Entire Dataset & DataLoader Pipeline from Pure Scratch
=================================================================================
NO imports from `src`. NO third-party wrappers.
Only standard Python, OpenCV, and pure PyTorch.

This script teaches you:
1. How raw label files are parsed from scratch using plain Python file I/O.
2. The exact math for bounding box coordinate transformations:
   - Normalized YOLO format (x_center, y_center, width, height) in [0, 1]
   - Pixel coordinates (xmin, ymin, xmax, ymax) in [0, 200]
3. How to write a PyTorch `Dataset` from scratch (`__init__`, `__len__`, `__getitem__`).
4. Why PyTorch's default DataLoader crashes on Object Detection, and how
   to write a custom `collate_fn` to fix it.
"""

from pathlib import Path
from typing import overload

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

# The 6 metal defect classes in NEU-DET
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


# ==============================================================================
# PART 1: PARSING RAW LABEL FILES FROM SCRATCH
# ==============================================================================
def parse_label_file_from_scratch(label_file_path: Path) -> torch.Tensor:
    """
    Reads a raw YOLO label .txt file using standard Python file reading.
    Each line in the file is: <class_id> <x_center> <y_center> <width> <height>

    Returns:
        torch.Tensor of shape (N, 5), where each row is [class_id, xc, yc, w, h]
    """
    if not label_file_path.exists() or label_file_path.stat().st_size == 0:
        # If the image has no defects, return an empty tensor with 5 columns
        return torch.zeros((0, 5), dtype=torch.float32)

    boxes = []
    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = float(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # Clamp between 0.0 and 1.0 in case of minor annotation imperfections
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                boxes.append([class_id, xc, yc, w, h])

    if len(boxes) == 0:
        return torch.zeros((0, 5), dtype=torch.float32)

    return torch.tensor(boxes, dtype=torch.float32)


# ==============================================================================
# PART 2: BOUNDING BOX COORDINATE CONVERSIONS
# ==============================================================================
def yolo_to_pixel_box(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """
    Converts normalized YOLO coordinates (xc, yc, w, h) in range [0, 1]
    into absolute pixel corners (xmin, ymin, xmax, ymax) on the image.

    Math:
        xmin = (xc - w / 2) * img_w
        ymin = (yc - h / 2) * img_h
        xmax = (xc + w / 2) * img_w
        ymax = (yc + h / 2) * img_h
    """
    xmin = int((xc - w / 2.0) * img_w)
    ymin = int((yc - h / 2.0) * img_h)
    xmax = int((xc + w / 2.0) * img_w)
    ymax = int((yc + h / 2.0) * img_h)

    # Clip to image boundary
    xmin = max(0, min(img_w - 1, xmin))
    ymin = max(0, min(img_h - 1, ymin))
    xmax = max(0, min(img_w - 1, xmax))
    ymax = max(0, min(img_h - 1, ymax))

    return xmin, ymin, xmax, ymax


# ==============================================================================
# PART 3: WRITING A PYTORCH DATASET FROM SCRATCH
# ==============================================================================
class ScratchMetalDefectDataset(Dataset):
    """
    A 100% custom PyTorch Dataset implementing the 3 required methods:
    1. __init__: Finds all matching image and label file paths on disk.
    2. __len__: Returns the total number of images in the dataset.
    3. __getitem__: Loads one image, normalizes it, and loads its bounding boxes.
    """

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        target_size: tuple[int, int] = (200, 200),
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.target_size = target_size

        # Find all .jpg files
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        print(
            f"[Dataset Init] Found {len(self.image_paths)} images in {self.image_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Loads the sample at index `idx`.

        Returns:
            image_tensor: shape (3, 200, 200), float32 in range [0, 1]
            labels_tensor: shape (N, 5), where each row is [class_id, xc, yc, w, h]
            image_path: string filename for debugging
        """
        img_path = self.image_paths[idx]

        # 1. Load image using OpenCV (OpenCV loads BGR by default)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image from {img_path}")

        # 2. Convert BGR -> RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 3. Resize to target size (200, 200)
        resized = cv2.resize(rgb, self.target_size)

        # 4. Convert NumPy (Height, Width, Channels) in [0, 255]
        #    to PyTorch (Channels, Height, Width) in [0.0, 1.0]
        #    OpenCV shape: (200, 200, 3) -> permute(2, 0, 1) -> (3, 200, 200)
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # 5. Load corresponding YOLO label file
        label_path = self.label_dir / f"{img_path.stem}.txt"
        labels_tensor = parse_label_file_from_scratch(label_path)

        return image_tensor, labels_tensor, str(img_path)


@overload
def add(a: int, b: int): ...
@overload
def add(a: float, b: float): ...
@overload
def add(a: str, b: str): ...
def add(a, b):
    return a + b


# ==============================================================================
# PART 4: WRITING CUSTOM COLLATE_FN (WHY DEFAULT COLLATER CRASHES!)
# ==============================================================================
def custom_yolo_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Why PyTorch's default DataLoader collate crashes on object detection:
    -------------------------------------------------------------------
    In classification, every image has 1 label integer: [0, 2, 5, 1] -> can be stacked into a tensor!
    In object detection:
        Image 0 might have 3 defect boxes -> shape (3, 5)
        Image 1 might have 0 defect boxes -> shape (0, 5)
        Image 2 might have 6 defect boxes -> shape (6, 5)
    If you call torch.stack([tensor(3, 5), tensor(0, 5), tensor(6, 5)]), PyTorch crashes with:
    `RuntimeError: stack expects each tensor to be equal size`!

    The Solution (Custom Collate):
    ------------------------------
    We prepend the batch index (0, 1, 2...) to each bounding box:
    [image_idx_in_batch, class_id, xc, yc, w, h]
    and concatenate all boxes across the entire batch into one clean (Total_Boxes, 6) tensor!
    """
    images, targets, paths = zip(*batch)

    # Images all have the exact same shape (3, 200, 200), so they CAN be stacked!
    images_batch = torch.stack(images, dim=0)  # Shape: (Batch_Size, 3, 200, 200)
    # Package targets with their batch index
    all_targets = []
    for image_idx, target_boxes in enumerate(targets):
        breakpoint()
        num_boxes = target_boxes.shape[0]
        if num_boxes > 0:
            # Create a column of the current batch index: [image_idx, image_idx, ...]
            batch_col = torch.full(
                (num_boxes, 1), fill_value=image_idx, dtype=torch.float32
            )
            # Combine: (num_boxes, 1) + (num_boxes, 5) -> (num_boxes, 6)
            annotated_boxes = torch.cat((batch_col, target_boxes), dim=1)
            all_targets.append(annotated_boxes)

    if len(all_targets) > 0:
        targets_batch = torch.cat(all_targets, dim=0)  # Shape: (Total_N_Boxes, 6)
    else:
        targets_batch = torch.zeros((0, 6), dtype=torch.float32)

    return images_batch, targets_batch, list(paths)


# ==============================================================================
# DEMONSTRATION & VERIFICATION
# ==============================================================================
def main():
    print("=" * 75)
    print("DEMO: ZERO-DEPENDENCY DATASET & DATALOADER FROM SCRATCH")
    print("=" * 75)

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data" / "NEU-DET" / "train"
    if not data_dir.exists():
        data_dir = script_dir.parent / "learn" / "data" / "NEU-DET" / "train"

    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"

    if not lbl_dir.exists():
        raise FileNotFoundError(f"Label directory not found at: {lbl_dir}")

    lbl_files = sorted(lbl_dir.glob("*.txt"))
    if not lbl_files:
        raise FileNotFoundError(f"No .txt label files found in {lbl_dir}")
    sample_lbl = lbl_files[0]
    print(f"\n[1] Parsing raw label file: {sample_lbl.name}")
    raw_boxes = parse_label_file_from_scratch(sample_lbl)
    print(f"    Raw Tensor from scratch:\n{raw_boxes}")

    for i, row in enumerate(raw_boxes):
        cid, xc, yc, w, h = row.tolist()
        cid = int(cid)
        xmin, ymin, xmax, ymax = yolo_to_pixel_box(xc, yc, w, h, img_w=200, img_h=200)
        print(
            f"    Box {i+1}: Class={CLASS_NAMES[cid]} | YOLO=({xc:.2f}, {yc:.2f}, {w:.2f}, {h:.2f}) -> Pixels=[{xmin}, {ymin}, {xmax}, {ymax}]"
        )

    # Test Part 3: Test custom Dataset
    print("\n[2] Testing custom ScratchMetalDefectDataset:")
    dataset = ScratchMetalDefectDataset(image_dir=img_dir, label_dir=lbl_dir)
    print(f"    Dataset length: {len(dataset)}")

    sample_img_t, sample_lbl_t, _sample_path = dataset[0]
    print(
        f"    Sample 0 image tensor:  shape={sample_img_t.shape}, min={sample_img_t.min():.2f}, max={sample_img_t.max():.2f}"
    )
    print(f"    Sample 0 labels tensor: shape={sample_lbl_t.shape}")

    # Test Part 4: Test custom DataLoader with collate_fn
    print("\n[3] Testing PyTorch DataLoader with custom_yolo_collate_fn:")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_yolo_collate_fn,
    )

    batch_imgs, batch_targets, _batch_paths = next(iter(loader))
    print(
        f"    Batch Images Tensor:  {batch_imgs.shape}  (Batch=4, Channels=3, H=200, W=200)"
    )
    print(
        f"    Batch Targets Tensor: {batch_targets.shape}  (Total boxes across all 4 images)"
    )
    print(
        "    Format of targets:    [batch_index, class_id, x_center, y_center, width, height]"
    )
    print("\n    First 4 rows of batch_targets:")
    for row in batch_targets[:4]:
        b_idx, c_id, xc, yc, w, h = row.tolist()
        print(
            f"      Img {int(b_idx)} | Class {int(c_id)} ({CLASS_NAMES[int(c_id)]:<15}) | Box: ({xc:.3f}, {yc:.3f}, {w:.3f}, {h:.3f})"
        )

    print("\n" + "=" * 75)
    print(
        "SUCCESS: You just wrote a complete Object Detection dataset parser from scratch!"
    )
    print(
        "You now understand every single line without relying on any helper libraries."
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
