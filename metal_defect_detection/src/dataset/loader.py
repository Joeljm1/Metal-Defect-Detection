"""
PyTorch Dataset and DataLoader for NEU-DET Metal Surface Defect Detection.
"""

import random
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.parser import parse_yolo_label_file


class NEUDataset(Dataset):
    """
    Dataset loader for NEU-DET metallic surface defect detection.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
        target_size: tuple[int, int] = (200, 200),
        is_training: bool = False,
        augment: bool = False,
        image_paths: list[Path] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.preprocessor = preprocessor
        self.target_size = target_size
        self.is_training = is_training
        self.augment = augment and is_training

        # Scan for images, or use an explicit subset (e.g. a train/val split)
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if image_paths is not None:
            self.image_paths = sorted(image_paths)
        else:
            self.image_paths = sorted(
                [p for p in self.image_dir.iterdir() if p.suffix.lower() in supported_exts]
            )

        if len(self.image_paths) == 0 and image_paths is None:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentations(
        self, image: np.ndarray, labels: torch.Tensor
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Lightweight geometric and photometric augmentations for training.
        """
        # Horizontal flip (p=0.5)
        if random.random() > 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            if labels.numel() > 0:
                labels[:, 1] = 1.0 - labels[:, 1]  # flip xc

        # Vertical flip (p=0.5)
        if random.random() > 0.5:
            image = np.ascontiguousarray(np.flipud(image))
            if labels.numel() > 0:
                labels[:, 2] = 1.0 - labels[:, 2]  # flip yc

        return image, labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse corresponding label file
        label_file = self.label_dir / f"{img_path.stem}.txt"
        labels = parse_yolo_label_file(label_file)

        # Apply data augmentation if active
        if self.augment:
            image, labels = self._apply_augmentations(image, labels)

        # Apply preprocessing pipeline (CLAHE / Bilateral Filter if specified)
        if self.preprocessor is not None:
            image = self.preprocessor(image)

        # Resize if required
        h, w = image.shape[:2]
        if (w, h) != tuple(self.target_size):
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Convert HWC uint8 [0, 255] to CHW float32 [0.0, 1.0]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return img_tensor, labels, str(img_path)


def yolo_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]]
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Collate function to batch images and assemble variable-length targets.
    
    Target format: (N, 6) where each row is [batch_idx, class_id, xc, yc, w, h]
    """
    images, targets, paths = zip(*batch)
    stacked_images = torch.stack(images, dim=0)

    batched_targets = []
    for batch_idx, target in enumerate(targets):
        if target.numel() > 0:
            # Prepend batch_idx to each target box: [batch_idx, class_id, xc, yc, w, h]
            idx_column = torch.full((target.shape[0], 1), batch_idx, dtype=torch.float32)
            batched_targets.append(torch.cat([idx_column, target], dim=1))

    if len(batched_targets) > 0:
        stacked_targets = torch.cat(batched_targets, dim=0)
    else:
        stacked_targets = torch.zeros((0, 6), dtype=torch.float32)

    return stacked_images, stacked_targets, list(paths)


def create_dataloaders(
    data_dir: str | Path,
    preprocessor: Callable | None = None,
    batch_size: int = 16,
    num_workers: int = 2,
    target_size: tuple[int, int] = (200, 200),
    augment_train: bool = True,
    val_fraction: float = 0.1,
    split_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create train, validation, and test DataLoaders.

    The validation split is carved deterministically from the training images
    (``val_fraction`` of them, shuffled with ``split_seed``) so that checkpoint
    selection never sees the held-out test split. The test split is only meant
    for final evaluation.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    train_img_dir = data_dir / "train" / "images"
    train_lbl_dir = data_dir / "train" / "labels"
    test_img_dir = data_dir / "test" / "images"
    test_lbl_dir = data_dir / "test" / "labels"

    # Deterministic train / validation split over the training images
    all_train_paths = NEUDataset(
        image_dir=train_img_dir, label_dir=train_lbl_dir, target_size=target_size
    ).image_paths
    indices = list(range(len(all_train_paths)))
    random.Random(split_seed).shuffle(indices)
    n_val = round(len(indices) * val_fraction)
    if len(indices) > 1:
        n_val = min(max(n_val, 1), len(indices) - 1)
    else:
        n_val = 0
    val_indices = set(indices[:n_val])
    train_paths = [p for i, p in enumerate(all_train_paths) if i not in val_indices]
    val_paths = [p for i, p in enumerate(all_train_paths) if i in val_indices]

    train_dataset = NEUDataset(
        image_dir=train_img_dir,
        label_dir=train_lbl_dir,
        preprocessor=preprocessor,
        target_size=target_size,
        is_training=True,
        augment=augment_train,
        image_paths=train_paths,
    )

    val_dataset = NEUDataset(
        image_dir=train_img_dir,
        label_dir=train_lbl_dir,
        preprocessor=preprocessor,
        target_size=target_size,
        is_training=False,
        augment=False,
        image_paths=val_paths,
    )

    test_dataset = NEUDataset(
        image_dir=test_img_dir,
        label_dir=test_lbl_dir,
        preprocessor=preprocessor,
        target_size=target_size,
        is_training=False,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=yolo_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=yolo_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=yolo_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
