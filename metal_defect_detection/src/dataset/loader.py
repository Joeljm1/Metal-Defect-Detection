"""
PyTorch Dataset and DataLoader for NEU-DET Metal Surface Defect Detection.
"""

from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataset.parser import parse_yolo_label_file, CLASS_NAMES


class NEUDataset(Dataset):
    """
    Dataset loader for NEU-DET metallic surface defect detection.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        target_size: Tuple[int, int] = (200, 200),
        is_training: bool = False,
        augment: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.preprocessor = preprocessor
        self.target_size = target_size
        self.is_training = is_training
        self.augment = augment and is_training

        # Scan for images
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in supported_exts]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentations(
        self, image: np.ndarray, labels: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
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
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
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
    preprocessor: Optional[Callable] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    target_size: Tuple[int, int] = (200, 200),
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create train and validation DataLoaders.
    """
    data_dir = Path(data_dir)
    train_img_dir = data_dir / "train" / "images"
    train_lbl_dir = data_dir / "train" / "labels"
    test_img_dir = data_dir / "test" / "images"
    test_lbl_dir = data_dir / "test" / "labels"

    train_dataset = NEUDataset(
        image_dir=train_img_dir,
        label_dir=train_lbl_dir,
        preprocessor=preprocessor,
        target_size=target_size,
        is_training=True,
        augment=augment_train,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=yolo_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
