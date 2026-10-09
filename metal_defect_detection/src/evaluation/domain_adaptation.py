"""
Cross-Dataset Domain Adaptation and Generalization on GC10-DET (Hypothesis H4).

Evaluates model robustness under industrial domain shift:
1. Zero-shot cross-domain transfer on shared defect morphologies.
2. Few-shot fine-tuning and domain adaptation to 10 GC10-DET defect classes.
3. Comparative resilience: M1 (Plain Baseline) vs M4 (CLAHE + Bilateral + Attention).
"""

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import evaluate_detections, evaluate_model_on_loader
from src.models.detector import DefectDetector
from src.models.head import DetectHead
from src.training.loss import ComputeLoss

from collections import defaultdict

GC10_CLASSES: list[str] = [
    "crease",          # Class 0
    "crescent_gap",    # Class 1
    "inclusion",       # Class 2
    "oil_spot",        # Class 3
    "punching_hole",   # Class 4
    "rolled_pit",      # Class 5
    "silk_spot",       # Class 6
    "waist_folding",   # Class 7
    "water_spot",      # Class 8
    "welding_line",    # Class 9
]

# Exploratory morphological cross-domain mapping between NEU-DET (6 source classes)
# and GC10-DET (10 target classes).
# Note: In real-world transfer evaluation, zero-shot mapping across disparate defect taxonomies
# serves only as an exploratory heuristic. The authoritative cross-dataset benchmark uses
# honest few-shot linear probing on the real GC10-DET dataset (Lv et al., 2020) without class remapping.
NEU_TO_GC10_MAPPING: dict[str, str] = {
    "inclusion": "inclusion",            # Physical defect match
    "pitted_surface": "rolled_pit",      # Surface indentation / pit depressions
    "scratches": "crease",               # Linear directional surface deformities
    "patches": "water_spot",             # Surface oxidation / liquid residues
    "crazing": "welding_line",           # Linear fracture / joint stress marks
    "rolled-in_scale": "waist_folding",  # Heavy mechanical rolling deformation
}


class GC10Dataset(Dataset):
    """
    Dataset loader for GC10-DET metallic surface defects.
    Reads images and YOLO formatted labels.
    """

    def __init__(
        self,
        image_dir: Path | str,
        label_dir: Path | str,
        target_size: tuple[int, int] = (200, 200),
        preprocessor: Any | None = None,
        is_training: bool = False,
        image_paths: list[Path] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.target_size = target_size
        self.preprocessor = preprocessor
        self.is_training = is_training

        if image_paths is not None:
            self.image_paths = sorted(image_paths)
        elif self.image_dir.exists():
            self.image_paths = sorted(
                [p for p in self.image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".bmp"]]
            )
        else:
            self.image_paths = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Failed to load GC10 image: {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.preprocessor is not None:
            if callable(self.preprocessor):
                rgb = self.preprocessor(rgb)
            elif hasattr(self.preprocessor, "process"):
                rgb = self.preprocessor.process(rgb)

        tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        lbl_path = self.label_dir / f"{img_path.stem}.txt"
        targets = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = float(parts[0])
                        xc, yc, w, h = [float(x) for x in parts[1:5]]
                        targets.append([cls_id, xc, yc, w, h])
                    except ValueError:
                        continue

        if len(targets) > 0:
            target_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            target_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return tensor_img, target_tensor, str(img_path)


def sample_few_shot_subset(
    image_dir: Path | str,
    label_dir: Path | str,
    shots_per_class: int = 10,
    seed: int = 42,
) -> list[Path]:
    """
    Deterministically samples up to N real images per defect class for few-shot adaptation.
    Draws images containing each class to balance representation across all 10 GC10 defect types.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    rng = random.Random(seed)

    all_images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".bmp"]]
    ) if image_dir.exists() else []

    class_to_images: dict[int, list[Path]] = defaultdict(list)
    for img_path in all_images:
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(float(parts[0]))
                        if 0 <= cls_id < len(GC10_CLASSES):
                            class_to_images[cls_id].append(img_path)
                    except ValueError:
                        continue

    selected_stems = set()
    selected_paths: list[Path] = []

    for c in range(len(GC10_CLASSES)):
        candidates = list(dict.fromkeys(class_to_images[c]))  # unique, preserve order
        rng.shuffle(candidates)
        picked = candidates[:shots_per_class]
        for p in picked:
            if p.stem not in selected_stems:
                selected_stems.add(p.stem)
                selected_paths.append(p)

    return sorted(selected_paths)


def create_mock_gc10_test_fixtures(
    base_dir: Path | str = Path("data/mock_gc10_fixtures"),
    num_train_per_class: int = 20,
    num_test_per_class: int = 15,
    seed: int = 42,
) -> Path:
    """
    Generates lightweight synthetic mock image and label fixtures exclusively for fast,
    isolated unit tests (e.g. tests/test_domain_adaptation.py) without requiring external
    dataset downloads. Real cross-domain evaluations must always use data/GC10-DET.
    """
    base_dir = Path(base_dir)
    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"
    test_img_dir = base_dir / "test" / "images"
    test_lbl_dir = base_dir / "test" / "labels"

    for d in [train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Check if already generated
    existing = list(test_img_dir.glob("*.jpg"))
    if len(existing) >= len(GC10_CLASSES) * num_test_per_class:
        return base_dir

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    for split, n_per_class in [("train", num_train_per_class), ("test", num_test_per_class)]:
        img_out = train_img_dir if split == "train" else test_img_dir
        lbl_out = train_lbl_dir if split == "train" else test_lbl_dir

        for cls_idx, cls_name in enumerate(GC10_CLASSES):
            for i in range(n_per_class):
                # Synthesize metal surface with domain shift:
                # GC10-DET has darker, higher contrast, and directional brushed steel patterns
                img = np_rng.normal(loc=115, scale=28, size=(200, 200)).clip(40, 220).astype(np.uint8)
                # Add horizontal brush grain
                grain = np.tile(np_rng.normal(loc=0, scale=12, size=(200, 1)), (1, 200))
                img = np.clip(img.astype(np.float32) + grain, 20, 240).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # Add distinct illumination gradient across the steel plate
                grad = np.linspace(-35, 35, 200).reshape(1, 200, 1)
                img = np.clip(img.astype(np.float32) + grad, 0, 255).astype(np.uint8)

                boxes = []
                num_boxes = rng.randint(1, 3)
                for _ in range(num_boxes):
                    if cls_name == "punch_hole":
                        cx, cy = rng.randint(40, 160), rng.randint(40, 160)
                        r = rng.randint(10, 22)
                        cv2.circle(img, (cx, cy), r, (15, 15, 15), -1)
                        boxes.append([cls_idx, cx / 200.0, cy / 200.0, (2 * r) / 200.0, (2 * r) / 200.0])
                    elif cls_name == "welding_line":
                        x1, y1 = rng.randint(20, 60), rng.randint(20, 180)
                        x2, y2 = rng.randint(140, 180), rng.randint(20, 180)
                        cv2.line(img, (x1, y1), (x2, y2), (235, 230, 210), rng.randint(3, 6))
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        w, h = max(abs(x2 - x1), 10), max(abs(y2 - y1), 10)
                        boxes.append([cls_idx, cx / 200.0, cy / 200.0, w / 200.0, h / 200.0])
                    elif cls_name in ["water_spot", "oil_spot", "silk_spot"]:
                        cx, cy = rng.randint(40, 160), rng.randint(40, 160)
                        ax, ay = rng.randint(15, 35), rng.randint(10, 25)
                        color = (40, 45, 55) if cls_name == "oil_spot" else (180, 185, 200)
                        cv2.ellipse(img, (cx, cy), (ax, ay), rng.randint(0, 180), 0, 360, color, -1)
                        boxes.append([cls_idx, cx / 200.0, cy / 200.0, (2 * ax) / 200.0, (2 * ay) / 200.0])
                    elif cls_name in ["rolled_pit", "inclusion"]:
                        cx, cy = rng.randint(30, 170), rng.randint(30, 170)
                        w, h = rng.randint(12, 30), rng.randint(12, 30)
                        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (30, 30, 35), -1)
                        boxes.append([cls_idx, cx / 200.0, cy / 200.0, w / 200.0, h / 200.0])
                    else:  # crease, waist_folding, crescent_gap
                        x1, y1 = rng.randint(30, 170), rng.randint(30, 170)
                        w, h = rng.randint(30, 70), rng.randint(8, 18)
                        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (55, 55, 60), 2)
                        boxes.append([cls_idx, (x1 + w / 2) / 200.0, (y1 + h / 2) / 200.0, w / 200.0, h / 200.0])

                filename = f"gc10_{cls_name}_{i:03d}"
                cv2.imwrite(str(img_out / f"{filename}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                with open(lbl_out / f"{filename}.txt", "w", encoding="utf-8") as f:
                    f.writelines(f"{b[0]} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f} {b[4]:.4f}\n" for b in boxes)

    return base_dir


# Backward-compatible alias for unit tests and legacy fixtures
generate_synthetic_gc10_benchmark = create_mock_gc10_test_fixtures


class DomainAdaptedDetector(nn.Module):
    """
    Wraps an NEU-DET trained DefectDetector and replaces the classification head
    with a 10-class head for GC10-DET adaptation.
    """

    def __init__(
        self,
        base_model: DefectDetector,
        num_target_classes: int = len(GC10_CLASSES),
        freeze_backbone: bool = True,
        freeze_neck: bool = False,
    ):
        super().__init__()
        self.variant_name = f"{base_model.variant_name}_GC10_Adapted"
        self.use_preprocessing = base_model.use_preprocessing
        self.preprocessor = base_model.preprocessor
        self.backbone = base_model.backbone
        self.neck = base_model.neck
        self.freeze_backbone = freeze_backbone
        self.freeze_neck = freeze_neck
        self.img_size = getattr(base_model, "img_size", (200, 200))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        if freeze_neck:
            for p in self.neck.parameters():
                p.requires_grad = False
            self.neck.eval()
            for m in self.neck.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Re-initialize head with target number of defect classes (10)
        self.head = DetectHead(
            num_classes=num_target_classes,
            in_channels=self.neck.out_channels,
            anchors=base_model.head.anchors.cpu().numpy().tolist(),
            img_size=self.img_size,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.freeze_backbone:
                self.backbone.eval()
                for m in self.backbone.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            if self.freeze_neck:
                self.neck.eval()
                for m in self.neck.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

    def forward(self, x: torch.Tensor):
        img_size = (x.shape[3], x.shape[2]) if x.dim() >= 4 else getattr(self, "img_size", (200, 200))
        features = self.backbone(x)
        fused = self.neck(features)
        return self.head(fused, img_size=img_size)


def evaluate_zero_shot_domain_transfer(
    model: DefectDetector,
    gc10_test_loader: DataLoader,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Evaluates zero-shot cross-domain transferability on overlapping/analogue classes.
    Maps predicted NEU-DET class scores to the closest morphological GC10-DET category.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    neu_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in gc10_test_loader:
            images = images.to(device)
            decoded, _ = model(images)

            # Map NEU class index to corresponding GC10 class index
            # decoded: (B, num_boxes, 5 + 6)
            b, num_boxes, _ = decoded.shape
            gc10_decoded = torch.zeros((b, num_boxes, 5 + len(GC10_CLASSES)), device=device)
            gc10_decoded[..., :5] = decoded[..., :5]  # x, y, w, h, obj_conf

            # Remap class probabilities
            for neu_idx, neu_name in enumerate(neu_classes):
                if neu_name in NEU_TO_GC10_MAPPING:
                    gc10_name = NEU_TO_GC10_MAPPING[neu_name]
                    gc10_idx = GC10_CLASSES.index(gc10_name)
                    gc10_decoded[..., 5 + gc10_idx] = decoded[..., 5 + neu_idx]

            from src.utils.box_ops import non_max_suppression

            nms_preds = non_max_suppression(gc10_decoded, conf_thres=0.001, iou_thres=0.45)
            h, w = images.shape[2:]

            for b_idx in range(b):
                img_targets = targets[targets[:, 0] == b_idx]
                if img_targets.numel() > 0:
                    xc, yc = img_targets[:, 2] * w, img_targets[:, 3] * h
                    bw, bh = img_targets[:, 4] * w, img_targets[:, 5] * h
                    boxes = torch.stack(
                        [img_targets[:, 1], xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], dim=1
                    )
                else:
                    boxes = torch.zeros((0, 5), dtype=torch.float32)

                all_preds.append(nms_preds[b_idx].cpu())
                all_targets.append(boxes)

    metrics = evaluate_detections(
        all_preds,
        all_targets,
        iou_threshold=0.5,
        num_classes=len(GC10_CLASSES),
        class_names=GC10_CLASSES,
    )
    return metrics


def train_few_shot_adaptation(
    adapted_model: DomainAdaptedDetector,
    gc10_train_loader: DataLoader,
    gc10_test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Fine-tunes the adapted model on few-shot GC10-DET samples.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    adapted_model.to(device)
    adapted_model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, adapted_model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    loss_fn = ComputeLoss(obj_pos_weight=1.0)

    loss_history = []
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        for images, targets, _ in gc10_train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = adapted_model(images)
            loss, _ = loss_fn(preds, targets, adapted_model)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        loss_history.append(avg_loss)

    # Evaluate on GC10 test set
    adapted_model.eval()
    test_metrics = evaluate_model_on_loader(
        adapted_model,
        gc10_test_loader,
        device=device,
        num_classes=len(GC10_CLASSES),
        class_names=GC10_CLASSES,
    )
    return {
        "final_train_loss": loss_history[-1],
        "loss_history": loss_history,
        "mAP@0.5": test_metrics["mAP@0.5"],
        "precision": test_metrics["mean_precision"],
        "recall": test_metrics["mean_recall"],
        "per_class": test_metrics["per_class"],
    }
