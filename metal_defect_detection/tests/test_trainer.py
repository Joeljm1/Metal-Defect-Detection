"""
Unit test for Trainer class.
"""

import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.detector import DefectDetector
from src.training.trainer import Trainer


def dummy_collate(batch):
    images = torch.stack([item[0] for item in batch])
    # Targets: [img_idx, class_id, xc, yc, w, h]
    targets = torch.tensor([
        [0.0, 1.0, 0.5, 0.5, 0.2, 0.2],
        [1.0, 0.0, 0.3, 0.3, 0.15, 0.15],
    ])
    paths = ["dummy1.jpg", "dummy2.jpg"]
    return images, targets, paths


def test_trainer_single_step():
    model = DefectDetector.build_model(variant="M1")

    # 2 dummy images
    dummy_images = torch.randn(2, 3, 200, 200)
    dummy_dataset = TensorDataset(dummy_images)
    loader = DataLoader(dummy_dataset, batch_size=2, collate_fn=dummy_collate)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            device="cpu",
            learning_rate=1e-3,
            output_dir=tmp_dir,
        )

        train_metrics = trainer.train_epoch(epoch=1, dry_run=True)
        assert "loss" in train_metrics
        assert train_metrics["loss"] > 0.0

        val_metrics = trainer.validate(epoch=1, dry_run=True)
        assert "loss" in val_metrics
        assert val_metrics["loss"] > 0.0

        ckpt_path = trainer.save_checkpoint(epoch=1, val_loss=val_metrics["loss"], is_best=True)
        assert Path(ckpt_path).exists()
