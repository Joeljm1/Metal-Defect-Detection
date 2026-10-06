"""
Model Trainer for Ablation Matrix (M1 - M4).
"""

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.detector import DefectDetector
from src.training.loss import ComputeLoss


class Trainer:
    """
    Manages model training, validation, and checkpointing for defect detection models.
    """

    def __init__(
        self,
        model: DefectDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        output_dir: str | Path = "checkpoints",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compute_loss = ComputeLoss()

        # Parameter grouping: apply weight decay only to 2D+ weights (e.g. conv kernels)
        # and not to 1D biases or normalization affine parameters.
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        self.scheduler = None  # Created in fit() with T_max aligned to the epoch count

        self.best_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, epoch: int, dry_run: bool = False) -> dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        total_box = 0.0
        total_obj = 0.0
        total_cls = 0.0
        steps = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]")
        for batch_idx, (images, targets, _) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss, items = self.compute_loss(preds, targets, self.model)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            total_loss += items["loss_total"]
            total_box += items["loss_box"]
            total_obj += items["loss_obj"]
            total_cls += items["loss_cls"]
            steps += 1

            pbar.set_postfix({
                "loss": f"{items['loss_total']:.4f}",
                "box": f"{items['loss_box']:.4f}",
                "obj": f"{items['loss_obj']:.4f}",
            })

            if dry_run and batch_idx >= 2:
                break

        return {
            "loss": total_loss / max(1, steps),
            "loss_total": total_loss / max(1, steps),
            "loss_box": total_box / max(1, steps),
            "loss_obj": total_obj / max(1, steps),
            "loss_cls": total_cls / max(1, steps),
        }

    def validate(self, dry_run: bool = False, epoch: int | None = None) -> dict[str, float]:
        """Runs validation loss calculation in eval mode to prevent BatchNorm contamination."""
        self.model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch_idx, (images, targets, _) in enumerate(self.val_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(images)
                raw_preds = preds[1] if isinstance(preds, tuple) else preds
                _loss, items = self.compute_loss(raw_preds, targets, self.model)

                total_loss += items["loss_total"]
                steps += 1

                if dry_run and batch_idx >= 2:
                    break

        avg_loss = total_loss / max(1, steps)
        return {"loss": avg_loss, "val_loss": avg_loss}

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> Path:
        """Saves model checkpoint."""
        ckpt_name = f"{self.model.variant_name}_best.pt" if is_best else f"{self.model.variant_name}_epoch_{epoch}.pt"
        ckpt_path = self.output_dir / ckpt_name
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "variant": self.model.variant_name,
        }, ckpt_path)
        return ckpt_path

    def fit(self, epochs: int = 10, dry_run: bool = False) -> dict[str, Any]:
        """Trains for the specified number of epochs."""
        # Dynamically align CosineAnnealingLR cycle to total epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs), eta_min=1e-5
        )
        print(f"Starting training: {self.model.variant_name} on {self.device} (epochs={epochs}, dry_run={dry_run})")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch, dry_run=dry_run)
            val_metrics = self.validate(dry_run=dry_run)
            self.scheduler.step()

            self.history["train_loss"].append(train_metrics["loss_total"])
            self.history["val_loss"].append(val_metrics["val_loss"])

            print(
                f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_metrics['loss_total']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )

            # Checkpoint
            if val_metrics["val_loss"] < self.best_loss and not dry_run:
                self.best_loss = val_metrics["val_loss"]
                ckpt_path = self.save_checkpoint(epoch, self.best_loss, is_best=True)
                print(f" Saved new best model to {ckpt_path}")

            if dry_run:
                break

        if not dry_run and self.history["val_loss"]:
            # Persist end-of-training weights (e.g. for ONNX export / Grad-CAM analysis)
            final_path = self.save_checkpoint(epochs, self.history["val_loss"][-1], is_best=False)
            print(f" Saved final model to {final_path}")

        if not dry_run and self.best_loss < float("inf"):
            best_path = self.output_dir / f"{self.model.variant_name}_best.pt"
            if best_path.exists():
                print(f"Reloading best model weights from {best_path} (best val_loss={self.best_loss:.4f})")
                best_ckpt = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(best_ckpt["model_state"])

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f}s")
        return {"history": self.history, "elapsed_seconds": elapsed}
