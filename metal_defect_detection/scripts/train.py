"""
Training Script for Defect Detection Ablation Models (M1 to M4).
"""

from pathlib import Path
import argparse
import torch

from src.models.detector import DefectDetector
from src.dataset.loader import create_dataloaders
from src.preprocessing.pipeline import DefectPreprocessor
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Defect Detection Ablation Models")
    parser.add_argument("--model", type=str, default="M4", choices=["M1", "M2", "M3", "M4"], help="Ablation model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data/NEU-DET", help="Dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Run quick 2-batch sanity verification")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building Model {args.model} on device {device}...")

    # Build ablation model
    model = DefectDetector.build_model(variant=args.model, num_classes=6)

    # Setup preprocessor based on ablation configuration
    preprocessor = None
    if args.model in ("M2", "M4"):
        print("Enabling CLAHE + Bilateral Filtering Preprocessor for DataLoader...")
        preprocessor = DefectPreprocessor(use_clahe=True, use_bilateral=True)

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(200, 200),
        augment_train=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
    )

    results = trainer.fit(epochs=args.epochs, dry_run=args.dry_run)
    print(f"[SUCCESS] Finished training run for {args.model}.")


if __name__ == "__main__":
    main()
