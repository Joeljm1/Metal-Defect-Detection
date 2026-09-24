"""
Training Script for Defect Detection Ablation Models (M1 to M4).

Hyperparameter defaults (epochs / batch size / learning rate) come from the
selected variant's ``training:`` block in configs/model_mX.yaml; explicit CLI
flags override them.
"""

from pathlib import Path
import argparse
import torch
from src.models.detector import DefectDetector, load_variant_config
from src.dataset.loader import create_dataloaders
from src.preprocessing.pipeline import DefectPreprocessor
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model_on_loader


def main():
    parser = argparse.ArgumentParser(description="Train Defect Detection Ablation Models")
    parser.add_argument("--model", type=str, default="M4", choices=["M1", "M2", "M3", "M4"], help="Ablation model")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs (default: config training.epochs)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: config training.batch_size)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: config training.learning_rate)")
    parser.add_argument("--data-dir", type=str, default="data/NEU-DET", help="Dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Run quick 2-batch sanity verification")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time flips (exact reproducibility)")
    args = parser.parse_args()

    # Config-driven hyperparameter defaults (configs/model_mX.yaml `training:` block)
    t_cfg = load_variant_config(args.model).get("training", {})
    epochs = args.epochs if args.epochs is not None else int(t_cfg.get("epochs", 50))
    batch_size = args.batch_size if args.batch_size is not None else int(t_cfg.get("batch_size", 16))
    lr = args.lr if args.lr is not None else float(t_cfg.get("learning_rate", 1e-3))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building Model {args.model} on device {device} (epochs={epochs}, batch_size={batch_size}, lr={lr})...")

    # Build ablation model
    model = DefectDetector.build_model(variant=args.model, num_classes=6)

    # Setup preprocessor based on ablation configuration
    preprocessor = None
    if args.model in ("M2", "M4"):
        print("Enabling CLAHE + Bilateral Filtering Preprocessor for DataLoader...")
        preprocessor = DefectPreprocessor(use_clahe=True, use_bilateral=True)

    # Dataloaders: checkpoint selection uses the validation split; the test
    # split is evaluated once at the end.
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        batch_size=batch_size,
        num_workers=args.num_workers,
        target_size=(200, 200),
        augment_train=not args.no_augment,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=lr,
    )

    results = trainer.fit(epochs=epochs, dry_run=args.dry_run)

    if not args.dry_run:
        # Single final evaluation on the held-out test split
        metrics = evaluate_model_on_loader(model, test_loader, device)
        print(
            f"[TEST] mAP@0.5={metrics['mAP@0.5']:.4f} "
            f"precision={metrics['mean_precision']:.4f} "
            f"recall={metrics['mean_recall']:.4f} "
            f"f1={metrics['mean_f1']:.4f}"
        )

    print(f"[SUCCESS] Finished training run for {args.model}.")

if __name__ == "__main__":
    main()
