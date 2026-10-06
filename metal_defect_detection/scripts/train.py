"""
Training Script for Defect Detection Ablation Models (M1 to M4).

Hyperparameter defaults (epochs / batch size / learning rate) come from the
selected variant's ``training:`` block in configs/model_mX.yaml; explicit CLI
flags override them.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.dataset.loader import create_dataloaders
from src.dataset.parser import load_dataset_config
from src.evaluation.metrics import evaluate_model_on_loader
from src.models.detector import DefectDetector, load_variant_config
from src.preprocessing.pipeline import DefectPreprocessor
from src.training.trainer import Trainer


def set_seed(seed: int = 42):
    """Set random seed across random, numpy, and torch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    dataset_cfg = load_dataset_config()
    default_data_dir = dataset_cfg.get("path", "data/NEU-DET")
    raw_size = dataset_cfg.get("image_size", (200, 200))
    target_size: tuple[int, int] = (int(raw_size[0]), int(raw_size[1]))
    num_classes = dataset_cfg.get("num_classes", 6)

    parser = argparse.ArgumentParser(description="Train Defect Detection Ablation Models")
    parser.add_argument("--model", type=str, default="M4", choices=["M1", "M2", "M3", "M4"], help="Ablation model")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs (default: config training.epochs)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: config training.batch_size)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: config training.learning_rate)")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Run quick 2-batch sanity verification")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time flips (exact reproducibility)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use-clahe", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable CLAHE")
    parser.add_argument("--use-bilateral", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Bilateral filter")
    parser.add_argument("--use-eca", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable ECA channel attention")
    parser.add_argument("--use-spatial", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Spatial attention")
    args = parser.parse_args()

    set_seed(args.seed)

    # Config-driven hyperparameter defaults (configs/model_mX.yaml `training:` block)
    variant_cfg = load_variant_config(args.model)
    t_cfg = variant_cfg.get("training", {})
    m_cfg = variant_cfg.get("model", {})
    p_cfg = m_cfg.get("preprocessing", {})

    epochs = args.epochs if args.epochs is not None else int(t_cfg.get("epochs", 50))
    batch_size = args.batch_size if args.batch_size is not None else int(t_cfg.get("batch_size", 16))
    lr = args.lr if args.lr is not None else float(t_cfg.get("learning_rate", 1e-3))
    weight_decay = float(t_cfg.get("weight_decay", 5e-4))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building Model {args.model} on device {device} (epochs={epochs}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, seed={args.seed})...")

    # Build ablation model
    model = DefectDetector.build_model(variant=args.model, num_classes=num_classes)

    # Resolve individual preprocessing switches
    default_use_clahe = args.model in ("M2", "M4") or p_cfg.get("use_clahe", False)
    default_use_bilateral = args.model in ("M2", "M4") or p_cfg.get("use_bilateral", False)
    use_clahe = args.use_clahe if args.use_clahe is not None else default_use_clahe
    use_bilateral = args.use_bilateral if args.use_bilateral is not None else default_use_bilateral

    # Setup preprocessor based on resolved switches
    preprocessor = None
    if use_clahe or use_bilateral:
        print(f"Enabling Preprocessor for DataLoader (CLAHE={use_clahe}, Bilateral={use_bilateral})...")
        preprocessor = DefectPreprocessor.from_yaml(
            use_clahe=use_clahe, use_bilateral=use_bilateral, target_size=target_size
        )

    # Dataloaders: checkpoint selection uses the validation split; the test
    # split is evaluated once at the end.
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        batch_size=batch_size,
        num_workers=args.num_workers,
        target_size=target_size,
        augment_train=not args.no_augment,
        split_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
    )

    _results = trainer.fit(epochs=epochs, dry_run=args.dry_run)

    if not args.dry_run:
        # Load best checkpoint weights saved during training for final test evaluation
        best_ckpt_path = Path("checkpoints") / f"{model.variant_name}_best.pt"
        if best_ckpt_path.exists():
            print(f"Loading best checkpoint from {best_ckpt_path} for final test evaluation...")
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Single final evaluation on the held-out test split
        metrics = evaluate_model_on_loader(model, test_loader, device)
        print(
            f"[TEST - Best Model] mAP@0.5={metrics['mAP@0.5']:.4f} "
            f"precision={metrics['mean_precision']:.4f} "
            f"recall={metrics['mean_recall']:.4f} "
            f"f1={metrics['mean_f1']:.4f}"
        )

    print(f"[SUCCESS] Finished training run for {args.model}.")

if __name__ == "__main__":
    main()
