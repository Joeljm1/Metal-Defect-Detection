"""
Train Full Ablation Matrix (M1, M2, M3, M4).

Trains each variant on the training split, selects the best checkpoint on the
held-out validation split, and evaluates the test split using the shared
evaluation policy (full-curve mAP@0.5, P/R/F1 at conf 0.25).
Supports multi-seed runs (e.g. seeds 42, 43, 44) to quantify variance and spread (mean +/- std).

Hyperparameters (epochs / batch size / learning rate) are read from each
variant's ``training:`` block in configs/, with explicit arguments overriding.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dataset.loader import create_dataloaders
from src.evaluation.metrics import (
    EVAL_CONF_THRES,
    REPORT_CONF_THRES,
    evaluate_model_on_loader,
)
from src.models.detector import DefectDetector, load_variant_config
from src.preprocessing.pipeline import DefectPreprocessor
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ablation_matrix(
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    seeds: list[int] | None = None,
    data_dir: str = "data/NEU-DET",
    output_dir: str = "checkpoints",
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    active_seeds = seeds if seeds is not None and len(seeds) > 0 else [42]

    print(f"=== Task 5 Convergence Training on {device} ===")
    print(f"Active Seeds: {active_seeds}")
    print(
        f"Eval policy: checkpoint selection on validation split | "
        f"mAP full curve from conf {EVAL_CONF_THRES} | P/R/F1 at conf {REPORT_CONF_THRES}"
    )

    models = ["M1", "M2", "M3", "M4"]
    descriptions = {
        "M1": "Baseline (Plain YOLOv5s)",
        "M2": "Baseline + CLAHE & Bilateral",
        "M3": "Baseline + ECA/Spatial Attention",
        "M4": "Proposed Integrated Model",
    }

    summary: dict[str, Any] = {}

    for var in models:
        t_cfg = load_variant_config(var).get("training", {})
        v_epochs = epochs if epochs is not None else int(t_cfg.get("epochs", 50))
        v_batch = batch_size if batch_size is not None else int(t_cfg.get("batch_size", 16))
        v_lr = lr if lr is not None else float(t_cfg.get("learning_rate", 1e-3))
        v_weight_decay = float(t_cfg.get("weight_decay", 5e-4))

        print("\n=======================================================")
        print(f"[{var}] Training {descriptions[var]} across seeds {active_seeds}")
        print(f"      (epochs={v_epochs}, batch={v_batch}, lr={v_lr}, weight_decay={v_weight_decay})")
        print("=======================================================")

        runs = []
        best_primary_loss = float("inf")
        primary_fit_res = None
        primary_test_metrics = None

        preprocessor = None
        if var in ("M2", "M4"):
            preprocessor = DefectPreprocessor.from_yaml(use_clahe=True, use_bilateral=True)

        for s_idx, seed in enumerate(active_seeds):
            print(f"\n--- [{var}] Seed {seed} ({s_idx + 1}/{len(active_seeds)}) ---")
            set_seed(seed)

            model = DefectDetector.build_model(var, num_classes=6)

            train_loader, val_loader, test_loader = create_dataloaders(
                data_dir=data_dir,
                preprocessor=preprocessor,
                batch_size=v_batch,
                num_workers=4,
                augment_train=True,
                split_seed=seed,
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                learning_rate=v_lr,
                weight_decay=v_weight_decay,
                output_dir=output_dir,
            )

            fit_res = trainer.fit(epochs=v_epochs)
            best_ckpt = Path(output_dir) / f"{model.variant_name}_best.pt"

            # Load the best validation checkpoint before the single test evaluation
            if best_ckpt.exists():
                ckpt_data = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(ckpt_data["model_state"])

            val_metrics = evaluate_model_on_loader(model, val_loader, device)
            test_metrics = evaluate_model_on_loader(model, test_loader, device)

            print(
                f"[{var} seed={seed}] best_val_loss={trainer.best_loss:.4f} | "
                f"val mAP@0.5={val_metrics['mAP@0.5']:.4f} | "
                f"test mAP@0.5={test_metrics['mAP@0.5']:.4f} | "
                f"P={test_metrics['mean_precision']:.4f} | R={test_metrics['mean_recall']:.4f}"
            )

            run_record = {
                "seed": seed,
                "best_val_loss": round(trainer.best_loss, 4),
                "final_train_loss": round(fit_res["history"]["train_loss"][-1], 4),
                "val_mAP@0.5": round(val_metrics["mAP@0.5"], 4),
                "mAP@0.5": round(test_metrics["mAP@0.5"], 4),
                "precision": round(test_metrics["mean_precision"], 4),
                "recall": round(test_metrics["mean_recall"], 4),
                "f1": round(test_metrics["mean_f1"], 4),
                "per_class": {k: round(v["ap50"], 4) for k, v in test_metrics["per_class"].items()},
                "elapsed_seconds": round(fit_res["elapsed_seconds"], 2),
            }
            runs.append(run_record)

            # Keep best performing seed as primary for checkpoint and loss curves
            if s_idx == 0 or trainer.best_loss < best_primary_loss:
                best_primary_loss = trainer.best_loss
                primary_fit_res = fit_res
                primary_test_metrics = test_metrics

                # Keep primary checkpoint as canonical <variant_name>_best.pt
                torch.save({
                    "epoch": v_epochs,
                    "model_state": model.state_dict(),
                    "val_loss": trainer.best_loss,
                    "variant": model.variant_name,
                    "seed": seed,
                }, best_ckpt)

        # Aggregate statistics across seeds
        maps = [r["mAP@0.5"] for r in runs]
        precs = [r["precision"] for r in runs]
        recs = [r["recall"] for r in runs]
        f1s = [r["f1"] for r in runs]
        val_losses = [r["best_val_loss"] for r in runs]

        mean_map = float(np.mean(maps))
        std_map = float(np.std(maps)) if len(maps) > 1 else 0.0
        mean_prec = float(np.mean(precs))
        std_prec = float(np.std(precs)) if len(precs) > 1 else 0.0
        mean_rec = float(np.mean(recs))
        std_rec = float(np.std(recs)) if len(recs) > 1 else 0.0
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s)) if len(f1s) > 1 else 0.0

        summary[var] = {
            "name": descriptions[var],
            "epochs": v_epochs,
            "batch_size": v_batch,
            "learning_rate": v_lr,
            "seeds": active_seeds,
            "num_seeds": len(active_seeds),
            "best_val_loss": round(float(np.mean(val_losses)), 4),
            "final_train_loss": round(runs[0]["final_train_loss"], 4),
            "val_mAP@0.5": round(float(np.mean([r["val_mAP@0.5"] for r in runs])), 4),
            "mAP@0.5": round(mean_map, 4),
            "mAP@0.5_std": round(std_map, 4),
            "precision": round(mean_prec, 4),
            "precision_std": round(std_prec, 4),
            "recall": round(mean_rec, 4),
            "recall_std": round(std_rec, 4),
            "f1": round(mean_f1, 4),
            "f1_std": round(std_f1, 4),
            "per_class": {k: round(v["ap50"], 4) for k, v in primary_test_metrics["per_class"].items()} if (primary_test_metrics and "per_class" in primary_test_metrics) else runs[0]["per_class"],
            "eval_policy": {
                "checkpoint_selection": "validation split (10% of train)",
                "final_evaluation": "held-out test split",
                "nms_conf_thres": EVAL_CONF_THRES,
                "report_conf_thres": REPORT_CONF_THRES,
                "iou_match_threshold": 0.5,
            },
            "elapsed_seconds": round(sum(r["elapsed_seconds"] for r in runs), 2),
            "history": primary_fit_res["history"] if primary_fit_res else {},
            "runs": runs,
        }

    out_path = Path("reports/training_ablation_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Saved ablation convergence summary with {len(active_seeds)} seed(s) to {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Full Ablation Matrix (M1-M4) with Multi-Seed Support")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs per model (default: from config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: from config)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds (default: 42)")
    parser.add_argument("--data-dir", type=str, default="data/NEU-DET", help="Path to NEU-DET data directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()

    train_ablation_matrix(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seeds=args.seeds,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
