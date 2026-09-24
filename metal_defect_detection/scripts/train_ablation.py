"""
Train Full Ablation Matrix (M1, M2, M3, M4).

Trains each variant on the training split, selects the best checkpoint on the
held-out validation split, and evaluates the test split exactly once using the
shared evaluation policy (full-curve mAP@0.5, P/R/F1 at conf 0.25).

Hyperparameters (epochs / batch size / learning rate) are read from each
variant's ``training:`` block in configs/, with explicit arguments overriding.
"""

from pathlib import Path
import json
import torch

from src.models.detector import DefectDetector, load_variant_config
from src.dataset.loader import create_dataloaders
from src.preprocessing.pipeline import DefectPreprocessor
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    EVAL_CONF_THRES,
    REPORT_CONF_THRES,
    evaluate_model_on_loader,
)


def train_ablation_matrix(
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    data_dir: str = "data/NEU-DET",
    output_dir: str = "checkpoints",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Task 5 Convergence Training on {device} ===")
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

    summary = {}
    for var in models:
        t_cfg = load_variant_config(var).get("training", {})
        v_epochs = epochs if epochs is not None else int(t_cfg.get("epochs", 50))
        v_batch = batch_size if batch_size is not None else int(t_cfg.get("batch_size", 16))
        v_lr = lr if lr is not None else float(t_cfg.get("learning_rate", 1e-3))

        print(f"\n[{var}] Training {descriptions[var]} (epochs={v_epochs}, batch={v_batch}, lr={v_lr})...")
        model = DefectDetector.build_model(var, num_classes=6)

        preprocessor = None
        if var in ("M2", "M4"):
            preprocessor = DefectPreprocessor(use_clahe=True, use_bilateral=True)

        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            preprocessor=preprocessor,
            batch_size=v_batch,
            num_workers=4,
            augment_train=True,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=v_lr,
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
            f"[{var}] best_val_loss={trainer.best_loss:.4f} | "
            f"val mAP@0.5={val_metrics['mAP@0.5']:.4f} | "
            f"test mAP@0.5={test_metrics['mAP@0.5']:.4f}"
        )

        summary[var] = {
            "name": descriptions[var],
            "epochs": v_epochs,
            "batch_size": v_batch,
            "learning_rate": v_lr,
            "best_val_loss": round(trainer.best_loss, 4),
            "final_train_loss": round(fit_res["history"]["train_loss"][-1], 4),
            "val_mAP@0.5": round(val_metrics["mAP@0.5"], 4),
            "mAP@0.5": round(test_metrics["mAP@0.5"], 4),
            "precision": round(test_metrics["mean_precision"], 4),
            "recall": round(test_metrics["mean_recall"], 4),
            "f1": round(test_metrics["mean_f1"], 4),
            "per_class": {k: round(v["ap50"], 4) for k, v in test_metrics["per_class"].items()},
            "eval_policy": {
                "checkpoint_selection": "validation split (10% of train)",
                "final_evaluation": "held-out test split",
                "nms_conf_thres": EVAL_CONF_THRES,
                "report_conf_thres": REPORT_CONF_THRES,
                "iou_match_threshold": 0.5,
            },
            "elapsed_seconds": round(fit_res["elapsed_seconds"], 2),
            "history": fit_res["history"],
        }

    out_path = Path("reports/training_ablation_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Saved ablation convergence summary to {out_path}")
    return summary


if __name__ == "__main__":
    train_ablation_matrix()
