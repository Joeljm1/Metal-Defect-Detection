"""
Train Full Ablation Matrix (M1, M2, M3, M4) for Task 5.
"""

from pathlib import Path
import json
import time
import torch

from src.models.detector import DefectDetector
from src.dataset.loader import create_dataloaders
from src.preprocessing.pipeline import DefectPreprocessor
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_detections
from src.utils.box_ops import non_max_suppression


def evaluate_model_map(model, val_loader, device, conf_thres=0.10, iou_thres=0.45):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            preds, _ = model(images)
            nms = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)
            h, w = images.shape[2:]
            for b in range(images.shape[0]):
                t = targets[targets[:, 0] == b]
                if t.numel():
                    xc, yc, bw, bh = t[:, 2] * w, t[:, 3] * h, t[:, 4] * w, t[:, 5] * h
                    boxes = torch.stack([t[:, 1], xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], dim=1)
                else:
                    boxes = torch.zeros((0, 5))
                all_preds.append(nms[b].cpu())
                all_targets.append(boxes)
    return evaluate_detections(all_preds, all_targets, iou_threshold=0.5, num_classes=6)


def train_ablation_matrix(epochs=15, batch_size=16, lr=1e-3, data_dir="data/NEU-DET", output_dir="checkpoints"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Task 5 Convergence Training on {device} (Epochs={epochs}) ===")

    models = ["M1", "M2", "M3", "M4"]
    descriptions = {
        "M1": "Baseline (Plain YOLOv5s)",
        "M2": "Baseline + CLAHE & Bilateral",
        "M3": "Baseline + ECA/Spatial Attention",
        "M4": "Proposed Integrated Model",
    }

    summary = {}

    for var in models:
        print(f"\n[{var}] Training {descriptions[var]}...")
        model = DefectDetector.build_model(var, num_classes=6)

        preprocessor = None
        if var in ("M2", "M4"):
            preprocessor = DefectPreprocessor(use_clahe=True, use_bilateral=True)

        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            preprocessor=preprocessor,
            batch_size=batch_size,
            num_workers=4,
            augment_train=True,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=lr,
            output_dir=output_dir,
        )

        fit_res = trainer.fit(epochs=epochs)
        best_ckpt = Path(output_dir) / f"{model.variant_name}_best.pt"

        # Evaluate best checkpoint
        if best_ckpt.exists():
            ckpt_data = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt_data["model_state"])

        metrics = evaluate_model_map(model, val_loader, device, conf_thres=0.10)
        print(f"[{var}] Best val_loss={trainer.best_loss:.4f}, mAP@0.5={metrics['mAP@0.5']:.4f}")

        summary[var] = {
            "name": descriptions[var],
            "best_val_loss": round(trainer.best_loss, 4),
            "final_train_loss": round(fit_res["history"]["train_loss"][-1], 4),
            "mAP@0.5": round(metrics["mAP@0.5"], 4),
            "precision": round(metrics["mean_precision"], 4),
            "recall": round(metrics["mean_recall"], 4),
            "f1": round(metrics["mean_f1"], 4),
            "per_class": {k: round(v["ap50"], 4) for k, v in metrics["per_class"].items()},
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
    train_ablation_matrix(epochs=15, batch_size=16)
