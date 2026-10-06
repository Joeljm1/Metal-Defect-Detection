"""
Evaluate Cross-Dataset Domain Adaptation on GC10-DET (Task 6 — Hypothesis H4).

Compares:
1. Zero-shot transferability: M1 (Plain YOLOv5s) vs M4 (Proposed Integrated Model).
2. Few-shot domain adaptation: Adapting detection heads to 10 GC10-DET defect classes.
3. Quantifies domain gap Delta mAP and transfer efficiency.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.loader import yolo_collate_fn
from src.evaluation.domain_adaptation import (
    GC10_CLASSES,
    DomainAdaptedDetector,
    GC10Dataset,
    evaluate_zero_shot_domain_transfer,
    generate_synthetic_gc10_benchmark,
    train_few_shot_adaptation,
)
from src.models.detector import DefectDetector
from src.preprocessing.pipeline import DefectPreprocessor


def run_domain_adaptation_benchmark(
    data_dir: Path | str = Path("data/GC10-DET"),
    output_summary_path: Path | str = Path("reports/domain_adaptation_summary.json"),
    output_figure_path: Path | str = Path("reports/figures/domain_adaptation_comparison.png"),
    few_shot_epochs: int = 10,
) -> dict[str, Any]:
    output_summary_path = Path(output_summary_path)
    output_figure_path = Path(output_figure_path)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running GC10-DET Cross-Dataset Domain Adaptation on {device} ===")

    # Ensure GC10-DET benchmark data is available
    gc10_dir = generate_synthetic_gc10_benchmark(data_dir)

    # Load baseline M1 and proposed M4 checkpoints
    m1 = DefectDetector.build_model("M1").to(device).eval()
    m4 = DefectDetector.build_model("M4").to(device).eval()

    ckpt_m1 = Path("checkpoints/M1_Baseline_best.pt")
    ckpt_m4 = Path("checkpoints/M4_Proposed_Integrated_best.pt")

    if ckpt_m1.exists():
        m1.load_state_dict(torch.load(ckpt_m1, map_location=device)["model_state"])
        print(f"Loaded checkpoint: {ckpt_m1}")
    if ckpt_m4.exists():
        m4.load_state_dict(torch.load(ckpt_m4, map_location=device)["model_state"])
        print(f"Loaded checkpoint: {ckpt_m4}")

    # Create GC10 DataLoaders
    prep_m4 = DefectPreprocessor.from_yaml(use_clahe=True, use_bilateral=True)

    gc10_test_m1 = DataLoader(
        GC10Dataset(gc10_dir / "test" / "images", gc10_dir / "test" / "labels", preprocessor=None),
        batch_size=16,
        shuffle=False,
        collate_fn=yolo_collate_fn,
    )
    gc10_test_m4 = DataLoader(
        GC10Dataset(gc10_dir / "test" / "images", gc10_dir / "test" / "labels", preprocessor=prep_m4),
        batch_size=16,
        shuffle=False,
        collate_fn=yolo_collate_fn,
    )

    gc10_train_m1 = DataLoader(
        GC10Dataset(gc10_dir / "train" / "images", gc10_dir / "train" / "labels", preprocessor=None, is_training=True),
        batch_size=16,
        shuffle=True,
        collate_fn=yolo_collate_fn,
    )
    gc10_train_m4 = DataLoader(
        GC10Dataset(gc10_dir / "train" / "images", gc10_dir / "train" / "labels", preprocessor=prep_m4, is_training=True),
        batch_size=16,
        shuffle=True,
        collate_fn=yolo_collate_fn,
    )

    # 1. Zero-shot Evaluation on Analogue Classes
    print("\n[Phase 1] Evaluating Zero-Shot Cross-Domain Transfer...")
    m1_zero_shot = evaluate_zero_shot_domain_transfer(m1, gc10_test_m1, device=device)
    m4_zero_shot = evaluate_zero_shot_domain_transfer(m4, gc10_test_m4, device=device)

    print(f"  M1 Zero-Shot mAP@0.5: {m1_zero_shot['mAP@0.5']:.4f}")
    print(f"  M4 Zero-Shot mAP@0.5: {m4_zero_shot['mAP@0.5']:.4f}")

    # 2. Few-shot Adaptation (10 epochs on 10 GC10-DET classes)
    print("\n[Phase 2] Training Few-Shot Domain Adaptation Heads (10 classes)...")
    adapted_m1 = DomainAdaptedDetector(m1, num_target_classes=len(GC10_CLASSES), freeze_backbone=True)
    adapted_m4 = DomainAdaptedDetector(m4, num_target_classes=len(GC10_CLASSES), freeze_backbone=True)

    m1_adapt = train_few_shot_adaptation(
        adapted_m1, gc10_train_m1, gc10_test_m1, epochs=few_shot_epochs, device=device
    )
    m4_adapt = train_few_shot_adaptation(
        adapted_m4, gc10_train_m4, gc10_test_m4, epochs=few_shot_epochs, device=device
    )

    print(f"  M1 Adapted mAP@0.5: {m1_adapt['mAP@0.5']:.4f}")
    print(f"  M4 Adapted mAP@0.5: {m4_adapt['mAP@0.5']:.4f}")

    # Source domain (NEU-DET) baseline performance: dynamically load from summary or evaluate directly
    neu_m1_map = None
    neu_m4_map = None
    ablation_summary_path = Path("reports/training_ablation_summary.json")
    if ablation_summary_path.exists():
        try:
            with open(ablation_summary_path, "r", encoding="utf-8") as f:
                ab_data = json.load(f)
                if "M1" in ab_data and "mAP@0.5" in ab_data["M1"]:
                    neu_m1_map = float(ab_data["M1"]["mAP@0.5"])
                if "M4" in ab_data and "mAP@0.5" in ab_data["M4"]:
                    neu_m4_map = float(ab_data["M4"]["mAP@0.5"])
        except (OSError, json.JSONDecodeError):
            pass

    # If not found in summary, compute live on NEU-DET test split
    if neu_m1_map is None or neu_m4_map is None:
        print("\nEvaluating source domain baseline directly on NEU-DET test set...")
        from src.dataset.loader import create_dataloaders
        from src.evaluation.metrics import evaluate_model_on_loader
        _, _, neu_test_m1 = create_dataloaders(data_dir="data/NEU-DET", batch_size=16, num_workers=2)
        _, _, neu_test_m4 = create_dataloaders(data_dir="data/NEU-DET", preprocessor=prep_m4, batch_size=16, num_workers=2)
        if neu_m1_map is None:
            m1_neu = evaluate_model_on_loader(m1, neu_test_m1, device=device)
            neu_m1_map = m1_neu["mAP@0.5"]
        if neu_m4_map is None:
            m4_neu = evaluate_model_on_loader(m4, neu_test_m4, device=device)
            neu_m4_map = m4_neu["mAP@0.5"]

    summary = {
        "dataset": "GC10-DET-Synthetic",
        "is_synthetic": True,
        "note": "Synthetic metal defect benchmark simulating cross-domain steel inspection with directional texture and defect patterns.",
        "class_mapping_policy": "Heuristic morphological proxy (NEU 6 classes to GC10 10 classes)",
        "num_classes": len(GC10_CLASSES),
        "zero_shot_evaluation": {
            "M1_Baseline": {
                "mAP@0.5": round(m1_zero_shot["mAP@0.5"], 4),
                "precision": round(m1_zero_shot["mean_precision"], 4),
                "recall": round(m1_zero_shot["mean_recall"], 4),
                "domain_gap_drop": round(neu_m1_map - m1_zero_shot["mAP@0.5"], 4),
            },
            "M4_Proposed_Integrated": {
                "mAP@0.5": round(m4_zero_shot["mAP@0.5"], 4),
                "precision": round(m4_zero_shot["mean_precision"], 4),
                "recall": round(m4_zero_shot["mean_recall"], 4),
                "domain_gap_drop": round(neu_m4_map - m4_zero_shot["mAP@0.5"], 4),
                "relative_gain_over_m1": round(
                    ((m4_zero_shot["mAP@0.5"] - m1_zero_shot["mAP@0.5"]) / max(1e-4, m1_zero_shot["mAP@0.5"])) * 100, 2
                ),
            },
        },
        "few_shot_adaptation": {
            "epochs": few_shot_epochs,
            "M1_Baseline": {
                "mAP@0.5": round(m1_adapt["mAP@0.5"], 4),
                "precision": round(m1_adapt["precision"], 4),
                "recall": round(m1_adapt["recall"], 4),
                "per_class": {k: round(v["ap50"], 4) for k, v in m1_adapt["per_class"].items()},
            },
            "M4_Proposed_Integrated": {
                "mAP@0.5": round(m4_adapt["mAP@0.5"], 4),
                "precision": round(m4_adapt["precision"], 4),
                "recall": round(m4_adapt["recall"], 4),
                "per_class": {k: round(v["ap50"], 4) for k, v in m4_adapt["per_class"].items()},
                "relative_gain_over_m1": round(
                    ((m4_adapt["mAP@0.5"] - m1_adapt["mAP@0.5"]) / max(1e-4, m1_adapt["mAP@0.5"])) * 100, 2
                ),
            },
        },
        "hypothesis_h4_confirmed": bool(m4_adapt["mAP@0.5"] > m1_adapt["mAP@0.5"]),
    }

    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved domain adaptation summary to {output_summary_path}")

    # Generate Publication Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor("#fafbfc")

    # Plot 1: Source Domain (NEU-DET) vs Zero-Shot Target (GC10-DET)
    models = ["M1 (Baseline)", "M4 (Proposed)"]
    neu_scores = [neu_m1_map * 100, neu_m4_map * 100]
    zero_scores = [m1_zero_shot["mAP@0.5"] * 100, m4_zero_shot["mAP@0.5"] * 100]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width / 2, neu_scores, width, label="Source Domain (NEU-DET)", color="#2563eb", alpha=0.85)
    ax1.bar(x + width / 2, zero_scores, width, label="Zero-Shot Target (GC10-DET)", color="#dc2626", alpha=0.85)
    ax1.set_ylabel("mAP@0.5 (%)", fontweight="bold", fontsize=11)
    ax1.set_title("Cross-Domain Transfer Gap (Hypothesis H4)", fontweight="bold", fontsize=12, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 50)
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    for i in range(len(models)):
        ax1.text(x[i] - width / 2, neu_scores[i] + 1.0, f"{neu_scores[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax1.text(x[i] + width / 2, zero_scores[i] + 1.0, f"{zero_scores[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Plot 2: Few-Shot Adaptation Learning Curves
    epochs_range = list(range(1, few_shot_epochs + 1))
    ax2.plot(epochs_range, m1_adapt["loss_history"], marker="o", linewidth=2, color="#2563eb", label=f"M1 Loss ({m1_adapt['mAP@0.5']*100:.1f}% mAP)")
    ax2.plot(epochs_range, m4_adapt["loss_history"], marker="s", linewidth=2, color="#dc2626", label=f"M4 Loss ({m4_adapt['mAP@0.5']*100:.1f}% mAP)")
    ax2.set_xlabel("Adaptation Epoch", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Adaptation Multi-Task Loss", fontweight="bold", fontsize=11)
    ax2.set_title("Few-Shot GC10-DET Fine-Tuning Convergence", fontweight="bold", fontsize=12, pad=10)
    ax2.legend(frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_figure_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved domain adaptation figure to {output_figure_path}")

    return summary


if __name__ == "__main__":
    run_domain_adaptation_benchmark()
