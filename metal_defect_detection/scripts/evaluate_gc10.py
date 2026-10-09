"""
Evaluate Cross-Dataset Domain Adaptation on the Real GC10-DET Dataset (Task 6 — Hypothesis H4).

Implements rigorous, honest domain transfer evaluation:
1. Real steel-strip images from GC10-DET (10 industrial defect classes).
2. Linear Probe evaluation: freezes source-trained backbone & neck representations,
   adapting only detection heads on N real images per class across multiple random seeds.
3. Evaluates all 4 ablation variants (M1: Plain YOLOv5s, M2: CLAHE+Bilateral, M3: Attention, M4: Proposed Integrated).
4. Quantifies transfer efficiency, mean +/- std, and domain transfer gap Delta mAP.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.convert_gc10 import convert_and_setup_gc10
from src.dataset.loader import yolo_collate_fn
from src.evaluation.domain_adaptation import (
    GC10_CLASSES,
    DomainAdaptedDetector,
    GC10Dataset,
    sample_few_shot_subset,
    train_few_shot_adaptation,
)
from src.models.detector import DefectDetector
from src.preprocessing.pipeline import DefectPreprocessor


def evaluate_variant_linear_probe(
    variant: str,
    ckpt_path: Path,
    gc10_dir: Path,
    seeds: list[int] = [42, 123, 456],
    shots_per_class: int = 10,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """
    Evaluates a model variant under linear probe adaptation across multiple random seeds.
    Freezes backbone and neck; adapts only detection head.
    """
    print(f"\n--- Evaluating Variant {variant} (Linear Probe, {len(seeds)} seeds, {shots_per_class}-shot) ---")

    train_img_dir = gc10_dir / "train" / "images"
    train_lbl_dir = gc10_dir / "train" / "labels"
    test_img_dir = gc10_dir / "test" / "images"
    test_lbl_dir = gc10_dir / "test" / "labels"

    # Determine preprocessor according to ablation matrix
    if variant in ["M2", "M4"]:
        preprocessor = DefectPreprocessor.from_yaml(use_clahe=True, use_bilateral=True)
    else:
        preprocessor = None

    test_dataset = GC10Dataset(
        test_img_dir,
        test_lbl_dir,
        target_size=(200, 200),
        preprocessor=preprocessor,
        is_training=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=yolo_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    seed_results = []

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"  [Seed {seed} ({seed_idx}/{len(seeds)})]")
        # Set seeds for reproducible few-shot sampling and head weight init
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Sample few-shot subset of real images
        train_paths = sample_few_shot_subset(
            train_img_dir,
            train_lbl_dir,
            shots_per_class=shots_per_class,
            seed=seed,
        )
        print(f"    Sampled {len(train_paths)} few-shot training images across 10 classes.")

        train_dataset = GC10Dataset(
            train_img_dir,
            train_lbl_dir,
            target_size=(200, 200),
            preprocessor=preprocessor,
            image_paths=train_paths,
            is_training=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(train_dataset)),
            shuffle=True,
            collate_fn=yolo_collate_fn,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        # Build base detector and load source-trained checkpoint
        base_model = DefectDetector.build_model(variant).to(device)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            base_model.load_state_dict(ckpt["model_state"])
        else:
            print(f"    Warning: Checkpoint {ckpt_path} not found, using initialized weights!")

        # Wrap in Linear Probe: freeze backbone and neck completely
        adapted_model = DomainAdaptedDetector(
            base_model=base_model,
            num_target_classes=len(GC10_CLASSES),
            freeze_backbone=True,
            freeze_neck=True,
        ).to(device)

        # Train linear probe head
        metrics = train_few_shot_adaptation(
            adapted_model,
            train_loader,
            test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
        )

        print(
            f"    Seed {seed} -> mAP@0.5: {metrics['mAP@0.5']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}"
        )
        seed_results.append({
            "seed": seed,
            "mAP@0.5": metrics["mAP@0.5"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "final_train_loss": metrics["final_train_loss"],
            "loss_history": metrics["loss_history"],
            "per_class": {k: v["ap50"] for k, v in metrics["per_class"].items()},
        })

    # Aggregate mean +/- std across seeds
    maps = [r["mAP@0.5"] for r in seed_results]
    precs = [r["precision"] for r in seed_results]
    recs = [r["recall"] for r in seed_results]

    per_class_summary = {}
    for cls_name in GC10_CLASSES:
        cls_aps = [r["per_class"].get(cls_name, 0.0) for r in seed_results]
        per_class_summary[cls_name] = {
            "mean_ap50": float(np.mean(cls_aps)),
            "std_ap50": float(np.std(cls_aps)),
        }

    return {
        "variant": variant,
        "mean_mAP": float(np.mean(maps)),
        "std_mAP": float(np.std(maps)),
        "mean_precision": float(np.mean(precs)),
        "std_precision": float(np.std(precs)),
        "mean_recall": float(np.mean(recs)),
        "std_recall": float(np.std(recs)),
        "seeds": seed_results,
        "per_class": per_class_summary,
    }


def run_domain_adaptation_benchmark(
    data_dir: Path | str = Path("data/GC10-DET"),
    output_summary_path: Path | str = Path("reports/domain_adaptation_summary.json"),
    output_figure_path: Path | str = Path("reports/figures/domain_adaptation_comparison.png"),
    seeds: list[int] = [42, 123, 456],
    shots_per_class: int = 10,
    few_shot_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_summary_path = Path(output_summary_path)
    output_figure_path = Path(output_figure_path)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Real GC10-DET Domain Adaptation Benchmark (Linear Probe) on {device} ===")

    # Ensure dataset exists and is populated
    test_imgs = list((data_dir / "test" / "images").glob("*.*")) if (data_dir / "test" / "images").exists() else []
    if len(test_imgs) < 100:
        print("Real GC10-DET not yet structured in data/GC10-DET. Running conversion...")
        convert_and_setup_gc10(raw_dir="data/GC10-DET-real/data", output_dir=data_dir)

    test_imgs = list((data_dir / "test" / "images").glob("*.*"))
    train_imgs = list((data_dir / "train" / "images").glob("*.*"))
    print(f"Dataset verification: {len(train_imgs)} train images, {len(test_imgs)} test images.")

    # Checkpoints for ablation matrix
    checkpoints = {
        "M1": Path("checkpoints/M1_Baseline_best.pt"),
        "M2": Path("checkpoints/M2_Baseline_Preprocessing_best.pt"),
        "M3": Path("checkpoints/M3_Baseline_Attention_best.pt"),
        "M4": Path("checkpoints/M4_Proposed_Integrated_best.pt"),
    }

    # Load source domain NEU-DET performance
    source_map = {}
    ablation_summary_path = Path("reports/training_ablation_summary.json")
    if ablation_summary_path.exists():
        try:
            with open(ablation_summary_path, "r", encoding="utf-8") as f:
                ab_data = json.load(f)
                for var in ["M1", "M2", "M3", "M4"]:
                    if var in ab_data and "mAP@0.5" in ab_data[var]:
                        source_map[var] = float(ab_data[var]["mAP@0.5"])
        except (OSError, json.JSONDecodeError):
            pass

    # Evaluate M1 through M4
    results = {}
    for var in ["M1", "M2", "M3", "M4"]:
        res = evaluate_variant_linear_probe(
            variant=var,
            ckpt_path=checkpoints[var],
            gc10_dir=data_dir,
            seeds=seeds,
            shots_per_class=shots_per_class,
            epochs=few_shot_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        src_score = source_map.get(var, 0.0)
        res["source_neu_mAP"] = src_score
        res["domain_gap_drop"] = float(src_score - res["mean_mAP"])
        results[var] = res

    # Benchmark summary dictionary
    summary = {
        "dataset": "GC10-DET",
        "dataset_type": "Real Industrial Steel Strip Defect Dataset (Lv et al., 2020)",
        "num_classes": len(GC10_CLASSES),
        "classes": GC10_CLASSES,
        "protocol": "Linear Probe (Backbone & Neck frozen, Detection Head trained)",
        "shots_per_class": shots_per_class,
        "epochs": few_shot_epochs,
        "seeds": seeds,
        "models": {
            var: {
                "mean_mAP@0.5": round(results[var]["mean_mAP"], 4),
                "std_mAP@0.5": round(results[var]["std_mAP"], 4),
                "mean_precision": round(results[var]["mean_precision"], 4),
                "std_precision": round(results[var]["std_precision"], 4),
                "mean_recall": round(results[var]["mean_recall"], 4),
                "std_recall": round(results[var]["std_recall"], 4),
                "source_neu_mAP": round(results[var]["source_neu_mAP"], 4),
                "domain_gap_drop": round(results[var]["domain_gap_drop"], 4),
                "per_class": {
                    cls: {
                        "mean_ap50": round(results[var]["per_class"][cls]["mean_ap50"], 4),
                        "std_ap50": round(results[var]["per_class"][cls]["std_ap50"], 4),
                    }
                    for cls in GC10_CLASSES
                },
                "seed_runs": [
                    {
                        "seed": s["seed"],
                        "mAP@0.5": round(s["mAP@0.5"], 4),
                        "precision": round(s["precision"], 4),
                        "recall": round(s["recall"], 4),
                    }
                    for s in results[var]["seeds"]
                ],
            }
            for var in ["M1", "M2", "M3", "M4"]
        },
        "hypothesis_h4_outcome": {
            "m1_mean_map": round(results["M1"]["mean_mAP"], 4),
            "m4_mean_map": round(results["M4"]["mean_mAP"], 4),
            "delta_map_m4_vs_m1": round(results["M4"]["mean_mAP"] - results["M1"]["mean_mAP"], 4),
            "m4_outperformed_m1": bool(results["M4"]["mean_mAP"] > results["M1"]["mean_mAP"]),
            "transfer_efficiency_m4": round((results["M4"]["mean_mAP"] / max(1e-4, results["M4"]["source_neu_mAP"])) * 100, 2),
            "transfer_efficiency_m1": round((results["M1"]["mean_mAP"] / max(1e-4, results["M1"]["source_neu_mAP"])) * 100, 2),
        },
    }

    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved real domain adaptation summary to {output_summary_path}")

    # Generate Publication Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    variants = ["M1", "M2", "M3", "M4"]
    variant_names = ["M1 (Plain)", "M2 (Preproc)", "M3 (Attention)", "M4 (Integrated)"]
    mean_maps = [results[v]["mean_mAP"] * 100 for v in variants]
    std_maps = [results[v]["std_mAP"] * 100 for v in variants]
    colors = ["#64748b", "#0ea5e9", "#8b5cf6", "#10b981"]

    # Subplot 1: Linear Probe Transfer Performance with Std Error Bars
    bars = ax1.bar(
        variant_names,
        mean_maps,
        yerr=std_maps,
        capsize=5,
        color=colors,
        alpha=0.9,
        edgecolor="#1e293b",
        linewidth=1.2,
    )
    ax1.set_ylabel("Linear Probe mAP@0.5 (%)", fontweight="bold", fontsize=11)
    ax1.set_title(
        f"Cross-Domain Linear Probe Transfer on Real GC10-DET\n({shots_per_class}-shot per class, mean +/- std across {len(seeds)} seeds)",
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    max_y = max(mean_maps) if max(mean_maps) > 0 else 10
    ax1.set_ylim(0, max_y * 1.35)

    for bar, m, s in zip(bars, mean_maps, std_maps):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.3,
            f"{m:.1f} +/- {s:.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )

    # Subplot 2: Per-Class Comparison: M1 vs M4
    x = np.arange(len(GC10_CLASSES))
    width = 0.38
    m1_class_means = [results["M1"]["per_class"][c]["mean_ap50"] * 100 for c in GC10_CLASSES]
    m4_class_means = [results["M4"]["per_class"][c]["mean_ap50"] * 100 for c in GC10_CLASSES]

    ax2.bar(x - width / 2, m1_class_means, width, label="M1 (Baseline)", color="#64748b", alpha=0.85)
    ax2.bar(x + width / 2, m4_class_means, width, label="M4 (Integrated)", color="#10b981", alpha=0.85)
    ax2.set_ylabel("AP@0.5 (%)", fontweight="bold", fontsize=11)
    ax2.set_title("Per-Class Defect Transfer: M1 vs M4", fontweight="bold", fontsize=11, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(GC10_CLASSES, rotation=35, ha="right", fontsize=9, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.subplots_adjust(bottom=0.22, top=0.90, wspace=0.25)
    plt.savefig(output_figure_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved publication figure to {output_figure_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real GC10-DET Domain Adaptation Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/GC10-DET")
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_domain_adaptation_benchmark(
        data_dir=args.data_dir,
        shots_per_class=args.shots,
        few_shot_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
