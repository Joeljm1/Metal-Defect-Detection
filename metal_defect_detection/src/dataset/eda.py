"""
Exploratory Data Analysis (EDA) for NEU-DET Surface Defect Dataset.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset.parser import CLASS_NAMES, parse_yolo_label_file


def run_dataset_eda(dataset_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    """
    Analyzes NEU-DET dataset statistics and generates distribution plots.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    splits = ["train", "test"]

    for split in splits:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for lbl_file in lbl_dir.glob("*.txt"):
            boxes = parse_yolo_label_file(lbl_file)
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
                candidate = img_dir / f"{lbl_file.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            img_name = img_path.name if img_path is not None else f"{lbl_file.stem}.jpg"

            if boxes.numel() == 0:
                records.append({
                    "split": split,
                    "image_name": img_name,
                    "has_defects": False,
                    "class_id": None,
                    "class_name": "background",
                    "xc": None, "yc": None, "w": None, "h": None,
                    "area": None, "aspect_ratio": None
                })
            else:
                for b in boxes.numpy():
                    cls_id = int(b[0])
                    xc, yc, w, h = b[1:5]
                    records.append({
                        "split": split,
                        "image_name": img_name,
                        "has_defects": True,
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                        "xc": xc, "yc": yc, "w": w, "h": h,
                        "area": w * h,
                        "aspect_ratio": w / (h + 1e-6)
                    })

    df = pd.DataFrame(records)

    # Compute Summary Stats
    total_images = df["image_name"].nunique()
    defect_records = df[df["has_defects"] == True]
    total_boxes = len(defect_records)
    
    class_counts = defect_records["class_name"].value_counts().to_dict()
    split_counts = df.groupby("split")["image_name"].nunique().to_dict()
    boxes_per_split = defect_records.groupby("split").size().to_dict()

    summary = {
        "dataset_name": "NEU-DET",
        "total_images": total_images,
        "total_bounding_boxes": total_boxes,
        "boxes_per_image_mean": float(total_boxes / max(1, total_images)),
        "images_per_split": split_counts,
        "boxes_per_split": boxes_per_split,
        "defect_class_distribution": class_counts,
        "bbox_metrics": {
            "mean_normalized_width": float(defect_records["w"].mean()),
            "std_normalized_width": float(defect_records["w"].std()),
            "mean_normalized_height": float(defect_records["h"].mean()),
            "std_normalized_height": float(defect_records["h"].std()),
            "mean_area": float(defect_records["area"].mean()),
            "median_area": float(defect_records["area"].median()),
        }
    }

    # Save summary json
    json_path = output_dir / "dataset_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot Visualizations
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Class distribution bar chart
    sns.countplot(
        data=defect_records,
        x="class_name",
        order=CLASS_NAMES,
        palette="viridis",
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("NEU-DET Defect Instance Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Defect Class", fontsize=10)
    axes[0, 0].set_ylabel("Bounding Box Count", fontsize=10)
    axes[0, 0].tick_params(axis="x", rotation=30)

    # 2. Box Area distribution
    sns.histplot(
        defect_records["area"],
        bins=30,
        kde=True,
        color="crimson",
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("Normalized Bounding Box Area Distribution (W * H)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Normalized Area", fontsize=10)
    axes[0, 1].set_ylabel("Count", fontsize=10)

    # 3. Width vs Height Scatter / KDE
    sns.scatterplot(
        data=defect_records,
        x="w",
        y="h",
        hue="class_name",
        alpha=0.6,
        palette="tab10",
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Bounding Box Dimensions (Width vs Height)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Normalized Width", fontsize=10)
    axes[1, 0].set_ylabel("Normalized Height", fontsize=10)
    axes[1, 0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # 4. Box distribution across splits
    split_class_df = defect_records.groupby(["split", "class_name"]).size().reset_index(name="count")
    sns.barplot(
        data=split_class_df,
        x="class_name",
        y="count",
        hue="split",
        palette="Set2",
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Defect Distribution across Train / Test Splits", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Defect Class", fontsize=10)
    axes[1, 1].set_ylabel("Instance Count", fontsize=10)
    axes[1, 1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plot_path = output_dir / "dataset_eda_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    summary["plot_saved_to"] = str(plot_path)
    return summary
