"""
Generate Convergence Training Loss Curves for Task 5.
Plots train and validation loss across M1, M2, M3, and M4.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curves(
    json_path: Path = Path("reports/training_ablation_summary.json"),
    output_path: Path = Path("reports/figures/ablation_loss_curves.png"),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor("#fafbfc")

    colors = {
        "M1": "#2563eb",  # Blue
        "M2": "#059669",  # Green
        "M3": "#d97706",  # Orange
        "M4": "#dc2626",  # Red
    }

    labels = {
        "M1": "M1 (Baseline Plain YOLOv5s)",
        "M2": "M2 (Baseline + CLAHE & Bilateral)",
        "M3": "M3 (Baseline + ECA/Spatial Attention)",
        "M4": "M4 (Proposed Integrated Model)",
    }

    max_epochs = 0
    for var, info in data.items():
        epochs = list(range(1, len(info["history"]["train_loss"]) + 1))
        max_epochs = max(max_epochs, len(epochs))
        ax1.plot(
            epochs,
            info["history"]["train_loss"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=labels.get(var, var),
            color=colors.get(var, "#6b7280"),
        )
        ax2.plot(
            epochs,
            info["history"]["val_loss"],
            marker="s",
            markersize=3,
            linewidth=2,
            label=labels.get(var, var),
            color=colors.get(var, "#6b7280"),
        )

    ax1.set_title(f"Training Loss Convergence ({max_epochs} Epochs)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Total Loss (CIoU + BCE Obj + BCE Cls)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)

    ax2.set_title("Validation Loss Trajectory", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Validation Loss", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[DONE] Saved loss curves to {output_path}")


if __name__ == "__main__":
    plot_loss_curves()
