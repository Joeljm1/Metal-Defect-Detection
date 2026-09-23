"""
Generate Grad-CAM Visual Saliency Comparisons for Task 5.
Visualizes explainable heatmaps across all 6 defect classes comparing
Baseline (M1) vs Proposed Integrated Model with ECA Attention (M4).
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.models.detector import DefectDetector
from src.evaluation.gradcam import DefectGradCAM
from src.dataset.parser import CLASS_NAMES


def generate_gradcam_gallery(
    data_dir: Path = Path("data/NEU-DET/test/images"),
    output_path: Path = Path("reports/figures/gradcam_interpretability_gallery.png"),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained models
    m1 = DefectDetector.build_model("M1").to(device).eval()
    m4 = DefectDetector.build_model("M4").to(device).eval()

    ckpt1 = Path("checkpoints/M1_Baseline_best.pt")
    ckpt4 = Path("checkpoints/M4_Proposed_Integrated_best.pt")
    if ckpt1.exists():
        m1.load_state_dict(torch.load(ckpt1, map_location=device)["model_state"])
    if ckpt4.exists():
        m4.load_state_dict(torch.load(ckpt4, map_location=device)["model_state"])

    cam_m1 = DefectGradCAM(m1)
    cam_m4 = DefectGradCAM(m4)

    # Pick representative test image for each class
    selected_images = {}
    for img_p in sorted(data_dir.iterdir()):
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            if cls_name.lower().replace("-", "_") in img_p.stem.lower().replace("-", "_"):
                if cls_name not in selected_images:
                    selected_images[cls_name] = (cls_idx, img_p)

    fig, axes = plt.subplots(6, 4, figsize=(14, 18), dpi=150)
    fig.patch.set_facecolor("#fafbfc")

    col_titles = [
        "Raw Steel Surface",
        "M4 Preprocessed (CLAHE+Bilateral)",
        "M1 Baseline Grad-CAM",
        "M4 Integrated (ECA) Grad-CAM",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=12)

    for row, cls_name in enumerate(CLASS_NAMES):
        if cls_name not in selected_images:
            continue
        cls_idx, img_path = selected_images[cls_name]

        # Read image
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (200, 200))

        # Preprocessed version
        prep_rgb = m4.preprocessor.process(rgb_resized)

        # To tensor
        t_m1 = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        t_m4 = torch.from_numpy(prep_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        # Generate CAMs
        heatmap_m1 = cam_m1.generate_cam(t_m1, class_idx=cls_idx)
        heatmap_m4 = cam_m4.generate_cam(t_m4, class_idx=cls_idx)

        overlay_m1 = cam_m1.overlay_cam(rgb_resized, heatmap_m1, alpha=0.55)
        overlay_m4 = cam_m4.overlay_cam(prep_rgb, heatmap_m4, alpha=0.55)

        # Plot row
        axes[row, 0].imshow(rgb_resized)
        axes[row, 0].set_ylabel(cls_name.replace("_", " ").title(), fontsize=11, fontweight="bold")
        axes[row, 1].imshow(prep_rgb)
        axes[row, 2].imshow(overlay_m1)
        axes[row, 3].imshow(overlay_m4)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    cam_m1.remove_hooks()
    cam_m4.remove_hooks()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[DONE] Saved Grad-CAM gallery to {output_path}")


if __name__ == "__main__":
    generate_gradcam_gallery()
