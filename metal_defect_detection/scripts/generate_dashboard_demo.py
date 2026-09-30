"""
Generate Dashboard Demonstration Figure for Task 7 Final Submission.
Creates a publication-quality multi-panel visualization of the plant operator inspection pipeline.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.detector import DefectDetector
from src.deployment.dashboard import load_inspection_model, run_defect_inspection
from src.dataset.parser import CLASS_NAMES


def generate_dashboard_figure(
    output_path: Path | str = Path("reports/figures/dashboard_inspection_demo.png"),
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_inspection_model(variant="M4", device=device)

    # Pick 3 representative test defects: Scratches, Inclusion, Patches
    test_dir = Path("data/NEU-DET/test/images")
    targets = ["scratches", "inclusion", "patches"]
    selected = []
    for t in targets:
        for p in sorted(test_dir.glob("*.jpg")):
            if t in p.name.lower():
                selected.append((t, p))
                break

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5), dpi=150)
    fig.patch.set_facecolor("#fafbfc")

    col_headers = [
        "1. Raw Metallic Surface",
        "2. CLAHE + Bilateral Enhanced",
        "3. Localized Defect Detections",
        "4. Grad-CAM Saliency Attribution",
    ]
    for col, title in enumerate(col_headers):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=12)

    for row, (defect_type, img_path) in enumerate(selected):
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        res = run_defect_inspection(
            image_rgb=rgb,
            model=model,
            conf_thres=0.15,
            iou_thres=0.45,
            generate_gradcam=True,
            cam_alpha=0.55,
            device=device,
        )

        axes[row, 0].imshow(res["raw_image"])
        axes[row, 0].set_ylabel(defect_type.capitalize(), fontsize=11, fontweight="bold")
        axes[row, 1].imshow(res["preprocessed_image"])
        axes[row, 2].imshow(res["annotated_image"])
        axes[row, 3].imshow(res["gradcam_overlay"])

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[DONE] Saved dashboard inspection demo to {output_path}")


if __name__ == "__main__":
    generate_dashboard_figure()
