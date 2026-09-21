"""
Generate High-Quality Visual Comparison Plots for Preprocessing Pipeline.

Shows:
1. Raw Metallic Surface
2. Bilateral Filtered (Edge-Preserving Denoised)
3. CLAHE Enhanced (Contrast Equalized)
4. Combined Enhanced Pipeline
5. Edge Gradient Analysis (Sobel Filter) proving defect edge preservation
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing.pipeline import DefectPreprocessor
from src.dataset.parser import CLASS_NAMES, parse_yolo_label_file
from src.utils.visualization import draw_bounding_boxes


def generate_preprocessing_report_figures(
    dataset_dir: Path = Path("data/NEU-DET"),
    output_dir: Path = Path("reports/figures"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = DefectPreprocessor(use_clahe=True, use_bilateral=True)

    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"

    # Pick 1 sample image per class
    sample_images = {}
    for cls_name in CLASS_NAMES:
        matching = list(train_img_dir.glob(f"{cls_name}_*.jpg"))
        if matching:
            sample_images[cls_name] = matching[0]

    if not sample_images:
        print("[WARNING] No sample images found matching class patterns.")
        return

    print(f"Generating preprocessing comparisons for {len(sample_images)} classes...")

    # Create a comprehensive 6-row by 5-column figure
    # Columns: Raw (with GT box), Bilateral, CLAHE, Enhanced, Sobel Edge Map
    fig, axes = plt.subplots(len(sample_images), 5, figsize=(18, 3.2 * len(sample_images)))

    col_headers = [
        "1. Raw Image (+ GT Box)",
        "2. Bilateral Denoised",
        "3. CLAHE Contrast Boost",
        "4. Combined Enhanced",
        "5. Edge Preservation (Sobel)",
    ]

    for col_idx, h in enumerate(col_headers):
        axes[0, col_idx].set_title(h, fontsize=12, fontweight="bold", pad=10)

    for row_idx, (cls_name, img_path) in enumerate(sample_images.items()):
        img_bgr = cv2.imread(str(img_path))
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        boxes = parse_yolo_label_file(lbl_path)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        stages = preprocessor.get_stages(img_rgb)
        raw_rgb = stages["raw"]
        bilateral_rgb = stages["bilateral"]
        clahe_rgb = stages["clahe_only"]
        enhanced_rgb = stages["enhanced"]

        # Raw with GT boxes
        raw_with_boxes = draw_bounding_boxes(raw_rgb, boxes, is_normalized_xywh=True)

        # Sobel edge magnitude on enhanced image
        gray_enhanced = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_enhanced, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_mag = np.uint8(np.clip((edge_mag / (edge_mag.max() + 1e-6)) * 255, 0, 255))

        # Plot row (all images are RGB)
        row_imgs = [
            raw_with_boxes,
            bilateral_rgb,
            clahe_rgb,
            enhanced_rgb,
            edge_mag,
        ]

        for col_idx, display_img in enumerate(row_imgs):
            ax = axes[row_idx, col_idx]
            if col_idx == 4:
                ax.imshow(display_img, cmap="inferno")
            else:
                ax.imshow(display_img)
            ax.axis("off")

            if col_idx == 0:
                ax.text(
                    -15,
                    display_img.shape[0] // 2,
                    cls_name.upper().replace("_", " "),
                    rotation=90,
                    verticalalignment="center",
                    horizontalalignment="right",
                    fontsize=11,
                    fontweight="bold",
                )

    plt.tight_layout()
    output_path = output_dir / "preprocessing_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] High-res comparison plot saved to: {output_path}")


if __name__ == "__main__":
    generate_preprocessing_report_figures()
