"""
Batch Preprocessing Script for NEU-DET.

Applies:
1. Bilateral Filtering (edge-preserving denoising)
2. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
Generates an enhanced dataset directory ready for M2 and M4 training.
"""

from pathlib import Path
import shutil
import argparse
import cv2
from tqdm import tqdm

from src.preprocessing.pipeline import DefectPreprocessor


def preprocess_split(
    src_dir: Path,
    dst_dir: Path,
    preprocessor: DefectPreprocessor,
):
    src_images = src_dir / "images"
    src_labels = src_dir / "labels"

    dst_images = dst_dir / "images"
    dst_labels = dst_dir / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    img_files = list(src_images.glob("*.jpg"))
    print(f"Processing {len(img_files)} images from {src_images} -> {dst_images}")

    for img_path in tqdm(img_files, desc=f"Enhancing {src_dir.name}"):
        # Read image (cv2 loads as BGR)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # Convert to RGB for preprocessor pipeline, then back to BGR for writing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        enhanced_rgb = preprocessor.process(img_rgb)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

        # Write output image
        out_img_path = dst_images / img_path.name
        cv2.imwrite(str(out_img_path), enhanced_bgr)

        # Copy corresponding label file
        label_path = src_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, dst_labels / label_path.name)


def main():
    parser = argparse.ArgumentParser(description="Batch Preprocessing for NEU-DET")
    parser.add_argument("--src-dir", type=str, default="data/NEU-DET", help="Source dataset path")
    parser.add_argument(
        "--dst-dir",
        type=str,
        default="data/preprocessed/NEU-DET",
        help="Destination for enhanced dataset",
    )
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--bilateral-d", type=int, default=5, help="Bilateral filter diameter")
    args = parser.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)

    print("=" * 70)
    print("Executing Image Enhancement Pipeline (CLAHE + Bilateral Filtering)")
    print(f"Source: {src}")
    print(f"Target: {dst}")
    print("=" * 70)

    preprocessor = DefectPreprocessor(
        use_clahe=True,
        clahe_clip_limit=args.clip_limit,
        use_bilateral=True,
        bilateral_diameter=args.bilateral_d,
    )

    for split in ["train", "test"]:
        split_src = src / split
        split_dst = dst / split
        if split_src.exists():
            preprocess_split(split_src, split_dst, preprocessor)

    print("\n[SUCCESS] Batch dataset preprocessing complete!")


if __name__ == "__main__":
    main()
