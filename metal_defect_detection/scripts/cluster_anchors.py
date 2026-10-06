"""
K-Means Anchor Box Clustering for NEU-DET Metallic Defect Bounding Boxes.

Derives 9 anchor dimensions (3 per detection scale) by k-means clustering with
an IoU (1 - IoU) distance over the ground-truth box widths/heights, following
the standard YOLO anchor-adaptation procedure (YOLOv2/v5). Anchors are sorted
by area and grouped small -> large into the P3 / P4 / P5 detection scales.

Usage:
    uv run python scripts/cluster_anchors.py --data-dir data/NEU-DET --k 9
    uv run python scripts/cluster_anchors.py --output reports/anchors.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.dataset.parser import load_dataset_config

_DATASET_CFG = load_dataset_config()
PIXEL_SIZE = _DATASET_CFG.get("image_size", [200, 200])[0]  # Image width/height in pixels


def load_box_wh(label_dir: Path, img_size: int = PIXEL_SIZE) -> np.ndarray:
    """Load (w, h) of every YOLO label box, scaled to pixel units."""
    boxes = []
    for lbl in sorted(label_dir.glob("*.txt")):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                w, h = float(parts[3]) * img_size, float(parts[4]) * img_size
                if w > 0 and h > 0:
                    boxes.append((w, h))
    return np.array(boxes, dtype=np.float64)


def iou_wh(boxes: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """IoU between (w, h) rectangles anchored at the origin."""
    inter = np.minimum(boxes[:, None, :], clusters[None, :, :]).prod(axis=2)
    union = boxes[:, None, :].prod(2) + clusters[None, :, :].prod(2) - inter
    return inter / union


def kmeans(boxes: np.ndarray, k: int = 9, iters: int = 300, seed: int = 42) -> np.ndarray:
    """K-means with IoU distance. Deterministic for a given seed."""
    rng = random.Random(seed)
    clusters = boxes[rng.sample(range(len(boxes)), k)].copy()
    for _ in range(iters):
        assignment = iou_wh(boxes, clusters).argmax(axis=1)
        new_clusters = clusters.copy()
        for i in range(k):
            members = boxes[assignment == i]
            if len(members):
                new_clusters[i] = members.mean(axis=0)
        if np.allclose(new_clusters, clusters):
            break
        clusters = new_clusters
    return clusters


def group_into_scales(anchors: np.ndarray) -> list[list[list[float]]]:
    """Sort anchors by area and split into P3 (small) / P4 (mid) / P5 (large)."""
    order = np.argsort(anchors.prod(axis=1))
    anchors = anchors[order].round(1)
    per_scale = len(anchors) // 3
    return [
        anchors[i * per_scale : (i + 1) * per_scale].tolist() for i in range(3)
    ]


def main():
    parser = argparse.ArgumentParser(description="K-means anchor clustering for NEU-DET")
    parser.add_argument("--data-dir", type=str, default=_DATASET_CFG.get("path", "data/NEU-DET"), help="Dataset root")
    parser.add_argument("--k", type=int, default=9, help="Number of anchors (multiple of 3)")
    parser.add_argument("--seed", type=int, default=42, help="K-means init seed")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.k % 3 != 0:
        raise SystemExit("--k must be a multiple of 3 (3 anchors per detection scale)")

    data_dir = Path(args.data_dir)
    label_dir = data_dir / "train" / "labels"
    boxes = load_box_wh(label_dir)
    if len(boxes) == 0:
        raise SystemExit(f"No label boxes found under {label_dir}")

    anchors = kmeans(boxes, k=args.k, seed=args.seed)
    mean_iou = float(iou_wh(boxes, anchors).max(axis=1).mean())
    scales = group_into_scales(anchors)

    print(f"Clustering {len(boxes)} boxes from {label_dir} (k={args.k}, seed={args.seed})")
    print(f"Mean best-IoU of final anchors: {mean_iou:.4f}")
    names = ["P3 / stride 8 (small)", "P4 / stride 16 (mid)", "P5 / stride 32 (large)"]
    for name, scale in zip(names, scales):
        pretty = ", ".join(f"[{w}, {h}]" for w, h in scale)
        print(f"  {name}: {pretty}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "k": args.k,
                    "seed": args.seed,
                    "mean_best_iou": round(mean_iou, 4),
                    "anchors": scales,
                },
                f,
                indent=2,
            )
        print(f"[DONE] Saved anchors to {out_path}")


if __name__ == "__main__":
    main()
