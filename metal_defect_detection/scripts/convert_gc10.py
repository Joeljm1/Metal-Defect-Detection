"""
Conversion and formatting script for the real GC10-DET metallic defect dataset.

Prepares:
- data/GC10-DET/train/images and data/GC10-DET/train/labels (80% split)
- data/GC10-DET/test/images and data/GC10-DET/test/labels   (20% split)

Supports both pre-formatted YOLO format annotations and Pascal VOC XML annotations.
Reports class frequencies, defect counts, and splits.
"""

import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

GC10_CLASSES: list[str] = [
    "crease",          # Class 0
    "crescent_gap",    # Class 1
    "inclusion",       # Class 2
    "oil_spot",        # Class 3
    "punching_hole",   # Class 4
    "rolled_pit",      # Class 5
    "silk_spot",       # Class 6
    "waist_folding",   # Class 7
    "water_spot",      # Class 8
    "welding_line",    # Class 9
]

CLASS_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(GC10_CLASSES)}
# Alias mapping for VOC XML variations
CLASS_ALIASES: dict[str, str] = {
    "crease": "crease",
    "crescent_gap": "crescent_gap",
    "crescentgap": "crescent_gap",
    "inclusion": "inclusion",
    "oil_spot": "oil_spot",
    "oilspot": "oil_spot",
    "punching_hole": "punching_hole",
    "punchinghole": "punching_hole",
    "punch_hole": "punching_hole",
    "rolled_pit": "rolled_pit",
    "rolledpit": "rolled_pit",
    "silk_spot": "silk_spot",
    "silkspot": "silk_spot",
    "waist_folding": "waist_folding",
    "waistfolding": "waist_folding",
    "water_spot": "water_spot",
    "waterspot": "water_spot",
    "welding_line": "welding_line",
    "weldingline": "welding_line",
}


def parse_voc_xml(xml_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Parses Pascal VOC XML into normalized YOLO bounding boxes:
    Returns list of (class_id, xc, yc, w, h).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is None:
        return []
    width = float(size_elem.findtext("width", "0"))
    height = float(size_elem.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return []

    boxes = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", "").strip().lower()
        canon_name = CLASS_ALIASES.get(raw_name)
        if not canon_name or canon_name not in CLASS_NAME_TO_ID:
            continue
        cls_id = CLASS_NAME_TO_ID[canon_name]

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1e-4 or bh <= 1e-4:
            continue

        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        norm_w = bw / width
        norm_h = bh / height
        boxes.append((cls_id, xc, yc, norm_w, norm_h))

    return boxes


def backup_synthetic_dataset(target_dir: Path) -> None:
    """If target_dir contains synthetic surrogate data, moves it to backup."""
    if not target_dir.exists():
        return
    # Check if existing files contain synthetic prefixes
    sample_images = list(target_dir.rglob("gc10_*.jpg"))
    if len(sample_images) > 0:
        backup_path = target_dir.parent / "GC10-DET-synthetic-backup"
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(str(target_dir), str(backup_path))
        print(f"Moved legacy synthetic dataset from {target_dir} to {backup_path}")


def populate_split(
    img_sources: list[Path],
    lbl_sources: list[Path],
    dest_img_dir: Path,
    dest_lbl_dir: Path,
) -> dict[str, Any]:
    """Copies images and labels into standard project split directories."""
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    lbl_map: dict[str, Path] = {}
    for lbl_dir in lbl_sources:
        if lbl_dir.exists():
            for f in lbl_dir.glob("*.*"):
                if f.suffix.lower() in [".txt", ".xml"]:
                    lbl_map[f.stem] = f

    copied_images = 0
    copied_labels = 0
    class_instances = Counter()
    images_per_class = defaultdict(set)

    for img_dir in img_sources:
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            target_img = dest_img_dir / img_path.name
            if not target_img.exists() or target_img.stat().st_size != img_path.stat().st_size:
                shutil.copy2(img_path, target_img)
            copied_images += 1

            stem = img_path.stem
            target_lbl = dest_lbl_dir / f"{stem}.txt"

            if stem in lbl_map:
                src_lbl = lbl_map[stem]
                if src_lbl.suffix.lower() == ".xml":
                    boxes = parse_voc_xml(src_lbl)
                    lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in boxes]
                    target_lbl.write_text("".join(lines), encoding="utf-8")
                else:
                    if not target_lbl.exists() or target_lbl.stat().st_size != src_lbl.stat().st_size:
                        shutil.copy2(src_lbl, target_lbl)

                copied_labels += 1
                if target_lbl.exists():
                    for line in target_lbl.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if parts:
                            c = int(parts[0])
                            class_instances[c] += 1
                            images_per_class[c].add(stem)
            else:
                # Negative background image without annotations
                if not target_lbl.exists():
                    target_lbl.write_text("", encoding="utf-8")
                copied_labels += 1

    return {
        "images": copied_images,
        "labels": copied_labels,
        "class_instances": class_instances,
        "images_per_class": {c: len(stems) for c, stems in images_per_class.items()},
    }


def convert_and_setup_gc10(
    raw_dir: Path | str = Path("data/GC10-DET-real/data"),
    output_dir: Path | str = Path("data/GC10-DET"),
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    backup_synthetic_dataset(output_dir)

    output_train_img = output_dir / "train" / "images"
    output_train_lbl = output_dir / "train" / "labels"
    output_test_img = output_dir / "test" / "images"
    output_test_lbl = output_dir / "test" / "labels"

    # Held-out 20% test split: Combine valid (231) + test (231) = 462 images
    test_img_sources = [raw_dir / "images" / "test", raw_dir / "images" / "valid"]
    test_lbl_sources = [raw_dir / "labels" / "test", raw_dir / "labels" / "valid"]

    # 80% train split: 1841 images
    train_img_sources = [raw_dir / "images" / "train"]
    train_lbl_sources = [raw_dir / "labels" / "train"]

    print("Formatting GC10-DET test split (20% held-out evaluation)...")
    test_stats = populate_split(
        test_img_sources, test_lbl_sources, output_test_img, output_test_lbl
    )

    print("Formatting GC10-DET train split (80% adaptation split)...")
    train_stats = populate_split(
        train_img_sources, train_lbl_sources, output_train_img, output_train_lbl
    )

    print("\n" + "=" * 65)
    print("GC10-DET Dataset Standardization Report")
    print("=" * 65)
    print(f"Train split images: {train_stats['images']} | labels: {train_stats['labels']}")
    print(f"Test split images:  {test_stats['images']} | labels: {test_stats['labels']}")
    print(f"Total images:       {train_stats['images'] + test_stats['images']}")
    print("-" * 65)
    print(f"{'Class':<4} {'Name':<15} {'Train Imgs':<12} {'Train Boxes':<12} {'Test Imgs':<10} {'Test Boxes':<10}")
    print("-" * 65)
    for c, name in enumerate(GC10_CLASSES):
        tr_imgs = train_stats["images_per_class"].get(c, 0)
        tr_boxes = train_stats["class_instances"].get(c, 0)
        te_imgs = test_stats["images_per_class"].get(c, 0)
        te_boxes = test_stats["class_instances"].get(c, 0)
        print(f"{c:<4} {name:<15} {tr_imgs:<12} {tr_boxes:<12} {te_imgs:<10} {te_boxes:<10}")
    print("=" * 65)

    return {
        "train": train_stats,
        "test": test_stats,
    }


if __name__ == "__main__":
    convert_and_setup_gc10()
