"""
Evaluation Script for Defect Detection Models.
Computes Precision, Recall, F1, and mAP@0.5 on the test set.
"""

from pathlib import Path
import argparse
import torch
from rich.console import Console
from rich.table import Table

from src.models.detector import DefectDetector
from src.dataset.loader import create_dataloaders
from src.preprocessing.pipeline import DefectPreprocessor
from src.evaluation.metrics import evaluate_detections
from src.utils.box_ops import non_max_suppression


def run_evaluation(
    model_path: Path,
    data_dir: Path = Path("data/NEU-DET"),
    conf_thres: float = 0.20,
    iou_thres: float = 0.45,
    device_name: str | None = None,
):
    console = Console()
    device = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Loading checkpoint: {model_path} on {device}...[/bold cyan]")

    ckpt = torch.load(model_path, map_location=device)
    variant = ckpt.get("variant", "M4")
    model = DefectDetector.build_model(variant=variant, num_classes=6)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    preprocessor = None
    if "M2" in variant or "M4" in variant or "Integrated" in variant or "Preprocessing" in variant:
        preprocessor = DefectPreprocessor(use_clahe=True, use_bilateral=True)

    _, test_loader = create_dataloaders(
        data_dir=data_dir,
        preprocessor=preprocessor,
        batch_size=16,
        num_workers=2,
        augment_train=False,
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            preds, _ = model(images)  # decoded boxes: (B, num_boxes, 11)
            nms_preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)

            img_h, img_w = float(images.shape[2]), float(images.shape[3])
            for b_idx in range(images.shape[0]):
                # Target boxes for this image
                img_targets = targets[targets[:, 0] == b_idx]
                if img_targets.numel() > 0:
                    # Convert normalized xywh to pixel xyxy
                    xc = img_targets[:, 2] * img_w
                    yc = img_targets[:, 3] * img_h
                    w = img_targets[:, 4] * img_w
                    h = img_targets[:, 5] * img_h
                    boxes_xyxy = torch.stack(
                        [img_targets[:, 1], xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=1
                    )
                else:
                    boxes_xyxy = torch.zeros((0, 5), dtype=torch.float32)

                all_preds.append(nms_preds[b_idx].cpu())
                all_targets.append(boxes_xyxy)

    metrics = evaluate_detections(all_preds, all_targets, iou_threshold=0.5, num_classes=6)

    table = Table(title=f"Evaluation Results on NEU-DET Test Set ({variant})")
    table.add_column("Class", style="cyan")
    table.add_column("Ground Truths", style="white")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="yellow")
    table.add_column("F1-Score", style="magenta")
    table.add_column("AP@0.5", style="bold blue")

    for cls_name, m in metrics["per_class"].items():
        table.add_row(
            cls_name,
            str(m["num_gt"]),
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['ap50']:.3f}",
        )

    table.add_section()
    table.add_row(
        "[bold]Overall (mAP@0.5)[/bold]",
        "-",
        f"[bold]{metrics['mean_precision']:.3f}[/bold]",
        f"[bold]{metrics['mean_recall']:.3f}[/bold]",
        f"[bold]{metrics['mean_f1']:.3f}[/bold]",
        f"[bold cyan]{metrics['mAP@0.5']:.3f}[/bold cyan]",
    )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/M4_Proposed_Integrated_best.pt")
    parser.add_argument("--data-dir", type=str, default="data/NEU-DET")
    args = parser.parse_args()

    run_evaluation(Path(args.checkpoint), Path(args.data_dir))
