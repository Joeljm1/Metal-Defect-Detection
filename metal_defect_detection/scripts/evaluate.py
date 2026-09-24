"""
Evaluation Script for Defect Detection Models.

Computes Precision, Recall, F1, and mAP@0.5 on the held-out test split using
the shared evaluation policy: the AP curve spans the full confidence range
(NMS at conf 0.001) and P/R/F1 are reported at the fixed operating point
(conf 0.25).
"""

from pathlib import Path
import argparse
import torch
from rich.console import Console
from rich.table import Table

from src.models.detector import DefectDetector
from src.dataset.loader import create_dataloaders
from src.preprocessing.pipeline import DefectPreprocessor
from src.evaluation.metrics import (
    EVAL_CONF_THRES,
    REPORT_CONF_THRES,
    evaluate_model_on_loader,
)

def run_evaluation(
    model_path: Path,
    data_dir: Path = Path("data/NEU-DET"),
    conf_thres: float = EVAL_CONF_THRES,
    iou_thres: float = 0.45,
    device_name: str | None = None,
    report_conf_thres: float = REPORT_CONF_THRES,
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

    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        preprocessor=preprocessor,
        batch_size=16,
        num_workers=2,
        augment_train=False,
    )

    metrics = evaluate_model_on_loader(
        model,
        test_loader,
        device,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        report_conf_thres=report_conf_thres,
    )

    table = Table(
        title=(
            f"Evaluation Results on NEU-DET Test Set ({variant}) — "
            f"P/R/F1 @ conf {report_conf_thres}, AP over full curve"
        )
    )
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
