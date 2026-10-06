"""
CLI to run Exploratory Data Analysis on NEU-DET.
"""

import argparse

from rich.console import Console
from rich.table import Table

from src.dataset.eda import run_dataset_eda
from src.dataset.parser import load_dataset_config


def main():
    default_data_dir = load_dataset_config().get("path", "data/NEU-DET")
    parser = argparse.ArgumentParser(description="NEU-DET Exploratory Data Analysis")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Path to NEU-DET root")
    parser.add_argument("--out-dir", type=str, default="reports", help="Output directory for reports")
    args = parser.parse_args()

    console = Console()
    console.print("[bold blue]Starting NEU-DET Dataset Exploratory Data Analysis...[/bold blue]")

    summary = run_dataset_eda(args.data_dir, args.out_dir)

    table = Table(title="NEU-DET Dataset Statistics")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Total Images", str(summary["total_images"]))
    table.add_row("Total Bounding Boxes", str(summary["total_bounding_boxes"]))
    table.add_row("Mean Boxes / Image", f"{summary['boxes_per_image_mean']:.2f}")
    table.add_row("Train Images", str(summary["images_per_split"].get("train", 0)))
    table.add_row("Test Images", str(summary["images_per_split"].get("test", 0)))
    table.add_row("Mean Box Normalized Area", f"{summary['bbox_metrics']['mean_area']:.4f}")

    console.print(table)

    cls_table = Table(title="Class Breakdown")
    cls_table.add_column("Class", style="green")
    cls_table.add_column("Bounding Box Count", style="yellow")

    for cls_name, count in summary["defect_class_distribution"].items():
        cls_table.add_row(cls_name, str(count))

    console.print(cls_table)
    console.print(f"[bold green]Saved plot to:[/bold green] {summary['plot_saved_to']}")
    console.print(f"[bold green]Saved summary JSON to:[/bold green] {args.out_dir}/dataset_summary.json")


if __name__ == "__main__":
    main()
