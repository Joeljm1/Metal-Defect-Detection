"""
Inference Latency and FPS Benchmark for Ablation Models (M1 to M4).

Verifies Hypothesis H3 (Real-Time Deployment):
Integrated model maintains real-time inference throughput (>= 30-50 FPS).
"""

import argparse
from typing import Any
import torch
from rich.console import Console
from rich.table import Table

from src.models.detector import DefectDetector
from src.evaluation.benchmark import benchmark_model


def run_benchmark_matrix(device_name: str | None = None):
    device = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    console = Console()
    console.print(
        f"[bold cyan]Running Performance & FPS Benchmark on device: {device}[/bold cyan]\n"
    )

    variants = ["M1", "M2", "M3", "M4"]
    descriptions = {
        "M1": "Baseline (Plain YOLOv5s)",
        "M2": "Baseline + CLAHE & Bilateral",
        "M3": "Baseline + ECA/Spatial Attention",
        "M4": "Proposed Integrated Architecture",
    }

    results = []
    for var in variants:
        model = DefectDetector.build_model(variant=var)
        bench = benchmark_model(model, input_size=(200, 200), device=device)
        bench["variant"] = var
        bench["description"] = descriptions[var]
        results.append(bench)

    table = Table(title="Controlled Ablation Architecture Benchmark (Task 3 Matrix)")
    table.add_column("Model", style="bold cyan")
    table.add_column("Description", style="white")
    table.add_column("Params", style="green")
    table.add_column("Size (MB)", style="green")
    table.add_column("Preproc (ms)", style="cyan")
    table.add_column("Model (ms)", style="yellow")
    table.add_column("E2E Latency", style="bold yellow")
    table.add_column("Model FPS", style="magenta")
    table.add_column("E2E FPS", style="bold magenta")
    table.add_column("Real-Time (>=30 FPS)", style="bold blue")

    for r in results:
        status_str = (
            "[green]YES (PASS)[/green]"
            if r["realtime_feasible"]
            else "[red]NO (FAIL)[/red]"
        )
        table.add_row(
            r["variant"],
            r["description"],
            f"{r['total_parameters']:,}",
            f"{r['model_size_mb']:.2f} MB",
            (
                f"{r['preproc_latency_mean_ms']:.2f} ms"
                if r["preproc_latency_mean_ms"] > 0
                else "0.00 ms"
            ),
            f"{r['model_latency_mean_ms']:.2f} ms",
            f"{r['end_to_end_latency_ms']:.2f} ms",
            f"{r['model_fps']:.1f}",
            f"{r['end_to_end_fps']:.1f}",
            status_str,
        )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()
    run_benchmark_matrix(args.device)
