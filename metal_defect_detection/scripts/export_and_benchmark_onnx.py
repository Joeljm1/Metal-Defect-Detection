"""
ONNX Model Export and Multi-Format Edge Runtime Benchmark (Task 6 — Hypothesis H3).

Exports:
- M1 Baseline to checkpoints/M1_Baseline.onnx
- M4 Proposed Integrated Model to checkpoints/M4_Proposed_Integrated.onnx

Benchmarks across:
1. PyTorch FP32
2. PyTorch FP16
3. ONNX Runtime (CPU)
4. ONNX Runtime (CUDA / TensorRT acceleration if available)

Saves:
- reports/onnx_benchmark_summary.json
- reports/figures/edge_latency_quantization.png
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from src.deployment.export import benchmark_edge_runtime, export_model_to_onnx
from src.models.detector import DefectDetector


def run_export_and_benchmark(
    output_dir: Path | str = Path("checkpoints"),
    summary_path: Path | str = Path("reports/onnx_benchmark_summary.json"),
    figure_path: Path | str = Path("reports/figures/edge_latency_quantization.png"),
    benchmark_runs: int = 100,
) -> dict[str, Any]:
    console = Console()
    output_dir = Path(output_dir)
    summary_path = Path(summary_path)
    figure_path = Path(figure_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]=== Exporting ONNX Models & Benchmarking Edge Runtime ({device}) ===[/bold cyan]\n")

    variants = ["M1", "M4"]
    model_paths = {
        "M1": output_dir / "M1_Baseline.onnx",
        "M4": output_dir / "M4_Proposed_Integrated.onnx",
    }
    ckpt_paths = {
        "M1": output_dir / "M1_Baseline_best.pt",
        "M4": output_dir / "M4_Proposed_Integrated_best.pt",
    }

    all_benchmarks = {}

    for var in variants:
        console.print(f"[yellow]Exporting and Benchmarking [{var}]...[/yellow]")
        model = DefectDetector.build_model(var).eval()
        if ckpt_paths[var].exists():
            ckpt = torch.load(ckpt_paths[var], map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            console.print(f"  Loaded checkpoint weights: {ckpt_paths[var]}")

        onnx_file = model_paths[var]
        export_model_to_onnx(model, onnx_file, input_size=(200, 200))
        console.print(f"  Exported ONNX model -> {onnx_file} ({onnx_file.stat().st_size / (1024*1024):.2f} MB)")

        bench = benchmark_edge_runtime(
            model=model,
            onnx_path=onnx_file,
            input_size=(200, 200),
            benchmark_runs=benchmark_runs,
            device=device,
        )
        all_benchmarks[var] = bench

    # Display Rich Table
    table = Table(title="Edge Inference Runtime Benchmark (Task 6 Milestone — 100 Iterations)")
    table.add_column("Model Variant", style="bold cyan")
    table.add_column("Runtime Engine / Format", style="white")
    table.add_column("Mean Latency", style="yellow")
    table.add_column("p95 Latency", style="magenta")
    table.add_column("Throughput (FPS)", style="bold green")
    table.add_column("Real-Time (>=30 FPS)", style="bold blue")

    for var, bench in all_benchmarks.items():
        for fmt_name, fmt_data in bench["formats"].items():
            fps = fmt_data["fps"]
            rt_status = "[green]PASS[/green]" if fps >= 30.0 else "[red]FAIL[/red]"
            table.add_row(
                bench["variant"],
                fmt_name.upper().replace("_", " "),
                f"{fmt_data['mean_latency_ms']:.2f} ms",
                f"{fmt_data['p95_latency_ms']:.2f} ms",
                f"{fps:.1f} FPS",
                rt_status,
            )

    console.print(table)

    # Save summary JSON
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=2)
    console.print(f"\n[green]Saved benchmark summary to {summary_path}[/green]")

    # Generate Publication Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor("#fafbfc")

    # Latency Comparison (Bar Chart)
    m1_data = all_benchmarks["M1"]["formats"]
    m4_data = all_benchmarks["M4"]["formats"]

    common_formats = [f for f in m4_data if f in m1_data]
    format_labels = [f.replace("_", " ").upper() for f in common_formats]

    m1_latencies = [m1_data[f]["mean_latency_ms"] for f in common_formats]
    m4_latencies = [m4_data[f]["mean_latency_ms"] for f in common_formats]

    x = np.arange(len(common_formats))
    width = 0.35

    ax1.bar(x - width / 2, m1_latencies, width, label="M1 (Baseline Plain YOLOv5s)", color="#2563eb", alpha=0.85)
    ax1.bar(x + width / 2, m4_latencies, width, label="M4 (Proposed Integrated Model)", color="#dc2626", alpha=0.85)
    ax1.axhline(y=33.3, color="gray", linestyle="--", linewidth=1.5, label="30 FPS Threshold (33.3 ms)")
    ax1.set_ylabel("Inference Latency (ms)", fontweight="bold", fontsize=11)
    ax1.set_title("Edge Inference Latency by Runtime Engine (Lower is Better)", fontweight="bold", fontsize=11, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(format_labels, fontweight="bold", fontsize=9)
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    for i in range(len(common_formats)):
        ax1.text(x[i] - width / 2, m1_latencies[i] + 0.3, f"{m1_latencies[i]:.2f}ms", ha="center", fontsize=8, fontweight="bold")
        ax1.text(x[i] + width / 2, m4_latencies[i] + 0.3, f"{m4_latencies[i]:.2f}ms", ha="center", fontsize=8, fontweight="bold")

    # Throughput (FPS) Comparison (Bar Chart)
    m1_fps = [m1_data[f]["fps"] for f in common_formats]
    m4_fps = [m4_data[f]["fps"] for f in common_formats]

    ax2.bar(x - width / 2, m1_fps, width, label="M1 (Baseline Plain YOLOv5s)", color="#2563eb", alpha=0.85)
    ax2.bar(x + width / 2, m4_fps, width, label="M4 (Proposed Integrated Model)", color="#dc2626", alpha=0.85)
    ax2.axhline(y=30.0, color="gray", linestyle="--", linewidth=1.5, label="Industrial Real-Time (30 FPS)")
    ax2.set_ylabel("Throughput (Frames Per Second)", fontweight="bold", fontsize=11)
    ax2.set_title("Edge Inference Throughput (FPS) (Higher is Better)", fontweight="bold", fontsize=11, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(format_labels, fontweight="bold", fontsize=9)
    ax2.legend(frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    for i in range(len(common_formats)):
        ax2.text(x[i] - width / 2, m1_fps[i] + 5.0, f"{m1_fps[i]:.0f}", ha="center", fontsize=8, fontweight="bold")
        ax2.text(x[i] + width / 2, m4_fps[i] + 5.0, f"{m4_fps[i]:.0f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight", dpi=150)
    plt.close()
    console.print(f"[green]Saved edge quantization figure to {figure_path}[/green]")

    return all_benchmarks


if __name__ == "__main__":
    run_export_and_benchmark()
