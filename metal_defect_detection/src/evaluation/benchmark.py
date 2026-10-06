"""
Model Complexity and Inference Latency / FPS Benchmarking.

Evaluates:
- Parameter count and Model Size (MB)
- Inference latency per frame (ms)
- Throughput in Frames Per Second (FPS) to verify Real-Time requirement (>= 30-50 FPS)
"""

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from src.utils.box_ops import non_max_suppression


def benchmark_model(
    model: nn.Module,
    input_size: tuple[int, int] = (200, 200),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    warmup_runs: int = 15,
    benchmark_runs: int = 100,
) -> dict[str, Any]:
    """
    Measures parameter counts, model memory footprint, raw model FPS,
    and true end-to-end pipeline FPS (including preprocessing if configured).
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes

    dummy_input = torch.randn(1, 3, input_size[1], input_size[0], device=dev)

    # Warmup model forward
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Measure pure model forward latency
    model_latencies = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            model_latencies.append((end - start) * 1000.0)  # ms

    model_latencies = np.array(model_latencies)
    mean_model_latency = float(np.mean(model_latencies))
    std_model_latency = float(np.std(model_latencies))
    model_fps = float(1000.0 / mean_model_latency) if mean_model_latency > 0 else 0.0

    # Measure preprocessing latency (if configured on model, e.g. M2, M4)
    preproc = getattr(model, "preprocessor", None)
    if preproc is not None:
        dummy_img = np.random.randint(0, 256, (input_size[1], input_size[0], 3), dtype=np.uint8)
        for _ in range(warmup_runs):
            _ = preproc(dummy_img)

        preproc_latencies = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = preproc(dummy_img)
            end = time.perf_counter()
            preproc_latencies.append((end - start) * 1000.0)

        mean_preproc_latency = float(np.mean(preproc_latencies))
        std_preproc_latency = float(np.std(preproc_latencies))
    else:
        mean_preproc_latency = 0.0
        std_preproc_latency = 0.0

    # Measure NMS postprocessing latency
    with torch.no_grad():
        sample_out = model(dummy_input)
        sample_decoded = sample_out[0] if isinstance(sample_out, tuple) else sample_out
    nms_latencies = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        _ = non_max_suppression(sample_decoded, conf_thres=0.25, iou_thres=0.45)
        end = time.perf_counter()
        nms_latencies.append((end - start) * 1000.0)

    mean_nms_latency = float(np.mean(nms_latencies))
    std_nms_latency = float(np.std(nms_latencies))

    # True End-to-End Latency = Preprocessing + Model forward pass + NMS
    mean_e2e_latency = mean_model_latency + mean_preproc_latency + mean_nms_latency
    e2e_fps = float(1000.0 / max(1e-3, mean_e2e_latency))

    return {
        "device": str(dev),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 3),
        "model_latency_mean_ms": round(mean_model_latency, 3),
        "model_latency_std_ms": round(std_model_latency, 3),
        "model_fps": round(model_fps, 1),
        "preproc_latency_mean_ms": round(mean_preproc_latency, 3),
        "preproc_latency_std_ms": round(std_preproc_latency, 3),
        "nms_latency_mean_ms": round(mean_nms_latency, 3),
        "nms_latency_std_ms": round(std_nms_latency, 3),
        "end_to_end_latency_ms": round(mean_e2e_latency, 3),
        "end_to_end_fps": round(e2e_fps, 1),
        # Aliases for backward compatibility:
        "latency_mean_ms": round(mean_e2e_latency, 3),
        "latency_std_ms": round(std_model_latency, 3),
        "fps": round(e2e_fps, 1),
        "realtime_feasible": e2e_fps >= 30.0,
    }
