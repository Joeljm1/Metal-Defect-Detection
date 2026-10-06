"""
Unit test for Model Benchmarking (FPS and end-to-end latency).
"""

from src.evaluation.benchmark import benchmark_model
from src.models.detector import DefectDetector


def test_benchmark_model_m1_and_m4():
    # Test M1 (no preprocessor)
    model_m1 = DefectDetector.build_model(variant="M1")
    bench_m1 = benchmark_model(model_m1, input_size=(64, 64), device="cpu", warmup_runs=2, benchmark_runs=3)

    assert bench_m1["total_parameters"] > 0
    assert bench_m1["preproc_latency_mean_ms"] == 0.0
    assert bench_m1["nms_latency_mean_ms"] >= 0.0
    assert bench_m1["end_to_end_latency_ms"] >= bench_m1["model_latency_mean_ms"]
    assert abs(bench_m1["end_to_end_latency_ms"] - (bench_m1["model_latency_mean_ms"] + bench_m1["nms_latency_mean_ms"])) < 1e-2

    # Test M4 (with preprocessor)
    model_m4 = DefectDetector.build_model(variant="M4")
    bench_m4 = benchmark_model(model_m4, input_size=(64, 64), device="cpu", warmup_runs=2, benchmark_runs=3)

    assert bench_m4["preproc_latency_mean_ms"] > 0.0
    assert bench_m4["end_to_end_latency_ms"] > bench_m4["model_latency_mean_ms"]
    assert bench_m4["end_to_end_fps"] > 0.0
