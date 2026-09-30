"""
Unit tests for ONNX model export and edge inference benchmarking (Task 6).
"""

import tempfile
from pathlib import Path
import torch
import numpy as np

from src.models.detector import DefectDetector
from src.deployment.export import (
    ExportableDetectorWrapper,
    export_model_to_onnx,
    create_onnx_inference_session,
    predict_onnx,
    benchmark_edge_runtime,
)


def test_exportable_wrapper():
    model = DefectDetector.build_model("M4")
    wrapper = ExportableDetectorWrapper(model)
    x = torch.randn(1, 3, 200, 200)
    out = wrapper(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 1
    assert out.shape[2] == 11  # 5 + 6 classes


def test_export_and_onnx_session_execution():
    model = DefectDetector.build_model("M1").eval()
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = Path(tmp_dir) / "m1_test.onnx"
        export_model_to_onnx(model, onnx_file, input_size=(200, 200))
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 0

        session, provider = create_onnx_inference_session(onnx_file, prefer_cuda=False)
        assert session is not None
        assert "CPUExecutionProvider" in provider

        dummy_img = np.random.randn(1, 3, 200, 200).astype(np.float32)
        detections = predict_onnx(session, dummy_img, conf_thres=0.1, iou_thres=0.45)
        assert isinstance(detections, list)
        assert len(detections) == 1
        assert isinstance(detections[0], torch.Tensor)


def test_benchmark_edge_runtime_smoke():
    model = DefectDetector.build_model("M1").eval()
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = Path(tmp_dir) / "m1_bench.onnx"
        export_model_to_onnx(model, onnx_file, input_size=(200, 200))

        bench = benchmark_edge_runtime(
            model=model,
            onnx_path=onnx_file,
            input_size=(200, 200),
            warmup_runs=2,
            benchmark_runs=5,
            device="cpu",
        )
        assert "formats" in bench
        assert "pytorch_fp32" in bench["formats"]
        assert "onnxruntime_cpu" in bench["formats"]
        assert bench["formats"]["pytorch_fp32"]["fps"] > 0
        assert bench["formats"]["onnxruntime_cpu"]["fps"] > 0
