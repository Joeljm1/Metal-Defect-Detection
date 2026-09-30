"""
Edge Deployment and Model Export for Metal Defect Detectors (Hypothesis H3).

Provides:
1. ONNX model export with dynamic batch axes and verification.
2. FP16 half-precision model export and quantization.
3. ONNX Runtime inference sessions and benchmarking across hardware execution providers.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from src.models.detector import DefectDetector
from src.utils.box_ops import non_max_suppression


class ExportableDetectorWrapper(nn.Module):
    """
    Wraps DefectDetector for clean deployment export, returning only the
    decoded detection tensor (B, num_candidates, 5 + num_classes).
    """

    def __init__(self, model: DefectDetector):
        super().__init__()
        self.model = model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In eval mode, model(x) returns (decoded_detections, raw_feature_outputs)
        decoded, _ = self.model(x)
        return decoded


def export_model_to_onnx(
    model: DefectDetector,
    output_path: Path | str,
    input_size: Tuple[int, int] = (200, 200),
    opset_version: int = 18,
    half_precision: bool = False,
    device: str = "cpu",
) -> Path:
    """
    Exports a trained DefectDetector to ONNX format with dynamic batch dimension.

    Args:
        model: DefectDetector instance
        output_path: Destination path for .onnx file
        input_size: (H, W) input spatial resolution
        opset_version: ONNX operator set version (default: 18)
        half_precision: If True, exports in FP16 precision
        device: Device to use during export ('cpu' or 'cuda')

    Returns:
        Path to exported ONNX model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_to_export = ExportableDetectorWrapper(model).to(device).eval()
    if half_precision:
        model_to_export = model_to_export.half()
        dummy_input = torch.randn(1, 3, *input_size, device=device, dtype=torch.float16)
    else:
        dummy_input = torch.randn(1, 3, *input_size, device=device, dtype=torch.float32)

    export_kwargs: Dict[str, Any] = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["detections"],
    )

    # Keep exporter progress chatter out of console output
    torch_onnx_logger = logging.getLogger("torch.onnx")
    previous_level = torch_onnx_logger.level
    torch_onnx_logger.setLevel(logging.WARNING)
    try:
        # Preferred: torch.export-based (dynamo) ONNX exporter — the supported
        # path going forward (the legacy TorchScript exporter is deprecated).
        torch.onnx.export(
            model_to_export,
            (dummy_input,),
            str(output_path),
            dynamo=True,
            dynamic_shapes={"x": {0: torch.export.Dim("batch_size", min=1, max=64)}},
            **export_kwargs,
        )
    except Exception:
        # Fallback: legacy TorchScript exporter with dynamic_axes
        torch.onnx.export(
            model_to_export,
            (dummy_input,),
            str(output_path),
            dynamo=False,
            dynamic_axes={
                "images": {0: "batch_size"},
                "detections": {0: "batch_size"},
            },
            **export_kwargs,
        )
    finally:
        torch_onnx_logger.setLevel(previous_level)

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    return output_path


def create_onnx_inference_session(
    onnx_path: Path | str,
    prefer_cuda: bool = True,
) -> Tuple[ort.InferenceSession, str]:
    """
    Creates an ONNX Runtime InferenceSession with optimal execution providers.

    Returns:
        (session, active_provider_name)
    """
    onnx_path = Path(onnx_path)
    available_providers = ort.get_available_providers()

    selected_providers = []
    if prefer_cuda and "CUDAExecutionProvider" in available_providers and torch.cuda.is_available():
        selected_providers.append("CUDAExecutionProvider")
    selected_providers.append("CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    session = ort.InferenceSession(str(onnx_path), sess_options, providers=selected_providers)
    active_provider = session.get_providers()[0]
    return session, active_provider


def predict_onnx(
    session: ort.InferenceSession,
    image_tensor: Union[torch.Tensor, np.ndarray],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> List[torch.Tensor]:
    """
    Executes an ONNX Runtime inference session on image tensor and applies NMS.

    Args:
        session: Active InferenceSession
        image_tensor: (B, 3, H, W) numpy array or torch tensor in [0, 1]
        conf_thres: Confidence threshold
        iou_thres: NMS IoU threshold

    Returns:
        List of detection tensors for each batch image, shape (N, 6): [x1, y1, x2, y2, conf, cls]
    """
    if isinstance(image_tensor, torch.Tensor):
        np_input = image_tensor.detach().cpu().numpy()
    else:
        np_input = image_tensor

    if np_input.ndim == 3:
        np_input = np.expand_dims(np_input, axis=0)

    np_input = np_input.astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    raw_preds = session.run([output_name], {input_name: np_input})[0]
    tensor_preds = torch.from_numpy(raw_preds)
    return non_max_suppression(tensor_preds, conf_thres=conf_thres, iou_thres=iou_thres)


def benchmark_edge_runtime(
    model: DefectDetector,
    onnx_path: Optional[Path | str] = None,
    input_size: Tuple[int, int] = (200, 200),
    warmup_runs: int = 15,
    benchmark_runs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Comprehensive multi-format edge benchmark comparing:
    - PyTorch FP32
    - PyTorch FP16
    - ONNX Runtime (CPU)
    - ONNX Runtime (GPU/CUDA if available)
    """
    results: Dict[str, Any] = {
        "variant": model.variant_name,
        "input_size": list(input_size),
        "benchmark_runs": benchmark_runs,
        "formats": {},
    }

    # 1. PyTorch FP32 Benchmark
    model_fp32 = model.to(device).eval()
    dummy_fp32 = torch.randn(1, 3, *input_size, device=device, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model_fp32(dummy_fp32)
        if device == "cuda":
            torch.cuda.synchronize()

        times_fp32 = []
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _ = model_fp32(dummy_fp32)
            if device == "cuda":
                torch.cuda.synchronize()
            times_fp32.append((time.perf_counter() - t0) * 1000.0)

    mean_fp32 = float(np.mean(times_fp32))
    results["formats"]["pytorch_fp32"] = {
        "mean_latency_ms": round(mean_fp32, 2),
        "std_latency_ms": round(float(np.std(times_fp32)), 2),
        "p95_latency_ms": round(float(np.percentile(times_fp32, 95)), 2),
        "fps": round(1000.0 / mean_fp32, 1),
        "device": device,
    }

    # 2. PyTorch FP16 Benchmark (if CUDA available, else simulated)
    if device == "cuda":
        model_fp16 = model.half().to("cuda").eval()
        dummy_fp16 = torch.randn(1, 3, *input_size, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model_fp16(dummy_fp16)
            torch.cuda.synchronize()

            times_fp16 = []
            for _ in range(benchmark_runs):
                t0 = time.perf_counter()
                _ = model_fp16(dummy_fp16)
                torch.cuda.synchronize()
                times_fp16.append((time.perf_counter() - t0) * 1000.0)

        mean_fp16 = float(np.mean(times_fp16))
        results["formats"]["pytorch_fp16"] = {
            "mean_latency_ms": round(mean_fp16, 2),
            "std_latency_ms": round(float(np.std(times_fp16)), 2),
            "p95_latency_ms": round(float(np.percentile(times_fp16, 95)), 2),
            "fps": round(1000.0 / mean_fp16, 1),
            "device": "cuda",
        }
        # Revert model to float
        model.float()

    # 3. ONNX Runtime Benchmark
    if onnx_path is None:
        temp_onnx = Path(f"checkpoints/{model.variant_name}_temp.onnx")
        export_model_to_onnx(model, temp_onnx, input_size=input_size)
        onnx_file = temp_onnx
        cleanup_temp = True
    else:
        onnx_file = Path(onnx_path)
        cleanup_temp = False

    file_size_mb = onnx_file.stat().st_size / (1024 * 1024)
    results["model_size_onnx_mb"] = round(file_size_mb, 2)

    # Benchmark ONNX on CPU
    session_cpu = ort.InferenceSession(
        str(onnx_file),
        providers=["CPUExecutionProvider"],
    )
    dummy_np = np.random.randn(1, 3, *input_size).astype(np.float32)
    in_name = session_cpu.get_inputs()[0].name
    out_name = session_cpu.get_outputs()[0].name

    for _ in range(warmup_runs):
        _ = session_cpu.run([out_name], {in_name: dummy_np})

    times_onnx_cpu = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = session_cpu.run([out_name], {in_name: dummy_np})
        times_onnx_cpu.append((time.perf_counter() - t0) * 1000.0)

    mean_onnx_cpu = float(np.mean(times_onnx_cpu))
    results["formats"]["onnxruntime_cpu"] = {
        "mean_latency_ms": round(mean_onnx_cpu, 2),
        "std_latency_ms": round(float(np.std(times_onnx_cpu)), 2),
        "p95_latency_ms": round(float(np.percentile(times_onnx_cpu, 95)), 2),
        "fps": round(1000.0 / mean_onnx_cpu, 1),
        "provider": "CPUExecutionProvider",
    }

    # Benchmark ONNX on CUDA if available
    if "CUDAExecutionProvider" in ort.get_available_providers() and torch.cuda.is_available():
        session_cuda = ort.InferenceSession(
            str(onnx_file),
            providers=["CUDAExecutionProvider"],
        )
        for _ in range(warmup_runs):
            _ = session_cuda.run([out_name], {in_name: dummy_np})
        torch.cuda.synchronize()

        times_onnx_cuda = []
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _ = session_cuda.run([out_name], {in_name: dummy_np})
            torch.cuda.synchronize()
            times_onnx_cuda.append((time.perf_counter() - t0) * 1000.0)

        mean_onnx_cuda = float(np.mean(times_onnx_cuda))
        results["formats"]["onnxruntime_cuda"] = {
            "mean_latency_ms": round(mean_onnx_cuda, 2),
            "std_latency_ms": round(float(np.std(times_onnx_cuda)), 2),
            "p95_latency_ms": round(float(np.percentile(times_onnx_cuda, 95)), 2),
            "fps": round(1000.0 / mean_onnx_cuda, 1),
            "provider": "CUDAExecutionProvider",
        }

    if cleanup_temp and onnx_file.exists():
        onnx_file.unlink()

    return results
