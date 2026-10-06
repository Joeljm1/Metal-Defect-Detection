"""
Verification tests for audit and architectural fixes:
1. Real grid stride decoding (no 4% / 12% box inflation).
2. Dashboard checkpoint resolution for all variants and missing checkpoint warning.
3. Domain adaptation frozen backbone BatchNorm staying in eval mode.
4. Edge deployment benchmark not mutating model weights to half in place.
5. Independent attention (ECA vs SAM) and preprocessing (CLAHE vs Bilateral) switches.
6. Trainer reloading best checkpoint weights.
7. End-to-end benchmark latency including preprocessing and NMS.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.deployment.dashboard import load_inspection_model, run_defect_inspection
from src.deployment.export import benchmark_edge_runtime
from src.evaluation.benchmark import benchmark_model
from src.evaluation.domain_adaptation import DomainAdaptedDetector
from src.models.detector import DefectDetector
from src.training.loss import ComputeLoss


def test_stride_real_grid_alignment():
    """Verify that 200x200 inputs decode with real grid strides without out-of-bounds inflation."""
    model = DefectDetector.build_model("M1", num_classes=6).eval()
    x = torch.zeros(1, 3, 200, 200)

    decoded, _raw = model(x)
    # Check that decoded bounding boxes are scaled within 200x200
    # Center x, y and width, height should never exceed 200
    max_xy = decoded[..., 0:2].max().item()
    assert max_xy <= 200.0, f"Max decoded coordinate {max_xy} exceeded 200!"

    # Test with 224x224 input
    x_224 = torch.zeros(1, 3, 224, 224)
    decoded_224, _ = model(x_224)
    max_xy_224 = decoded_224[..., 0:2].max().item()
    assert max_xy_224 <= 224.0, f"Max decoded coordinate {max_xy_224} exceeded 224!"


def test_compute_loss_real_stride_anchors():
    """Verify ComputeLoss normalizes anchors using real grid strides."""
    model = DefectDetector.build_model("M1", num_classes=6)
    loss_fn = ComputeLoss()
    model.train()

    images = torch.zeros(2, 3, 200, 200)
    targets = torch.tensor([
        [0.0, 0.0, 0.5, 0.5, 0.2, 0.2],
        [1.0, 1.0, 0.3, 0.3, 0.1, 0.1],
    ])
    preds = model(images)
    _tcls, _tbox, indices, _anch = loss_fn._build_targets(preds, targets, model)

    assert len(indices) == 3
    # P4 has 13 cells, real stride is 200 / 13 = 15.385
    # P5 has 7 cells, real stride is 200 / 7 = 28.571
    # Check that tbox coordinates are properly bounded by grid shapes
    assert (indices[0][2] < 25).all()
    assert (indices[1][2] < 13).all()
    assert (indices[2][2] < 7).all()


def test_dashboard_checkpoint_lookup_and_warning(tmp_path):
    """Verify that all M1-M4 variants resolve checkpoints (named as Trainer saves them) and missing ones warn."""
    for var in ["M1", "M2", "M3", "M4"]:
        # Same filename convention as Trainer.save_checkpoint: <variant_name>_best.pt
        src_model = DefectDetector.build_model(var, num_classes=6)
        ckpt_file = tmp_path / f"{src_model.variant_name}_best.pt"
        torch.save({"model_state": src_model.state_dict(), "variant": src_model.variant_name}, ckpt_file)

        model = load_inspection_model(variant=var, checkpoint_dir=tmp_path, device="cpu")
        assert model.loaded_checkpoint is not None, f"Variant {var} failed to find checkpoint!"
        assert Path(model.loaded_checkpoint) == ckpt_file

    # Nonexistent checkpoint should issue a UserWarning
    with pytest.warns(UserWarning, match="Checkpoint not found"):
        untrained_model = load_inspection_model(
            variant="M4", checkpoint_dir=tmp_path / "nonexistent_checkpoints_dir", device="cpu"
        )
    assert untrained_model.loaded_checkpoint is None


def test_dashboard_realtime_includes_preprocessing_and_reports_gradcam_errors(monkeypatch):
    """Real-time status must use prep+infer latency; Grad-CAM failures must be surfaced, not hidden."""
    import src.deployment.dashboard as dash

    model = DefectDetector.build_model("M4", num_classes=6).eval()
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    res = run_defect_inspection(img, model, generate_gradcam=False, device="cpu")
    t = res["telemetry"]
    assert t["pipeline_latency_ms"] == pytest.approx(t["prep_latency_ms"] + t["infer_latency_ms"], abs=0.02)
    assert t["pipeline_fps"] <= t["infer_fps"] + 0.1
    assert t["realtime_pass"] == (t["pipeline_fps"] >= 30.0)
    assert res["gradcam_error"] is None

    class _BrokenCAM:
        def __init__(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(dash, "DefectGradCAM", _BrokenCAM)
    res = run_defect_inspection(img, model, generate_gradcam=True, device="cpu", raise_on_cam_error=False)
    assert res["gradcam_error"] is not None and "boom" in res["gradcam_error"]

    with pytest.raises(RuntimeError, match="boom"):
        run_defect_inspection(img, model, generate_gradcam=True, device="cpu", raise_on_cam_error=True)



def test_domain_adaptation_frozen_bn_mode():
    """Verify that frozen backbone BatchNorm layers remain in eval mode during training."""
    base_model = DefectDetector.build_model("M1", num_classes=6)
    adapted = DomainAdaptedDetector(base_model, num_target_classes=10, freeze_backbone=True)

    # Calling train() on the adapted detector should NOT set backbone BN to train()
    adapted.train()
    for name, m in adapted.backbone.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            assert not m.training, f"BatchNorm {name} was put in training mode despite frozen backbone!"

    # Head on the other hand should be in training mode
    assert adapted.head.training, "Adapted head should be in training mode!"


def test_benchmark_fp16_does_not_mutate_model():
    """Verify that benchmark_edge_runtime does not mutate the original model's parameter dtype."""
    model = DefectDetector.build_model("M1", num_classes=6)
    orig_dtype = next(model.parameters()).dtype

    _ = benchmark_edge_runtime(model, input_size=(64, 64), benchmark_runs=2, warmup_runs=1, device="cpu")

    after_dtype = next(model.parameters()).dtype
    assert after_dtype == orig_dtype == torch.float32, "Model parameters were mutated!"


def test_attention_and_preprocessing_independent_switches():
    """Verify that ECA/Spatial attention and CLAHE/Bilateral can be toggled independently."""
    # Attention: ECA only
    m_eca = DefectDetector(use_attention=True, use_eca=True, use_spatial=False)
    assert m_eca.neck.att_p3.use_eca is True
    assert m_eca.neck.att_p3.use_spatial is False
    assert m_eca.neck.att_p3.channel_att is not None
    assert m_eca.neck.att_p3.spatial_att is None

    # Attention: Spatial only
    m_sam = DefectDetector(use_attention=True, use_eca=False, use_spatial=True)
    assert m_sam.neck.att_p3.use_eca is False
    assert m_sam.neck.att_p3.use_spatial is True
    assert m_sam.neck.att_p3.channel_att is None
    assert m_sam.neck.att_p3.spatial_att is not None

    # Preprocessing: CLAHE only
    m_clahe = DefectDetector(use_preprocessing=True, use_clahe=True, use_bilateral=False)
    assert m_clahe.preprocessor is not None
    assert m_clahe.preprocessor.use_clahe is True
    assert m_clahe.preprocessor.use_bilateral is False
    assert m_clahe.preprocessor.clahe is not None
    assert m_clahe.preprocessor.bilateral is None

    # Preprocessing: Bilateral only
    m_bil = DefectDetector(use_preprocessing=True, use_clahe=False, use_bilateral=True)
    assert m_bil.preprocessor is not None
    assert m_bil.preprocessor.use_clahe is False
    assert m_bil.preprocessor.use_bilateral is True
    assert m_bil.preprocessor.clahe is None
    assert m_bil.preprocessor.bilateral is not None


def test_end_to_end_benchmark_includes_nms_and_preproc():
    """Verify benchmark_model and benchmark_edge_runtime report NMS and full pipeline latency."""
    model = DefectDetector.build_model("M4", num_classes=6)
    bench = benchmark_model(model, input_size=(64, 64), device="cpu", warmup_runs=2, benchmark_runs=3)

    assert "nms_latency_mean_ms" in bench
    assert "preproc_latency_mean_ms" in bench
    assert bench["nms_latency_mean_ms"] >= 0.0
    assert bench["preproc_latency_mean_ms"] >= 0.0
    assert bench["end_to_end_latency_ms"] >= (bench["model_latency_mean_ms"] + bench["preproc_latency_mean_ms"])

    edge_bench = benchmark_edge_runtime(model, input_size=(64, 64), benchmark_runs=2, warmup_runs=1, device="cpu")
    assert "nms_latency_ms" in edge_bench
    assert "preprocessing_latency_ms" in edge_bench
    assert "end_to_end_latency_ms" in edge_bench["formats"]["pytorch_fp32"]
