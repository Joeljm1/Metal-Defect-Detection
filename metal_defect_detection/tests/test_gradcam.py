import numpy as np
import torch

from src.evaluation.gradcam import DefectGradCAM
from src.models.detector import DefectDetector


def test_gradcam_instantiation_and_forward():
    model = DefectDetector.build_model("M4")
    gradcam = DefectGradCAM(model)
    x = torch.randn(1, 3, 200, 200)
    cam = gradcam.generate_cam(x)
    assert cam.shape == (200, 200)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_gradcam_overlay():
    model = DefectDetector.build_model("M1")
    gradcam = DefectGradCAM(model)
    img_np = np.zeros((200, 200, 3), dtype=np.uint8)
    cam = np.ones((200, 200), dtype=np.float32) * 0.5
    overlay = gradcam.overlay_cam(img_np, cam)
    assert overlay.shape == (200, 200, 3)
    assert overlay.dtype == np.uint8
    gradcam.remove_hooks()
    assert len(gradcam.hook_handles) == 0


def test_gradcam_context_manager():
    model = DefectDetector.build_model("M4")
    with DefectGradCAM(model) as gradcam:
        x = torch.randn(1, 3, 200, 200)
        cam = gradcam.generate_cam(x)
        assert cam.shape == (200, 200)
        # Forward + backward hook per multi-scale target layer
        assert len(gradcam.hook_handles) == 2 * len(gradcam.target_layers)
    assert len(gradcam.hook_handles) == 0


def test_gradcam_multiscale_target_layers():
    # M4: hooks on c3_fpn2 / c3_pan1 / att_p3 (multi-scale PANet neck layers)
    gradcam_m4 = DefectGradCAM(DefectDetector.build_model("M4"))
    assert len(gradcam_m4.target_layers) == 3
    gradcam_m4.remove_hooks()

    # M1 has no attention module: Identity blocks are skipped
    gradcam_m1 = DefectGradCAM(DefectDetector.build_model("M1"))
    assert len(gradcam_m1.target_layers) == 2
    gradcam_m1.remove_hooks()
