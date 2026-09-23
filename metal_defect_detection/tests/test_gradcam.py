import torch
import numpy as np
from src.models.detector import DefectDetector
from src.evaluation.gradcam import DefectGradCAM


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
        assert len(gradcam.hook_handles) == 2
    assert len(gradcam.hook_handles) == 0
