"""
Unit tests for Detector Architecture and Ablation Models (M1 to M4).
"""

import torch
import pytest

from src.models.backbone import CSPDarknetBackbone
from src.models.neck import PANetNeck
from src.models.head import DetectHead
from src.models.detector import DefectDetector


def test_backbone_feature_scales():
    backbone = CSPDarknetBackbone(in_channels=3)
    x = torch.randn(2, 3, 200, 200)
    p3, p4, p5 = backbone(x)

    # Stride 8, 16, 32
    assert p3.shape[1] == 128
    assert p4.shape[1] == 256
    assert p5.shape[1] == 512
    assert p3.shape[2] == 25 and p3.shape[3] == 25
    assert p4.shape[2] == 13 and p4.shape[3] == 13
    assert p5.shape[2] == 7 and p5.shape[3] == 7


def test_neck_with_and_without_attention():
    # Without attention
    neck_vanilla = PANetNeck(in_channels=[128, 256, 512], use_attention=False)
    p3 = torch.randn(2, 128, 25, 25)
    p4 = torch.randn(2, 256, 13, 13)
    p5 = torch.randn(2, 512, 7, 7)

    o3, o4, o5 = neck_vanilla((p3, p4, p5))
    assert o3.shape == p3.shape
    assert o4.shape == p4.shape
    assert o5.shape == p5.shape

    # With attention (M3, M4)
    neck_att = PANetNeck(in_channels=[128, 256, 512], use_attention=True)
    ao3, ao4, ao5 = neck_att((p3, p4, p5))
    assert ao3.shape == p3.shape
    assert ao4.shape == p4.shape
    assert ao5.shape == p5.shape


def test_ablation_variants_forward_pass():
    variants = ["M1", "M2", "M3", "M4"]
    x = torch.randn(2, 3, 200, 200)

    for var in variants:
        model = DefectDetector.build_model(variant=var, num_classes=6)

        # Training mode: returns raw head outputs (list of 3 tensors)
        model.train()
        train_out = model(x)
        assert isinstance(train_out, list)
        assert len(train_out) == 3
        assert train_out[0].shape[-1] == 5 + 6  # 11 channels: 4 bbox + 1 obj + 6 cls

        # Eval mode: returns (decoded_boxes, raw_outputs)
        model.eval()
        with torch.no_grad():
            eval_out, _ = model(x)
            assert isinstance(eval_out, torch.Tensor)
            assert eval_out.shape[0] == 2
            assert eval_out.shape[2] == 5 + 6
