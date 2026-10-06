"""
Unit tests for Lightweight Attention Modules (ECA, Spatial Attention).
"""

import torch

from src.models.attention import ECABlock, ECASpatialAttention, SpatialAttentionBlock


def test_eca_block_forward_and_backward():
    batch_size = 2
    channels = 128
    height, width = 25, 25

    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    eca = ECABlock(channels=channels, gamma=2, b=1)

    out = eca(x)
    assert out.shape == x.shape

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_spatial_attention_forward_and_backward():
    batch_size = 2
    channels = 64
    height, width = 25, 25

    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    sam = SpatialAttentionBlock(kernel_size=7)

    out = sam(x)
    assert out.shape == x.shape

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_combined_eca_spatial_attention():
    batch_size = 2
    channels = 256
    height, width = 13, 13

    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    attention = ECASpatialAttention(channels=channels, gamma=2, b=1, spatial_kernel=7)

    out = attention(x)
    assert out.shape == x.shape

    # Parameter count should be very small (< 200 parameters)
    param_count = sum(p.numel() for p in attention.parameters())
    assert param_count < 200, f"Attention should be lightweight, but got {param_count} params"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
