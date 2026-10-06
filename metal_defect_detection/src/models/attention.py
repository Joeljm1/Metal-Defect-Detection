"""
Lightweight Attention Modules for Metal Surface Defect Detection.

Implements:
1. ECA (Efficient Channel Attention): Avoids dimensionality reduction and captures
   cross-channel interaction using 1D adaptive convolution.
2. Spatial Attention Module (SAM): Emphasizes spatially localized defect regions
   (cracks, pits, scratches) by pooling across channels.
3. ECASpatialAttention: Combined sequential channel + spatial attention block.
"""

import math

import torch
from torch import nn


class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) block (Wang et al., CVPR 2020).

    Generates channel attention via fast 1D convolution with adaptive kernel size k,
    proportional to channel dimension C:
        k = |(log2(C) / gamma) + (b / gamma)|_odd
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        self.channels = channels
        # Calculate adaptive kernel size k
        t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = t if t % 2 == 1 else t + 1
        self.kernel_size = max(k, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling: (B, C, 1, 1) -> squeeze to (B, 1, C)
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        # 1D convolution along channel dimension: (B, 1, C)
        y = self.conv1d(y)
        # Transpose back and reshape: (B, C, 1, 1)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        # Scale original feature map
        return x * y.expand_as(x)


class SpatialAttentionBlock(nn.Module):
    """
    Spatial Attention Module (SAM).

    Extracts spatial importance maps by aggregating channel statistics via
    mean and max pooling, followed by a spatial convolution and sigmoid gating.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concat along channel axis: (B, 2, H, W)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        # Convolution + Sigmoid: (B, 1, H, W)
        spatial_att = self.sigmoid(self.conv(cat_out))
        return x * spatial_att


class ECASpatialAttention(nn.Module):
    """
    Combined Lightweight ECA Channel + Spatial Attention Block.

    Applies channel attention first to focus on 'what' defect features are relevant,
    followed by spatial attention to pinpoint 'where' subtle defects are located.
    Supports enabling either or both mechanisms independently.
    Equipped with a residual shortcut to guarantee stable gradient flow.
    """

    def __init__(
        self,
        channels: int,
        gamma: int = 2,
        b: int = 1,
        spatial_kernel: int = 7,
        use_eca: bool = True,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.use_eca = use_eca
        self.use_spatial = use_spatial
        self.channel_att = (
            ECABlock(channels=channels, gamma=gamma, b=b) if use_eca else None
        )
        self.spatial_att = (
            SpatialAttentionBlock(kernel_size=spatial_kernel) if use_spatial else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = x
        if self.channel_att is not None:
            out = self.channel_att(out)
        if self.spatial_att is not None:
            out = self.spatial_att(out)
        return residual + out
