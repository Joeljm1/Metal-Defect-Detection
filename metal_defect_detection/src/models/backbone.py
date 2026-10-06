"""
CSPDarknet Backbone for Multi-Scale Feature Extraction.

Based on the CSP (Cross Stage Partial) Darknet architecture used in YOLOv5s.
Extracts hierarchical features at 3 detection scales:
- P3 (stride 8): fine-scale features for small defects (pits, micro-cracks)
- P4 (stride 16): intermediate-scale features (scratches, patches)
- P5 (stride 32): high-level semantic features with large receptive fields (rolled-in scale)
"""

import torch
from torch import nn


def autopad(k: int, p: int | None = None) -> int:
    """Pad to 'same' shape outputs."""
    if p is None:
        return k // 2
    return p



class ConvBNSiLU(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard residual bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBNSiLU(hidden_channels, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Block(nn.Module):
    """CSP Bottleneck with 3 convolutions (Cross Stage Partial block)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv3 = ConvBNSiLU(2 * hidden_channels, out_channels, 1)
        self.m = nn.Sequential(
            *(Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBNSiLU(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class CSPDarknetBackbone(nn.Module):
    """
    CSPDarknet Backbone producing multi-scale feature representations.
    
    Returns:
        Tuple of (P3, P4, P5) feature maps.
    """

    def __init__(self, in_channels: int = 3, depth_multiple: float = 0.33, width_multiple: float = 0.50):
        super().__init__()
        # Channel scaling for YOLOv5s: 64 -> 32, 128 -> 64, 256 -> 128, 512 -> 256, 1024 -> 512
        c1 = max(round(64 * width_multiple), 1)     # 32
        c2 = max(round(128 * width_multiple), 1)    # 64
        c3 = max(round(256 * width_multiple), 1)    # 128 (P3)
        c4 = max(round(512 * width_multiple), 1)    # 256 (P4)
        c5 = max(round(1024 * width_multiple), 1)   # 512 (P5)

        n1 = max(round(3 * depth_multiple), 1)
        n2 = max(round(6 * depth_multiple), 1)
        n3 = max(round(9 * depth_multiple), 1)

        # Stem & Stage 1
        self.stem = ConvBNSiLU(in_channels, c1, 6, 2, 2)  # Stride 2
        self.stage1 = nn.Sequential(
            ConvBNSiLU(c1, c2, 3, 2),                      # Stride 4
            C3Block(c2, c2, n=n1),
        )

        # Stage 2 -> Output P3 (Stride 8)
        self.stage2 = nn.Sequential(
            ConvBNSiLU(c2, c3, 3, 2),
            C3Block(c3, c3, n=n2),
        )

        # Stage 3 -> Output P4 (Stride 16)
        self.stage3 = nn.Sequential(
            ConvBNSiLU(c3, c4, 3, 2),
            C3Block(c4, c4, n=n3),
        )

        # Stage 4 -> Output P5 (Stride 32)
        self.stage4 = nn.Sequential(
            ConvBNSiLU(c4, c5, 3, 2),
            C3Block(c5, c5, n=n1),
            SPPF(c5, c5, k=5),
        )

        self.out_channels = [c3, c4, c5]  # [128, 256, 512] for YOLOv5s

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5
