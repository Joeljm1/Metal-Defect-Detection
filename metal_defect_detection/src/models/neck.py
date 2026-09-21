"""
PANet (Path Aggregation Network) Neck with Lightweight Attention Hooks.

Fuses multi-scale features across bottom-up and top-down pathways.
When enabled (ablation variants M3 and M4), injects ECA channel attention and
spatial attention blocks into the feature pyramid to preserve fine defect gradients.
"""

from typing import List, Tuple
import torch
import torch.nn as nn

from src.models.backbone import ConvBNSiLU, C3Block
from src.models.attention import ECASpatialAttention


import torch.nn.functional as F


class PANetNeck(nn.Module):
    """
    PANet Feature Pyramid Neck with optional lightweight ECA + Spatial Attention.
    """

    def __init__(
        self,
        in_channels: List[int],  # [c3, c4, c5], e.g. [128, 256, 512]
        use_attention: bool = False,
        eca_gamma: int = 2,
        eca_b: int = 1,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        c3, c4, c5 = in_channels
        self.use_attention = use_attention

        # Top-down FPN pathway
        self.reduce_p5 = ConvBNSiLU(c5, c4, 1, 1)
        self.c3_fpn1 = C3Block(c4 + c4, c4, n=1, shortcut=False)

        self.reduce_p4 = ConvBNSiLU(c4, c3, 1, 1)
        self.c3_fpn2 = C3Block(c3 + c3, c3, n=1, shortcut=False)

        # Bottom-up PANet pathway
        self.down_p3 = ConvBNSiLU(c3, c3, 3, 2)
        self.c3_pan1 = C3Block(c3 + c4, c4, n=1, shortcut=False)

        self.down_p4 = ConvBNSiLU(c4, c4, 3, 2)
        self.c3_pan2 = C3Block(c4 + c5, c5, n=1, shortcut=False)

        # Attention modules (M3, M4)
        if self.use_attention:
            self.att_p3 = ECASpatialAttention(c3, gamma=eca_gamma, b=eca_b, spatial_kernel=spatial_kernel)
            self.att_p4 = ECASpatialAttention(c4, gamma=eca_gamma, b=eca_b, spatial_kernel=spatial_kernel)
            self.att_p5 = ECASpatialAttention(c5, gamma=eca_gamma, b=eca_b, spatial_kernel=spatial_kernel)
        else:
            self.att_p3 = nn.Identity()
            self.att_p4 = nn.Identity()
            self.att_p5 = nn.Identity()

        self.out_channels = [c3, c4, c5]

    def forward(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = features

        # --- Top-Down FPN Pathway ---
        p5_reduced = self.reduce_p5(p5)
        p5_upsampled = F.interpolate(p5_reduced, size=p4.shape[2:], mode="nearest")
        p4_fused = self.c3_fpn1(torch.cat([p5_upsampled, p4], dim=1))

        p4_reduced = self.reduce_p4(p4_fused)
        p4_upsampled = F.interpolate(p4_reduced, size=p3.shape[2:], mode="nearest")
        p3_out = self.c3_fpn2(torch.cat([p4_upsampled, p3], dim=1))

        # --- Bottom-Up PANet Pathway ---
        p3_down = self.down_p3(p3_out)
        p3_down = F.interpolate(p3_down, size=p4_fused.shape[2:], mode="nearest")
        p4_out = self.c3_pan1(torch.cat([p3_down, p4_fused], dim=1))

        p4_down = self.down_p4(p4_out)
        p4_down = F.interpolate(p4_down, size=p5.shape[2:], mode="nearest")
        p5_out = self.c3_pan2(torch.cat([p4_down, p5], dim=1))

        # --- Attention Enhancement (M3 / M4) ---
        p3_out = self.att_p3(p3_out)
        p4_out = self.att_p4(p4_out)
        p5_out = self.att_p5(p5_out)

        return p3_out, p4_out, p5_out
