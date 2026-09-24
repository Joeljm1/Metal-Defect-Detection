"""
Multi-Scale Detection Head for Metal Defect Detection.

Predicts bounding box coordinates, objectness scores, and multi-class defect logits
across 3 feature pyramid scales (P3, P4, P5).
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class DetectHead(nn.Module):
    """
    YOLOv5-style Multi-Scale Detection Head.
    """

    def __init__(
        self,
        num_classes: int = 6,
        anchors: Optional[List[List[Tuple[float, float]]]] = None,
        in_channels: Optional[List[int]] = None,
        strides: Tuple[int, ...] = (8, 16, 32),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.no = num_classes + 5  # outputs per anchor: [x, y, w, h, obj_conf, class_probs...]
        self.nl = len(strides)     # number of detection layers (3)
        self.strides = strides

        if anchors is None:
            # Anchors matched to 200x200 NEU-DET defect aspect ratios (small pits,
            # elongated scratches, large patches), grouped small->large per scale.
            # Reference k-means derivation (IoU distance, k=9):
            #   uv run python scripts/cluster_anchors.py
            anchors = [
                [[29, 38], [34, 76], [79, 56]],       # P3 / stride 8 (small / compact defects)
                [[33, 185], [57, 122], [178, 51]],    # P4 / stride 16 (elongated scratches / horizontal scale)
                [[127, 80], [95, 182], [179, 186]],  # P5 / stride 32 (large defect clusters & patches)
            ]

        self.na = len(anchors[0])  # number of anchors per scale (3)
        # Register anchors as buffers (shape: nl, na, 2)
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))

        if in_channels is None:
            in_channels = [128, 256, 512]

        # 1x1 convolutions producing predictions for each scale
        self.heads = nn.ModuleList(
            [nn.Conv2d(c, self.na * self.no, kernel_size=1) for c in in_channels]
        )

    def forward(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: Tuple of (P3, P4, P5) feature maps from neck.
            
        Returns:
            In training mode: List of 3 raw prediction tensors of shape (B, na, ny, nx, no)
            In evaluation mode: Tuple of (concatenated_detections, raw_predictions)
        """
        raw_outputs = []
        decoded_outputs = []

        for i, (feat, head) in enumerate(zip(features, self.heads)):
            b, _, ny, nx = feat.shape
            # Reshape: (B, na * no, ny, nx) -> (B, na, no, ny, nx) -> (B, na, ny, nx, no)
            out = head(feat).view(b, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            raw_outputs.append(out)

            if not self.training:
                # Decode predictions during inference
                grid = self._make_grid(nx, ny, feat.device)
                anchor_grid = self.anchors[i].view(1, self.na, 1, 1, 2).to(feat.device)
                stride = self.strides[i]

                # Bounding box offsets with sigmoid activation
                xy = (out[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride
                wh = (out[..., 2:4].sigmoid() * 2.0) ** 2 * anchor_grid
                box = torch.cat((xy, wh), dim=-1)

                # Objectness and class probabilities
                conf = out[..., 4:5].sigmoid()
                cls_prob = out[..., 5:].sigmoid()

                # Flatten spatial dimensions: (B, na * ny * nx, 5 + num_classes)
                pred = torch.cat((box, conf, cls_prob), dim=-1).view(b, -1, self.no)
                decoded_outputs.append(pred)

        if self.training:
            return raw_outputs
        else:
            return torch.cat(decoded_outputs, dim=1), raw_outputs

    @staticmethod
    def _make_grid(nx: int, ny: int, device: torch.device) -> torch.Tensor:
        yv, xv = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
