"""
Multi-Scale Detection Head for Metal Defect Detection.

Predicts bounding box coordinates, objectness scores, and multi-class defect logits
across 3 feature pyramid scales (P3, P4, P5).
"""

from typing import Any

import torch
from torch import nn


class DetectHead(nn.Module):
    """
    YOLOv5-style Multi-Scale Detection Head.
    """

    anchors: torch.Tensor

    def __init__(
        self,
        num_classes: int = 6,
        anchors: Any = None,
        in_channels: list[int] | None = None,
        strides: tuple[int, ...] = (8, 16, 32),
        img_size: tuple[int, int] | None = (200, 200),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.no = num_classes + 5  # outputs per anchor: [x, y, w, h, obj_conf, class_probs...]
        self.nl = len(strides)     # number of detection layers (3)
        self.strides = strides
        self.img_size = img_size

        if anchors is None:
            # Anchors matched to 200x200 NEU-DET defect aspect ratios (small pits,
            # elongated scratches, large patches), grouped small->large per scale.
            # Reference k-means derivation (IoU distance, k=9):
            #   uv run python scripts/cluster_anchors.py
            raw_anchors = [
                [[29, 38], [34, 76], [79, 56]],       # P3 / stride 8 (small / compact defects)
                [[33, 185], [57, 122], [178, 51]],    # P4 / stride 16 (elongated scratches / horizontal scale)
                [[127, 80], [95, 182], [179, 186]],  # P5 / stride 32 (large defect clusters & patches)
            ]
        else:
            raw_anchors = anchors

        self.na = len(raw_anchors[0])  # number of anchors per scale (3)
        # Register anchors as buffers (shape: nl, na, 2)
        self.register_buffer("anchors", torch.tensor(raw_anchors, dtype=torch.float32))

        if in_channels is None:
            in_channels = [128, 256, 512]

        # 1x1 convolutions producing predictions for each scale
        self.heads = nn.ModuleList(
            [nn.Conv2d(c, self.na * self.no, kernel_size=1) for c in in_channels]
        )

        # Objectness bias initialization:
        # Initialize objectness bias to -4.6 (sigmoid(-4.6) ~ 0.01) to prevent
        # early loss explosion and extreme gradients on thousands of background cells.
        for module in self.heads:
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                b = module.bias.view(self.na, -1)
                b.data[:, 4].fill_(-4.6)
                module.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        img_size: tuple[int, int] | None = None,
        return_decoded: bool | None = None,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: Tuple of (P3, P4, P5) feature maps from neck.
            img_size: Optional (width, height) of the input image to compute exact
                      grid-to-pixel strides when input is not an exact multiple of 32.
            return_decoded: Deprecated parameter kept for backwards compatibility.
                      Decoding is always performed to maintain a consistent return type.
            
        Returns:
            Tuple of (concatenated_detections, raw_predictions) of shapes:
            ((B, total_anchors, 5 + num_classes), [(B, na, ny, nx, no), ...])
            consistently across both training and evaluation modes.
        """
        raw_outputs = []
        decoded_outputs = []
        anchors_buf: torch.Tensor = self.anchors

        for i, feat in enumerate(features):
            head_conv = self.heads[i]
            assert isinstance(head_conv, nn.Module)
            b, _, ny, nx = feat.shape
            # Reshape: (B, na * no, ny, nx) -> (B, na, no, ny, nx) -> (B, na, ny, nx, no)
            out = head_conv(feat).view(b, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            raw_outputs.append(out)

            # Decode predictions using true grid strides
            grid = self._make_grid(nx, ny, feat.device)
            anchor_grid = anchors_buf[i].view(1, self.na, 1, 1, 2).to(feat.device)

            isize = img_size or self.img_size
            if isize is not None:
                w_val, h_val = isize[0], isize[1]
                if isinstance(w_val, torch.Tensor) and isinstance(h_val, torch.Tensor):
                    sx = w_val.to(feat.device, dtype=torch.float32) / nx
                    sy = h_val.to(feat.device, dtype=torch.float32) / ny
                    stride_t = torch.stack([sx, sy]).view(1, 1, 1, 1, 2)
                else:
                    stride_t = feat.new_tensor(
                        [float(w_val) / nx, float(h_val) / ny]
                    ).view(1, 1, 1, 1, 2)
            else:
                stride_t = feat.new_tensor(
                    [float(self.strides[i]), float(self.strides[i])]
                ).view(1, 1, 1, 1, 2)

            # Bounding box offsets with sigmoid activation scaled by real strides
            xy = (out[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride_t
            wh = (out[..., 2:4].sigmoid() * 2.0) ** 2 * anchor_grid
            box = torch.cat((xy, wh), dim=-1)

            # Objectness and class probabilities
            conf = out[..., 4:5].sigmoid()
            cls_prob = out[..., 5:].sigmoid()

            # Flatten spatial dimensions: (B, na * ny * nx, 5 + num_classes)
            pred = torch.cat((box, conf, cls_prob), dim=-1).view(b, -1, self.no)
            decoded_outputs.append(pred)

        return torch.cat(decoded_outputs, dim=1), raw_outputs

    @staticmethod
    def _make_grid(nx: int, ny: int, device: torch.device) -> torch.Tensor:
        yv, xv = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
