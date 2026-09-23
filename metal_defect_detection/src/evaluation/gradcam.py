"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for Defect Inspection.

Computes visual saliency heatmaps highlighting spatial regions that contribute
to defect classification, validating feature extraction and attention mechanisms (Phase 3).
"""

from typing import Tuple, Optional, List, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detector import DefectDetector


class DefectGradCAM:
    """
    Grad-CAM implementation specifically designed for multi-scale YOLO defect detectors.
    Extracts gradient-weighted feature activation maps from neck attention/PANet layers.
    """

    def __init__(self, model: DefectDetector, target_layer: Optional[nn.Module] = None):
        self.model = model.eval()
        self.device = next(model.parameters()).device

        # Target high-level multi-scale fusion layer in neck for optimal spatial-semantic localization
        if target_layer is None:
            if hasattr(model.neck, "c3_pan1"):
                self.target_layer = model.neck.c3_pan1
            elif hasattr(model.neck, "c3_fpn2"):
                self.target_layer = model.neck.c3_fpn2
            else:
                self.target_layer = model.neck
        else:
            self.target_layer = target_layer

        self.activations: List[torch.Tensor] = []
        self.gradients: List[torch.Tensor] = []
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        self.hook_handles = [
            self.target_layer.register_forward_hook(forward_hook),
            self.target_layer.register_full_backward_hook(backward_hook),
        ]

    def remove_hooks(self):
        """Removes forward and backward hooks to prevent memory leaks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def __del__(self):
        self.remove_hooks()

    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        target_scale: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generates normalized 2D Grad-CAM heatmap for an image tensor (1, 3, H, W).
        """
        self.activations.clear()
        self.gradients.clear()

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        self.model.zero_grad()

        # Forward pass (eval mode gives tuple: decoded, raw)
        decoded, _ = self.model(image_tensor)

        # Target score: either class-specific probability or peak objectness detection
        if class_idx is not None:
            # Score across bounding boxes for target class
            score = (decoded[0, :, 4] * decoded[0, :, 5 + class_idx]).max()
        else:
            # Overall highest confidence detection
            class_scores = decoded[0, :, 5:].max(dim=1)[0]
            score = (decoded[0, :, 4] * class_scores).max()

        score.backward(retain_graph=False)

        if not self.activations or not self.gradients:
            # Fallback if no gradients flowed
            h, w = image_tensor.shape[2:]
            return np.zeros((h, w), dtype=np.float32)

        act = self.activations[0]    # (1, C, h_feat, w_feat)
        grad = self.gradients[0]    # (1, C, h_feat, w_feat)

        # Global average pooling over spatial dimensions
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * act, dim=1, keepdim=True)    # (1, 1, h_feat, w_feat)
        cam = F.relu(cam)

        # Resize to input spatial resolution
        h_out, w_out = target_scale or image_tensor.shape[2:]
        cam_resized = F.interpolate(cam, size=(h_out, w_out), mode="bilinear", align_corners=False)
        cam_np = cam_resized.squeeze().detach().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)

    def overlay_cam(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlays heatmap onto original RGB/BGR image.
        Args:
            image: (H, W, 3) image in uint8
            heatmap: (H, W) in [0, 1]
            alpha: Transparency factor
        Returns:
            Blended uint8 RGB image
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize((heatmap * 255).astype(np.uint8), (w, h))
        heatmap_color = cv2.applyColorMap(heatmap_resized, colormap)
        # Convert BGR colormap output to RGB
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(image, 1.0 - alpha, heatmap_color, alpha, 0)
        return blended
