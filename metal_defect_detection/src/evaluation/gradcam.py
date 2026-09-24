"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for Defect Inspection.

Computes visual saliency heatmaps highlighting spatial regions that contribute
to defect classification, validating feature extraction and attention mechanisms (Phase 3).

Saliency is computed on three multi-scale PANet neck layers (``c3_fpn2``,
``c3_pan1``, and the ``att_p3`` attention block) and averaged, yielding
attribution across the P3/P4/P5 feature pyramid. Identity blocks (M1/M2,
which carry no attention module) are skipped automatically.
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

    DEFAULT_TARGET_LAYER_NAMES = ("c3_fpn2", "c3_pan1", "att_p3")

    def __init__(self, model: DefectDetector, target_layer: Optional[nn.Module] = None):
        self.model = model.eval()
        self.device = next(model.parameters()).device

        # Target multi-scale fusion layers in the neck for spatial-semantic localization
        if target_layer is not None:
            self.target_layers = [target_layer]
        else:
            layers = []
            for name in self.DEFAULT_TARGET_LAYER_NAMES:
                module = getattr(model.neck, name, None)
                if module is not None and not isinstance(module, nn.Identity):
                    layers.append(module)
            self.target_layers = layers or [model.neck]
        # Backward-compatible alias for the primary target layer
        self.target_layer = self.target_layers[0]

        # Per-layer capture buffers (index-aligned with self.target_layers)
        self.activations: List[List[torch.Tensor]] = [[] for _ in self.target_layers]
        self.gradients: List[List[torch.Tensor]] = [[] for _ in self.target_layers]
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        for idx, layer in enumerate(self.target_layers):
            def forward_hook(module, inp, out, idx=idx):
                self.activations[idx].append(out)

            def backward_hook(module, grad_in, grad_out, idx=idx):
                self.gradients[idx].append(grad_out[0])

            self.hook_handles.append(layer.register_forward_hook(forward_hook))
            self.hook_handles.append(layer.register_full_backward_hook(backward_hook))

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
        Generates a normalized 2D Grad-CAM heatmap for an image tensor (1, 3, H, W),
        averaged over all hooked multi-scale neck layers.
        """
        for buf in self.activations:
            buf.clear()
        for buf in self.gradients:
            buf.clear()

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

        h_out, w_out = target_scale or image_tensor.shape[2:]

        # Grad-CAM per hooked layer, then average across layers
        cams = []
        for acts, grads in zip(self.activations, self.gradients):
            if not acts or not grads:
                # Fallback if no gradients flowed to this layer
                continue
            act = acts[0]    # (1, C, h_feat, w_feat)
            grad = grads[0]  # (1, C, h_feat, w_feat)

            # Global average pooling of gradients over spatial dimensions
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activation maps, rectified
            cam = torch.sum(weights * act, dim=1, keepdim=True)    # (1, 1, h_feat, w_feat)
            cam = F.relu(cam)

            cam_resized = F.interpolate(cam, size=(h_out, w_out), mode="bilinear", align_corners=False)
            cam_np = cam_resized.squeeze().detach().cpu().numpy()

            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.zeros_like(cam_np)
            cams.append(cam_np.astype(np.float32))

        if not cams:
            return np.zeros((h_out, w_out), dtype=np.float32)

        cam_np = np.mean(np.stack(cams, axis=0), axis=0)

        # Re-normalize the multi-layer average to [0, 1]
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
