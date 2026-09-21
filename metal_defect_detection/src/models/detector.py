"""
Integrated Defect Detector Supporting Controlled Ablation Matrix (M1, M2, M3, M4).

Configurations:
- M1: Baseline Detector (YOLOv5s) - Raw images, standard PANet neck.
- M2: Baseline + CLAHE & Bilateral Preprocessing - Preprocessed images, standard PANet neck.
- M3: Baseline + Multi-Scale Attention - Raw images, PANet neck with ECA + Spatial attention.
- M4: Proposed Integrated Architecture - CLAHE/Bilateral Preprocessing + ECA + Spatial attention.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import yaml
import numpy as np
import torch
import torch.nn as nn

from src.models.backbone import CSPDarknetBackbone
from src.models.neck import PANetNeck
from src.models.head import DetectHead
from src.preprocessing.pipeline import DefectPreprocessor
from src.utils.box_ops import non_max_suppression


class DefectDetector(nn.Module):
    """
    End-to-end defect detection model for metal surfaces.
    Supports M1, M2, M3, and M4 ablation configurations.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        use_attention: bool = False,
        use_preprocessing: bool = False,
        eca_gamma: int = 2,
        eca_b: int = 1,
        spatial_kernel: int = 7,
        anchors: Optional[List[List[Tuple[float, float]]]] = None,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.50,
        variant_name: str = "M1_Baseline",
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
    ):
        super().__init__()
        self.variant_name = variant_name
        self.use_attention = use_attention
        self.use_preprocessing = use_preprocessing
        self.num_classes = num_classes

        # Preprocessor instance
        self.preprocessor = (
            DefectPreprocessor(
                use_clahe=True,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid=clahe_tile_grid,
                use_bilateral=True,
                bilateral_diameter=bilateral_d,
                bilateral_sigma_color=bilateral_sigma_color,
                bilateral_sigma_space=bilateral_sigma_space,
            )
            if use_preprocessing
            else None
        )

        # Backbone: CSPDarknet
        self.backbone = CSPDarknetBackbone(
            in_channels=in_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
        )

        # Neck: PANet with optional attention
        self.neck = PANetNeck(
            in_channels=self.backbone.out_channels,
            use_attention=use_attention,
            eca_gamma=eca_gamma,
            eca_b=eca_b,
            spatial_kernel=spatial_kernel,
        )

        # Head: Multi-scale detection head
        self.head = DetectHead(
            num_classes=num_classes,
            anchors=anchors,
            in_channels=self.neck.out_channels,
        )

    def preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies CLAHE + Bilateral filtering to a batch of image tensors (B, C, H, W) in [0, 1].
        """
        if not self.use_preprocessing or self.preprocessor is None:
            return x

        device = x.device
        processed_list = []
        for i in range(x.shape[0]):
            # Convert CHW [0, 1] tensor to HWC uint8 numpy array
            img_np = (x[i].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            enhanced = self.preprocessor.process(img_np)
            # Convert back to tensor [0, 1]
            enhanced_t = torch.from_numpy(enhanced).permute(2, 0, 1).float() / 255.0
            processed_list.append(enhanced_t)

        return torch.stack(processed_list, dim=0).to(device)

    def forward(
        self, x: torch.Tensor, apply_preprocessor: bool = False
    ) -> List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input image tensor (B, 3, H, W) in [0, 1].
            apply_preprocessor: If True and model uses preprocessing (M2/M4),
                               runs CLAHE + Bilateral filtering before backbone.
        """
        if apply_preprocessor and self.use_preprocessing:
            x = self.preprocess_tensor(x)

        features = self.backbone(x)
        fused_features = self.neck(features)
        return self.head(fused_features)

    def predict(
        self,
        image: Union[np.ndarray, torch.Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> torch.Tensor:
        """
        Standalone end-to-end inference on a single image.
        Accepts raw uint8 image numpy array or float tensor, applies preprocessing (if M2/M4),
        runs network, and applies NMS.
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(image, np.ndarray):
            img_in = image.copy()
            if self.use_preprocessing and self.preprocessor is not None:
                img_in = self.preprocessor.process(img_in)
            # To tensor
            tensor_in = (
                torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ).to(device)
        else:
            tensor_in = image if image.dim() == 4 else image.unsqueeze(0)
            if self.use_preprocessing:
                tensor_in = self.preprocess_tensor(tensor_in)
            tensor_in = tensor_in.to(device)

        with torch.no_grad():
            decoded, _ = self.forward(tensor_in, apply_preprocessor=False)
            nms_results = non_max_suppression(
                decoded, conf_thres=conf_thres, iou_thres=iou_thres
            )

        return nms_results[0]

    @classmethod
    def build_model(
        cls, variant: str = "M1", config_dir: Optional[Union[str, Path]] = None, num_classes: int = 6
    ) -> "DefectDetector":
        """
        Factory method to construct ablation models M1, M2, M3, or M4,
        loading parameters from YAML configurations in configs/.
        """
        v = variant.upper().strip()
        var_key = "m1"
        if "M4" in v or "INTEGRATED" in v:
            var_key = "m4"
            default_name = "M4_Proposed_Integrated"
            default_att = True
            default_prep = True
        elif "M3" in v or "ATTENTION" in v:
            var_key = "m3"
            default_name = "M3_Baseline_Attention"
            default_att = True
            default_prep = False
        elif "M2" in v or "CLAHE" in v or "PREPROCESSING" in v:
            var_key = "m2"
            default_name = "M2_Baseline_Preprocessing"
            default_att = False
            default_prep = True
        elif "M1" in v or "BASELINE" in v:
            var_key = "m1"
            default_name = "M1_Baseline"
            default_att = False
            default_prep = False
        else:
            raise ValueError(f"Unknown variant '{variant}'. Expected one of: M1, M2, M3, M4")

        # Resolve config path
        cfg_files = {
            "m1": "model_m1_baseline.yaml",
            "m2": "model_m2_clahe.yaml",
            "m3": "model_m3_attention.yaml",
            "m4": "model_m4_integrated.yaml",
        }

        # Search paths for config
        search_dirs = []
        if config_dir:
            search_dirs.append(Path(config_dir))
        search_dirs.extend([
            Path("configs"),
            Path("../configs"),
            Path(__file__).resolve().parent.parent.parent / "configs",
        ])

        cfg_data = {}
        for d in search_dirs:
            p = d / cfg_files[var_key]
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        cfg_data = yaml.safe_load(f) or {}
                    break
                except Exception:
                    pass

        m_cfg = cfg_data.get("model", {})
        p_cfg = m_cfg.get("preprocessing", {})
        a_cfg = m_cfg.get("attention", {})

        return cls(
            num_classes=m_cfg.get("num_classes", num_classes),
            in_channels=m_cfg.get("in_channels", 3),
            use_attention=a_cfg.get("use_eca", default_att) or a_cfg.get("use_spatial", default_att),
            use_preprocessing=p_cfg.get("use_clahe", default_prep) or p_cfg.get("use_bilateral", default_prep),
            eca_gamma=a_cfg.get("eca_gamma", 2),
            eca_b=a_cfg.get("eca_b", 1),
            spatial_kernel=a_cfg.get("spatial_kernel_size", 7),
            anchors=m_cfg.get("anchors", None),
            variant_name=m_cfg.get("name", default_name),
            clahe_clip_limit=p_cfg.get("clahe_clip_limit", 2.0),
            clahe_tile_grid=tuple(p_cfg.get("clahe_tile_grid", (8, 8))),
            bilateral_d=p_cfg.get("bilateral_d", 5),
            bilateral_sigma_color=p_cfg.get("bilateral_sigma_color", 50.0),
            bilateral_sigma_space=p_cfg.get("bilateral_sigma_space", 50.0),
        )
