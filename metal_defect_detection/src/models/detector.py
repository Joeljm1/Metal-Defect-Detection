"""
Integrated Defect Detector Supporting Controlled Ablation Matrix (M1, M2, M3, M4).

Configurations:
- M1: Baseline Detector (YOLOv5s) - Raw images, standard PANet neck.
- M2: Baseline + CLAHE & Bilateral Preprocessing - Preprocessed images, standard PANet neck.
- M3: Baseline + Multi-Scale Attention - Raw images, PANet neck with ECA + Spatial attention.
- M4: Proposed Integrated Architecture - CLAHE/Bilateral Preprocessing + ECA + Spatial attention.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from src.models.backbone import CSPDarknetBackbone
from src.models.head import DetectHead
from src.models.neck import PANetNeck
from src.preprocessing.pipeline import DefectPreprocessor, load_preprocessing_config
from src.utils.box_ops import non_max_suppression

_VARIANT_CFG_FILES = {
    "m1": "model_m1_baseline.yaml",
    "m2": "model_m2_clahe.yaml",
    "m3": "model_m3_attention.yaml",
    "m4": "model_m4_integrated.yaml",
}


def _resolve_variant(variant: str) -> tuple[str, str, bool, bool]:
    """Map a variant name to (config key, display name, use_attention, use_preprocessing)."""
    v = variant.upper().strip()
    if "M4" in v or "INTEGRATED" in v:
        return "m4", "M4_Proposed_Integrated", True, True
    if "M3" in v or "ATTENTION" in v:
        return "m3", "M3_Baseline_Attention", True, False
    if "M2" in v or "CLAHE" in v or "PREPROCESSING" in v:
        return "m2", "M2_Baseline_Preprocessing", False, True
    if "M1" in v or "BASELINE" in v:
        return "m1", "M1_Baseline", False, False
    raise ValueError(f"Unknown variant '{variant}'. Expected one of: M1, M2, M3, M4")


def load_variant_config(variant: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Load the YAML configuration for an ablation variant (M1-M4).

    Returns the full config dict (``{}`` if no config file is found). Used both
    by :meth:`DefectDetector.build_model` (``model:`` section) and the training
    scripts (``training:`` section).
    """
    var_key, _, _, _ = _resolve_variant(variant)
    search_dirs = []
    if config_dir:
        search_dirs.append(Path(config_dir))
    search_dirs.extend([
        Path("configs"),
        Path("../configs"),
        Path(__file__).resolve().parent.parent.parent / "configs",
    ])

    for d in search_dirs:
        p = d / _VARIANT_CFG_FILES[var_key]
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError):
                continue
    return {}


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
        use_eca: bool = True,
        use_spatial: bool = True,
        use_clahe: bool = True,
        use_bilateral: bool = True,
        eca_gamma: int = 2,
        eca_b: int = 1,
        spatial_kernel: int = 7,
        anchors: Any = None,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.50,
        variant_name: str = "M1_Baseline",
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        img_size: tuple[int, int] = (200, 200),
    ):
        super().__init__()
        self.loaded_checkpoint: str | None = None
        self.variant_name = variant_name
        self.use_attention = use_attention and (use_eca or use_spatial)
        self.use_preprocessing = use_preprocessing and (use_clahe or use_bilateral)
        self.use_eca = use_eca
        self.use_spatial = use_spatial
        self.use_clahe = use_clahe
        self.use_bilateral = use_bilateral
        self.num_classes = num_classes
        self.img_size = img_size

        # Preprocessor instance
        self.preprocessor = (
            DefectPreprocessor.from_yaml(
                use_clahe=use_clahe,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid=clahe_tile_grid,
                use_bilateral=use_bilateral,
                bilateral_diameter=bilateral_d,
                bilateral_sigma_color=bilateral_sigma_color,
                bilateral_sigma_space=bilateral_sigma_space,
                target_size=img_size,
            )
            if self.use_preprocessing
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
            use_attention=self.use_attention,
            use_eca=use_eca,
            use_spatial=use_spatial,
            eca_gamma=eca_gamma,
            eca_b=eca_b,
            spatial_kernel=spatial_kernel,
        )

        # Head: Multi-scale detection head with dynamic stride support
        self.head = DetectHead(
            num_classes=num_classes,
            anchors=anchors,
            in_channels=self.neck.out_channels,
            img_size=img_size,
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
        self, x: torch.Tensor
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass on a batch of (possibly preprocessed) image tensors.

        Preprocessing is NOT applied here: during training/evaluation the
        DataLoader applies CLAHE + bilateral filtering (M2/M4), and single-image
        inference goes through :meth:`predict`, which preprocesses exactly once.
        This keeps a single preprocessing path and prevents double-filtering.

        Args:
            x: Input image tensor (B, 3, H, W) in [0, 1].
        """
        features = self.backbone(x)
        fused_features = self.neck(features)
        img_size = (x.shape[3], x.shape[2]) if x.dim() >= 4 else getattr(self, "img_size", (200, 200))
        return self.head(fused_features, img_size=img_size)

    def predict(
        self,
        image: np.ndarray | torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> torch.Tensor:
        """
        Standalone end-to-end inference on a single image.

        Accepts a raw uint8 numpy image in RGB channel order (convert BGR output
        of ``cv2.imread`` with ``cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`` first)
        or a float tensor; applies preprocessing exactly once (if M2/M4), runs
        the network, and applies NMS.
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
            decoded, _ = self.forward(tensor_in)
            nms_results = non_max_suppression(
                decoded, conf_thres=conf_thres, iou_thres=iou_thres
            )

        return nms_results[0]

    @classmethod
    def build_model(
        cls, variant: str = "M1", config_dir: str | Path | None = None, num_classes: int = 6
    ) -> "DefectDetector":
        """
        Factory method to construct ablation models M1, M2, M3, or M4,
        loading parameters from YAML configurations in configs/.
        """
        _, default_name, default_att, default_prep = _resolve_variant(variant)
        cfg_data = load_variant_config(variant, config_dir=config_dir)
        m_cfg = cfg_data.get("model", {})
        p_cfg = m_cfg.get("preprocessing", {})
        a_cfg = m_cfg.get("attention", {})

        # Load global preprocessing.yaml as single source of truth for preprocessing params
        prep_yaml = load_preprocessing_config(
            Path(config_dir) / "preprocessing.yaml" if config_dir else None
        )
        clahe_yaml = prep_yaml.get("clahe", {})
        bilateral_yaml = prep_yaml.get("bilateral_filter", {})
        norm_cfg = prep_yaml.get("normalization", {})
        target_size = tuple(norm_cfg.get("target_size", (200, 200)))

        clahe_clip = clahe_yaml.get("clip_limit", p_cfg.get("clahe_clip_limit", 2.0))
        clahe_grid = tuple(clahe_yaml.get("tile_grid_size", p_cfg.get("clahe_tile_grid", (8, 8))))
        b_d = bilateral_yaml.get("diameter", p_cfg.get("bilateral_d", 5))
        b_sc = bilateral_yaml.get("sigma_color", p_cfg.get("bilateral_sigma_color", 50.0))
        b_ss = bilateral_yaml.get("sigma_space", p_cfg.get("bilateral_sigma_space", 50.0))

        cfg_use_eca = a_cfg.get("use_eca", default_att)
        cfg_use_spatial = a_cfg.get("use_spatial", default_att)
        cfg_use_clahe = p_cfg.get("use_clahe", default_prep)
        cfg_use_bilateral = p_cfg.get("use_bilateral", default_prep)

        return cls(
            num_classes=m_cfg.get("num_classes", num_classes),
            in_channels=m_cfg.get("in_channels", 3),
            use_attention=cfg_use_eca or cfg_use_spatial,
            use_preprocessing=cfg_use_clahe or cfg_use_bilateral,
            use_eca=cfg_use_eca,
            use_spatial=cfg_use_spatial,
            use_clahe=cfg_use_clahe,
            use_bilateral=cfg_use_bilateral,
            eca_gamma=a_cfg.get("eca_gamma", 2),
            eca_b=a_cfg.get("eca_b", 1),
            spatial_kernel=a_cfg.get("spatial_kernel_size", 7),
            anchors=m_cfg.get("anchors", None),
            variant_name=m_cfg.get("name", default_name),
            clahe_clip_limit=clahe_clip,
            clahe_tile_grid=clahe_grid,
            bilateral_d=b_d,
            bilateral_sigma_color=b_sc,
            bilateral_sigma_space=b_ss,
            img_size=target_size,
        )
