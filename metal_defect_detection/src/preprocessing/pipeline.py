"""
Unified Image Preprocessing & Enhancement Pipeline.

Combines:
1. CLAHE (Contrast-Limited Adaptive Histogram Equalization) for local illumination normalization.
2. Bilateral Filtering for edge-preserving denoising.
3. Contrast & intensity normalization.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

from src.preprocessing.clahe import CLAHEEnhancer
from src.preprocessing.bilateral import BilateralFilter


class DefectPreprocessor:
    """
    Composite pipeline for metallic surface defect image preprocessing.
    """

    def __init__(
        self,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        use_bilateral: bool = True,
        bilateral_diameter: int = 5,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        target_size: Optional[Tuple[int, int]] = (200, 200),
        color_format: str = "RGB",
    ):
        self.use_clahe = use_clahe
        self.use_bilateral = use_bilateral
        self.target_size = target_size
        self.color_format = color_format.upper()

        self.clahe = (
            CLAHEEnhancer(
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid,
                color_format=color_format,
            )
            if use_clahe
            else None
        )
        self.bilateral = (
            BilateralFilter(
                diameter=bilateral_diameter,
                sigma_color=bilateral_sigma_color,
                sigma_space=bilateral_sigma_space,
            )
            if use_bilateral
            else None
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Execute preprocessing pipeline on an image.
        Pipeline sequence:
        1. Resize to target size (if provided and different)
        2. Bilateral Filtering (edge-preserving denoising)
        3. CLAHE (local contrast enhancement)
        
        Args:
            image: Input uint8 image numpy array.
            
        Returns:
            Preprocessed uint8 image numpy array.
        """
        if image is None:
            raise ValueError("Input image is None")

        img = image.copy()

        # Resize if specified
        if self.target_size is not None:
            h, w = img.shape[:2]
            if (w, h) != tuple(self.target_size):
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Apply Bilateral Filter first to suppress high frequency noise before contrast stretch
        if self.use_bilateral and self.bilateral is not None:
            img = self.bilateral.apply(img)

        # Apply CLAHE to enhance defect features
        if self.use_clahe and self.clahe is not None:
            img = self.clahe.apply(img)

        return img

    def get_stages(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process and return intermediate stages for visualization and ablation analysis.
        
        Returns dict with keys: 'raw', 'bilateral', 'clahe_only', 'enhanced'
        """
        stages = {"raw": image.copy()}

        if self.target_size is not None:
            h, w = image.shape[:2]
            if (w, h) != tuple(self.target_size):
                stages["raw"] = cv2.resize(image, self.target_size)

        bilateral_filter = self.bilateral or BilateralFilter()
        clahe_enhancer = self.clahe or CLAHEEnhancer(color_format=self.color_format)

        stages["bilateral"] = bilateral_filter.apply(stages["raw"])
        stages["clahe_only"] = clahe_enhancer.apply(stages["raw"])
        stages["enhanced"] = clahe_enhancer.apply(stages["bilateral"])

        return stages

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.process(image)
