"""
Bilateral Filtering Module for Edge-Preserving Denoising.

Metal component surfaces have micro-texture noise and grain from rolling and milling.
Standard Gaussian blurring blurs out fine defect boundaries (e.g. subtle crazing lines).
Bilateral filtering smooths high-frequency grain noise while strictly preserving sharp
defect edges by weighting pixels based on both spatial distance and radiometric intensity difference.
"""

from typing import Union
import cv2
import numpy as np


class BilateralFilter:
    """
    Bilateral filter for edge-preserving smoothing of metallic surface images.
    """

    def __init__(
        self,
        diameter: int = 5,
        sigma_color: float = 50.0,
        sigma_space: float = 50.0,
    ):
        """
        Args:
            diameter: Diameter of each pixel neighborhood used during filtering.
                      If non-positive, it is computed from sigma_space.
            sigma_color: Filter sigma in color/intensity space. Larger values mean
                         farther colors within the neighborhood will be mixed together.
            sigma_space: Filter sigma in the coordinate space. Larger values mean
                         farther pixels will influence each other.
        """
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to the image.
        
        Args:
            image: Input uint8 numpy array (H, W) or (H, W, 3).
            
        Returns:
            Filtered image with identical shape and uint8 dtype.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        return cv2.bilateralFilter(
            image,
            d=self.diameter,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)
