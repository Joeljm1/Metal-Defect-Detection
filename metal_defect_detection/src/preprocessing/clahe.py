"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Module.

In metallic surface defect detection, uneven lighting and severe specular reflections
frequently cause loss of defect contrast. CLAHE divides the image into contextual tiles,
computes histogram equalization locally, and limits contrast amplification to prevent
over-amplifying noise.
"""

import cv2
import numpy as np


class CLAHEEnhancer:
    """
    CLAHE image contrast enhancement for metallic surface defect images.
    
    Supports both grayscale and BGR/RGB images. For color images, enhancement
    is applied to the Luminance (L) channel in CIELAB color space to enhance
    contrast without introducing chromatic distortion.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        color_format: str = "RGB",
    ):
        """
        Args:
            clip_limit: Threshold for contrast limiting (default 2.0).
            tile_grid_size: Size of grid for histogram equalization (default 8x8).
            color_format: Color space of 3-channel input images, either "RGB" (default) or "BGR".
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.color_format = color_format.upper()
        self.clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tuple(tile_grid_size)
        )

    def __deepcopy__(self, memo):
        return CLAHEEnhancer(
            clip_limit=self.clip_limit,
            tile_grid_size=self.tile_grid_size,
            color_format=self.color_format,
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement to an image.
        
        Args:
            image: Input image as uint8 numpy array (H, W) or (H, W, 3) in RGB (default) or BGR.
            
        Returns:
            Enhanced image with identical shape, dtype uint8, and color space.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        if len(image.shape) == 2:
            # Grayscale image
            return self.clahe.apply(image)

        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                return self.clahe.apply(image[:, :, 0])[:, :, np.newaxis]
            elif image.shape[2] == 3:
                # Convert to LAB color space based on input color format
                if self.color_format == "RGB":
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    enhanced_l = self.clahe.apply(l_channel)
                    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
                    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
                else:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    enhanced_l = self.clahe.apply(l_channel)
                    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
                    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported image dimension: {image.shape}")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)
