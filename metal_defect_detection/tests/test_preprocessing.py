"""
Unit tests for CLAHE, Bilateral filtering, and Preprocessing Pipeline.
"""

import numpy as np

from src.preprocessing.bilateral import BilateralFilter
from src.preprocessing.clahe import CLAHEEnhancer
from src.preprocessing.pipeline import DefectPreprocessor


def test_clahe_enhancement():
    enhancer = CLAHEEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Test grayscale
    gray_img = np.random.randint(50, 150, (200, 200), dtype=np.uint8)
    enhanced_gray = enhancer(gray_img)
    assert enhanced_gray.shape == gray_img.shape
    assert enhanced_gray.dtype == np.uint8

    # Test RGB color image preserves channel order (no R-B swap)
    rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_img[:, :, 0] = 200  # High Red
    rgb_img[:, :, 2] = 20   # Low Blue
    enhanced_rgb = enhancer(rgb_img)
    assert enhanced_rgb.shape == rgb_img.shape
    # Red channel must remain significantly higher than Blue channel
    assert enhanced_rgb[:, :, 0].mean() > enhanced_rgb[:, :, 2].mean() + 50


def test_bilateral_filtering():
    bfilter = BilateralFilter(diameter=5, sigma_color=50.0, sigma_space=50.0)
    
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    filtered = bfilter(img)
    assert filtered.shape == img.shape
    assert filtered.dtype == np.uint8


def test_defect_preprocessor_pipeline():
    pipeline = DefectPreprocessor(
        use_clahe=True,
        clahe_clip_limit=2.0,
        use_bilateral=True,
        bilateral_diameter=5,
        target_size=(200, 200),
    )
    
    img = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    processed = pipeline(img)
    
    assert processed.shape == (200, 200, 3)
    assert processed.dtype == np.uint8

    # Test intermediate stages extraction
    stages = pipeline.get_stages(img)
    assert "raw" in stages
    assert "bilateral" in stages
    assert "clahe_only" in stages
    assert "enhanced" in stages
    for v in stages.values():
        assert v.shape == (200, 200, 3)


def test_defect_preprocessor_from_yaml(tmp_path):
    # Test loading from default configs/preprocessing.yaml
    prep = DefectPreprocessor.from_yaml()
    assert prep.clahe is not None
    assert prep.clahe.clip_limit == 2.0
    assert prep.bilateral is not None
    assert prep.bilateral.diameter == 5

    # Test that modifying the YAML dynamically alters the preprocessor
    custom_yaml = tmp_path / "custom_prep.yaml"
    custom_yaml.write_text("""
clahe:
  enabled: true
  clip_limit: 4.5
  tile_grid_size: [16, 16]
bilateral_filter:
  enabled: true
  diameter: 9
  sigma_color: 75.0
  sigma_space: 75.0
normalization:
  target_size: [320, 320]
""")
    custom_prep = DefectPreprocessor.from_yaml(custom_yaml)
    assert custom_prep.clahe is not None
    assert custom_prep.bilateral is not None
    assert custom_prep.clahe.clip_limit == 4.5
    assert custom_prep.clahe.tile_grid_size == (16, 16)
    assert custom_prep.bilateral.diameter == 9
    assert custom_prep.bilateral.sigma_color == 75.0
    assert custom_prep.target_size == (320, 320)

