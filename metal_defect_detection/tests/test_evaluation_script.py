"""
Unit test for evaluate script workflow.
"""

import tempfile
from pathlib import Path
import torch

from src.models.detector import DefectDetector
from scripts.evaluate import run_evaluation


def test_evaluate_script_run():
    # Build model and save a temporary checkpoint
    model = DefectDetector.build_model(variant="M1")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "M1_test.pt"
        torch.save({
            "epoch": 1,
            "model_state": model.state_dict(),
            "variant": "M1",
        }, ckpt_path)

        # Run evaluation on NEU-DET data
        data_dir = Path("data/NEU-DET")
        if data_dir.exists():
            # Test that it executes without errors
            run_evaluation(model_path=ckpt_path, data_dir=data_dir, conf_thres=0.5, device_name="cpu")
