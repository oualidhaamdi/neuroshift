"""
BrainExtractor implementation using HD-BET (Neural Network).
High-quality skull stripping using deep learning.
"""

from __future__ import annotations

import os
import tempfile
import SimpleITK as sitk

from src.backends.base import BrainExtractor
from src.core import setup_logger

logger = setup_logger(__name__)

class HDBetExtractor(BrainExtractor):
    """Deep-learning based brain extraction using HD-BET."""

    def __init__(self, mode: str = 'fast', device: int | str = 0):
        """
        Args:
            mode: 'fast' or 'accurate'.
            device: GPU ID (e.g. 0) or 'cpu'.
        """
        self.mode = mode
        self.device = device

    def extract(self, image: sitk.Image) -> sitk.Image:
        logger.info(f"  [HDBetExtractor] Running HD-BET (mode={self.mode}, device={self.device})...")
        
        try:
            from HD_BET.hd_bet import run_hd_bet
        except ImportError:
            logger.error("HD_BET library not installed. Please install it to use this backend.")
            raise

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.nii.gz")
            sitk.WriteImage(image, input_path)
            output_base = os.path.join(tmpdir, "output")

            run_hd_bet(
                input_path,
                output_base,
                mode=self.mode,
                device=self.device,
                postprocess=True,
                save_mask=True,
            )

            mask_path = output_base + "_mask.nii.gz"
            if os.path.exists(mask_path):
                mask = sitk.ReadImage(mask_path)
                return sitk.Cast(mask, sitk.sitkUInt8)
            
            raise FileNotFoundError("HD-BET failed to produce a mask file.")
