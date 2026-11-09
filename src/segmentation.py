"""
SEGMENTATION MODULE - Automated Brain & Cavity Extraction.
"""

import SimpleITK as sitk
import numpy as np

from src.core import setup_logger
from src.backends.bet_sitk import OtsuExtractor

logger = setup_logger(__name__)

class SegmentationModule:
    """Handles automated segmentation of brain and surgical cavity."""

    def __init__(self, brain_extractor=None):
        """
        Args:
            brain_extractor: A BrainExtractor instance. Defaults to HDBet with Otsu fallback.
        """
        if brain_extractor is None:
            try:
                from src.backends.bet_hdbet import HDBetExtractor
                brain_extractor = HDBetExtractor()
            except (ImportError, Exception) as e:
                logger.warning(f"HDBetExtractor unavailable ({e}). Falling back to Otsu.")
                brain_extractor = OtsuExtractor()
        self.brain_extractor = brain_extractor
        self._fallback = OtsuExtractor()

    def extract_brain_mask(self, image: sitk.Image) -> sitk.Image:
        """Extract brain mask using the primary extractor with fallback."""
        try:
            return self.brain_extractor.extract(image)
        except Exception as e:
            logger.warning(f"  [Seg] Primary extractor failed ({e}). Falling back to Otsu.")
            return self._fallback.extract(image)

    def segment_cavity(self, fixed: sitk.Image, moving_registered: sitk.Image,
                       brain_mask: sitk.Image, threshold_std: float = 2.5,
                       min_object_size: int = 500) -> sitk.Image:
        """
        Segment the resection cavity area by analyzing the difference between
        pre-op and post-op registered images.
        """
        logger.info("  [Seg] Segmenting surgical cavity area...")

        # 1. Intensity difference
        diff = sitk.Abs(fixed - moving_registered)
        diff_masked = sitk.Mask(diff, brain_mask)

        # 2. Statistics-based thresholding
        diff_np = sitk.GetArrayViewFromImage(diff_masked)
        mask_np = sitk.GetArrayViewFromImage(brain_mask)
        valid_diffs = diff_np[mask_np > 0]
        
        threshold_val = float(np.mean(valid_diffs) + (threshold_std * np.std(valid_diffs)))
        upper_val = float(np.max(valid_diffs))

        if threshold_val >= upper_val:
            logger.warning("  [Seg] No cavity detected (threshold too high).")
            empty = sitk.Image(fixed.GetSize(), sitk.sitkUInt8)
            empty.CopyInformation(fixed)
            return empty

        cavity_mask = sitk.BinaryThreshold(
            diff_masked, lowerThreshold=threshold_val, upperThreshold=upper_val + 1.0, 
            insideValue=1, outsideValue=0
        )

        # 3. Clean-up: Morphological opening + Connected Components
        kernel = [3] * fixed.GetDimension()
        cavity_clean = sitk.BinaryMorphologicalOpening(cavity_mask, kernel)

        # Relabel and filter by size
        cc = sitk.ConnectedComponent(cavity_clean)
        final = sitk.RelabelComponent(cc, min_object_size)
        
        # Keep only the largest component (label 1)
        final_mask = sitk.BinaryThreshold(final, lowerThreshold=1, upperThreshold=1)

        logger.info("  [Seg] Cavity mask generated.")
        return final_mask
