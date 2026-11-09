"""
BrainExtractor fallback using SimpleITK (Otsu + morphology).
"""

from __future__ import annotations

import SimpleITK as sitk

from src.backends.base import BrainExtractor
from src.core import setup_logger

logger = setup_logger(__name__)


class OtsuExtractor(BrainExtractor):
    """Approximate brain extraction via Otsu thresholding + morphological ops."""

    def extract(self, image: sitk.Image) -> sitk.Image:
        logger.info("  [OtsuExtractor] Running SimpleITK fallback brain extraction...")
        smooth = sitk.SmoothingRecursiveGaussian(image, 2.0)
        otsu = sitk.OtsuThreshold(sitk.Cast(smooth, sitk.sitkFloat32), 0, 1)
        closed = sitk.BinaryMorphologicalClosing(sitk.Cast(otsu, sitk.sitkUInt8), [3, 3, 3])

        cc = sitk.ConnectedComponent(closed)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        labels_sorted = sorted(stats.GetLabels(), key=lambda l: -stats.GetNumberOfPixels(l))

        if labels_sorted:
            mask = sitk.Cast(sitk.Equal(cc, labels_sorted[0]), sitk.sitkUInt8)
        else:
            mask = closed

        return sitk.BinaryFillhole(mask)
