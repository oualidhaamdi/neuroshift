"""
TissueSegmenter fallback using SimpleITK K-Means.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from src.backends.base import TissueSegmenter
from src.core import setup_logger

logger = setup_logger(__name__)

_MAX_DIM = 256


class KMeansSegmenter(TissueSegmenter):
    """Tissue segmentation via SimpleITK K-Means (fallback when FSL is unavailable)."""

    def __init__(self, n_classes: int = 3):
        self.n_classes = n_classes

    def segment(self, brain_img: sitk.Image, mask: sitk.Image) -> sitk.Image:
        sz = brain_img.GetSize()
        max_inplane = max(sz[0], sz[1])

        if max_inplane > _MAX_DIM:
            factor = max_inplane / _MAX_DIM
            new_sz = [max(1, int(sz[0] / factor)), max(1, int(sz[1] / factor)), sz[2]]
            new_sp = [brain_img.GetSpacing()[i] * sz[i] / new_sz[i] for i in range(3)]
            brain_ds = sitk.Resample(brain_img, new_sz, sitk.Transform(), sitk.sitkLinear,
                                     brain_img.GetOrigin(), new_sp, brain_img.GetDirection(),
                                     0.0, brain_img.GetPixelID())
            mask_ds = sitk.Resample(mask, new_sz, sitk.Transform(), sitk.sitkNearestNeighbor,
                                    mask.GetOrigin(), new_sp, mask.GetDirection(),
                                    0.0, mask.GetPixelID())
            logger.info(f"  [KMeansSegmenter] K-Means on {new_sz} (downsample ×{factor:.1f})")
        else:
            brain_ds, mask_ds = brain_img, mask

        arr = sitk.GetArrayFromImage(brain_ds).astype(np.float32)
        marr = sitk.GetArrayFromImage(mask_ds).astype(np.uint8)

        brain_vals = arr[marr > 0]
        if len(brain_vals) == 0:
            logger.warning("  [KMeansSegmenter] Empty mask — returning zeros")
            out = sitk.Image(brain_img.GetSize(), sitk.sitkUInt8)
            out.CopyInformation(brain_img)
            return out

        p1, p99 = float(np.percentile(brain_vals, 1)), float(np.percentile(brain_vals, 99))
        arr_norm = np.clip((arr - p1) / (p99 - p1 + 1e-9) * 1000.0, 0, 1000).astype(np.float32)

        km_img = sitk.GetImageFromArray(arr_norm)
        km_img.CopyInformation(brain_ds)
        km_masked = sitk.Mask(km_img, sitk.Cast(mask_ds > 0, sitk.sitkUInt8))

        otsu = sitk.OtsuMultipleThresholdsImageFilter()
        otsu.SetNumberOfThresholds(self.n_classes)
        result_ds = otsu.Execute(km_masked)

        result_arr = sitk.GetArrayFromImage(result_ds).astype(np.uint8)
        seg_arr_ds = np.where(marr > 0, result_arr + 1, 0).astype(np.uint8)
        seg_ds = sitk.GetImageFromArray(seg_arr_ds)
        seg_ds.CopyInformation(brain_ds)

        if max_inplane > _MAX_DIM:
            seg = sitk.Resample(seg_ds, brain_img.GetSize(), sitk.Transform(),
                                sitk.sitkNearestNeighbor, brain_img.GetOrigin(),
                                brain_img.GetSpacing(), brain_img.GetDirection(),
                                0.0, sitk.sitkUInt8)
            seg_arr = sitk.GetArrayFromImage(seg).astype(np.uint8)
            seg_arr[sitk.GetArrayFromImage(mask).astype(np.uint8) == 0] = 0
            out = sitk.GetImageFromArray(seg_arr)
            out.CopyInformation(brain_img)
            return out

        return seg_ds
