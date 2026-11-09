"""
TissueSegmenter implementation using FSL FAST (HMRF-GMM).
"""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from src.backends.base import TissueSegmenter
from src.core import setup_logger
from src.fsl_wrapper import FSLWrapper

logger = setup_logger(__name__)

_MAX_DIM = 256  # downsample threshold for FSL FAST


class FASTSegmenter(TissueSegmenter):
    """Tissue segmentation via FSL FAST (Hidden Markov Random Field + GMM)."""

    def __init__(self, fsl: FSLWrapper, out_dir: Path, n_classes: int = 3):
        self.fsl = fsl
        self.out_dir = out_dir
        self.n_classes = n_classes

    def _run_fast_with_fallback(self, temp_in: Path, pref: Path, brain_img: sitk.Image):
        try:
            self.fsl.fast(temp_in, pref, n_classes=self.n_classes)
        except Exception as e:
            logger.warning(f"  [FASTSegmenter] 3-class segmentation failed ({e}). Trying 2 classes...")
            try:
                self.fsl.fast(temp_in, pref, n_classes=2)
            except Exception as e2:
                logger.warning(f"  [FASTSegmenter] 2-class failed ({e2}). Falling back to SimpleITK Otsu...")
                otsu = sitk.OtsuMultipleThresholdsImageFilter()
                otsu.SetNumberOfThresholds(2)
                seg = otsu.Execute(brain_img)
                sitk.WriteImage(sitk.Cast(seg, sitk.sitkUInt8), str(pref) + "_seg.nii.gz")

    def segment(self, brain_img: sitk.Image, mask: sitk.Image) -> sitk.Image:
        """
        Writes brain_img to disk and runs FSL FAST. If the image is larger
        than _MAX_DIM in-plane it is downsampled first and the result is
        upsampled back to original resolution with NearestNeighbor.
        """
        obj = brain_img
        sz = obj.GetSize()
        max_inplane = max(sz[0], sz[1])
        pref = self.out_dir / "fast_seg"

        if max_inplane > _MAX_DIM:
            factor = max_inplane / float(_MAX_DIM)
            new_sz = [max(1, int(sz[0] / factor)), max(1, int(sz[1] / factor)), sz[2]]
            new_sp = [obj.GetSpacing()[i] * sz[i] / new_sz[i] for i in range(3)]
            ds = sitk.Resample(obj, new_sz, sitk.Transform(), sitk.sitkLinear,
                               obj.GetOrigin(), new_sp, obj.GetDirection(), 0.0, obj.GetPixelID())
            temp_in = self.out_dir / "fast_input_ds.nii.gz"
            sitk.WriteImage(ds, str(temp_in))
            self.fsl._wait_for_file_wsl(str(temp_in))
            self._run_fast_with_fallback(temp_in, pref, ds)
            seg_ds = sitk.ReadImage(str(pref) + "_seg.nii.gz")
            seg = sitk.Resample(seg_ds, obj, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0.0, seg_ds.GetPixelID())
        else:
            tmp = self.out_dir / "fast_input.nii.gz"
            sitk.WriteImage(obj, str(tmp))
            self._run_fast_with_fallback(tmp, pref, obj)
            seg = sitk.ReadImage(str(pref) + "_seg.nii.gz")

        logger.info(f"  [FASTSegmenter] segmentation complete — {pref}_seg.nii.gz")
        return seg
