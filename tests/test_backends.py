"""SimpleITK fallback backends and FSL detection (REQ-PRE-01..03, REQ-SYS-01)."""

import numpy as np
import SimpleITK as sitk

from src.backends.bet_sitk import OtsuExtractor
from src.backends.rigid_sitk import Euler3DAligner
from src.backends.segmenter_sitk import KMeansSegmenter
from src.fsl_wrapper import check_fsl_availability


def _dice(a, b):
    a, b = a > 0, b > 0
    return 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


def test_otsu_brain_extraction_matches_phantom(phantom):
    img, mask = phantom
    est = OtsuExtractor().extract(sitk.Cast(img, sitk.sitkFloat32))
    assert _dice(sitk.GetArrayFromImage(est), sitk.GetArrayFromImage(mask)) > 0.9


def test_kmeans_segmentation_produces_three_classes_inside_mask(phantom):
    img, mask = phantom
    seg = sitk.GetArrayFromImage(KMeansSegmenter().segment(sitk.Cast(img, sitk.sitkFloat32), mask))
    inside = sitk.GetArrayFromImage(mask) > 0
    assert set(np.unique(seg[inside])) <= {1, 2, 3, 4}
    assert len(np.unique(seg[inside])) >= 3
    assert np.all(seg[~inside] == 0)


def test_euler3d_recovers_small_translation(phantom):
    img, _ = phantom
    shift = (3.0, -2.0, 1.5)  # mm
    moving = sitk.Resample(img, img, sitk.TranslationTransform(3, shift), sitk.sitkLinear, 0.0)
    tx = Euler3DAligner(n_iterations=100, sampling_pct=0.5).align(img, moving)

    # The recovered transform maps fixed -> moving points: compare on a test point
    p = img.TransformContinuousIndexToPhysicalPoint((24.0, 24.0, 24.0))
    err = np.linalg.norm(np.subtract(tx.TransformPoint(p), np.add(p, shift)))
    assert err < 1.0  # mm


def test_fsl_check_never_raises():
    ok, msg = check_fsl_availability()
    assert isinstance(ok, bool) and isinstance(msg, str)
