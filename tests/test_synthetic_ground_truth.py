"""Synthetic Gaussian ground truth and validation metrics (REQ-VAL-01, REQ-VAL-02)."""

import numpy as np
import pytest
import SimpleITK as sitk

from src.qc.create_synthetic_patients import apply_gauss3d_shift
from src.qc.run_synthetic_sweep import compute_metrics


def test_gaussian_field_peak_direction_and_support(phantom):
    img, mask = phantom
    _, field = apply_gauss3d_shift(img, mask, [1, 0, 0], A_mm=5.0, sigma_mm=20.0)
    arr = sitk.GetArrayFromImage(field)          # (Z, Y, X, 3)
    mag = np.linalg.norm(arr, axis=-1)
    inside = sitk.GetArrayFromImage(mask) > 0

    assert mag.max() == pytest.approx(5.0, rel=0.05)     # peak = amplitude
    assert np.all(mag[~inside] == 0)                     # zero outside the brain
    assert np.allclose(arr[..., 1:], 0.0)                # pure X shift


def test_gaussian_shift_changes_image(phantom):
    img, mask = phantom
    shifted, _ = apply_gauss3d_shift(img, mask, [0, 1, 0], A_mm=5.0, sigma_mm=20.0)
    diff = np.abs(sitk.GetArrayFromImage(shifted) - sitk.GetArrayFromImage(img))
    assert diff.mean() > 1.0


def _write_pair(tmp_path, gt_field, pred_arr, mask):
    gt_p, pred_p, mask_p = (tmp_path / n for n in ("gt.nii.gz", "pred.nii.gz", "mask.nii.gz"))
    sitk.WriteImage(gt_field, str(gt_p))
    pred = sitk.GetImageFromArray(pred_arr.astype(np.float32), isVector=True)
    pred.CopyInformation(gt_field)
    sitk.WriteImage(pred, str(pred_p))
    sitk.WriteImage(mask, str(mask_p))
    return pred_p, gt_p, mask_p


def test_metrics_perfect_recovery_is_zero(tmp_path, phantom):
    img, mask = phantom
    _, gt = apply_gauss3d_shift(img, mask, [1, 1, 0], A_mm=5.0, sigma_mm=20.0)
    gt_arr = sitk.GetArrayFromImage(gt)
    # Convention: GT is a pull-back field, so perfect recovery is pred = -GT
    m = compute_metrics(*_write_pair(tmp_path, gt, -gt_arr, mask))
    assert m["epe"]["mean"] == pytest.approx(0.0, abs=1e-5)
    assert m["dot_ratio"]["mean"] == pytest.approx(1.0, abs=1e-4)
    assert m["angular_err"]["mean"] == pytest.approx(0.0, abs=0.1)


def test_metrics_no_recovery_equals_gt_magnitude(tmp_path, phantom):
    img, mask = phantom
    _, gt = apply_gauss3d_shift(img, mask, [0, 0, 1], A_mm=5.0, sigma_mm=20.0)
    gt_arr = sitk.GetArrayFromImage(gt)
    m = compute_metrics(*_write_pair(tmp_path, gt, np.zeros_like(gt_arr), mask))
    mag = np.linalg.norm(gt_arr, axis=-1)
    sig = mag[(sitk.GetArrayFromImage(mask) > 0) & (mag > 0.5)]
    assert m["epe"]["mean"] == pytest.approx(sig.mean(), rel=1e-4)
    assert m["dot_ratio"]["mean"] == pytest.approx(0.0, abs=1e-6)
