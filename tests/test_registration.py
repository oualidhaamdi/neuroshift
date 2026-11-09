"""
Deformable registration on the synthetic phantom (REQ-REG-01, REQ-REG-02, REQ-CFM-01).

These are end-to-end checks of the intensity-based B-Spline registrar on a
48^3 phantom: they verify that a known Gaussian displacement is recovered and
that Cost Function Masking (excluding a simulated surgical cavity from the
metric) does not make the result worse near the cavity.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from src.backends.deformable_bspline_intensity import BSplineIntensityRegistrar
from src.qc.create_synthetic_patients import apply_gauss3d_shift


def _predicted_field(tx, ref):
    disp = sitk.TransformToDisplacementField(
        tx, sitk.sitkVectorFloat64, ref.GetSize(), ref.GetOrigin(),
        ref.GetSpacing(), ref.GetDirection(),
    )
    return sitk.GetArrayFromImage(disp)


def _epe(pred, gt, region):
    # GT is a pull-back field: perfect recovery is pred = -GT
    return float(np.linalg.norm(pred + gt, axis=-1)[region].mean())


@pytest.fixture(scope="module")
def shifted_case(phantom):
    img, mask = phantom
    moving, gt = apply_gauss3d_shift(img, mask, [1, 0.5, 0], A_mm=5.0, sigma_mm=20.0)
    gt_arr = sitk.GetArrayFromImage(gt)
    region = (sitk.GetArrayFromImage(mask) > 0) & (np.linalg.norm(gt_arr, axis=-1) > 0.5)
    return img, mask, moving, gt_arr, region


def test_bspline_recovers_known_gaussian_shift(shifted_case, reg_config):
    fixed, mask, moving, gt, region = shifted_case
    tx = BSplineIntensityRegistrar().register(fixed, moving, mask, reg_config)
    pred = _predicted_field(tx, fixed)

    epe_before = _epe(np.zeros_like(gt), gt, region)
    epe_after = _epe(pred, gt, region)
    assert epe_after < 0.5 * epe_before, (epe_before, epe_after)


def test_bspline_field_is_diffeomorphic_on_phantom(shifted_case, reg_config):
    fixed, mask, moving, _, _ = shifted_case
    tx = BSplineIntensityRegistrar().register(fixed, moving, mask, reg_config)
    disp = sitk.TransformToDisplacementField(
        tx, sitk.sitkVectorFloat64, fixed.GetSize(), fixed.GetOrigin(),
        fixed.GetSpacing(), fixed.GetDirection(),
    )
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(disp))
    folding_pct = 100.0 * np.mean(jac[sitk.GetArrayFromImage(mask) > 0] <= 0)
    assert folding_pct < 0.5


def test_cost_function_masking_near_cavity(shifted_case, reg_config):
    fixed, mask, moving, gt, region = shifted_case

    # Simulated resection cavity: dark sphere present only in the moving image
    z, y, x = np.indices(gt.shape[:3])
    cavity = (x - 30) ** 2 + (y - 24) ** 2 + (z - 24) ** 2 < 5 ** 2
    mov_arr = sitk.GetArrayFromImage(moving)
    mov_arr[cavity] = 0.0
    moving_cav = sitk.GetImageFromArray(mov_arr)
    moving_cav.CopyInformation(moving)

    mask_arr = sitk.GetArrayFromImage(mask)
    cfm_mask = sitk.GetImageFromArray((mask_arr * ~cavity).astype(np.uint8))
    cfm_mask.CopyInformation(mask)

    reg = BSplineIntensityRegistrar()
    pred_plain = _predicted_field(reg.register(fixed, moving_cav, mask, reg_config), fixed)
    pred_cfm = _predicted_field(reg.register(fixed, moving_cav, cfm_mask, reg_config), fixed)

    # Evaluate in a shell around the cavity, where the intensity mismatch bites
    shell = (x - 30) ** 2 + (y - 24) ** 2 + (z - 24) ** 2 < 9 ** 2
    near = shell & region
    epe_plain, epe_cfm = _epe(pred_plain, gt, near), _epe(pred_cfm, gt, near)
    assert epe_cfm <= epe_plain * 1.05, (epe_plain, epe_cfm)
