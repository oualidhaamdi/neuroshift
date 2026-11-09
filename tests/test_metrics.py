"""Similarity and deformation metrics (REQ-MET-01, REQ-MET-02)."""

import numpy as np
import pytest
import SimpleITK as sitk

from src.metrics.deformation import compute_jacobian_determinant
from src.metrics.similarity import compute_mi, compute_ncc


def test_ncc_identical_is_one(phantom):
    arr = sitk.GetArrayFromImage(phantom[0])
    assert compute_ncc(arr, arr) == pytest.approx(1.0, abs=1e-6)


def test_ncc_inverted_is_minus_one(phantom):
    arr = sitk.GetArrayFromImage(phantom[0])
    assert compute_ncc(arr, -arr) == pytest.approx(-1.0, abs=1e-6)


def test_mi_self_higher_than_noise(phantom):
    arr = sitk.GetArrayFromImage(phantom[0])
    noise = np.random.default_rng(0).normal(size=arr.shape)
    assert compute_mi(arr, arr) > compute_mi(arr, noise)


def test_jacobian_identity_is_one(phantom):
    jac = compute_jacobian_determinant(sitk.Transform(3, sitk.sitkIdentity), phantom[0])
    assert np.allclose(jac, 1.0, atol=1e-6)


def test_jacobian_uniform_scaling(phantom):
    tx = sitk.ScaleTransform(3, (1.1, 1.1, 1.1))
    jac = compute_jacobian_determinant(tx, phantom[0])
    # Ignore the one-voxel border where finite differences are one-sided
    assert np.allclose(jac[2:-2, 2:-2, 2:-2], 1.1 ** 3, atol=1e-3)
