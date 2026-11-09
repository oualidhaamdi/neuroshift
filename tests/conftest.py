"""
Shared fixtures: a small, fully synthetic brain-like phantom.

No patient data is used anywhere in the test-suite. The phantom is an
ellipsoidal "brain" with three nested intensity classes (CSF-, GM- and
WM-like) plus smooth texture, so that intensity-based registration has
gradients to work with. Volumes are small (48^3 voxels at 2.5 mm) so the
whole suite runs in seconds on a CI runner.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIZE = 48
SPACING = 2.5  # mm  -> 120 mm field of view


def _make_phantom(seed: int = 0) -> tuple[sitk.Image, sitk.Image]:
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*(np.arange(SIZE) - SIZE / 2 + 0.5,) * 3, indexing="ij")
    r = np.sqrt((x / 18.0) ** 2 + (y / 21.0) ** 2 + (z / 16.0) ** 2)

    img = np.zeros((SIZE,) * 3, dtype=np.float32)
    img[r < 1.00] = 300.0   # CSF-like rim
    img[r < 0.90] = 600.0   # GM-like
    img[r < 0.60] = 900.0   # WM-like core

    # Smooth texture so the B-Spline metric has gradients inside each class
    noise = rng.normal(0.0, 1.0, img.shape).astype(np.float32)
    tex = sitk.GetArrayFromImage(
        sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(noise), 2.0)
    )
    tex = tex / (np.abs(tex).max() + 1e-9)
    img = np.where(r < 1.0, img + 120.0 * tex, 0.0).astype(np.float32)

    image = sitk.GetImageFromArray(img)
    image.SetSpacing((SPACING,) * 3)
    mask = sitk.GetImageFromArray((r < 1.0).astype(np.uint8))
    mask.CopyInformation(image)
    return image, mask


@pytest.fixture(scope="session")
def phantom():
    """(image, brain_mask) tuple of a synthetic brain-like phantom."""
    return _make_phantom()


@pytest.fixture(scope="session")
def reg_config():
    """Minimal config object with a fast, seeded registration profile."""
    return SimpleNamespace(
        SELECTED_REGISTRATION_MODE="test",
        REGISTRATION_MODES={
            "test": {
                "learning_rate": 0.5,
                "max_iterations": 40,
                "sampling_percentage": 0.5,   # x2 for small ROI -> 100 %
                "grid_spacing": [15.0],
                "distance_map_ds_factor": 1,
                "random_seed": 42,
            }
        },
    )
