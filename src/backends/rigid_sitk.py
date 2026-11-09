"""
RigidAligner fallback using SimpleITK Euler3D + Mattes Mutual Information.
"""

from __future__ import annotations

import SimpleITK as sitk

from src.backends.base import RigidAligner
from src.core import setup_logger

logger = setup_logger(__name__)


class Euler3DAligner(RigidAligner):
    """6-DOF rigid registration via SimpleITK (Euler3D + Mattes MI)."""

    def __init__(self, n_iterations: int = 100, n_bins: int = 50,
                 sampling_pct: float = 0.05):
        self.n_iterations = n_iterations
        self.n_bins = n_bins
        self.sampling_pct = sampling_pct

    def align(self, fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
        fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
        moving_f = sitk.Cast(moving, sitk.sitkFloat32)

        initial_tx = sitk.CenteredTransformInitializer(
            fixed_f, moving_f,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=self.n_bins)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(self.sampling_pct)
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=self.n_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetInitialTransform(initial_tx)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        tx = reg.Execute(fixed_f, moving_f)
        logger.info(f"  [Euler3DAligner] metric={reg.GetMetricValue():.4f} "
                    f"stop='{reg.GetOptimizerStopConditionDescription()}'")
        return tx
