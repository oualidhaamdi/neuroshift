"""
Baseline DeformableRegistrar — B-Spline on raw intensity images.

Purpose
-------
This is the *intensity-based* counterpart to BSplineDistanceMapRegistrar.
It exists to provide a fair comparison baseline for the paper:

  tissue-guided  (BSplineDistanceMapRegistrar)  →  matches distance maps of
                                                   tissue boundaries
  intensity-based (this class)                  →  matches raw intensities
                                                   inside the brain mask

Everything else — B-Spline grid spacing, optimizer (LBFGS2 / GradientDescent),
pyramid, sampling strategy, mask-weighted metric, two-stage coarse→fine
composition — is identical to the tissue-guided version, so any performance
delta isolates the contribution of the distance-map representation.

Metric
------
MeanSquares on normalised intensity (each input rescaled to [0, 1] within the
brain mask).  Mattes mutual information would also be valid; MeanSquares was
chosen for direct comparability with the tissue-guided pipeline.

Usage
-----
Selected in src/register_tissue.py when env var BASELINE_REGISTRAR=intensity
or when the registrar is explicitly passed to the pipeline constructor.
"""

from __future__ import annotations

import time

import numpy as np
import SimpleITK as sitk

from src.backends.base import DeformableRegistrar
from src.core import setup_logger

logger = setup_logger(__name__)


def _normalize_intensity(img: sitk.Image, mask: sitk.Image | None = None) -> sitk.Image:
    """Rescale intensities to [0, 1] using the 1st-99th percentile within the brain mask."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if mask is not None:
        m = sitk.GetArrayFromImage(mask) > 0
        if m.shape == arr.shape and m.any():
            ref = arr[m]
        else:
            ref = arr
    else:
        ref = arr
    lo, hi = np.percentile(ref, [1, 99])
    arr_n = np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    out = sitk.GetImageFromArray(arr_n)
    out.CopyInformation(img)
    return out


class BSplineIntensityRegistrar(DeformableRegistrar):
    """B-Spline on raw intensity images (baseline). Same grid/optimizer as tissue-guided."""

    def _run_stage(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        mesh_size: list,
        max_iter: int,
        params: dict,
        is_small_roi: bool,
        sampling: float,
        mask_ds: sitk.Image | None,
        stage_num: int,
    ) -> tuple[sitk.Transform, float, str]:
        tx = sitk.BSplineTransformInitializer(
            image1=fixed, transformDomainMeshSize=mesh_size, order=3
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
        pct = min(1.0, sampling * (2.0 if is_small_roi else 10.0))
        seed = params.get("random_seed")  # optional: makes metric sampling reproducible
        if seed is not None:
            reg.SetMetricSamplingPercentage(pct, int(seed))
        else:
            reg.SetMetricSamplingPercentage(pct)
        reg.SetInterpolator(sitk.sitkLinear)
        if mask_ds is not None:
            reg.SetMetricFixedMask(mask_ds)

        if is_small_roi:
            lr = params.get("learning_rate", 0.2)
            n_iter = max(40, max_iter)
            reg.SetOptimizerAsGradientDescent(
                learningRate=lr,
                numberOfIterations=n_iter,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=15,
                estimateLearningRate=reg.EachIteration,
            )
            logger.info(f"  [Intensity Stage {stage_num}] GradientDescent LR={lr}  iter={n_iter}")
        else:
            reg.SetOptimizerAsLBFGS2(
                solutionAccuracy=1e-3,
                numberOfIterations=max_iter,
                deltaConvergenceTolerance=1e-4,
            )
            logger.info(f"  [Intensity Stage {stage_num}] LBFGS2  iter={max_iter}")

        reg.SetOptimizerScalesFromPhysicalShift()

        if is_small_roi:
            reg.SetShrinkFactorsPerLevel([2, 1])
            reg.SetSmoothingSigmasPerLevel([0.5, 0.0])
        else:
            reg.SetShrinkFactorsPerLevel([4, 2, 1])
            reg.SetSmoothingSigmasPerLevel([1.0, 0.5, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        reg.SetInitialTransform(tx, inPlace=True)

        _state = {"count": 0, "level": -1, "t0": time.time()}

        def _log():
            lvl = reg.GetCurrentLevel()
            if lvl != _state["level"]:
                _state["level"] = lvl
                _state["count"] = 0
                logger.info(f"  [Intensity Stage {stage_num} | Level {lvl}] started")
            _state["count"] += 1
            if _state["count"] % 10 == 0:
                logger.info(
                    f"    iter {_state['count']:3d} | "
                    f"metric={reg.GetMetricValue():.5f} | "
                    f"elapsed {time.time()-_state['t0']:.0f}s"
                )

        reg.AddCommand(sitk.sitkIterationEvent, _log)
        out_tx = reg.Execute(fixed, moving)
        return out_tx, reg.GetMetricValue(), reg.GetOptimizerStopConditionDescription()

    def register(
        self,
        fixed_intensity: sitk.Image,
        moving_intensity: sitk.Image,
        mask: sitk.Image,
        config,
        rigid_tx: sitk.Transform | None = None,
        override_grid_sp: float | None = None,
    ) -> sitk.Transform:
        params    = config.REGISTRATION_MODES[config.SELECTED_REGISTRATION_MODE]
        grid_sp   = params.get("grid_spacing", [30.0])
        if isinstance(grid_sp, list):
            grid_sp = grid_sp[0]
        if override_grid_sp is not None:
            grid_sp = float(override_grid_sp)
            logger.info(f"  [BSplineIntensityRegistrar] grid_sp overridden to {grid_sp}mm")
        max_iter  = params.get("max_iterations", 100)
        sampling  = params.get("sampling_percentage", 0.05)
        ds_factor = params.get("distance_map_ds_factor", 4)

        phys_size    = [fixed_intensity.GetSize()[i] * fixed_intensity.GetSpacing()[i] for i in range(3)]
        is_small_roi = (override_grid_sp is not None) or (min(phys_size) <= 200.0)

        if is_small_roi:
            shrink_factors = [2, 2, 1]
            logger.info(
                f"  [BSplineIntensityRegistrar] Small ROI "
                f"({[int(p) for p in phys_size]}mm) — fine settings"
            )
            if override_grid_sp is None:
                grid_sp = 10.0
        else:
            sf_xy = max(1, ds_factor)
            sf_z  = max(1, int(ds_factor * fixed_intensity.GetSpacing()[2]
                               / fixed_intensity.GetSpacing()[0]))
            sf_z  = min(sf_z, fixed_intensity.GetSize()[2] // 10 + 1)
            shrink_factors = [sf_xy, sf_xy, max(1, sf_z // 2)]

        logger.info(
            f"  [BSplineIntensityRegistrar] bin_shrink={shrink_factors}  "
            f"grid={grid_sp}mm  iter={max_iter}"
        )

        # Normalize intensities in the brain mask, then downsample
        fixed_n  = _normalize_intensity(fixed_intensity,  mask)
        moving_n = _normalize_intensity(moving_intensity, mask)

        bin_shrink = sitk.BinShrinkImageFilter()
        bin_shrink.SetShrinkFactors(shrink_factors)
        fixed_ds  = bin_shrink.Execute(fixed_n)
        moving_ds = bin_shrink.Execute(moving_n)

        mask_ds = None
        if mask is not None:
            try:
                mask_ds = sitk.Cast(
                    bin_shrink.Execute(sitk.Cast(mask, sitk.sitkFloat32)) > 0.5,
                    sitk.sitkUInt8,
                )
                logger.info("  [BSplineIntensityRegistrar] Brain mask applied to metric")
            except Exception as e:
                logger.warning(f"  Mask downsampling failed ({e}) — proceeding without mask")

        # Two-stage coarse → fine, same as tissue-guided
        coarse_mult = 2.0 if is_small_roi else 1.5
        coarse_grid = grid_sp * coarse_mult
        coarse_mesh = [
            max(1, int(fixed_ds.GetSize()[i] * fixed_ds.GetSpacing()[i] / coarse_grid))
            for i in range(3)
        ]
        coarse_iter = max(20, max_iter // 3) if is_small_roi else max_iter // 3
        logger.info(f"  [Intensity Stage 1] coarse grid={coarse_grid:.1f}mm  iter={coarse_iter}")
        tx1, m1, stop1 = self._run_stage(
            fixed_ds, moving_ds, coarse_mesh,
            coarse_iter, params, is_small_roi, sampling, mask_ds, stage_num=1,
        )
        logger.info(f"  [Intensity Stage 1] metric={m1:.5f}  stop='{stop1}'")

        moving_stage2 = sitk.Resample(moving_ds, fixed_ds, tx1, sitk.sitkLinear, 0.0)

        fine_mesh = [
            max(1, int(fixed_ds.GetSize()[i] * fixed_ds.GetSpacing()[i] / grid_sp))
            for i in range(3)
        ]
        logger.info(f"  [Intensity Stage 2] fine grid={grid_sp:.1f}mm  iter={max_iter}")
        tx2, m2, stop2 = self._run_stage(
            fixed_ds, moving_stage2, fine_mesh,
            max_iter, params, is_small_roi, sampling, mask_ds, stage_num=2,
        )
        logger.info(f"  [Intensity Stage 2] metric={m2:.5f}  stop='{stop2}'")

        composite = sitk.CompositeTransform(3)
        composite.AddTransform(tx2)
        composite.AddTransform(tx1)
        logger.info("  [BSplineIntensityRegistrar] Composite (coarse ∘ fine) assembled")
        return composite
