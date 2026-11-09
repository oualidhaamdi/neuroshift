"""
DeformableRegistrar implementation: B-Spline on Danielsson Distance Maps.

Extracts edges from tissue segmentation maps, computes Danielsson distance
maps (one per image), then runs a multi-stage B-Spline registration.

Improvements over the original single-stage approach:
  1. Multi-stage B-Spline: coarse grid captures global deformation, fine grid
     refines residuals — avoids local minima near brain boundaries.
  2. Mask-weighted metric: restricts MSE computation to within the brain mask,
     so boundary voxels carry more weight and background noise is ignored.
  3. 3-level pyramid [4, 2, 1]: coarser levels first for robust initialisation,
     then progressively refine — especially helpful for anterior/edge cases.
"""

from __future__ import annotations

import time

import SimpleITK as sitk

from src.backends.base import DeformableRegistrar
from src.core import setup_logger

logger = setup_logger(__name__)


def extract_tissue_edges(tissue_seg: sitk.Image, sigma_mm: float = 1.0) -> sitk.Image:
    """
    Compute gradient magnitude of the tissue segmentation map.

    The segmentation {0=BG, 1=CSF, 2=GM, 3=WM} is treated as a scalar field;
    its gradient is non-zero at every tissue boundary (CSF/GM and GM/WM)
    simultaneously, producing a single combined edge map in [0, 1].
    """
    seg_float = sitk.Cast(tissue_seg, sitk.sitkFloat32)
    seg_smooth = sitk.SmoothingRecursiveGaussian(seg_float, sigma_mm)
    grad = sitk.GradientMagnitudeRecursiveGaussian(seg_smooth, sigma_mm * 0.5)
    arr = sitk.GetArrayFromImage(grad).astype("float32")
    m = float(arr.max())
    if m > 0:
        arr /= m
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(tissue_seg)
    return out  # edge map [0,1]: 1 = any tissue boundary, 0 = interior


class BSplineDistanceMapRegistrar(DeformableRegistrar):
    """
    B-Spline registration driven by Danielsson Distance Maps of tissue edges.

    Key idea: instead of matching raw MRI intensities (unreliable post-op),
    match the distance-to-nearest-tissue-boundary field, which is dense,
    smooth, and anatomically meaningful regardless of post-op artefacts.

    ── Multi-stage rationale ────────────────────────────────────────────────────
    The register() method always runs two consecutive B-Spline optimisations
    (coarse → fine), regardless of whether the input is a full brain or a small
    synthetic ROI.  The two-stage strategy was chosen after observing that a
    single fine-grid optimisation frequently fell into local minima, especially
    near cavity boundaries where the distance map has sharp discontinuities.

    Stage 1 — coarse grid (grid_sp × 1.5 for full brain, × 2 for small ROI):
        Few nodes → wide receptive field → can move tissue in bulk without
        being trapped by local texture.  Uses a fraction of the total iteration
        budget (max_iter // 3).

    Stage 2 — fine grid (grid_sp from config):
        Works on the moving image already pre-aligned by Stage 1.  Many nodes
        → captures millimetre-scale residuals that the coarse grid missed.
        Uses the full iteration budget.

    The two transforms are composed as composite(p) = tx1(tx2(p)), i.e. the
    fine correction is applied first in the moving-image domain, then the
    coarse offset.

    ── Grid spacing & mesh-size calculation ────────────────────────────────────
    grid_spacing (mm) is read from config.REGISTRATION_MODES[mode]['grid_spacing']
    and can be overridden by register_tissue.py for synthetic cases (10 mm).

    For each spatial axis i:
        mesh_nodes[i] = max(1, int(physical_size_mm[i] / grid_spacing_mm))

    where physical_size_mm[i] = downsampled_size[i] × downsampled_spacing[i]
    (BinShrink preserves physical extent, so this equals the original extent).

    Example — full brain 256 × 256 × 180 mm, precise mode (grid_sp = 10 mm):
        coarse (15 mm): [17, 17, 12] nodes → 17 × 17 × 12 × 3 ≈ 10 k params
        fine   (10 mm): [25, 25, 18] nodes → 25 × 25 × 18 × 3 ≈ 34 k params

    Example — synthetic ROI 80 × 80 × 60 mm, override (grid_sp = 10 mm):
        coarse (20 mm): [4,  4,  3 ] nodes →  4 ×  4 ×  3 × 3 ≈  144 params
        fine   (10 mm): [8,  8,  6 ] nodes →  8 ×  8 ×  6 × 3 ≈ 1152 params

    ── Optimizer & pyramid ──────────────────────────────────────────────────────
    is_small_roi = (override_grid_sp is not None) OR (min_physical_dim ≤ 200 mm)

    The optimizer choice is driven by the number of free parameters:

    Full brain  (LBFGS2, ~10-34k params):
        Quasi-Newton with Hessian approximation; converges in 20-50 iterations
        regardless of parameter count.  sampling × 10 (up to 100%) gives the
        dense gradient estimates needed for an accurate Hessian.
        learning_rate from config is ignored by LBFGS2.

    Small ROI   (GradientDescent, ~150-1200 params):
        First-order method, sufficient for the small parameter space.
        learning_rate from config controls the initial step size.
        sampling × 2 is enough for a stable gradient in the small domain.

    NOTE: the synthetic test-case results (EPE 0.73 mm) were obtained on small
    ROI cases using GradientDescent.  The improvement for real patients comes
    from the finer grid_spacing (10 mm in 'precise' mode), not the optimizer.

    Pyramid: full brain [4, 2, 1] (3 levels), small ROI [2, 1] (2 levels).
    Coarse multiplier: 1.5× (full brain), 2× (small ROI).
    """

    def _run_stage(
        self,
        fixed_dist: sitk.Image,
        moving_dist: sitk.Image,
        mesh_size: list,
        max_iter: int,
        params: dict,
        is_small_roi: bool,
        sampling: float,
        mask_ds: sitk.Image | None,
        stage_num: int,
    ) -> tuple[sitk.Transform, float, str]:
        """Run one B-Spline optimisation stage. Returns (transform, final_metric, stop_cond)."""
        tx = sitk.BSplineTransformInitializer(
            image1=fixed_dist, transformDomainMeshSize=mesh_size, order=3
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
        # Sampling scales with ROI size: full brain needs denser sampling for
        # LBFGS2's Hessian approximation; small ROI is accurate with less.
        reg.SetMetricSamplingPercentage(min(1.0, sampling * (2.0 if is_small_roi else 10.0)))
        reg.SetInterpolator(sitk.sitkLinear)

        # Mask-weighted metric: restrict to brain interior
        if mask_ds is not None:
            reg.SetMetricFixedMask(mask_ds)

        if is_small_roi:
            # GradientDescent: better for small parameter spaces (~1k params);
            # learning_rate from config controls the step explicitly.
            lr = params.get("learning_rate", 0.2)
            n_iter = max(40, max_iter)
            reg.SetOptimizerAsGradientDescent(
                learningRate=lr,
                numberOfIterations=n_iter,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=15,
                estimateLearningRate=reg.EachIteration,
            )
            logger.info(f"  [Stage {stage_num}] GradientDescent LR={lr}  iter={n_iter}")
        else:
            # LBFGS2: quasi-Newton method, converges in ~20-50 iterations even
            # with 10-30k parameters; far more efficient than GradientDescent
            # at full-brain scale.
            reg.SetOptimizerAsLBFGS2(
                solutionAccuracy=1e-3,
                numberOfIterations=max_iter,
                deltaConvergenceTolerance=1e-4,
            )
            logger.info(f"  [Stage {stage_num}] LBFGS2  iter={max_iter}")

        reg.SetOptimizerScalesFromPhysicalShift()

        # Pyramid: more levels for full brain (more voxels to traverse coarse→fine)
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
                logger.info(f"  [Stage {stage_num} | Level {lvl}] started")
            _state["count"] += 1
            if _state["count"] % 10 == 0:
                logger.info(
                    f"    iter {_state['count']:3d} | "
                    f"metric={reg.GetMetricValue():.5f} | "
                    f"elapsed {time.time()-_state['t0']:.0f}s"
                )

        reg.AddCommand(sitk.sitkIterationEvent, _log)
        out_tx = reg.Execute(fixed_dist, moving_dist)
        return out_tx, reg.GetMetricValue(), reg.GetOptimizerStopConditionDescription()

    def register(
        self,
        fixed_edge: sitk.Image,
        moving_edge: sitk.Image,
        mask: sitk.Image,
        config,
        rigid_tx: sitk.Transform | None = None,  # kept for API compat, not used
        override_grid_sp: float | None = None,
    ) -> sitk.Transform:
        params     = config.REGISTRATION_MODES[config.SELECTED_REGISTRATION_MODE]
        grid_sp    = params.get("grid_spacing", [30.0])
        if isinstance(grid_sp, list):
            grid_sp = grid_sp[0]
        if override_grid_sp is not None:
            grid_sp = float(override_grid_sp)
            logger.info(f"  [BSplineDistanceMapRegistrar] grid_sp overridden to {grid_sp}mm")
        max_iter   = params.get("max_iterations", 100)
        sampling   = params.get("sampling_percentage", 0.05)
        ds_factor  = params.get("distance_map_ds_factor", 4)

        # ── Determine ROI type ─────────────────────────────────────────────────
        phys_size = [fixed_edge.GetSize()[i] * fixed_edge.GetSpacing()[i] for i in range(3)]
        is_small_roi = (override_grid_sp is not None) or (min(phys_size) <= 200.0)

        if is_small_roi:
            shrink_factors = [2, 2, 1]
            logger.info(f"  [BSplineDistanceMapRegistrar] Small ROI ({[int(p) for p in phys_size]}mm) — fine settings")
            if override_grid_sp is None:
                grid_sp = 10.0
        else:
            sf_xy = max(1, ds_factor)
            sf_z  = max(1, int(ds_factor * fixed_edge.GetSpacing()[2] / fixed_edge.GetSpacing()[0]))
            sf_z  = min(sf_z, fixed_edge.GetSize()[2] // 10 + 1)
            shrink_factors = [sf_xy, sf_xy, max(1, sf_z // 2)]

        logger.info(f"  [BSplineDistanceMapRegistrar] bin_shrink={shrink_factors} | grid={grid_sp}mm | iter={max_iter}")

        # ── Downsample edge maps ───────────────────────────────────────────────
        bin_shrink = sitk.BinShrinkImageFilter()
        bin_shrink.SetShrinkFactors(shrink_factors)
        fixed_ds  = bin_shrink.Execute(fixed_edge)
        moving_ds = bin_shrink.Execute(moving_edge)

        # ── 2. Mask-weighted metric ────────────────────────────────────────────
        mask_ds = None
        if mask is not None:
            try:
                mask_ds = sitk.Cast(
                    bin_shrink.Execute(sitk.Cast(mask, sitk.sitkFloat32)) > 0.5,
                    sitk.sitkUInt8,
                )
                logger.info("  [BSplineDistanceMapRegistrar] Brain mask applied to metric")
            except Exception as e:
                logger.warning(f"  Mask downsampling failed ({e}) — proceeding without mask")

        # ── Danielsson Distance Maps ───────────────────────────────────────────
        dist_filter = sitk.DanielssonDistanceMapImageFilter()
        dist_filter.SetInputIsBinary(True)
        dist_filter.SetUseImageSpacing(True)

        fixed_dist  = sitk.Cast(
            dist_filter.Execute(sitk.Cast(fixed_ds  > 0.01, sitk.sitkUInt8)), sitk.sitkFloat32
        )
        moving_dist = sitk.Cast(
            dist_filter.Execute(sitk.Cast(moving_ds > 0.01, sitk.sitkUInt8)), sitk.sitkFloat32
        )

        # ── Small ROI: two-stage (coarse 20mm → fine 10mm) ───────────────────
        if is_small_roi:
            coarse_grid_roi = grid_sp * 2.0   # 20mm coarse pass
            coarse_mesh_roi = [
                max(1, int(fixed_ds.GetSize()[i] * fixed_ds.GetSpacing()[i] / coarse_grid_roi))
                for i in range(3)
            ]
            logger.info(f"  [BSpline Stage 1] coarse grid={coarse_grid_roi:.1f}mm | iter={max(20, max_iter // 3)}")
            tx1, m1, stop1 = self._run_stage(
                fixed_dist, moving_dist, coarse_mesh_roi,
                max(20, max_iter // 3), params, is_small_roi, sampling, mask_ds, stage_num=1,
            )
            logger.info(f"  [Stage 1] metric={m1:.5f}  stop='{stop1}'")

            moving_stage2 = sitk.Resample(moving_dist, fixed_dist, tx1, sitk.sitkLinear, 0.0)

            fine_mesh_roi = [
                max(1, int(fixed_ds.GetSize()[i] * fixed_ds.GetSpacing()[i] / grid_sp))
                for i in range(3)
            ]
            logger.info(f"  [BSpline Stage 2] fine grid={grid_sp:.1f}mm | iter={max_iter}")
            tx2, m2, stop2 = self._run_stage(
                fixed_dist, moving_stage2, fine_mesh_roi,
                max_iter, params, is_small_roi, sampling, mask_ds, stage_num=2,
            )
            logger.info(f"  [Stage 2] metric={m2:.5f}  stop='{stop2}'")

            composite_roi = sitk.CompositeTransform(3)
            composite_roi.AddTransform(tx2)
            composite_roi.AddTransform(tx1)
            return composite_roi

        # ── 1. Multi-stage B-Spline (full brain) ──────────────────────────────
        # Stage 1: coarse grid — captures global/large-scale deformation
        coarse_grid = grid_sp * 1.5
        coarse_mesh = [
            max(1, int(fixed_ds.GetSize()[i] * fixed_ds.GetSpacing()[i] / coarse_grid))
            for i in range(3)
        ]
        logger.info(f"  [BSpline Stage 1] coarse grid={coarse_grid:.1f}mm | iter={max_iter // 3}")
        tx1, m1, stop1 = self._run_stage(
            fixed_dist, moving_dist, coarse_mesh,
            max_iter // 3, params, is_small_roi, sampling, mask_ds, stage_num=1,
        )
        logger.info(f"  [Stage 1] metric={m1:.5f}  stop='{stop1}'")

        # Pre-register moving distance map with coarse result
        moving_stage2 = sitk.Resample(moving_dist, fixed_dist, tx1, sitk.sitkLinear, 0.0)

        # Stage 2: fine grid — corrects residuals that the coarse stage missed
        fine_mesh = [
            max(1, int(fixed_ds.GetSize()[i] * fixed_ds.GetSpacing()[i] / grid_sp))
            for i in range(3)
        ]
        logger.info(f"  [BSpline Stage 2] fine grid={grid_sp:.1f}mm | iter={max_iter}")
        tx2, m2, stop2 = self._run_stage(
            fixed_dist, moving_stage2, fine_mesh,
            max_iter, params, is_small_roi, sampling, mask_ds, stage_num=2,
        )
        logger.info(f"  [Stage 2] metric={m2:.5f}  stop='{stop2}'")

        # Compose: for point p in fixed space, apply tx2 first (residual offset into
        # pre-aligned moving), then tx1 (coarse offset into original moving).
        # In ITK CompositeTransform semantics (first added = innermost = applied first):
        #   composite(p) = tx1(tx2(p))  ✓
        composite = sitk.CompositeTransform(3)
        composite.AddTransform(tx2)  # innermost — applied first
        composite.AddTransform(tx1)  # outermost — applied second
        logger.info("  [BSpline] Composite (coarse ∘ fine) assembled")
        return composite
