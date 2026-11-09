"""
Parametric B-Spline Validation Sweep.

Sweeps the full A × σ × direction × location space to characterise
registration accuracy as a function of each parameter.

Test matrix
-----------
Core grid       : A(4) x s(3) x direction(4) = 48 cases  (center only)
Location grid   : A=5, s=25, X & Y, 4 off-centre locations = 8 cases
──────────────────────────────────────────────────────────────────────
Total           : 56 cases

Pilot subset (--pilot flag): 6 representative cases (~1-1.5 h)

Metrics computed per voxel inside the brain mask
-------------------------------------------------
epe             : ‖pred + gt‖₂  (pred ≈ -gt for perfect recovery)
angular_error   : angle between pred and -gt vectors (degrees)
dot_ratio       : (pred · (-gt)) / ‖gt‖²  (1=perfect, 0=none, <0=wrong dir)
err_x/y/z       : |pred_k + gt_k| per component

Usage
-----
python -m src.qc.run_synthetic_sweep                    # full sweep PRE+POST (112 cases)
python -m src.qc.run_synthetic_sweep --pre              # PRE only  (56 cases)
python -m src.qc.run_synthetic_sweep --post             # POST only (56 cases)
python -m src.qc.run_synthetic_sweep --pilot            # pilot PRE+POST (12 cases)
python -m src.qc.run_synthetic_sweep --pilot --pre      # pilot PRE only (6 cases)
python -m src.qc.run_synthetic_sweep --pilot --post     # pilot POST only (6 cases)
python -m src.qc.run_synthetic_sweep --single ID        # one specific case (PRE+POST)
python -m src.qc.run_synthetic_sweep --single ID --pre  # PRE variant only
python -m src.qc.run_synthetic_sweep --single ID --post # POST variant only
python -m src.qc.run_synthetic_sweep --dry-run          # preview without running
python -m src.qc.run_synthetic_sweep --resume           # skip already completed cases
python -m src.qc.run_synthetic_sweep --no-cleanup       # keep intermediate files
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from itertools import product
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import load_config, setup_logger
from src.data_manager import DataManager
from src.qc.create_synthetic_patients import (
    generate_gauss3d_test,
    generate_gauss3d_test_post_variant,
    LOCATIONS,
)
from src.qc import sanitize_for_json, restore_nan_from_json

logger = setup_logger("sweep")

# ── Parameter grid ─────────────────────────────────────────────────────────────
AMP_MM_VALUES   = [3.0, 5.0, 8.0, 15.0]
SIGMA_MM_VALUES = [15.0, 25.0, 40.0]

# ── Isotropic resampling of high-resolution sources ─────────────────────────
# A very fine, anisotropic source grid (e.g. sub-0.3 mm in-plane, >100M
# voxels) yields a weak/ill-scaled MeanSquares gradient and the B-Spline optimizer
# stalls at identity (pred≈0 → angular=90°, dot_ratio=0). Sources finer than the
# threshold are resampled to the target isotropic spacing before synthetic
# generation, bringing them into the same regime as the rest of the cohort.
# Typical clinical T1 grids (≥0.5 mm) are above the threshold and left untouched.
ISO_RESAMPLE_THRESHOLD_MM = 0.5
ISO_RESAMPLE_TARGET_MM    = 1.0

DIRECTIONS = {
    "X":   [1.0, 0.0, 0.0],
    "Y":   [0.0, 1.0, 0.0],
    "Z":   [0.0, 0.0, 1.0],
    "XYZ": [1.0, 1.0, 0.5],
}

LOCATION_GRID_DIRECTION = ["X", "Y"]   # directions used for location sweep
LOCATION_GRID_A    = 5.0
LOCATION_GRID_S    = 25.0
LOCATION_EXTRA     = ["anterior", "posterior", "lateral_R", "superior"]  # skip center (in core)

# Pilot: 6 cases covering all key dimensions
PILOT_CASES = [
    {"name": "Pilot_A050_S250_X_C",   "direction": "X",   "A_mm": 5.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "Pilot_A050_S250_Y_C",   "direction": "Y",   "A_mm": 5.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "Pilot_A050_S250_Z_C",   "direction": "Z",   "A_mm": 5.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "Pilot_A050_S250_XYZ_C", "direction": "XYZ", "A_mm": 5.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "Pilot_A150_S250_X_C",   "direction": "X",   "A_mm": 15.0, "sigma_mm": 25.0, "location": "center"},
    {"name": "Pilot_A050_S250_X_ANT", "direction": "X",   "A_mm": 5.0,  "sigma_mm": 25.0, "location": "anterior"},
]

# Multi-brain reduced set: 10 cases covering 4 directions, 4 amplitudes, 3 sigmas, 2 locations.
# Used by multi_brain_sweep.py orchestrator for cross-brain validation.
MULTIBRAIN_CASES = [
    # 4 directions @ A=5mm σ=25mm center
    {"name": "MB_A050_S250_X_C",   "direction": "X",   "A_mm": 5.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "MB_A050_S250_Y_C",   "direction": "Y",   "A_mm": 5.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "MB_A050_S250_Z_C",   "direction": "Z",   "A_mm": 5.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "MB_A050_S250_XYZ_C", "direction": "XYZ", "A_mm": 5.0,  "sigma_mm": 25.0, "location": "center"},
    # 3 amplitudes (3, 8, 15mm) @ X σ=25
    {"name": "MB_A030_S250_X_C",   "direction": "X",   "A_mm": 3.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "MB_A080_S250_X_C",   "direction": "X",   "A_mm": 8.0,  "sigma_mm": 25.0, "location": "center"},
    {"name": "MB_A150_S250_X_C",   "direction": "X",   "A_mm": 15.0, "sigma_mm": 25.0, "location": "center"},
    # 2 sigmas (15, 40mm) @ X A=5
    {"name": "MB_A050_S150_X_C",   "direction": "X",   "A_mm": 5.0,  "sigma_mm": 15.0, "location": "center"},
    {"name": "MB_A050_S400_X_C",   "direction": "X",   "A_mm": 5.0,  "sigma_mm": 40.0, "location": "center"},
    # 1 off-center anterior
    {"name": "MB_A050_S250_X_ANT", "direction": "X",   "A_mm": 5.0,  "sigma_mm": 25.0, "location": "anterior"},
]


def _resample_isotropic(img: sitk.Image, target_mm: float, is_mask: bool = False) -> sitk.Image:
    """Resample img to target_mm isotropic spacing, preserving origin/direction.

    Linear interpolation for intensity images, nearest-neighbour for masks/labels
    (to keep them binary/categorical).
    """
    sp = img.GetSpacing()
    sz = img.GetSize()
    new_size = [max(1, int(round(sz[i] * sp[i] / target_mm))) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((target_mm, target_mm, target_mm))
    rs.SetSize(new_size)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0)
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return rs.Execute(img)


def _maybe_resample_source(
    pre_img: sitk.Image,
    mask_img: sitk.Image,
    post_op_img: sitk.Image | None,
    target_mm: float | None,
):
    """Resample pre/mask/post to isotropic if the source is finer than the threshold.

    ``target_mm`` is the CLI override (``--iso-resample-mm``): ``None`` → use the
    module default ``ISO_RESAMPLE_TARGET_MM``; ``<=0`` → disable resampling entirely.
    Only sources with ``min(spacing) < ISO_RESAMPLE_THRESHOLD_MM`` are touched, so
    coarser source brains keep their native grid.
    """
    if target_mm is None:
        target_mm = ISO_RESAMPLE_TARGET_MM
    if target_mm <= 0:
        logger.info("  [Resample] disabled (--iso-resample-mm 0) — using native grid")
        return pre_img, mask_img, post_op_img

    sp = pre_img.GetSpacing()
    if min(sp) >= ISO_RESAMPLE_THRESHOLD_MM:
        logger.info(
            f"  [Resample] native spacing {tuple(round(s, 3) for s in sp)}mm "
            f">= {ISO_RESAMPLE_THRESHOLD_MM}mm threshold — no resample"
        )
        return pre_img, mask_img, post_op_img

    old_sz = pre_img.GetSize()
    pre_r  = _resample_isotropic(pre_img,  target_mm, is_mask=False)
    mask_r = _resample_isotropic(mask_img, target_mm, is_mask=True)
    post_r = (_resample_isotropic(post_op_img, target_mm, is_mask=False)
              if post_op_img is not None else None)
    logger.warning(
        f"  [Resample] high-res source {old_sz}@{tuple(round(s, 3) for s in sp)}mm "
        f"→ {pre_r.GetSize()}@{target_mm}mm iso (intensity=linear, mask=nearest)"
    )
    return pre_r, mask_r, post_r


def _case_name(A: float, sigma: float, dir_tag: str, loc: str) -> str:
    loc_tag = loc[:3].upper() if loc != "center" else "C"
    return f"Sweep_A{int(A*10):03d}_S{int(sigma*10):03d}_{dir_tag}_{loc_tag}"


def build_test_cases(pilot: bool = False, cases_set: str = "full") -> list[dict]:
    """Return the test case list. cases_set takes precedence over pilot."""
    if cases_set == "multibrain":
        return list(MULTIBRAIN_CASES)
    if cases_set == "pilot" or pilot:
        return list(PILOT_CASES)

    cases = []

    # Core: full A × σ × direction grid, all at center
    for dir_tag, d_vec in DIRECTIONS.items():
        for a, s in product(AMP_MM_VALUES, SIGMA_MM_VALUES):
            cases.append({
                "name":      _case_name(a, s, dir_tag, "center"),
                "direction": dir_tag,
                "A_mm":      a,
                "sigma_mm":  s,
                "location":  "center",
            })

    # Location grid: A=5, σ=25, X and Y, off-centre locations
    for dir_tag in LOCATION_GRID_DIRECTION:
        for loc in LOCATION_EXTRA:
            cases.append({
                "name":      _case_name(LOCATION_GRID_A, LOCATION_GRID_S, dir_tag, loc),
                "direction": dir_tag,
                "A_mm":      LOCATION_GRID_A,
                "sigma_mm":  LOCATION_GRID_S,
                "location":  loc,
            })

    return cases


GT_SIG_THRESHOLD_MM = 0.5  # only evaluate voxels where GT displacement is meaningful


def compute_metrics(pred_path: Path, gt_path: Path, mask_path: Path | None) -> dict:
    """
    Compute EPE, angular error, dot ratio, per-component error, and Jacobian stats.

    Convention: gt stores pull-back field (+d); tissue moved in -d direction.
    For a perfect recovery: pred ≈ -gt, so pred + gt ≈ 0.

    Metrics are computed only on voxels where ||gt|| > GT_SIG_THRESHOLD_MM.
    Jacobian stats are computed over all brain-mask voxels (not just significant ones).

    n_voxels  = total brain-mask voxels
    n_sig     = voxels used for displacement metrics (GT > threshold)
    """
    pred_img = sitk.ReadImage(str(pred_path))
    gt_img   = sitk.ReadImage(str(gt_path))

    pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.float32)  # (Z,Y,X,3)
    gt_arr   = sitk.GetArrayFromImage(gt_img).astype(np.float32)

    if pred_arr.shape != gt_arr.shape:
        pred_img = sitk.Resample(pred_img, gt_img, sitk.Transform(), sitk.sitkLinear, 0.0)
        pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.float32)

    # Apply brain mask
    if mask_path and mask_path.exists():
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
        if mask_arr.shape == pred_arr.shape[:3]:
            pred_arr = pred_arr[mask_arr]   # (N, 3)
            gt_arr   = gt_arr[mask_arr]
        else:
            pred_arr = pred_arr.reshape(-1, 3)
            gt_arr   = gt_arr.reshape(-1, 3)
    else:
        pred_arr = pred_arr.reshape(-1, 3)
        gt_arr   = gt_arr.reshape(-1, 3)

    # Magnitude field (mm)
    gt_mag = np.linalg.norm(gt_arr, axis=-1)
    
    # Filter for significant voxels (GT > 0.5mm)
    sig_mask = gt_mag > GT_SIG_THRESHOLD_MM
    n_sig = int(np.sum(sig_mask))
    
    if n_sig < 10:
        logger.warning(f"  Insufficient significant voxels ({n_sig}) — results may be noisy.")
        # Fallback to all brain mask voxels if nothing passes threshold
        sig_mask = np.ones_like(gt_mag, dtype=bool)
        n_sig = len(gt_mag)

    p = pred_arr[sig_mask]
    g = gt_arr[sig_mask]
    g_m = gt_mag[sig_mask]

    # 1. EPE (L2 norm of pred + gt)
    # Since pred should be -gt, pred + gt should be 0.
    epe_vectors = p + g
    epe = np.linalg.norm(epe_vectors, axis=-1)
    
    # 2. Angular error (degrees)
    # Angle between pred and -gt
    neg_g = -g
    dot = np.sum(p * neg_g, axis=-1)
    norm_p = np.linalg.norm(p, axis=-1)
    norm_neg_g = np.linalg.norm(neg_g, axis=-1)

    cos_theta = dot / (norm_p * norm_neg_g + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_theta))

    # 2b. Clinically-relevant angular metrics
    #   ang_mean_central   — mean angle over voxels with ‖GT‖ > 2mm
    #                        (excludes peripheral noise where direction is random)
    #   ang_weighted_mean  — magnitude-weighted mean: sum(ang * ‖GT‖) / sum(‖GT‖)
    #                        (every voxel contributes proportionally to its GT)
    #   epe_mean_central   — mean EPE on the same central mask
    central_mask = g_m > 2.0
    if central_mask.any():
        ang_mean_central  = float(np.mean(angles[central_mask]))
        epe_mean_central  = float(np.mean(epe[central_mask]))
    else:
        ang_mean_central  = float("nan")
        epe_mean_central  = float("nan")
    sum_gm = float(np.sum(g_m))
    ang_weighted_mean = float(np.sum(angles * g_m) / sum_gm) if sum_gm > 0 else float("nan")

    # 2c. Binned angular & EPE stats by GT magnitude
    #   Decomposes ang/epe across windows of GT magnitude so periphery (small GT, noise)
    #   is separated from clinically-relevant central voxels (large GT).
    #   Bins: [0.5,1), [1,2), [2,5), [5,∞) mm
    GT_BINS = [(0.5, 1.0, "lt1"), (1.0, 2.0, "1to2"),
               (2.0, 5.0, "2to5"), (5.0, float("inf"), "gt5")]
    binned: dict[str, float] = {}
    for lo, hi, tag in GT_BINS:
        m_bin = (g_m >= lo) & (g_m < hi)
        n_bin = int(m_bin.sum())
        binned[f"ang_{tag}_mean"]   = float(np.mean(angles[m_bin]))         if n_bin else float("nan")
        binned[f"ang_{tag}_median"] = float(np.median(angles[m_bin]))       if n_bin else float("nan")
        binned[f"ang_{tag}_p95"]    = float(np.percentile(angles[m_bin], 95)) if n_bin else float("nan")
        binned[f"epe_{tag}_mean"]   = float(np.mean(epe[m_bin]))            if n_bin else float("nan")
        binned[f"pct_gt90_{tag}"]   = float(np.mean(angles[m_bin] > 90.0) * 100.0) if n_bin else float("nan")
        binned[f"n_{tag}"]          = n_bin

    # 3. Dot ratio (Directional recovery quality)
    # (pred · -gt) / ||gt||²
    # 1.0 = perfect recovery, 0.0 = no recovery, <0 = wrong direction
    dot_ratio = dot / (g_m**2 + 1e-12)

    # 4. Per-component absolute error
    err_x = np.abs(p[:, 0] + g[:, 0])
    err_y = np.abs(p[:, 1] + g[:, 1])
    err_z = np.abs(p[:, 2] + g[:, 2])

    # 5. Jacobian stats — all brain-mask voxels (not just significant ones)
    jac_result = {"min": np.nan, "mean": np.nan, "pct_fold": np.nan}
    jac_path = pred_path.parent / "brainshift_jacobian.nii.gz"
    if jac_path.exists():
        try:
            jac_img = sitk.ReadImage(str(jac_path))
            jac_arr = sitk.GetArrayFromImage(jac_img).astype(np.float32)
            if mask_path and mask_path.exists():
                mask_arr_jac = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
                if mask_arr_jac.shape == jac_arr.shape:
                    jac_arr = jac_arr[mask_arr_jac]
                else:
                    jac_arr = jac_arr.ravel()
            else:
                jac_arr = jac_arr.ravel()
            jac_result = {
                "min":      float(np.min(jac_arr)),
                "mean":     float(np.mean(jac_arr)),
                "pct_fold": float(np.mean(jac_arr < 0) * 100),
            }
        except Exception as e:
            logger.warning(f"  Jacobian stats failed: {e}")

    return {
        "epe": {
            "mean":   float(np.mean(epe)),
            "median": float(np.median(epe)),
            "p95":    float(np.percentile(epe, 95)),
            "max":    float(np.max(epe)),
        },
        "angular_err": {
            "mean":          float(np.mean(angles)),
            "p95":           float(np.percentile(angles, 95)),
            "max":           float(np.max(angles)),
            "mean_central":  ang_mean_central,
            "weighted_mean": ang_weighted_mean,
        },
        "epe_central":   epe_mean_central,
        "n_central":     int(central_mask.sum()),
        "binned":        binned,
        "dot_ratio": {
            "mean":   float(np.mean(dot_ratio)),
            "median": float(np.median(dot_ratio)),
        },
        "err_x": {"mean": float(np.mean(err_x))},
        "err_y": {"mean": float(np.mean(err_y))},
        "err_z": {"mean": float(np.mean(err_z))},
        "jacobian":     jac_result,
        "gt_peak_mm":   float(np.max(gt_mag)),
        "n_voxels":     len(gt_arr),
        "n_sig":        n_sig,
        "ang_pct_gt90": float(np.mean(angles > 90) * 100),
    }


def run_pipeline(patient_id: str) -> bool:
    """Invokes main.py to run registration and analysis."""
    cmd = [
        sys.executable, "main.py",
        "--patient", patient_id,
        "--register-tissue", "--analyze",
    ]
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  Pipeline failed for {patient_id} (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Parametric B-Spline validation sweep")
    parser.add_argument("--pilot", action="store_true", help="Run only the pilot cases (X, Y, Z, XYZ)")
    parser.add_argument("--single", type=str, help="Run only a specific case ID (e.g. Pilot_A050_S250_X_ANT)")
    parser.add_argument("--dry-run", action="store_true", help="Show cases without running")
    parser.add_argument("--resume",     action="store_true", help="Skip already-completed cases")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep intermediate files")
    parser.add_argument("--pre",  action="store_true", default=False,
                        help="Run only PRE cases (pre-op as source)")
    parser.add_argument("--post", action="store_true", default=False,
                        help="Run only POST cases (real post-op as source)")
    parser.add_argument("--source-brain", type=str, default="RHUH-0002",
                        help="Case ID under RunBatch_Input/ used as source anatomy "
                             "for synthetic generation (default: RHUH-0002)")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Override default JSON output path (default: RunBatch_Output/sweep_results.json)")
    parser.add_argument("--cases-set", type=str, default="full",
                        choices=["full", "pilot", "multibrain"],
                        help="Test case set: full (56), pilot (6), or multibrain (10)")
    parser.add_argument("--tissue-guided", action="store_true",
                        help="Opt-in to the tissue-guided B-Spline registrar (distance maps). "
                             "Use this as the ABLATION mode; default is intensity-based.")
    parser.add_argument("--no-cfm", action="store_true",
                        help="Disable Cost Function Masking even when a cavity mask is present. "
                             "Used for the CFM ablation study (DISABLE_CFM=1 env var).")
    parser.add_argument("--cache-source", action="store_true",
                        help="Pre-compute BET+FAST on the source brain (RAS-oriented) ONCE and reuse "
                             "for every synthetic case. Sets REUSE_PRE_CACHE=1. ~30-40% speedup.")
    parser.add_argument("--pid-prefix", type=str, default="",
                        help="Prefix prepended to output directory names (RunBatch_Input/Output). "
                             "Used by multi_brain_sweep --parallel to avoid collisions when "
                             "several brains write to the same case dirs concurrently. "
                             "JSON result keys are stored WITHOUT the prefix.")
    parser.add_argument("--iso-resample-mm", type=float, default=None,
                        help=f"Target isotropic spacing (mm) for sources finer than "
                             f"{ISO_RESAMPLE_THRESHOLD_MM}mm in-plane (default: "
                             f"{ISO_RESAMPLE_TARGET_MM}mm). Pass 0 to disable. Fixes the "
                             "B-Spline identity-stall on hi-res anisotropic brains.")
    args = parser.parse_args()

    if args.tissue_guided:
        os.environ["REGISTRAR"] = "tissue"
        logger.info("Tissue-guided ablation mode: REGISTRAR=tissue (will affect child main.py runs)")
    if args.no_cfm:
        os.environ["DISABLE_CFM"] = "1"
        logger.info("CFM ablation: DISABLE_CFM=1 (cavity will not be excluded from metric)")
    if args.cache_source:
        os.environ["REUSE_PRE_CACHE"] = "1"
        logger.info("Cache-source mode: REUSE_PRE_CACHE=1 (BET/FAST on pre will be skipped per case)")

    pid_prefix = args.pid_prefix  # empty string → no prefix (single-brain runs)
    if pid_prefix:
        logger.info(f"PID prefix: '{pid_prefix}' (output dirs namespaced per-brain for parallel safety)")

    # Determine which passes to run:
    #   --pre only  → PRE pass, no POST
    #   --post only → POST pass, no PRE
    #   both or neither → run both passes
    if args.pre and not args.post:
        run_pre, run_post = True, False
    elif args.post and not args.pre:
        run_pre, run_post = False, True
    else:
        run_pre, run_post = True, True

    config = load_config()
    dm = DataManager(config)
    if args.out_json:
        results_path = Path(args.out_json)
        results_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_path = config.PROJECT_ROOT / "RunBatch_Output" / "sweep_results.json"

    # Load source brain (parameterised via --source-brain)
    source_brain = args.source_brain
    paths = dm.get_patient_paths(source_brain)
    if not paths["pre_op"] or not paths["pre_op"].exists():
        logger.error(f"{source_brain} pre_op not found. Run --patient {source_brain} first.")
        return
    logger.info(f"Source brain: {source_brain}  →  JSON: {results_path}")

    pre_img = sitk.DICOMOrient(
        sitk.ReadImage(str(paths["pre_op"])), "RAS"
    )
    if paths["brain_mask"] and paths["brain_mask"].exists():
        mask_img = sitk.DICOMOrient(
            sitk.ReadImage(str(paths["brain_mask"])), "RAS"
        )
    else:
        logger.warning("Brain mask not found — using Otsu fallback")
        from src.backends.bet_sitk import OtsuExtractor
        mask_img = OtsuExtractor().extract(sitk.Cast(pre_img, sitk.sitkFloat32))

    # Load real post-op for POST-variant generation
    post_op_img: sitk.Image | None = None
    if run_post:
        post_op_path = paths.get("post_op")
        if post_op_path and post_op_path.exists():
            post_op_img = sitk.DICOMOrient(sitk.ReadImage(str(post_op_path)), "RAS")
            logger.info(f"Loaded {source_brain} post-op: {post_op_path}")
        else:
            logger.error(
                f"{source_brain} post_op not found. Cannot generate POST variants. "
                f"Expected: RunBatch_Completed/{source_brain}/1/original_post_op.nii.gz"
            )
            return

    # ── Resample very-fine anisotropic sources to isotropic (fixes identity-stall) ──
    # Must run BEFORE the cache pre-warm and case generation so mask/seg/GT/registration
    # all share the resampled grid.
    pre_img, mask_img, post_op_img = _maybe_resample_source(
        pre_img, mask_img, post_op_img, args.iso_resample_mm
    )

    # ── Pre-warm source cache: BET+FAST on the RAS-oriented source once ──────
    # When --cache-source is set, compute these once and reuse for every case.
    cache_mask_path: Path | None = None
    cache_seg_path:  Path | None = None
    if args.cache_source:
        cache_dir = config.PROJECT_ROOT / "RunBatch_Output" / f"_cache_{source_brain}_RAS"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_mask_path = cache_dir / "brain_mask.nii.gz"
        cache_seg_path  = cache_dir / "seg_pre.nii.gz"

        if not cache_mask_path.exists():
            logger.info(f"  [Pre-warm] Writing RAS source brain mask → {cache_mask_path}")
            sitk.WriteImage(sitk.Cast(mask_img, sitk.sitkUInt8), str(cache_mask_path))

        if not cache_seg_path.exists():
            # MUST match the segmenter the synthetic cases use for the moving (post)
            # seg. Synthetic cases force use_fsl=False (register_tissue.py), so post_seg
            # is k-means. If the cached pre_seg used FAST instead, FAST(3 tissue classes)
            # vs k-means(4 intensity levels) label mismatch makes the report Dice collapse
            # to ~0 and biases the tissue-guided ablation. Keep both on k-means.
            logger.info(f"  [Pre-warm] Running k-means segmentation on RAS source pre-op...")
            from src.backends.segmenter_sitk import KMeansSegmenter
            segmenter = KMeansSegmenter()
            pre_img_f32 = sitk.Cast(pre_img, sitk.sitkFloat32)
            mask_u8     = sitk.Cast(mask_img, sitk.sitkUInt8)
            pre_seg_cached = segmenter.segment(sitk.Mask(pre_img_f32, mask_u8), mask_u8)
            sitk.WriteImage(sitk.Cast(pre_seg_cached, sitk.sitkUInt8), str(cache_seg_path))
            logger.success(f"  [Pre-warm] Cached seg_pre → {cache_seg_path}")
        else:
            logger.info(f"  [Pre-warm] Cache already populated: {cache_dir}")

    # Load existing results (for --resume)
    all_results: dict = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = restore_nan_from_json(json.load(f))

    # --- Selection Logic ---
    if args.single:
        # Search in ALL lists (pilot, multibrain, full)
        all_possible = (build_test_cases(cases_set="pilot")
                        + build_test_cases(cases_set="multibrain")
                        + build_test_cases(cases_set="full"))
        cases = [c for c in all_possible if c["name"] == args.single]
        if not cases:
            logger.error(f"Single case '{args.single}' not found in any definition.")
            return
        mode = f"SINGLE:{args.single}"
    else:
        cases = build_test_cases(pilot=args.pilot, cases_set=args.cases_set)
        mode = ("MULTIBRAIN" if args.cases_set == "multibrain"
                else "PILOT" if (args.pilot or args.cases_set == "pilot")
                else "FULL")

    passes_label = ("PRE+POST" if (run_pre and run_post)
                    else "PRE" if run_pre else "POST")
    logger.section(f"Sweep [{mode}] [{passes_label}]: {len(cases)} base cases")

    if args.dry_run:
        for i, c in enumerate(cases):
            if run_pre:
                logger.info(
                    f"  [{i+1:2d}/{len(cases)}] {c['name']:<40s} "
                    f"A={c['A_mm']:5.1f}mm  s={c['sigma_mm']:5.1f}mm  "
                    f"dir={c['direction']:<4s}  loc={c['location']}"
                )
            if run_post:
                logger.info(f"         + POST variant: {c['name']}_POST")
        n_pre  = len(cases) if run_pre  else 0
        n_post = len(cases) if run_post else 0
        logger.success(f"Dry-run: {n_pre} PRE + {n_post} POST cases would run.")
        return

    t_total = time.monotonic()

    pre_cases = cases if run_pre else []
    for i, case in enumerate(pre_cases):
        pid    = case["name"]           # JSON key — always unprefixed
        fs_pid = pid_prefix + pid       # file-system name — namespaced per brain when parallel
        logger.section(f"[PRE {i+1}/{len(pre_cases)}] {pid}")

        if args.resume and pid in all_results and all_results[pid].get("status") == "OK":
            logger.info("  Already OK — skipping.")
            continue

        t0 = time.monotonic()

        # 1. Generate synthetic patient
        loc     = case["location"]
        cf      = LOCATIONS.get(loc, (0.5, 0.5, 0.5))
        d_vec   = DIRECTIONS[case["direction"]]
        logger.info(
            f"  Generating: A={case['A_mm']}mm  s={case['sigma_mm']}mm  "
            f"dir={case['direction']}  loc={loc}"
        )
        try:
            generate_gauss3d_test(
                pre_img, mask_img,
                test_name=fs_pid,
                direction_vec=d_vec,
                A_mm=case["A_mm"],
                sigma_mm=case["sigma_mm"],
                center_frac=cf,
            )
        except Exception as e:
            logger.error(f"  Generation failed: {e}")
            continue

        # 2a. Seed case's tissue_registration dir with cached source mask + seg.
        # ALWAYS overwrite to avoid stale files from earlier runs with a different
        # source brain or different reorientation grid (was a real bug: stale 240×240
        # masks from older runs persisted into the case dir and caused size mismatches
        # downstream in brain_shift_analysis.run_segmentation).
        if args.cache_source and cache_mask_path and cache_seg_path:
            case_reg_dir = config.PROJECT_ROOT / "RunBatch_Output" / fs_pid / "tissue_registration"
            case_reg_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(cache_mask_path, case_reg_dir / "brain_mask.nii.gz")
            shutil.copy(cache_seg_path,  case_reg_dir / "seg_pre.nii.gz")

        # 2b. Run pipeline
        logger.info("  Running pipeline (register-tissue + analyze)...")
        if not run_pipeline(fs_pid):
            all_results[pid] = {**case, "direction": case["direction"],
                                "location": loc, "status": "FAILED"}
            _save(all_results, results_path)
            continue

        # 3. Compute metrics
        ppaths = dm.get_patient_paths(fs_pid)
        pred_path = ppaths["out_dir"] / "brainshift_displacement.nii.gz"
        gt_path   = ppaths.get("gt_displacement")

        if not pred_path.exists() or not gt_path or not gt_path.exists():
            logger.error(f"  Displacement files missing for {pid}")
            all_results[pid] = {**case, "status": "NO_OUTPUT"}
        else:
            mask_path = ppaths.get("brain_mask") or paths["brain_mask"]
            metrics = compute_metrics(pred_path, gt_path, mask_path)
            elapsed = time.monotonic() - t0
            all_results[pid] = {
                **case,
                "source_type":  "pre",
                "status":       "OK",
                "elapsed_min":  elapsed / 60.0,
                **{f"epe_{k}":         v for k, v in metrics["epe"].items()},
                **{f"ang_{k}":         v for k, v in metrics["angular_err"].items()},
                **{f"dot_{k}":         v for k, v in metrics["dot_ratio"].items()},
                **{f"err_x_{k}":       v for k, v in metrics["err_x"].items()},
                **{f"err_y_{k}":       v for k, v in metrics["err_y"].items()},
                **{f"err_z_{k}":       v for k, v in metrics["err_z"].items()},
                **{f"jac_{k}":         v for k, v in metrics["jacobian"].items()},
                "gt_peak_mm":          metrics["gt_peak_mm"],
                "n_voxels":            metrics["n_voxels"],
                "n_sig":               metrics["n_sig"],
                "ang_pct_gt90":        metrics["ang_pct_gt90"],
                "epe_mean_central":    metrics["epe_central"],
                "n_central":           metrics["n_central"],
                **metrics["binned"],
            }
            jac_fold = metrics["jacobian"].get("pct_fold", float("nan"))
            logger.success(
                f"  EPE={metrics['epe']['mean']:.2f}mm(P95={metrics['epe']['p95']:.2f})  "
                f"Ang={metrics['angular_err']['mean']:.1f}°(C={metrics['angular_err']['mean_central']:.1f} W={metrics['angular_err']['weighted_mean']:.1f})  "
                f"DotRatio={metrics['dot_ratio']['mean']:.3f}  "
                f"JacFold={jac_fold:.1f}%  "
                f"GT_peak={metrics['gt_peak_mm']:.1f}mm  "
                f"n_sig={metrics['n_sig']:,}  "
                f"[{elapsed/60:.1f} min]"
            )

        # 4. Save partial results after each case (crash-safe)
        _save(all_results, results_path)

        # 5. Cleanup (saves ~200 MB/case)
        if not args.no_cleanup:
            for d in [config.PROJECT_ROOT / "RunBatch_Input" / fs_pid,
                      ppaths["out_dir"]]:
                if d.exists():
                    try:
                        shutil.rmtree(d)
                    except Exception:
                        pass

    # ── SECOND PASS: POST VARIANTS ─────────────────────────────────────────────
    if run_post and post_op_img is not None:
        logger.section("POST-variant pass (real post-op + same Gaussian shift)")
        for i, case in enumerate(cases):
            pid         = case["name"]          # JSON key — unprefixed
            pid_post    = pid + "_POST"          # JSON key — unprefixed
            fs_pid_post = pid_prefix + pid_post  # file-system name — namespaced per brain
            logger.section(f"[POST {i+1}/{len(cases)}] {pid_post}")

            if args.resume and pid_post in all_results and all_results[pid_post].get("status") == "OK":
                logger.info("  Already OK — skipping.")
                continue

            t0_post = time.monotonic()
            loc   = case["location"]
            cf    = LOCATIONS.get(loc, (0.5, 0.5, 0.5))
            d_vec = DIRECTIONS[case["direction"]]

            try:
                generate_gauss3d_test_post_variant(
                    pre_img=pre_img, pre_mask=mask_img, post_img=post_op_img,
                    test_name=fs_pid_post, direction_vec=d_vec,
                    A_mm=case["A_mm"], sigma_mm=case["sigma_mm"], center_frac=cf,
                )
            except Exception as e:
                logger.error(f"  POST generation failed: {e}")
                all_results[pid_post] = {**case, "name": pid_post, "source_type": "post", "status": "FAILED"}
                _save(all_results, results_path)
                continue

            if args.cache_source and cache_mask_path and cache_seg_path:
                case_reg_dir_p = config.PROJECT_ROOT / "RunBatch_Output" / fs_pid_post / "tissue_registration"
                case_reg_dir_p.mkdir(parents=True, exist_ok=True)
                # Always overwrite (see PRE pass comment for rationale)
                shutil.copy(cache_mask_path, case_reg_dir_p / "brain_mask.nii.gz")
                shutil.copy(cache_seg_path,  case_reg_dir_p / "seg_pre.nii.gz")

            logger.info(f"  Running pipeline for {pid_post}...")
            if not run_pipeline(fs_pid_post):
                all_results[pid_post] = {**case, "name": pid_post, "source_type": "post",
                                         "location": loc, "status": "FAILED"}
                _save(all_results, results_path)
                continue

            ppaths_post    = dm.get_patient_paths(fs_pid_post)
            pred_path_post = ppaths_post["out_dir"] / "brainshift_displacement.nii.gz"
            gt_path_post   = ppaths_post.get("gt_displacement")

            if not pred_path_post.exists() or not gt_path_post or not gt_path_post.exists():
                logger.error(f"  Displacement files missing for {pid_post}")
                all_results[pid_post] = {**case, "name": pid_post, "source_type": "post", "status": "NO_OUTPUT"}
            else:
                mask_path_post = ppaths_post.get("brain_mask") or paths["brain_mask"]
                metrics_post   = compute_metrics(pred_path_post, gt_path_post, mask_path_post)
                elapsed_post   = time.monotonic() - t0_post
                all_results[pid_post] = {
                    **case,
                    "name":         pid_post,
                    "source_type":  "post",
                    "status":       "OK",
                    "elapsed_min":  elapsed_post / 60.0,
                    **{f"epe_{k}":    v for k, v in metrics_post["epe"].items()},
                    **{f"ang_{k}":    v for k, v in metrics_post["angular_err"].items()},
                    **{f"dot_{k}":    v for k, v in metrics_post["dot_ratio"].items()},
                    **{f"err_x_{k}":  v for k, v in metrics_post["err_x"].items()},
                    **{f"err_y_{k}":  v for k, v in metrics_post["err_y"].items()},
                    **{f"err_z_{k}":  v for k, v in metrics_post["err_z"].items()},
                    **{f"jac_{k}":    v for k, v in metrics_post["jacobian"].items()},
                    "gt_peak_mm":      metrics_post["gt_peak_mm"],
                    "n_voxels":        metrics_post["n_voxels"],
                    "n_sig":           metrics_post["n_sig"],
                    "ang_pct_gt90":    metrics_post["ang_pct_gt90"],
                    "epe_mean_central":metrics_post["epe_central"],
                    "n_central":       metrics_post["n_central"],
                    **metrics_post["binned"],
                }
                jac_fold_post = metrics_post["jacobian"].get("pct_fold", float("nan"))
                logger.success(
                    f"  [POST] EPE={metrics_post['epe']['mean']:.2f}mm  "
                    f"Ang={metrics_post['angular_err']['mean']:.1f}°(C={metrics_post['angular_err']['mean_central']:.1f} W={metrics_post['angular_err']['weighted_mean']:.1f})  "
                    f"Dot={metrics_post['dot_ratio']['mean']:.3f}  "
                    f"JacFold={jac_fold_post:.1f}%  "
                    f"[{elapsed_post/60:.1f} min]"
                )

            _save(all_results, results_path)

            if not args.no_cleanup:
                for d in [config.PROJECT_ROOT / "RunBatch_Input" / fs_pid_post,
                          ppaths_post["out_dir"]]:
                    if d.exists():
                        try:
                            shutil.rmtree(d)
                        except Exception:
                            pass

    total_min = (time.monotonic() - t_total) / 60
    ok_pre  = sum(1 for v in all_results.values()
                  if isinstance(v, dict) and v.get("status") == "OK"
                  and v.get("source_type", "pre") == "pre")
    ok_post = sum(1 for v in all_results.values()
                  if isinstance(v, dict) and v.get("status") == "OK"
                  and v.get("source_type") == "post")
    n_post_expected = len(cases) if run_post else 0
    logger.success(
        f"\nSweep [{mode}] completed: {ok_pre}/{len(cases)} PRE OK  |  "
        f"{ok_post}/{n_post_expected} POST OK  |  {total_min:.0f} min total"
    )
    logger.info(f"Results: {results_path}")


def _save(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sanitize_for_json(results), f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
