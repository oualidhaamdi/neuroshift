"""
Synthetic Patient Generator.

Applies a known 3D Gaussian displacement field to a real brain MRI so that
registration accuracy can be measured against ground truth.

Public API
----------
apply_gauss3d_shift(img, mask_img, direction_vec, A_mm, sigma_mm, center_frac)
generate_gauss3d_test(pre_img, mask_img, test_name, direction_vec, A_mm, sigma_mm, center_frac)
generate_gauss3d_test_post_variant(pre_img, pre_mask, post_img, test_name, direction_vec, A_mm, sigma_mm, center_frac)
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import load_config, setup_logger
from src.data_manager import DataManager

logger = setup_logger("create_synthetic")

# Anatomical location presets (fractions of brain bounding box in X,Y,Z)
LOCATIONS = {
    "center":    (0.50, 0.50, 0.50),
    "anterior":  (0.50, 0.25, 0.50),   # frontal lobe
    "posterior": (0.50, 0.75, 0.50),   # occipital lobe
    "lateral_R": (0.75, 0.50, 0.50),   # right hemisphere
    "superior":  (0.50, 0.50, 0.75),   # vertex
}


def apply_gauss3d_shift(
    img: sitk.Image,
    mask_img: sitk.Image,
    direction_vec,
    A_mm: float,
    sigma_mm: float,
    center_frac: tuple = (0.5, 0.5, 0.5),
) -> tuple[sitk.Image, sitk.Image]:
    """
    Apply a smooth 3D Gaussian displacement field to img.

    The Gaussian is centred at a point derived from the brain bounding box
    via center_frac = (fx, fy, fz) where each component is in [0, 1].
    (0.5, 0.5, 0.5) = brain centre of mass (default).

    The displacement field is stored as a PULL-BACK: each voxel stores the
    sampling offset used to build the synthetic post-op. The actual tissue
    movement direction is therefore -direction_vec.

    Returns
    -------
    shifted_img : sitk.Image   — synthetic post-op
    field_img   : sitk.Image   — pull-back displacement field (VectorFloat32)
    """
    sp = img.GetSpacing()
    sz = img.GetSize()

    mask_arr = sitk.GetArrayFromImage(mask_img)  # (Z, Y, X)

    # Bounding box of the brain mask
    z_idx, y_idx, x_idx = np.where(mask_arr > 0)
    bb_x = (x_idx.min(), x_idx.max())
    bb_y = (y_idx.min(), y_idx.max())
    bb_z = (z_idx.min(), z_idx.max())

    # Centre in voxel coordinates, then convert to mm
    cx_vox = bb_x[0] + center_frac[0] * (bb_x[1] - bb_x[0])
    cy_vox = bb_y[0] + center_frac[1] * (bb_y[1] - bb_y[0])
    cz_vox = bb_z[0] + center_frac[2] * (bb_z[1] - bb_z[0])
    cx_mm = cx_vox * sp[0]
    cy_mm = cy_vox * sp[1]
    cz_mm = cz_vox * sp[2]

    # Build Gaussian field in (Z, Y, X) array order
    xs = np.arange(sz[0]) * sp[0]
    ys = np.arange(sz[1]) * sp[1]
    zs = np.arange(sz[2]) * sp[2]
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")

    dist2 = (X - cx_mm) ** 2 + (Y - cy_mm) ** 2 + (Z - cz_mm) ** 2
    gauss = A_mm * np.exp(-dist2 / (2.0 * sigma_mm ** 2))

    d = np.array(direction_vec, dtype=float)
    d /= np.linalg.norm(d) + 1e-9

    inside = mask_arr > 0
    dx = np.zeros_like(gauss, dtype=np.float32)
    dy = np.zeros_like(gauss, dtype=np.float32)
    dz = np.zeros_like(gauss, dtype=np.float32)
    dx[inside] = (gauss * d[0]).astype(np.float32)[inside]
    dy[inside] = (gauss * d[1]).astype(np.float32)[inside]
    dz[inside] = (gauss * d[2]).astype(np.float32)[inside]

    def _to_sitk(arr):
        im = sitk.GetImageFromArray(arr)
        im.CopyInformation(img)
        return im

    field_img = sitk.Compose(_to_sitk(dx), _to_sitk(dy), _to_sitk(dz))

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(img)
    resample.SetTransform(
        sitk.DisplacementFieldTransform(sitk.Cast(field_img, sitk.sitkVectorFloat64))
    )
    resample.SetInterpolator(sitk.sitkLinear)
    shifted = resample.Execute(img)

    peak_mag = float(
        np.max(np.linalg.norm(np.stack([dx, dy, dz], axis=-1), axis=-1))
    )
    logger.info(
        f"  Gaussian 3D | A={A_mm}mm σ={sigma_mm}mm | peak={peak_mag:.2f}mm "
        f"| dir={np.round(d, 2)} | center_frac={center_frac}"
    )
    return shifted, sitk.Cast(field_img, sitk.sitkVectorFloat32)


def generate_gauss3d_test(
    pre_img: sitk.Image,
    mask_img: sitk.Image,
    test_name: str,
    direction_vec,
    A_mm: float = 10.0,
    sigma_mm: float = 25.0,
    center_frac: tuple = (0.5, 0.5, 0.5),
) -> None:
    """
    Write a synthetic test case to RunBatch_Input/<test_name>/.

    Directory layout
    ----------------
    0/t1.nii.gz          — pre-op (source brain, RAS-oriented)
    1/t1.nii.gz          — synthetic post-op
    brain_mask.nii.gz    — brain mask reused from source
    gt_displacement.nii.gz — pull-back field (see apply_gauss3d_shift docstring)
    test_info.txt        — metadata (direction, A, sigma, location, centre slice)
    """
    config = load_config()
    test_dir = config.PROJECT_ROOT / "RunBatch_Input" / test_name
    if test_dir.exists():
        shutil.rmtree(str(test_dir))
    (test_dir / "0").mkdir(parents=True, exist_ok=True)
    (test_dir / "1").mkdir(parents=True, exist_ok=True)

    pre_img  = sitk.DICOMOrient(pre_img,  "RAS")
    mask_img = sitk.DICOMOrient(mask_img, "RAS")

    post_img, field_img = apply_gauss3d_shift(
        pre_img, mask_img, direction_vec, A_mm, sigma_mm, center_frac
    )

    sitk.WriteImage(sitk.Cast(pre_img,  sitk.sitkFloat32), str(test_dir / "0" / "t1.nii.gz"))
    sitk.WriteImage(sitk.Cast(post_img, sitk.sitkFloat32), str(test_dir / "1" / "t1.nii.gz"))
    sitk.WriteImage(sitk.Cast(mask_img, sitk.sitkUInt8),   str(test_dir / "brain_mask.nii.gz"))
    sitk.WriteImage(field_img,                              str(test_dir / "gt_displacement.nii.gz"))

    # Axial centre slice for viewer golden-border highlight
    mask_arr = sitk.GetArrayFromImage(mask_img)
    z_idx, y_idx, x_idx = np.where(mask_arr > 0)
    bb_z = (z_idx.min(), z_idx.max())
    bb_y = (y_idx.min(), y_idx.max())
    bb_x = (x_idx.min(), x_idx.max())
    z_c = int(bb_z[0] + center_frac[2] * (bb_z[1] - bb_z[0]))

    d = np.array(direction_vec, dtype=float)
    d /= np.linalg.norm(d) + 1e-9

    # Derive location name from known presets (or "custom")
    loc_name = "custom"
    for name, frac in LOCATIONS.items():
        if np.allclose(center_frac, frac, atol=0.01):
            loc_name = name
            break

    with open(test_dir / "test_info.txt", "w") as f:
        f.write(f"type=gauss3d\n")
        f.write(f"direction={d[0]:.3f},{d[1]:.3f},{d[2]:.3f}\n")
        f.write(f"amplitude_mm={A_mm}\n")
        f.write(f"sigma_mm={sigma_mm}\n")
        f.write(f"center_frac={center_frac[0]:.2f},{center_frac[1]:.2f},{center_frac[2]:.2f}\n")
        f.write(f"location={loc_name}\n")
        f.write(f"axis=0\n")
        f.write(f"slice={z_c}\n")

    logger.success(
        f"Test '{test_name}' created — A={A_mm}mm σ={sigma_mm}mm "
        f"dir={np.round(d,2)} loc={loc_name} z_c={z_c}"
    )


def generate_cavity_mismatch_test(
    pre_img: sitk.Image,
    mask_img: sitk.Image,
    test_name: str,
    direction_vec,
    A_mm: float = 5.0,
    sigma_mm: float = 25.0,
    center_frac: tuple = (0.5, 0.5, 0.5),
    cavity_radius_mm: float = 15.0,
    cavity_intensity_factor: float = 0.05,
) -> None:
    """
    Write a synthetic case with REAL intensity discontinuity:
        Fixed (0/)  = source pre-op  (no cavity, no shift)
        Moving (1/) = source pre-op + Gaussian shift + DARK SPHERICAL CAVITY
        GT          = the same Gaussian field as the standard sweep

    This is the scenario the tissue-guided approach was designed for:
    intensity-based registration is expected to fail near the cavity because
    MSE on intensities tries to "explain" the dark hole by deforming nearby
    tissue inward.  Tissue-guided B-Spline (on distance maps of tissue edges)
    should be largely unaffected outside the cavity.

    Output directory layout (identical to generate_gauss3d_test):
      0/t1.nii.gz          — pre-op (no cavity)
      1/t1.nii.gz          — pre-op + shift + cavity carve-out
      brain_mask.nii.gz    — source brain mask (RAS)
      gt_displacement.nii.gz — pull-back Gaussian field
      mask_cavity_manual.nii.gz — sphere mask of the carve-out (for QC)
      test_info.txt
    """
    config = load_config()
    test_dir = config.PROJECT_ROOT / "RunBatch_Input" / test_name
    if test_dir.exists():
        shutil.rmtree(str(test_dir))
    (test_dir / "0").mkdir(parents=True, exist_ok=True)
    (test_dir / "1").mkdir(parents=True, exist_ok=True)

    pre_img  = sitk.DICOMOrient(pre_img,  "RAS")
    mask_img = sitk.DICOMOrient(mask_img, "RAS")

    # 1. Apply Gaussian shift → shifted image + GT field
    shifted_img, field_img = apply_gauss3d_shift(
        pre_img, mask_img, direction_vec, A_mm, sigma_mm, center_frac
    )

    # 2. Build a spherical cavity mask centred at center_frac
    mask_arr = sitk.GetArrayFromImage(mask_img)
    sp       = pre_img.GetSpacing()
    sz_arr   = mask_arr.shape  # (Z, Y, X)
    z_idx, y_idx, x_idx = np.where(mask_arr > 0)
    bb_x = (x_idx.min(), x_idx.max())
    bb_y = (y_idx.min(), y_idx.max())
    bb_z = (z_idx.min(), z_idx.max())
    cx_mm = (bb_x[0] + center_frac[0] * (bb_x[1] - bb_x[0])) * sp[0]
    cy_mm = (bb_y[0] + center_frac[1] * (bb_y[1] - bb_y[0])) * sp[1]
    cz_mm = (bb_z[0] + center_frac[2] * (bb_z[1] - bb_z[0])) * sp[2]

    xs = np.arange(sz_arr[2]) * sp[0]
    ys = np.arange(sz_arr[1]) * sp[1]
    zs = np.arange(sz_arr[0]) * sp[2]
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    dist2 = (X - cx_mm) ** 2 + (Y - cy_mm) ** 2 + (Z - cz_mm) ** 2
    cavity_arr = (dist2 < cavity_radius_mm ** 2) & (mask_arr > 0)

    # 3. Carve cavity in the shifted (moving) image → dark CSF-like intensity
    shifted_arr = sitk.GetArrayFromImage(shifted_img).astype(np.float32)
    fg = shifted_arr[shifted_arr > 0]
    bg_target = float(np.percentile(fg, 5)) * cavity_intensity_factor if fg.size else 0.0
    shifted_arr[cavity_arr] = bg_target

    moving_img = sitk.GetImageFromArray(shifted_arr)
    moving_img.CopyInformation(pre_img)

    cavity_img = sitk.GetImageFromArray(cavity_arr.astype(np.uint8))
    cavity_img.CopyInformation(pre_img)

    # 4. Write outputs
    sitk.WriteImage(sitk.Cast(pre_img,    sitk.sitkFloat32), str(test_dir / "0" / "t1.nii.gz"))
    sitk.WriteImage(sitk.Cast(moving_img, sitk.sitkFloat32), str(test_dir / "1" / "t1.nii.gz"))
    sitk.WriteImage(sitk.Cast(mask_img,   sitk.sitkUInt8),   str(test_dir / "brain_mask.nii.gz"))
    sitk.WriteImage(field_img,                                str(test_dir / "gt_displacement.nii.gz"))
    sitk.WriteImage(cavity_img,                               str(test_dir / "mask_cavity_manual.nii.gz"))

    d = np.array(direction_vec, dtype=float)
    d /= np.linalg.norm(d) + 1e-9
    with open(test_dir / "test_info.txt", "w") as f:
        f.write(f"type=cavity_mismatch\n")
        f.write(f"direction={d[0]:.3f},{d[1]:.3f},{d[2]:.3f}\n")
        f.write(f"amplitude_mm={A_mm}\n")
        f.write(f"sigma_mm={sigma_mm}\n")
        f.write(f"center_frac={center_frac[0]:.2f},{center_frac[1]:.2f},{center_frac[2]:.2f}\n")
        f.write(f"cavity_radius_mm={cavity_radius_mm}\n")
        f.write(f"cavity_intensity_factor={cavity_intensity_factor}\n")

    n_cavity = int(cavity_arr.sum())
    logger.success(
        f"Cavity-mismatch case '{test_name}' — A={A_mm}mm σ={sigma_mm}mm "
        f"cavity R={cavity_radius_mm}mm  voxels={n_cavity:,}  "
        f"darkened to {bg_target:.1f}"
    )


def generate_gauss3d_test_post_variant(
    pre_img: sitk.Image,
    pre_mask: sitk.Image,
    post_img: sitk.Image,
    test_name: str,
    direction_vec,
    A_mm: float = 10.0,
    sigma_mm: float = 25.0,
    center_frac: tuple = (0.5, 0.5, 0.5),
) -> None:
    """
    Write a POST-variant synthetic test case to RunBatch_Input/<test_name>/.

    The real Post-Op image (containing the surgical cavity) is rigidly aligned
    to Pre-Op space before the same Gaussian displacement is applied. The
    ground-truth displacement is computed from the Pre-Op brain mask bounding
    box, making it numerically identical to the PRE-variant GT.

    Both fixed and moving come from the real Post-Op (rigidly aligned to Pre-Op
    space so the Gaussian hits the same anatomical region). The B-Spline works
    entirely within the "cavity world".

    Experiment design:
      PRE case: Fixed=Pre-Op,      Moving=Pre-Op+Gaussian      (no cavity)
      POST case: Fixed=Post-aligned, Moving=Post-aligned+Gaussian (cavity present)

    Directory layout
    ----------------
    0/t1.nii.gz             — rigidly-aligned Post-Op (fixed, with cavity)
    1/t1.nii.gz             — rigidly-aligned Post-Op + Gaussian shift (moving)
    brain_mask.nii.gz       — pre-op brain mask (same space after alignment)
    gt_displacement.nii.gz  — pull-back field (identical to PRE variant GT)
    test_info.txt           — same fields as PRE + source_type=post
    """
    config = load_config()
    test_dir = config.PROJECT_ROOT / "RunBatch_Input" / test_name
    if test_dir.exists():
        shutil.rmtree(str(test_dir))
    (test_dir / "0").mkdir(parents=True, exist_ok=True)
    (test_dir / "1").mkdir(parents=True, exist_ok=True)

    pre_img  = sitk.DICOMOrient(pre_img,  "RAS")
    pre_mask = sitk.DICOMOrient(pre_mask, "RAS")
    post_img = sitk.DICOMOrient(post_img, "RAS")

    # Rigid alignment: align real post-op → pre-op space.
    # Use SetMetricFixedMask so the cavity region does not pull the optimizer.
    logger.info(f"  [{test_name}] Rigidly aligning real post-op → pre-op space...")
    fixed_f  = sitk.Cast(pre_img,  sitk.sitkFloat32)
    moving_f = sitk.Cast(post_img, sitk.sitkFloat32)

    initial_tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricFixedMask(sitk.Cast(pre_mask, sitk.sitkUInt8))
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.08)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=150,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_tx)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    rigid_tx = reg.Execute(fixed_f, moving_f)
    logger.info(
        f"  [{test_name}] Rigid done: metric={reg.GetMetricValue():.4f}  "
        f"stop='{reg.GetOptimizerStopConditionDescription()}'"
    )

    post_aligned = sitk.Resample(post_img, pre_img, rigid_tx, sitk.sitkLinear, 0.0)

    # Apply SAME Gaussian as PRE variant using pre_mask → identical GT field
    post_shifted, field_img = apply_gauss3d_shift(
        post_aligned, pre_mask, direction_vec, A_mm, sigma_mm, center_frac
    )

    # Fixed = aligned Post-Op (with cavity), Moving = aligned Post-Op + Gaussian.
    # Both images come from the real post-op, so the B-Spline works entirely
    # within the "cavity world" — this is the correct experiment design.
    sitk.WriteImage(sitk.Cast(post_aligned, sitk.sitkFloat32), str(test_dir / "0" / "t1.nii.gz"))
    sitk.WriteImage(sitk.Cast(post_shifted, sitk.sitkFloat32), str(test_dir / "1" / "t1.nii.gz"))
    sitk.WriteImage(sitk.Cast(pre_mask,     sitk.sitkUInt8),   str(test_dir / "brain_mask.nii.gz"))
    sitk.WriteImage(field_img,                                  str(test_dir / "gt_displacement.nii.gz"))

    mask_arr = sitk.GetArrayFromImage(pre_mask)
    z_idx, y_idx, x_idx = np.where(mask_arr > 0)
    bb_z = (z_idx.min(), z_idx.max())
    z_c  = int(bb_z[0] + center_frac[2] * (bb_z[1] - bb_z[0]))

    d = np.array(direction_vec, dtype=float)
    d /= np.linalg.norm(d) + 1e-9

    loc_name = "custom"
    for name, frac in LOCATIONS.items():
        if np.allclose(center_frac, frac, atol=0.01):
            loc_name = name
            break

    with open(test_dir / "test_info.txt", "w") as f:
        f.write(f"type=gauss3d\n")
        f.write(f"direction={d[0]:.3f},{d[1]:.3f},{d[2]:.3f}\n")
        f.write(f"amplitude_mm={A_mm}\n")
        f.write(f"sigma_mm={sigma_mm}\n")
        f.write(f"center_frac={center_frac[0]:.2f},{center_frac[1]:.2f},{center_frac[2]:.2f}\n")
        f.write(f"location={loc_name}\n")
        f.write(f"axis=0\n")
        f.write(f"slice={z_c}\n")
        f.write(f"source_type=post\n")

    logger.success(
        f"POST-variant '{test_name}' created — A={A_mm}mm σ={sigma_mm}mm "
        f"dir={np.round(d,2)} loc={loc_name}"
    )


def main(source_brain: str | None = None):
    import argparse
    if source_brain is None:
        parser = argparse.ArgumentParser(description="Generate the 6 pilot synthetic cases")
        parser.add_argument("--source-brain", default=None,
                            help="Case ID under RunBatch_Input/ used as source anatomy "
                                 "(default: config.ACTIVE_PATIENT)")
        source_brain = parser.parse_args().source_brain

    config = load_config()
    dm = DataManager(config)
    source_brain = source_brain or config.ACTIVE_PATIENT
    paths = dm.get_patient_paths(source_brain)

    if not paths["pre_op"] or not paths["pre_op"].exists():
        logger.error(f"{source_brain}: pre_op not found.")
        return

    pre_img = sitk.DICOMOrient(
        sitk.ReadImage(str(paths["pre_op"])), "RAS"
    )
    mask_path = paths["brain_mask"]
    if mask_path and mask_path.exists():
        mask_img = sitk.ReadImage(str(mask_path))
    else:
        logger.warning("No brain mask found — computing Otsu fallback")
        from src.backends.bet_sitk import OtsuExtractor
        mask_img = OtsuExtractor().extract(sitk.Cast(pre_img, sitk.sitkFloat32))

    logger.info(f"Source: {source_brain}  |  size: {pre_img.GetSize()}")

    # Six pilot test cases
    pilot_cases = [
        ("Pilot_A050_S250_X_C",   [1,0,0],     5.0, 25.0, (0.50, 0.50, 0.50)),
        ("Pilot_A050_S250_Y_C",   [0,1,0],     5.0, 25.0, (0.50, 0.50, 0.50)),
        ("Pilot_A050_S250_Z_C",   [0,0,1],     5.0, 25.0, (0.50, 0.50, 0.50)),
        ("Pilot_A050_S250_XYZ_C", [1,1,0.5],   5.0, 25.0, (0.50, 0.50, 0.50)),
        ("Pilot_A150_S250_X_C",   [1,0,0],    15.0, 25.0, (0.50, 0.50, 0.50)),
        ("Pilot_A050_S250_X_ANT", [1,0,0],     5.0, 25.0, (0.50, 0.25, 0.50)),
    ]
    for name, d, a, s, cf in pilot_cases:
        generate_gauss3d_test(pre_img, mask_img, name, d, a, s, cf)

    logger.success("Pilot test cases generated.")


if __name__ == "__main__":
    main()
