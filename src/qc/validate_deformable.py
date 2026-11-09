"""
Deformable Registration Validation.

Generates a comprehensive visual and statistical validation report for
the B-Spline deformable registration:
  - NCC and image comparison before/after
  - Jacobian determinant map with statistics
  - Displacement magnitude map
  - Folding analysis (negative Jacobians)

Outputs → figures/deformable_validation/

Usage:
    python main.py --validate-deformable
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import SimpleITK as sitk

sys.path.append(str(Path(__file__).parents[2]))

from src.core import setup_logger, load_config
from src.metrics.similarity import compute_ncc
from src.metrics.deformation import compute_jacobian_determinant

logger = setup_logger(__name__)


def _load_volume(path: Path, label: str):
    """Load a NIfTI volume or log an error and return None."""
    if not path.exists():
        logger.warning(f"{label} not found: {path}")
        return None
    return sitk.ReadImage(str(path))


def _to_fixed_space(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample image into the physical space of reference (same size/spacing/origin)."""
    if (image.GetSize() == reference.GetSize() and
            np.allclose(image.GetSpacing(), reference.GetSpacing()) and
            np.allclose(image.GetOrigin(), reference.GetOrigin())):
        return image
    return sitk.Resample(
        image, reference,
        sitk.Transform(),
        sitk.sitkLinear, 0.0,
        image.GetPixelID()
    )


def _save_fig(fig, path: Path, name: str):
    fig.savefig(path / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {name}")


def validate_deformable(config, paths: dict):
    logger.section("Validating Deformable Registration")

    output_dir = paths["out_dir"] / "validation" / "deformable"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if not paths["transform"] or not paths["transform"].exists():
        logger.error(f"Deformable transform not found for {paths['id']}")
        logger.info("Run --analyze first.")
        return

    fixed = _load_volume(paths["pre_op"], "Pre-Op")
    if fixed is None:
        return

    transform = sitk.ReadTransform(str(paths["transform"]))
    logger.info(f"Loaded B-Spline transform: {paths['transform'].name}")

    # Try to get the deformed post-op (with B-Spline applied)
    deformed_path = paths["out_dir"] / "post_registered_tissue.nii.gz"
    deformed = _load_volume(deformed_path, "Deformed post-op")

    # If deformed not saved, regenerate it from affine + transform
    if deformed is None:
        affine = _load_volume(paths["post_rigid"], "Affine post-op")
        if affine is not None:
            logger.info("Regenerating deformed volume from transform…")
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(transform)
            deformed = resampler.Execute(sitk.Cast(affine, sitk.sitkFloat32))
            sitk.WriteImage(deformed, str(deformed_path))
            logger.info(f"  Saved: {deformed_path.name}")

    # Displacement and Jacobian – ensure they are in fixed's space
    disp_mag_path = paths["magnitude"]
    jac_path = paths["jacobian"]

    disp_mag_img = _load_volume(disp_mag_path, "Displacement magnitude")
    jac_img = _load_volume(jac_path, "Jacobian")

    if jac_img is None:
        logger.info("Computing Jacobian from transform…")
        jac_arr = compute_jacobian_determinant(transform, fixed)
    else:
        jac_img = _to_fixed_space(jac_img, fixed)
        jac_arr = sitk.GetArrayFromImage(jac_img)

    if disp_mag_img is not None:
        disp_mag_img = _to_fixed_space(disp_mag_img, fixed)

    fixed_arr = sitk.GetArrayFromImage(fixed)
    spacing = fixed.GetSpacing()
    aspect_ax = spacing[1] / spacing[0]

    # ── Metrics ───────────────────────────────────────────────────────────────
    logger.section("Quantitative Metrics")

    # Ensure deformed is in fixed's space before computing NCC
    if deformed is not None:
        deformed = _to_fixed_space(sitk.Cast(deformed, sitk.sitkFloat32), fixed)
        deformed_arr = sitk.GetArrayFromImage(deformed)
        ncc_val = compute_ncc(fixed_arr, deformed_arr)
        logger.info(f"  NCC (Pre vs Deformed Post): {ncc_val:.4f}")
    else:
        deformed_arr = None
        ncc_val = None
        logger.warning("  Deformed volume unavailable – NCC skipped.")

    # NCC vs Affine (improvement)
    affine_img = _load_volume(paths["post_rigid"], "Affine post-op")
    ncc_affine = None
    if affine_img is not None:
        affine_img = _to_fixed_space(sitk.Cast(affine_img, sitk.sitkFloat32), fixed)
        affine_arr = sitk.GetArrayFromImage(affine_img)
        ncc_affine = compute_ncc(fixed_arr, affine_arr)
        logger.info(f"  NCC (Pre vs Affine Post):   {ncc_affine:.4f}")
        if ncc_val is not None:
            delta = ncc_val - ncc_affine
            logger.info(f"  NCC improvement (Deformable - Affine): {delta:+.4f}")

    # Jacobian statistics
    jac_flat = jac_arr.flatten()
    neg_count = int(np.sum(jac_flat < 0))
    pct_neg = 100.0 * neg_count / len(jac_flat)
    logger.info(f"\n  Jacobian Statistics:")
    logger.info(f"    Min:           {np.min(jac_flat):.4f}")
    logger.info(f"    Max:           {np.max(jac_flat):.4f}")
    logger.info(f"    Mean:          {np.mean(jac_flat):.4f}")
    logger.info(f"    Std:           {np.std(jac_flat):.4f}")
    logger.info(f"    Negative vox:  {neg_count:,} ({pct_neg:.2f}%)")

    if neg_count == 0:
        logger.success("  No folding detected – topology preserved.")
    elif pct_neg < 0.5:
        logger.warning(f"  Minor folding: {pct_neg:.2f}% voxels. Consider mild regularisation increase.")
    else:
        logger.error(
            f"  CRITICAL: {pct_neg:.2f}% voxels folded ({neg_count:,}/{len(jac_flat):,}). "
            "Increase B-Spline grid spacing or reduce learning rate."
        )

    # Displacement stats
    if disp_mag_img is not None:
        disp_arr = sitk.GetArrayFromImage(disp_mag_img)
        logger.info(f"\n  Displacement Statistics (mm):")
        logger.info(f"    Mean:   {np.mean(disp_arr):.2f}")
        logger.info(f"    Median: {np.median(disp_arr):.2f}")
        logger.info(f"    P95:    {np.percentile(disp_arr, 95):.2f}")
        logger.info(f"    Max:    {np.max(disp_arr):.2f}")
    else:
        disp_arr = None

    # ── Figure 1: Pre / Post-Affine / Post-Deformable + Diff ─────────────────
    n_slices = fixed_arr.shape[0]
    slices_to_show = [n_slices // 4, n_slices // 2, 3 * n_slices // 4]

    n_rows = 4 if deformed_arr is not None else 3
    fig1, axes = plt.subplots(n_rows, len(slices_to_show), figsize=(15, n_rows * 4))

    vmin, vmax = np.percentile(fixed_arr, [1, 99])
    row_labels = ["Pre-Op", "Post-Op (Affine)", "Post-Op (Deformable)", "Difference (Def - Pre)"]

    for col, z in enumerate(slices_to_show):
        z = min(z, fixed_arr.shape[0] - 1)

        axes[0, col].imshow(fixed_arr[z], cmap='gray', vmin=vmin, vmax=vmax, aspect=aspect_ax)
        axes[0, col].set_title(f"Slice {z}", fontsize=9)
        axes[0, col].axis('off')

        if affine_img is not None:
            az = min(z, affine_arr.shape[0] - 1)
            axes[1, col].imshow(affine_arr[az], cmap='gray', vmin=vmin, vmax=vmax, aspect=aspect_ax)
        axes[1, col].axis('off')

        if deformed_arr is not None:
            dz = min(z, deformed_arr.shape[0] - 1)
            axes[2, col].imshow(deformed_arr[dz], cmap='gray', vmin=vmin, vmax=vmax, aspect=aspect_ax)
            axes[2, col].axis('off')
            diff = deformed_arr[dz] - fixed_arr[z]
            im_diff = axes[3, col].imshow(diff, cmap='RdBu_r',
                                          vmin=np.percentile(diff, 5),
                                          vmax=np.percentile(diff, 95),
                                          aspect=aspect_ax)
            axes[3, col].axis('off')

    for row, label in enumerate(row_labels[:n_rows]):
        axes[row, 0].set_ylabel(label, fontsize=10, labelpad=5)

    ncc_str = f"NCC={ncc_val:.4f}" if ncc_val else "NCC=N/A"
    affine_str = f"| Affine NCC={ncc_affine:.4f}" if ncc_affine else ""
    plt.suptitle(f"Deformable Registration Comparison\n{ncc_str} {affine_str}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig1, output_dir, "01_comparison_multislice.png")

    # ── Figure 2: Jacobian maps ───────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, len(slices_to_show), figsize=(15, 5))

    for col, z in enumerate(slices_to_show):
        z = min(z, jac_arr.shape[0] - 1)
        im = axes2[col].imshow(jac_arr[z], cmap='RdBu_r', vmin=0.5, vmax=1.5, aspect=aspect_ax)
        axes2[col].set_title(f"Slice {z}", fontsize=9)
        axes2[col].axis('off')
        if col == len(slices_to_show) - 1:
            plt.colorbar(im, ax=axes2[col], fraction=0.046, pad=0.04, label='Jacobian det')

    plt.suptitle(
        f"Jacobian Determinant Map\n"
        f"<1: Compression (blue) | >1: Expansion (red) | Neg voxels: {neg_count:,} ({pct_neg:.2f}%)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    _save_fig(fig2, output_dir, "02_jacobian_map.png")

    # ── Figure 3: Jacobian histogram ──────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    # Brain mask preference: resolved path > auto fallback
    brain_mask_img = _load_volume(paths["brain_mask"], "Brain mask")

    if brain_mask_img is not None:
        brain_mask_arr = sitk.GetArrayFromImage(brain_mask_img)
        if brain_mask_arr.shape == jac_arr.shape:
            jac_brain = jac_arr[brain_mask_arr > 0]
        else:
            jac_brain = jac_flat
    else:
        jac_brain = jac_flat

    jac_plot = jac_brain[np.abs(jac_brain) < 5]  # remove wild outliers for display
    ax3.hist(jac_plot, bins=100, color='#3b82f6', edgecolor='none', alpha=0.8, density=True)
    ax3.axvline(1.0, color='#1e293b', lw=2, ls='-', label='J=1 (no deformation)')
    ax3.axvline(0.0, color='#ef4444', lw=2, ls='--', label=f'J=0 ({neg_count:,} vox folded)')
    ax3.set_xlabel("Jacobian Determinant", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title("Jacobian Distribution (brain region)\n"
                  "Ideal: tight peak at 1.0, no mass at J≤0", fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    stats_text = (
        f"Mean = {np.mean(jac_brain):.3f}\n"
        f"Std  = {np.std(jac_brain):.3f}\n"
        f"Min  = {np.min(jac_brain):.3f}\n"
        f"Max  = {np.max(jac_brain):.3f}\n"
        f"J<0  = {neg_count:,} ({pct_neg:.2f}%)"
    )
    ax3.text(0.97, 0.97, stats_text, transform=ax3.transAxes, fontsize=9,
             family='monospace', va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))
    plt.tight_layout()
    _save_fig(fig3, output_dir, "03_jacobian_histogram.png")

    # ── Figure 4: Displacement magnitude map ──────────────────────────────────
    if disp_arr is not None:
        fig4, axes4 = plt.subplots(1, len(slices_to_show), figsize=(15, 5))
        vmax_disp = np.percentile(disp_arr, 99)

        for col, z in enumerate(slices_to_show):
            z = min(z, disp_arr.shape[0] - 1)
            im4 = axes4[col].imshow(disp_arr[z], cmap='inferno',
                                    vmin=0, vmax=vmax_disp, aspect=aspect_ax)
            axes4[col].set_title(f"Slice {z}", fontsize=9)
            axes4[col].axis('off')
            if col == len(slices_to_show) - 1:
                plt.colorbar(im4, ax=axes4[col], fraction=0.046, pad=0.04, label='mm')

        plt.suptitle(
            f"Displacement Magnitude (mm) – Mean={np.mean(disp_arr):.2f}  "
            f"P95={np.percentile(disp_arr, 95):.2f}  Max={np.max(disp_arr):.2f}",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        _save_fig(fig4, output_dir, "04_displacement_magnitude.png")

    # ── Text report ───────────────────────────────────────────────────────────
    report_path = output_dir / "deformable_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("DEFORMABLE REGISTRATION VALIDATION REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Patient: {paths['id']}\n")
        f.write(f"Transform: {paths['transform'].name}\n\n")
        f.write("IMAGE METRICS:\n")
        f.write(f"  NCC (Pre vs Deformable Post): {ncc_val:.4f}\n" if ncc_val else "  NCC: unavailable\n")
        f.write(f"  NCC (Pre vs Affine Post):     {ncc_affine:.4f}\n" if ncc_affine else "")
        f.write("\nJACOBIAN ANALYSIS:\n")
        f.write(f"  Min: {np.min(jac_flat):.4f}  Max: {np.max(jac_flat):.4f}\n")
        f.write(f"  Mean: {np.mean(jac_flat):.4f}  Std: {np.std(jac_flat):.4f}\n")
        f.write(f"  Negative voxels: {neg_count:,} ({pct_neg:.2f}%)\n")
        if neg_count == 0:
            f.write("  STATUS: OK – topology preserved.\n")
        elif pct_neg < 0.5:
            f.write("  STATUS: WARN – minor folding.\n")
        else:
            f.write("  STATUS: CRITICAL – excessive folding. Increase grid_spacing.\n")
        if disp_arr is not None:
            f.write("\nDISPLACEMENT (mm):\n")
            f.write(f"  Mean:   {np.mean(disp_arr):.2f}\n")
            f.write(f"  Median: {np.median(disp_arr):.2f}\n")
            f.write(f"  P95:    {np.percentile(disp_arr, 95):.2f}\n")
            f.write(f"  Max:    {np.max(disp_arr):.2f}\n")

    logger.info(f"  Saved: deformable_validation_report.txt")
    logger.success(f"Validation complete. Output: {output_dir}")


def main():
    import os
    from src.data_manager import DataManager
    config = load_config()
    dm = DataManager(config)
    patient_id = os.environ.get("BRAIN_SHIFT_ACTIVE_PATIENT", config.ACTIVE_PATIENT)
    paths = dm.get_patient_paths(patient_id)
    validate_deformable(config, paths)

if __name__ == "__main__":
    main()
