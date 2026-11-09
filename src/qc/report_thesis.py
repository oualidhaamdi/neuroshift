"""
Thesis Report Generator - Final publication-ready figures.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from src.core import setup_logger

logger = setup_logger(__name__)

def generate_thesis_report(config, paths: dict):
    """Generates the 10 final figures for the thesis report."""
    patient_id = paths["id"]
    output_dir = paths["out_dir"]
    figures_dir = output_dir / "thesis_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Clean old figures
    for f in figures_dir.glob("*.png"):
        f.unlink()

    logger.info(f"Generating thesis figures for: {patient_id}")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not paths["pre_op"] or not paths["pre_op"].exists():
        logger.error(f"Pre-Op image not found for {patient_id}")
        return

    reference = sitk.ReadImage(str(paths["pre_op"]))

    # Displacement magnitude
    disp_path = paths["magnitude"]
    if disp_path and disp_path.exists():
        displacement = sitk.ReadImage(str(disp_path))
    else:
        logger.warning("  Displacement field not found, using zeros placeholder.")
        displacement = sitk.Image(reference.GetSize(), sitk.sitkFloat64)
        displacement.CopyInformation(reference)

    # Jacobian
    jac_path = paths["jacobian"]
    if not jac_path or not jac_path.exists():
        logger.error("Jacobian field not found. Run --analyze first.")
        return
    jacobian = sitk.ReadImage(str(jac_path))

    # Brain Mask
    if not paths["brain_mask"] or not paths["brain_mask"].exists():
        logger.error(f"Brain mask not found for {patient_id}")
        return
    brain_mask = sitk.ReadImage(str(paths["brain_mask"]))

    # Cavity Mask
    cavity_path = output_dir / "mask_cavity_auto.nii.gz"
    if not cavity_path.exists():
        cavity_path = output_dir / "mask_cavity_manual.nii.gz"

    if cavity_path.exists():
        cavity_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(cavity_path)))
    else:
        cavity_arr = np.zeros(sitk.GetArrayFromImage(brain_mask).shape, dtype=np.uint8)

    # Transform for 3D vectors
    if not paths["transform"] or not paths["transform"].exists():
        logger.error(f"Transform not found for {patient_id}")
        return

    transform = sitk.ReadTransform(str(paths["transform"]))
    disp_field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64,
        reference.GetSize(), reference.GetOrigin(),
        reference.GetSpacing(), reference.GetDirection()
    )

    ref_arr       = sitk.GetArrayFromImage(reference)
    disp_arr      = sitk.GetArrayFromImage(displacement)
    jac_arr       = sitk.GetArrayFromImage(jacobian)
    mask_arr      = sitk.GetArrayFromImage(brain_mask)
    disp_field_arr = sitk.GetArrayFromImage(disp_field)

    spacing = reference.GetSpacing()
    z_mid, y_mid, x_mid = (s // 2 for s in ref_arr.shape)
    aspect_axial    = spacing[1] / spacing[0]
    aspect_coronal  = spacing[2] / spacing[0]
    aspect_sagittal = spacing[2] / spacing[1]

    brain_disp = disp_arr[mask_arr > 0]
    brain_jac  = jac_arr[mask_arr > 0]

    if len(brain_disp) > 0:
        stats_vals = {
            "mean_shift":    brain_disp.mean(),
            "max_shift":     brain_disp.max(),
            "p95_shift":     float(np.percentile(brain_disp, 95)),
            "mean_jacobian": brain_jac.mean(),
            "max_jacobian":  brain_jac.max(),
            "min_jacobian":  brain_jac.min(),
        }
    else:
        stats_vals = {"mean_shift": 0, "max_shift": 0, "p95_shift": 0, "mean_jacobian": 1.0}

    def _save(fig, name):
        p = figures_dir / name
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    plt.style.use("seaborn-v0_8-paper")

    # ── FIG 01: Multislice Overview ───────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 5, figsize=(18, 8))
    z_slices = np.linspace(z_mid-20, z_mid+20, 5, dtype=int)
    for i, z in enumerate(z_slices):
        axes1[0, i].imshow(ref_arr[z], cmap='gray', vmin=0, vmax=1200, aspect=aspect_axial)
        axes1[0, i].contour(mask_arr[z], levels=[0.5], colors='lime', linewidths=1, alpha=0.6)
        axes1[0, i].axis('off')
        axes1[1, i].imshow(ref_arr[z], cmap='gray', alpha=0.3, vmin=0, vmax=1200, aspect=aspect_axial)
        dm = np.ma.masked_where(mask_arr[z] == 0, disp_arr[z])
        axes1[1, i].imshow(dm, cmap='turbo', vmin=0, vmax=8, aspect=aspect_axial)
        axes1[1, i].axis('off')
    plt.tight_layout()
    _save(fig1, "Fig01_Multislice_Overview.png")

    # ── FIG 02: Statistical Distributions ─────────────────────────────────────
    fig2, (ax_d, ax_j) = plt.subplots(1, 2, figsize=(14, 5))
    ax_d.hist(brain_disp, bins=100, color='#3b82f6', alpha=0.7)
    ax_d.axvline(stats_vals['mean_shift'], color='red', ls='--')
    ax_j.hist(brain_jac, bins=100, color='#10b981', alpha=0.7)
    ax_j.axvline(1.0, color='black', lw=1)
    plt.tight_layout()
    _save(fig2, "Fig02_Statistical_Distributions.png")

    # ── FIG 03: Detailed Slice ────────────────────────────────────────────────
    fig3 = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig3)
    ax_ref = fig3.add_subplot(gs[0, 0]); ax_ref.imshow(ref_arr[z_mid], cmap='gray', aspect=aspect_axial); ax_ref.axis('off')
    ax_disp = fig3.add_subplot(gs[0, 1]); dm_mid = np.ma.masked_where(mask_arr[z_mid] == 0, disp_arr[z_mid])
    im3 = ax_disp.imshow(dm_mid, cmap='turbo', vmin=0, vmax=10, aspect=aspect_axial); plt.colorbar(im3, ax=ax_disp); ax_disp.axis('off')
    ax_jac = fig3.add_subplot(gs[1, 0]); jm_mid = np.ma.masked_where(mask_arr[z_mid] == 0, jac_arr[z_mid])
    im_j = ax_jac.imshow(jm_mid, cmap='RdBu_r', vmin=0.7, vmax=1.3, aspect=aspect_axial); plt.colorbar(im_j, ax=ax_jac); ax_jac.axis('off')
    ax_txt = fig3.add_subplot(gs[1, 1]); ax_txt.axis('off')
    summary = f"ID: {patient_id}\nMean Shift: {stats_vals['mean_shift']:.2f} mm\nMax Shift: {stats_vals['max_shift']:.2f} mm"
    ax_txt.text(0.1, 0.5, summary, fontsize=12, family='monospace')
    plt.tight_layout()
    _save(fig3, "Fig03_Detailed_Slice.png")

    # ── FIG 04: Triplanar Basic ───────────────────────────────────────────────
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
    axes4[0].imshow(ref_arr[z_mid], cmap='gray', aspect=aspect_axial)
    axes4[1].imshow(ref_arr[:, y_mid, :], cmap='gray', aspect=aspect_coronal, origin='lower')
    axes4[2].imshow(ref_arr[:, :, x_mid], cmap='gray', aspect=aspect_sagittal, origin='lower')
    for ax in axes4: ax.axis('off')
    _save(fig4, "Fig04_Triplanar_Basic.png")

    # ── FIG 05: 3D Vectors ────────────────────────────────────────────────────
    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111, projection='3d')
    step = 15
    z, y, x = np.mgrid[0:ref_arr.shape[0]:step, 0:ref_arr.shape[1]:step, 0:ref_arr.shape[2]:step]
    u, v, w = disp_field_arr[z, y, x, 0], disp_field_arr[z, y, x, 1], disp_field_arr[z, y, x, 2]
    mask_sub = mask_arr[z, y, x] > 0
    ax5.quiver(x[mask_sub], y[mask_sub], z[mask_sub], u[mask_sub], v[mask_sub], w[mask_sub], length=2, alpha=0.6)
    _save(fig5, "Fig05_3D_Vectors.png")

    # ── FIG 06: Displacement Triplanar ────────────────────────────────────────
    fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6))
    axes6[0].imshow(ref_arr[z_mid], cmap='gray', alpha=0.3, aspect=aspect_axial)
    axes6[0].imshow(np.ma.masked_where(mask_arr[z_mid]==0, disp_arr[z_mid]), cmap='turbo', vmin=0, vmax=8, aspect=aspect_axial)
    axes6[1].imshow(ref_arr[:, y_mid, :], cmap='gray', alpha=0.3, aspect=aspect_coronal, origin='lower')
    axes6[1].imshow(np.ma.masked_where(mask_arr[:, y_mid, :]==0, disp_arr[:, y_mid, :]), cmap='turbo', vmin=0, vmax=8, aspect=aspect_coronal, origin='lower')
    axes6[2].imshow(ref_arr[:, :, x_mid], cmap='gray', alpha=0.3, aspect=aspect_sagittal, origin='lower')
    im6 = axes6[2].imshow(np.ma.masked_where(mask_arr[:, :, x_mid]==0, disp_arr[:, :, x_mid]), cmap='turbo', vmin=0, vmax=8, aspect=aspect_sagittal, origin='lower')
    for ax in axes6: ax.axis('off')
    plt.colorbar(im6, ax=axes6, label='mm', shrink=0.8)
    _save(fig6, "Fig06_Displacement_Triplanar.png")

    # ── FIG 07: Cavity Detail ─────────────────────────────────────────────────
    if cavity_arr.sum() > 0:
        fig7, ax7 = plt.subplots(figsize=(8, 8))
        coords = np.argwhere(cavity_arr > 0); cz, cy, cx = coords.mean(axis=0).astype(int); roi = 60
        ax7.imshow(ref_arr[cz, cy-roi:cy+roi, cx-roi:cx+roi], cmap='gray', aspect='equal')
        ax7.contour(cavity_arr[cz, cy-roi:cy+roi, cx-roi:cx+roi], levels=[0.5], colors='red', linewidths=2)
        ax7.axis('off'); _save(fig7, "Fig07_Cavity_Detail.png")

    # ── FIG 08: Pre/Post Overlay ──────────────────────────────────────────────
    fig8, ax8 = plt.subplots(figsize=(8, 8))
    ax8.imshow(ref_arr[z_mid], cmap='Reds', alpha=0.5, aspect=aspect_axial)
    if paths["post_rigid"].exists():
        post_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(paths["post_rigid"])))
        ax8.imshow(post_arr[z_mid], cmap='Blues', alpha=0.5, aspect=aspect_axial)
    ax8.axis('off'); _save(fig8, "Fig08_PrePost_Overlay.png")

    # ── FIG 09: Jacobian Map ──────────────────────────────────────────────────
    fig9, ax9 = plt.subplots(figsize=(8, 8))
    ax9.imshow(ref_arr[z_mid], cmap='gray', alpha=0.2, aspect=aspect_axial)
    im9 = ax9.imshow(np.ma.masked_where(mask_arr[z_mid]==0, jac_arr[z_mid]), cmap='RdBu_r', vmin=0.8, vmax=1.2, aspect=aspect_axial)
    plt.colorbar(im9, ax=ax9, label='Jacobian'); ax9.axis('off'); _save(fig9, "Fig09_Jacobian.png")

    # ── FIG 10: Final Dashboard ───────────────────────────────────────────────
    fig10, ax10 = plt.subplots(figsize=(10, 6)); ax10.axis('off')
    summary_box = f"FINAL REPORT: {patient_id}\nMean Shift: {stats_vals['mean_shift']:.2f} mm\nMean J: {stats_vals['mean_jacobian']:.3f}"
    ax10.text(0.5, 0.5, summary_box, ha='center', va='center', fontsize=14, family='monospace', bbox=dict(boxstyle='round,pad=1', facecolor='#f8fafc', edgecolor='#3b82f6'))
    _save(fig10, "Fig10_Summary.png")

    logger.success(f"Thesis report generated for {patient_id}")
