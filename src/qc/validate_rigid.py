"""
Validation script for Rigid Registration.
Generates a publication-ready tri-planar figure with light theme,
contour overlay, color-coded difference map, and metric summary.
"""

import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(str(Path(__file__).parents[2]))

from src.core import setup_logger, load_config
from src.metrics.similarity import compute_ncc, compute_mi

logger = setup_logger(__name__)


def _norm(arr):
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - p1) / (p99 - p1 + 1e-9), 0.0, 1.0)


def _resample_to(src: sitk.Image, ref: sitk.Image) -> sitk.Image:
    return sitk.Resample(src, ref, sitk.Transform(),
                         sitk.sitkLinear, 0.0, src.GetPixelID())


def validate_rigid(config, paths: dict):
    logger.section("Validating Rigid Registration")

    fixed_path      = paths["pre_op"]
    moving_reg_path = paths["post_rigid"]

    if not moving_reg_path or not moving_reg_path.exists():
        logger.warning(f"Rigid result not found: {moving_reg_path}"); return

    fixed  = sitk.ReadImage(str(fixed_path),      sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_reg_path), sitk.sitkFloat32)
    moving_r = _resample_to(moving, fixed)

    fixed_arr  = sitk.GetArrayFromImage(fixed)
    moving_arr = sitk.GetArrayFromImage(moving_r)

    ncc = compute_ncc(fixed_arr, moving_arr)
    mi  = compute_mi(fixed_arr, moving_arr)
    logger.info(f"  NCC: {ncc:.4f}  MI: {mi:.4f}")

    verdict_str = "EXCELLENT" if ncc > 0.70 else "GOOD" if ncc > 0.55 else "MARGINAL"
    verdict_clr = '#1a7a3a'   if ncc > 0.70 else '#c97a00' if ncc > 0.55 else '#c0392b'

    diff_arr = np.abs(fixed_arr - moving_arr)
    diff_p95 = np.percentile(diff_arr, 95)

    sp  = fixed.GetSpacing()
    asp_ax  = sp[1] / sp[0]
    asp_cor = sp[2] / sp[0]
    asp_sag = sp[2] / sp[1]
    sz      = fixed.GetSize()
    c       = [s // 2 for s in sz]

    # ── Figure (light theme) ──────────────────────────────────────────────────
    plt.rcdefaults()
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.suptitle("Rigid Registration Validation — Tri-Planar Comparison",
                 fontsize=14, fontweight='bold', y=0.97, color='#1e293b')

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.06, wspace=0.04,
                           left=0.06, right=0.96, top=0.92, bottom=0.13)

    planes      = ['Axial', 'Coronal', 'Sagittal']
    col_titles  = ['Pre-Op (Fixed)', 'Post-Op Affine  +  Fixed contour', 'Difference  |Fixed − Affine|']

    def _sl(arr, pi):
        if pi == 0:   return arr[c[2], :, :]
        elif pi == 1: return arr[:, c[1], :]
        else:         return arr[:, :, c[0]]

    asps = [asp_ax, asp_cor, asp_sag]
    origs = ['upper', 'lower', 'lower']  # coronal+sagittal: head up

    for row in range(3):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#f8fafc')

            sl_fix = _norm(_sl(fixed_arr,  row))
            sl_mov = _norm(_sl(moving_arr, row))
            sl_dif = _sl(diff_arr, row)

            if col == 0:
                ax.imshow(sl_fix, cmap='gray', aspect=asps[row],
                          origin=origs[row], interpolation='bilinear')
            elif col == 1:
                ax.imshow(sl_mov, cmap='gray', aspect=asps[row],
                          origin=origs[row], interpolation='bilinear')
                # Green contour of the fixed image on the moving image
                ax.contour(sl_fix, levels=[0.35, 0.60],
                           colors=['#16a34a', '#22c55e'],
                           linewidths=[0.6, 0.4], alpha=0.85,
                           origin=origs[row])
            else:
                im = ax.imshow(sl_dif, cmap='hot', aspect=asps[row],
                               origin=origs[row], vmin=0, vmax=diff_p95,
                               interpolation='bilinear')

            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, pad=4,
                             color='#1e293b', fontweight='semibold')
            if col == 0:
                ax.set_ylabel(planes[row], fontsize=9, color='#475569',
                              fontweight='bold', labelpad=3)
            ax.set_xticks([]); ax.set_yticks([])
            for sp_obj in ax.spines.values():
                sp_obj.set_edgecolor('#cbd5e1')

    # Colorbar for difference column
    cax = fig.add_axes([0.68, 0.06, 0.27, 0.022])
    sm  = plt.cm.ScalarMappable(cmap='hot',
                                 norm=plt.Normalize(vmin=0, vmax=diff_p95))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('|Intensity difference| — 95th percentile clip', fontsize=8, color='#475569')
    cb.ax.tick_params(labelsize=7, colors='#475569')

    # Metrics text (bottom-left)
    txt = (f"NCC = {ncc:.4f}   ·   MI = {mi:.4f}   ·   "
           f"Verdict: {verdict_str}   ·   Green contour = Fixed reference boundary")
    fig.text(0.06, 0.05, txt, fontsize=9.5, color=verdict_clr,
             bbox=dict(fc='#f0fdf4', ec='#86efac', boxstyle='round,pad=0.4'))

    # Legend: green = fixed contour
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], color='#16a34a', lw=1.2, label='Fixed image contour')]
    fig.legend(handles=leg, loc='lower right', fontsize=8,
               framealpha=0.9, edgecolor='#cbd5e1', bbox_to_anchor=(0.96, 0.04))

    # Save
    output_dir = paths["out_dir"] / "validation" / "rigid"
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "rigid_comparison_triplanar.png"
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"  NCC: {ncc:.4f}  MI: {mi:.4f}")
    logger.success(f"Rigid validation complete. Output: {out}")


def main():
    import os
    from src.data_manager import DataManager
    config = load_config()
    dm = DataManager(config)
    patient_id = os.environ.get("BRAIN_SHIFT_ACTIVE_PATIENT", config.ACTIVE_PATIENT)
    paths = dm.get_patient_paths(patient_id)
    validate_rigid(config, paths)

if __name__ == "__main__":
    main()
