"""
Preprocessing Validation.

Generates intensity histograms, before/after overlays, and statistical
summaries for the Pre-Op and Post-Op volumes BEFORE any registration.
This validates that the raw inputs are suitable (correct orientation,
reasonable intensity ranges, no outlier artifacts).

Usage:
    python main.py --validate-preprocessing
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.core import setup_logger, load_config

logger = setup_logger(__name__)


def _compute_stats(arr: np.ndarray, label: str) -> dict:
    """Compute and log descriptive statistics for a volume array."""
    stats = {
        'label': label,
        'min':   float(arr.min()),
        'max':   float(arr.max()),
        'mean':  float(arr.mean()),
        'std':   float(arr.std()),
        'p1':    float(np.percentile(arr, 1)),
        'p99':   float(np.percentile(arr, 99)),
        'shape': arr.shape,
    }
    logger.info(f"\n  [{label}]")
    logger.info(f"    Shape : {stats['shape']}")
    logger.info(f"    Min   : {stats['min']:.1f}   Max: {stats['max']:.1f}")
    logger.info(f"    Mean  : {stats['mean']:.1f}  ±{stats['std']:.1f}")
    logger.info(f"    P1/P99: {stats['p1']:.1f} / {stats['p99']:.1f}")
    return stats


def _normalize_for_display(arr: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
    """Clip and normalise to [0,1] for display."""
    lo, hi = np.percentile(arr, p_low), np.percentile(arr, p_high)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def validate_preprocessing(config, paths: dict):
    """Main entry point for preprocessing validation."""
    output_dir = paths["out_dir"] / "validation" / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.section("Preprocessing Validation")

    # ── Load volumes ──────────────────────────────────────────────────────────
    if not paths["pre_op"] or not paths["pre_op"].exists():
        logger.error(f"Pre-Op volume not found for {paths['id']}")
        return

    # Post-Op: try several candidate paths in priority order
    post_op_path = paths["post_op"]
    if post_op_path is None or not post_op_path.exists():
        logger.error("No Post-Op volume found.")
        return
    logger.info(f"Using Post-Op volume: {post_op_path.name}")

    pre_img = sitk.ReadImage(str(paths["pre_op"]))
    post_img = sitk.ReadImage(str(post_op_path))

    pre_arr = sitk.GetArrayFromImage(pre_img).astype(np.float32)
    post_arr = sitk.GetArrayFromImage(post_img).astype(np.float32)

    logger.info("Volumes loaded. Computing statistics…")

    stats_pre = _compute_stats(pre_arr, "Pre-Op")
    stats_post = _compute_stats(post_arr, "Post-Op")

    # ── Orientation / spacing check ───────────────────────────────────────────
    logger.section("Metadata Check")
    logger.info(f"  Pre-Op  spacing : {pre_img.GetSpacing()}")
    logger.info(f"  Post-Op spacing : {post_img.GetSpacing()}")
    logger.info(f"  Pre-Op  origin  : {np.round(pre_img.GetOrigin(), 1)}")
    logger.info(f"  Post-Op origin  : {np.round(post_img.GetOrigin(), 1)}")

    if pre_img.GetDirection() != post_img.GetDirection():
        logger.warning("Direction cosines differ between Pre-Op and Post-Op!")
    else:
        logger.success("Direction cosines match.")

    # ── Figure 1: Multi-slice overview ────────────────────────────────────────
    n_slices = pre_arr.shape[0]
    slice_positions = [n_slices // 6, n_slices // 3, n_slices // 2,
                       2 * n_slices // 3, 5 * n_slices // 6]

    fig1, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig1.suptitle("Preprocessing Validation – Multi-Slice Overview\n"
                  "(Check alignment, orientation, artefacts)", fontsize=14, fontweight='bold')

    for col, z in enumerate(slice_positions):
        pre_z = z if z < pre_arr.shape[0] else pre_arr.shape[0] // 2
        post_z = z if z < post_arr.shape[0] else post_arr.shape[0] // 2

        sl_pre = _normalize_for_display(pre_arr[pre_z])
        sl_post_display = pre_z if post_z < post_arr.shape[0] else 0
        sl_post = _normalize_for_display(post_arr[min(post_z, post_arr.shape[0]-1)])

        vmin, vmax = 0, 1

        # Row 0: Pre-Op
        axes[0, col].imshow(sl_pre, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"z={pre_z}", fontsize=9)
        axes[0, col].axis('off')

        # Row 1: Post-Op
        axes[1, col].imshow(sl_post, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, col].axis('off')

        # Row 2: Difference (only if same shape)
        if pre_arr.shape == post_arr.shape:
            diff = sl_pre - sl_post
            im = axes[2, col].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        else:
            axes[2, col].text(0.5, 0.5, "Different\nshapes", ha='center', va='center',
                              transform=axes[2, col].transAxes, color='gray')
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel("Pre-Op", fontsize=10, labelpad=5)
    axes[1, 0].set_ylabel("Post-Op", fontsize=10, labelpad=5)
    axes[2, 0].set_ylabel("Difference", fontsize=10, labelpad=5)

    plt.tight_layout()
    out_path = output_dir / "01_multislice_overview.png"
    fig1.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"  Saved: {out_path.name}")

    # ── Figure 2: Intensity histograms ────────────────────────────────────────
    fig2, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Intensity Distribution Analysis", fontsize=13, fontweight='bold')

    # Flatten, remove near-zero background
    pre_flat = pre_arr[pre_arr > 10].flatten()
    post_flat = post_arr[post_arr > 10].flatten()

    bins = np.linspace(0, max(stats_pre['p99'], stats_post['p99']), 100)

    ax_hist.hist(pre_flat, bins=bins, alpha=0.6, color='#3b82f6', label='Pre-Op', density=True)
    ax_hist.hist(post_flat, bins=bins, alpha=0.6, color='#ef4444', label='Post-Op', density=True)
    ax_hist.axvline(stats_pre['mean'], color='#1d4ed8', lw=1.5, ls='--',
                    label=f"Pre mean={stats_pre['mean']:.0f}")
    ax_hist.axvline(stats_post['mean'], color='#b91c1c', lw=1.5, ls='--',
                    label=f"Post mean={stats_post['mean']:.0f}")
    ax_hist.set_xlabel("Intensity (HU / a.u.)", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.set_title("Intensity Histograms (foreground only)")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # Stats table text
    stats_str = (
        f"{'':20s} {'Pre-Op':>10s} {'Post-Op':>10s}\n"
        f"{'─'*42}\n"
        f"{'Min':20s} {stats_pre['min']:>10.1f} {stats_post['min']:>10.1f}\n"
        f"{'Max':20s} {stats_pre['max']:>10.1f} {stats_post['max']:>10.1f}\n"
        f"{'Mean':20s} {stats_pre['mean']:>10.1f} {stats_post['mean']:>10.1f}\n"
        f"{'Std':20s} {stats_pre['std']:>10.1f} {stats_post['std']:>10.1f}\n"
        f"{'P1':20s} {stats_pre['p1']:>10.1f} {stats_post['p1']:>10.1f}\n"
        f"{'P99':20s} {stats_pre['p99']:>10.1f} {stats_post['p99']:>10.1f}\n"
        f"{'Shape':20s} {str(stats_pre['shape']):>10s} {str(stats_post['shape']):>10s}\n"
    )
    ax_scatter.axis('off')
    ax_scatter.text(0.05, 0.95, stats_str, transform=ax_scatter.transAxes,
                    fontsize=9, family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))
    ax_scatter.set_title("Volume Statistics Summary", fontsize=11)

    plt.tight_layout()
    out_path2 = output_dir / "02_intensity_histograms.png"
    fig2.savefig(out_path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"  Saved: {out_path2.name}")

    # ── Figure 3: Central slice checkerboard overlay ──────────────────────────
    z_center = pre_arr.shape[0] // 2
    z_post = min(z_center, post_arr.shape[0] - 1)

    pre_norm = _normalize_for_display(pre_arr[z_center])
    post_norm_slice = _normalize_for_display(post_arr[z_post])

    # Checkerboard (only if same 2D shape for central slice)
    h, w = pre_norm.shape
    ph, pw = post_norm_slice.shape
    min_h, min_w = min(h, ph), min(w, pw)

    checker = np.zeros((min_h, min_w))
    block = 32
    for i in range(0, min_h, block * 2):
        for j in range(0, min_w, block * 2):
            checker[i:i+block, j:j+block] = 1
            checker[i+block:i+block*2, j+block:j+block*2] = 1

    overlay = (checker * pre_norm[:min_h, :min_w] +
               (1 - checker) * post_norm_slice[:min_h, :min_w])

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    fig3.suptitle(f"Central Slice Overlay (z={z_center}) – Pre vs Post (BEFORE Registration)",
                  fontsize=12, fontweight='bold')

    axes3[0].imshow(pre_norm, cmap='gray')
    axes3[0].set_title('Pre-Op', fontsize=11, fontweight='bold')
    axes3[0].axis('off')

    axes3[1].imshow(post_norm_slice, cmap='gray')
    axes3[1].set_title('Post-Op', fontsize=11, fontweight='bold')
    axes3[1].axis('off')

    axes3[2].imshow(overlay, cmap='gray')
    axes3[2].set_title('Checkerboard Overlay\n(Misalignment visible as edges)', fontsize=11, fontweight='bold')
    axes3[2].axis('off')

    plt.tight_layout()
    out_path3 = output_dir / "03_checkerboard_overlay.png"
    fig3.savefig(out_path3, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    logger.info(f"  Saved: {out_path3.name}")

    # ── Automated warnings ────────────────────────────────────────────────────
    logger.section("Automated Quality Checks")

    mean_ratio = stats_post['mean'] / (stats_pre['mean'] + 1e-8)
    if mean_ratio < 0.7 or mean_ratio > 1.4:
        logger.warning(
            f"Large intensity difference between volumes (ratio={mean_ratio:.2f}). "
            "Consider intensity normalisation."
        )
    else:
        logger.success(f"Intensity ranges comparable (ratio={mean_ratio:.2f}).")

    if pre_arr.shape != post_arr.shape:
        logger.warning(f"Volume shapes differ: Pre={pre_arr.shape}, Post={post_arr.shape}.")
    else:
        logger.success("Volumes have identical shapes.")

    logger.success(f"Preprocessing validation complete. Outputs in: {output_dir}")


def main():
    import os
    from src.data_manager import DataManager
    config = load_config()
    dm = DataManager(config)
    patient_id = os.environ.get("BRAIN_SHIFT_ACTIVE_PATIENT", config.ACTIVE_PATIENT)
    paths = dm.get_patient_paths(patient_id)
    validate_preprocessing(config, paths)

if __name__ == "__main__":
    main()
