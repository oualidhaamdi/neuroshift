"""
Segmentation Validation Module.
Generates metrics (Volume, Dice) and visual reports for brain and cavity segmentation.
"""

import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from src.core import setup_logger, load_config

logger = setup_logger(__name__)

def compute_volume(mask, spacing):
    """Compute volume in cubic centimeters (cc)."""
    voxel_vol = np.prod(spacing)
    count = np.sum(sitk.GetArrayViewFromImage(mask) > 0)
    return (count * voxel_vol) / 1000.0

def compute_overlap_metrics(mask1, mask2):
    """Compute Dice and Jaccard coefficients."""
    overlap = sitk.LabelOverlapMeasuresImageFilter()
    overlap.Execute(mask1, mask2)
    return {
        'Dice': overlap.GetDiceCoefficient(),
        'Jaccard': overlap.GetJaccardCoefficient(),
        'VolumeSimilarity': overlap.GetVolumeSimilarity()
    }

def validate_segmentation(config, paths: dict):
    logger.section("Validating Brain Segmentation")

    output_dir = paths["out_dir"] / "validation" / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Paths
    path_mri = paths["pre_op"]
    path_mask_auto = paths["brain_mask"]
    path_mask_manual = paths["out_dir"] / "mask_brain_manual.nii.gz"

    if not path_mri or not path_mri.exists():
        logger.error(f"MRI not found for {paths['id']}")
        return

    mri = sitk.ReadImage(str(path_mri))
    spacing = mri.GetSpacing()
    
    results = {}
    
    # 2. Analyze Auto vs Manual
    masks = {}
    
    if path_mask_manual.exists():
        logger.info(f"Loading Manual Mask: {path_mask_manual.name}")
        mask_manual = sitk.ReadImage(str(path_mask_manual))
        masks['Manual'] = sitk.Cast(mask_manual, sitk.sitkUInt8)
        vol = compute_volume(masks['Manual'], spacing)
        results['Manual_Volume_cc'] = vol
        logger.info(f"  Volume: {vol:.2f} cc")
        
    if path_mask_auto.exists():
        logger.info(f"Loading Auto Mask: {path_mask_auto.name}")
        mask_auto = sitk.ReadImage(str(path_mask_auto))
        masks['Auto'] = sitk.Cast(mask_auto, sitk.sitkUInt8)
        vol = compute_volume(masks['Auto'], spacing)
        results['Auto_Volume_cc'] = vol
        logger.info(f"  Volume: {vol:.2f} cc")
        
    # Overlap?
    if 'Manual' in masks and 'Auto' in masks:
        metrics = compute_overlap_metrics(masks['Manual'], masks['Auto'])
        results.update(metrics)
        logger.info(f"  Overlap: Dice={metrics['Dice']:.4f}, Jaccard={metrics['Jaccard']:.4f}")
    
    # 3. Visualization
    logger.info("Generating visualization...")
    
    # Slice selection (center)
    size = mri.GetSize()
    c = [s//2 for s in size]
    
    mri_arr = sitk.GetArrayFromImage(mri)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Aspect ratios for proper display
    aspect_ax = spacing[1] / spacing[0]
    aspect_cor = spacing[2] / spacing[0]
    aspect_sag = spacing[2] / spacing[1]

    # Axial
    axes[0].imshow(mri_arr[c[2], :, :], cmap='gray', aspect=aspect_ax)
    if 'Manual' in masks:
        mask_arr = sitk.GetArrayFromImage(masks['Manual'])
        axes[0].contour(mask_arr[c[2], :, :], colors='lime', linewidths=1.0, levels=[0.5], label='Manual')
    if 'Auto' in masks:
        mask_arr = sitk.GetArrayFromImage(masks['Auto'])
        axes[0].contour(mask_arr[c[2], :, :], colors='red', linewidths=1.0, linestyles='dashed', levels=[0.5], label='Auto')
    axes[0].set_title("Axial")
    
    # Coronal
    axes[1].imshow(mri_arr[:, c[1], :], cmap='gray', origin='lower', aspect=aspect_cor)
    if 'Manual' in masks:
        mask_arr = sitk.GetArrayFromImage(masks['Manual'])
        axes[1].contour(mask_arr[:, c[1], :], colors='lime', linewidths=1.0, levels=[0.5], origin='lower')
    if 'Auto' in masks:
        mask_arr = sitk.GetArrayFromImage(masks['Auto'])
        axes[1].contour(mask_arr[:, c[1], :], colors='red', linewidths=1.0, linestyles='dashed', levels=[0.5], origin='lower')
    axes[1].set_title("Coronal")
    
    # Sagittal
    axes[2].imshow(mri_arr[:, :, c[0]], cmap='gray', origin='lower', aspect=aspect_sag)
    if 'Manual' in masks:
        mask_arr = sitk.GetArrayFromImage(masks['Manual'])
        axes[2].contour(mask_arr[:, :, c[0]], colors='lime', linewidths=1.0, levels=[0.5], origin='lower')
    if 'Auto' in masks:
        mask_arr = sitk.GetArrayFromImage(masks['Auto'])
        axes[2].contour(mask_arr[:, :, c[0]], colors='red', linewidths=1.0, linestyles='dashed', levels=[0.5], origin='lower')
    axes[2].set_title("Sagittal")
    
    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='lime', lw=2),
                    Line2D([0], [0], color='red', linestyle='--', lw=2)]
    fig.legend(custom_lines, ['Manual', 'Auto'], loc='upper right')
    
    # Text Stats
    stats_text = "Metrics:\n"
    if 'Manual_Volume_cc' in results:
        stats_text += f"Manual Vol: {results['Manual_Volume_cc']:.1f} cc\n"
    if 'Auto_Volume_cc' in results:
        stats_text += f"Auto Vol: {results['Auto_Volume_cc']:.1f} cc\n"
    if 'Dice' in results:
        stats_text += f"Dice: {results['Dice']:.3f}\n"
    if 'Jaccard' in results:
        stats_text += f"Jaccard: {results['Jaccard']:.3f}"
        
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle("Segmentation Validation (Green=Manual, Red=Auto)")
    plt.tight_layout()
    plt.savefig(output_dir / "segmentation_check.png")
    logger.info(f"Saved figure to {output_dir}")
    
    # Save Report
    df = pd.DataFrame([results])
    df.to_csv(output_dir / "segmentation_metrics.csv", index=False)
    logger.info("Saved metrics CSV.")


def main():
    import os
    from src.data_manager import DataManager
    config = load_config()
    dm = DataManager(config)
    patient_id = os.environ.get("BRAIN_SHIFT_ACTIVE_PATIENT", config.ACTIVE_PATIENT)
    paths = dm.get_patient_paths(patient_id)
    validate_segmentation(config, paths)

if __name__ == "__main__":
    main()
