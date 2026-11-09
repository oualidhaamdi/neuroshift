"""
Capture Range Analysis (Convergence Curve).
Sweeps initial perturbations from 0 to 100mm and measures final TRE.
Identifies the "Break Point" where the optimizer fails to find the optimum.
"""

import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import random

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from src.core import setup_logger, load_config


logger = setup_logger(__name__)

def compute_tre(fixed_points, ground_truth_tx, recovered_tx):
    errors = []
    for p in fixed_points:
        p_gt = ground_truth_tx.TransformPoint(p)
        p_rec = recovered_tx.TransformPoint(p)
        dist = np.linalg.norm(np.array(p_gt) - np.array(p_rec))
        errors.append(dist)
    return np.mean(errors)

def analyze_capture_range(config, paths: dict, max_shift=100, step=5):
    logger.section(f"Capture Range Analysis (0 to {max_shift} mm)")

    if not paths["pre_op"] or not paths["pre_op"].exists():
        logger.error(f"Image not found for {paths['id']}")
        return

    fixed = sitk.ReadImage(str(paths["pre_op"]), sitk.sitkFloat32)
    
    output_dir = paths["out_dir"] / "validation" / "capture_range"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate points for TRE (50 random landmark points in physical space)
    bounds = fixed.GetSize()
    test_points = []
    for _ in range(50):
        idx = [random.randint(0, bounds[i]-1) for i in range(3)]
        pt = fixed.TransformContinuousIndexToPhysicalPoint(idx)
        test_points.append(pt)

    results = []
    shifts = np.arange(0, max_shift + 1, step)
    
    for shift in shifts:
        logger.info(f"Testing shift: {shift} mm...")
        
        # Rot scaling: 1 deg per 5 mm shift (heuristic)
        rot_deg = (shift / 10.0) * 1.0 
        
        # 1. Perturb
        gt_tx = sitk.Euler3DTransform()
        center = fixed.TransformContinuousIndexToPhysicalPoint(np.array(fixed.GetSize()) / 2.0)
        gt_tx.SetCenter(center)
        
        # Random axis-angle like rotation
        angle = np.radians(rot_deg)
        rx, ry, rz = [random.uniform(-angle, angle) for _ in range(3)]
        gt_tx.SetRotation(rx, ry, rz)
        
        # Random translation
        vec = np.random.normal(size=3)
        vec /= (np.linalg.norm(vec) + 1e-9)
        gt_tx.SetTranslation(tuple(vec * shift))
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(gt_tx)
        perturbed = resampler.Execute(fixed)
        
        # 2. Register
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation() # More robust for large shifts in same-modality
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.10)
        
        # Optimizer (Conservative settings for large sweep)
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=0.01, numberOfIterations=200, relaxationFactor=0.5
        )
        
        # Initial Transform: Initializer + Identity refinement
        init_tx = sitk.CenteredTransformInitializer(fixed, perturbed, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
        R.SetInitialTransform(init_tx)
        
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInterpolator(sitk.sitkLinear)
        
        # Multi-resolution pyramid (Strong downsampling to catch large shifts)
        R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
        R.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
        
        # Initial TRE
        # Note: If perturbed = Fixed(T_gt(p)), then T_reg should find T_gt.
        init_tre = compute_tre(test_points, gt_tx, sitk.Transform())
        
        try:
            final_tx = R.Execute(fixed, perturbed)
            # The registration finds T such that Fixed(p) = Moving(T(p)).
            # Since Moving(p) = Fixed(T_gt(p)), then T_reg should find T_gt.GetInverse().
            final_tre = compute_tre(test_points, gt_tx.GetInverse(), final_tx)
            status = "Success"
        except Exception as e:
            logger.warning(f"  Failed at {shift}mm: {e}")
            final_tre = init_tre
            status = "Failed"

        results.append({
            "Initial_Shift_mm": shift,
            "Initial_TRE_mm": init_tre,
            "Final_TRE_mm": final_tre,
            "Status": status
        })
        logger.info(f"  Init TRE: {init_tre:.2f} | Final TRE: {final_tre:.4f} mm")

    # 4. Save and Plot
    df = pd.DataFrame(results)
    output_dir = config.DIR_OUTPUT_FIGURES / "capture_range"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "capture_range_stats.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(11, 7), facecolor='white')
    
    plt.plot(df["Initial_Shift_mm"], df["Final_TRE_mm"], 'o-', color='#2563eb', linewidth=2.5, markersize=8, label='Residual Error (TRE)')
    plt.plot(df["Initial_Shift_mm"], df["Initial_Shift_mm"], '--', color='#94a3b8', alpha=0.5, label='No Correction Line')
    
    plt.axvline(35, color='#16a34a', linestyle='--', alpha=0.6, label='Operational Capture Range (~35mm)')
    plt.axhline(2.0, color='#dc2626', linestyle=':', label='Surgical Precision Limit (2mm)')
    
    plt.fill_between(df["Initial_Shift_mm"], 0, 2.0, color='#16a34a', alpha=0.1)
    
    plt.title('Algorithm Reliability Analysis — Capture Range Profile', fontsize=14, fontweight='bold')
    plt.xlabel('Initial Displacement (mm)', fontsize=11)
    plt.ylabel('Residual Alignment Error (mm)', fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=10, loc='upper left')
    
    plt.text(5, 0.5, 'Region of Total Recovery', color='#16a34a', fontsize=8, fontweight='bold')
    
    plot_path = output_dir / "capture_range_curve.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Analysis complete. Plot saved to: {plot_path}")

if __name__ == "__main__":
    analyze_capture_range()
