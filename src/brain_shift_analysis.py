"""
Brain Shift Analysis Pipeline.
Computes displacement, Jacobian, and regional statistics from registration results.
"""

import sys
import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.core import setup_logger, load_config
from src.data_manager import DataManager
from src.segmentation import SegmentationModule

logger = setup_logger(__name__)

def resample_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Ensure mask shares the physical space of reference."""
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    if (mask.GetSize() == reference.GetSize() and 
        np.allclose(mask.GetSpacing(), reference.GetSpacing())):
        return mask
    return sitk.Resample(mask, reference, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)

class BrainShiftAnalyzer:
    """Handles quantification of brain displacement and deformation."""
    
    def __init__(self, config, paths: dict):
        self.config = config
        self.paths = paths
        self.output_dir = paths["out_dir"]
        self.seg_module = SegmentationModule()
        self._normalize_paths()

    def _normalize_paths(self):
        """Replace None path values with safe default paths."""
        defaults = {
            "brain_mask":   self.output_dir / "tissue_registration" / "brain_mask.nii.gz",
            "post_rigid":   self.output_dir / "tissue_registration" / "post_rigid.nii.gz",
            "transform":    self.output_dir / "tissue_registration" / "tissue_bspline.tfm",
            "segmentation": self.output_dir / "tissue_registration" / "seg_pre.nii.gz",
        }
        for key, default in defaults.items():
            if self.paths.get(key) is None:
                self.paths[key] = default

    def run_segmentation(self, fixed: sitk.Image, moving_affine: sitk.Image):
        """Load or generate necessary masks for analysis."""
        # Brain mask preference: Existing > Generate
        if self.paths["brain_mask"].exists():
            logger.info(f"Using existing brain mask: {self.paths['brain_mask'].name}")
            brain_mask = sitk.ReadImage(str(self.paths["brain_mask"]))
        else:
            logger.info("Brain mask not found, generating auto...")
            brain_mask = self.seg_module.extract_brain_mask(fixed)
            sitk.WriteImage(brain_mask, str(self.paths["brain_mask"]))

        # Resample brain_mask to the fixed image grid BEFORE using it for cavity
        # segmentation, otherwise sitk.Mask in segment_cavity raises a size mismatch
        # when the cached mask has different dimensions than the current fixed.
        brain_mask = resample_mask(brain_mask, fixed)

        # Cavity mask: Manual > Auto > Generate
        manual_cavity = self.output_dir / "mask_cavity_manual.nii.gz"
        auto_cavity = self.output_dir / "mask_cavity_auto.nii.gz"

        if manual_cavity.exists():
            logger.info("Using manual cavity mask.")
            cavity_mask = sitk.ReadImage(str(manual_cavity))
        elif auto_cavity.exists():
            logger.info("Using existing auto cavity mask.")
            cavity_mask = sitk.ReadImage(str(auto_cavity))
        else:
            logger.info("Generating automatic cavity mask...")
            cavity_mask = self.seg_module.segment_cavity(fixed, moving_affine, brain_mask)
            sitk.WriteImage(cavity_mask, str(auto_cavity))

        return brain_mask, resample_mask(cavity_mask, fixed)

    def quantify(self, fixed: sitk.Image, brain_mask: sitk.Image, 
                 cavity_mask: sitk.Image, transform: sitk.Transform):
        """Compute displacement field and deformation metrics."""
        logger.section("Quantification")

        # 1. Displacement Field
        disp_field = sitk.TransformToDisplacementField(
            transform, sitk.sitkVectorFloat64,
            fixed.GetSize(), fixed.GetOrigin(),
            fixed.GetSpacing(), fixed.GetDirection()
        )
        disp_mag = sitk.VectorMagnitude(disp_field)

        # 2. Jacobian Determinant
        jac_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
        jacobian = jac_filter.Execute(disp_field)

        # Save volumes using standardized paths
        sitk.WriteImage(disp_field, str(self.paths["displacement"]))
        sitk.WriteImage(disp_mag, str(self.paths["magnitude"]))
        sitk.WriteImage(jacobian, str(self.paths["jacobian"]))

        # 3. Statistics
        mag_arr = sitk.GetArrayFromImage(disp_mag)
        mask_arr = sitk.GetArrayFromImage(brain_mask)
        jac_arr = sitk.GetArrayFromImage(jacobian)

        shifts = mag_arr[mask_arr > 0]
        logger.info(f"Mean Shift: {np.mean(shifts):.2f} mm")
        logger.info(f"Max Shift:  {np.max(shifts):.2f} mm")

        neg_jac = np.sum(jac_arr[mask_arr > 0] < 0)
        if neg_jac > 0:
            logger.error(f"CRITICAL: {neg_jac} voxels with negative Jacobian (folding).")
        else:
            logger.success("Topology preserved (no folding).")

        return shifts, disp_mag, jacobian

def main():
    config = load_config()
    dm = DataManager(config)
    
    patient_id = os.environ.get("BRAIN_SHIFT_ACTIVE_PATIENT", config.ACTIVE_PATIENT)
    paths = dm.get_patient_paths(patient_id)
    
    logger.setup_file_logging(str(paths["out_dir"]))
    logger.section(f"Brain Shift Analysis: {patient_id}")

    if not paths["pre_op"] or not paths["pre_op"].exists():
        logger.error(f"Pre-Op file not found for {patient_id}.")
        return

    # Check for transform (support both rigid and bspline)
    transform_path = paths["transform"] if paths["transform"].exists() else paths["transform_rigid"]
    if not transform_path.exists():
        logger.error("No registration transform found. Run registration first.")
        return

    # Check for post-op (prefer rigid registered if available)
    post_op_path = paths["post_rigid"] if paths["post_rigid"].exists() else paths["post_op"]

    fixed = sitk.ReadImage(str(paths["pre_op"]), sitk.sitkFloat32)
    moving_affine = sitk.ReadImage(str(post_op_path), sitk.sitkFloat32)
    transform = sitk.ReadTransform(str(transform_path))

    analyzer = BrainShiftAnalyzer(config, paths)
    brain_mask, cavity_mask = analyzer.run_segmentation(fixed, moving_affine)
    analyzer.quantify(fixed, brain_mask, cavity_mask, transform)

    logger.success("Analysis complete.")

if __name__ == "__main__":
    main()
