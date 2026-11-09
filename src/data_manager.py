"""
Data Management Module.
Centralizes file discovery, path resolution, and validation for patients.
"""

import os
from pathlib import Path
from src.core import setup_logger

logger = setup_logger(__name__)

# Standard naming convention for all pipeline outputs
FILENAMES = {
    "PRE_OP": "reference_pre_op.nii.gz",
    "POST_OP": "original_post_op.nii.gz",
    "BRAIN_MASK": "brain_mask.nii.gz",
    "POST_RIGID": "post_rigid.nii.gz",
    "TRANSFORM_RIGID": "rigid_brain_guided.tfm",
    "TRANSFORM_BSPLINE": "tissue_bspline.tfm",
    "SEGMENTATION": "seg_pre.nii.gz",
    "DISPLACEMENT": "brainshift_displacement.nii.gz",
    "MAGNITUDE": "brainshift_magnitude.nii.gz",
    "JACOBIAN": "brainshift_jacobian.nii.gz",
}

# Fallback names for outputs written by earlier versions of the pipeline
FALLBACKS = {
    "BRAIN_MASK": ["pre_bet_mask.nii.gz", "mask_brain_auto.nii.gz", "brain_mask.nii.gz", "healthy_registration_mask.nii.gz"],
    "POST_RIGID": ["post_rigid.nii.gz", "post_affine.nii.gz", "post_registered_tissue.nii.gz"],
    "TRANSFORM_RIGID": ["rigid_brain_guided.tfm", "rigid.tfm"],
    "TRANSFORM_BSPLINE": ["tissue_bspline.tfm", "bspline.tfm"],
    "SEGMENTATION": ["seg_pre.nii.gz", "pre_segmented.nii.gz"],
}

class DataManager:
    """Handles patient data discovery and path resolution across the project."""

    def __init__(self, config):
        self.config = config
        self.project_root = Path(config.PROJECT_ROOT)
        self.output_bases = [
            self.project_root / "RunBatch_Output",
            self.project_root / "outputs"
        ]
        self.input_bases = [
            self.project_root / "RunBatch_Completed",
            self.project_root / "RunBatch_Input"
        ]

    def get_available_patients(self) -> list[str]:
        """Scan output directories for processed patients that have both outputs and inputs."""
        patients = set()
        for base in self.output_bases:
            if base.exists():
                for d in base.iterdir():
                    if d.is_dir() and not d.name.startswith("."):
                        bspline_path = d / "tissue_registration" / "tissue_bspline.tfm"
                        rigid_path = d / "tissue_registration" / "post_rigid.nii.gz"
                        if not (bspline_path.exists() or rigid_path.exists()):
                            continue
                        # Require pre-op input to exist so the viewer can load it
                        if self._find_input_file(d.name, "0") is not None:
                            patients.add(d.name)
        return sorted(list(patients))

    def get_patient_paths(self, patient_id: str) -> dict:
        """Resolve all critical paths for a specific patient with legacy fallback support."""
        # 1. Determine Output Directory
        out_dir = None
        for base in self.output_bases:
            if (base / patient_id).exists():
                out_dir = base / patient_id
                break
        
        if not out_dir:
            out_dir = self.project_root / "outputs" / patient_id

        # 2. Resolve Inputs (Pre-Op/Post-Op)
        pre_op = self._find_input_file(patient_id, folder_idx="0")
        post_op = self._find_input_file(patient_id, folder_idx="1")

        # 3. Resolve Derivative paths with smart discovery
        # We check both the main out_dir and the "tissue_registration" subdirectory
        reg_dir = out_dir / "tissue_registration"
        
        def resolve(key, default_name):
            # 1. Check official name in out_dir or reg_dir
            for d in [out_dir, reg_dir]:
                if (d / default_name).exists():
                    return d / default_name
            # 2. Check fallbacks
            for fb_name in FALLBACKS.get(key, []):
                for d in [out_dir, reg_dir]:
                    if (d / fb_name).exists():
                        return d / fb_name
            # 3. Return None if not found (callers must check with .exists() or 'is not None')
            return None

        gt_disp_path = self.project_root / "RunBatch_Input" / patient_id / "gt_displacement.nii.gz"

        # For synthetic patients, prefer the pre-computed brain mask saved in the input
        # directory by create_synthetic_patients.py (same quality as the real patient run).
        # Fall back to this only when the output mask doesn't exist yet.
        brain_mask_resolved = resolve("BRAIN_MASK", FILENAMES["BRAIN_MASK"])
        if brain_mask_resolved is None:
            input_mask = self.project_root / "RunBatch_Input" / patient_id / "brain_mask.nii.gz"
            if input_mask.exists():
                brain_mask_resolved = input_mask
        if brain_mask_resolved is None:
            # Last resort: binarize seg_pre (any tissue label > 0 = brain).
            # Saves the result so subsequent calls find brain_mask.nii.gz directly.
            seg_path = resolve("SEGMENTATION", FILENAMES["SEGMENTATION"])
            if seg_path is not None and seg_path.exists():
                brain_mask_resolved = self._derive_mask_from_seg(seg_path, reg_dir)

        return {
            "id": patient_id,
            "out_dir": out_dir,
            "pre_op": pre_op,
            "post_op": post_op,
            "brain_mask": brain_mask_resolved,
            "post_rigid": resolve("POST_RIGID", FILENAMES["POST_RIGID"]),
            "transform_rigid": resolve("TRANSFORM_RIGID", FILENAMES["TRANSFORM_RIGID"]),
            "transform": resolve("TRANSFORM_BSPLINE", FILENAMES["TRANSFORM_BSPLINE"]),
            "segmentation": resolve("SEGMENTATION", FILENAMES["SEGMENTATION"]),
            "displacement": out_dir / FILENAMES["DISPLACEMENT"],
            "magnitude": out_dir / FILENAMES["MAGNITUDE"],
            "jacobian": out_dir / FILENAMES["JACOBIAN"],
            "gt_displacement": gt_disp_path if gt_disp_path.exists() else None
        }


    def _derive_mask_from_seg(self, seg_path: Path, save_dir: Path) -> Path | None:
        """Binarize seg_pre (CSF/GM/WM > 0) and save as brain_mask.nii.gz."""
        try:
            import SimpleITK as sitk
            seg = sitk.ReadImage(str(seg_path))
            mask = sitk.Cast(seg > 0, sitk.sitkUInt8)
            out_path = save_dir / "brain_mask.nii.gz"
            sitk.WriteImage(mask, str(out_path))
            logger.info(f"Brain mask derived from {seg_path.name} → {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"Could not derive brain mask from segmentation: {e}")
            return None

    def _find_input_file(self, patient_id: str, folder_idx: str) -> Path | None:
        """Helper to find NIfTI files in batch input/completed folders with modality priority."""
        for base in self.input_bases:
            p_dir = base / patient_id / folder_idx
            if p_dir.exists():
                niftis = sorted(p_dir.glob("*.nii*"))  # deterministic order
                if not niftis:
                    continue
                
                # Priority 1: Native T1 (no CE)
                for n in niftis:
                    name = n.name.lower()
                    if "t1" in name and "ce" not in name and "contrast" not in name:
                        return n
                
                # Priority 2: T1 Contrast Enhanced
                for n in niftis:
                    name = n.name.lower()
                    if "t1ce" in name or "t1_ce" in name or "contrast" in name:
                        return n

                # Priority 3: Generic T1 or anything with "pre" or "post" in the name
                for n in niftis:
                    name = n.name.lower()
                    if "t1" in name or "pre" in name or "post" in name:
                        return n

                # Fallback: Just return the first NIfTI if no priority matches
                return niftis[0]
        return None

    def validate_patient(self, patient_id: str) -> bool:
        """Check if essential input files exist for a patient."""
        paths = self.get_patient_paths(patient_id)
        if not paths["pre_op"] or not paths["pre_op"].exists():
            logger.warning(f"Patient {patient_id}: Missing Pre-Op file.")
            return False
        if not paths["post_op"] or not paths["post_op"].exists():
            logger.warning(f"Patient {patient_id}: Missing Post-Op file.")
            return False
        return True
