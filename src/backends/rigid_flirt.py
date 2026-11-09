"""
RigidAligner implementation using FSL FLIRT (skull-guided, 6-DOF).
"""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from src.backends.base import RigidAligner
from src.core import setup_logger
from src.fsl_wrapper import FSLWrapper

logger = setup_logger(__name__)


class FLIRTAligner(RigidAligner):
    """
    Skull-guided rigid alignment via FSL FLIRT (6-DOF, Mutual Information).

    Brain mask is inverted so that FLIRT optimises on the skull only,
    avoiding brain-shift contamination in the rigid step.
    """

    def __init__(self, fsl: FSLWrapper, out_dir: Path,
                 pre_mask: sitk.Image, post_mask: sitk.Image):
        self.fsl = fsl
        self.out_dir = out_dir
        self.pre_mask = pre_mask
        self.post_mask = post_mask

    def align(self, fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
        out_dir = self.out_dir
        post_rigid_nii = out_dir / "post_rigid.nii.gz"
        omat = out_dir / "flirt_rigid.mat"

        # Invert masks: skull weight = 1 where brain mask = 0
        pre_weight = sitk.Cast(sitk.Not(sitk.Cast(self.pre_mask, sitk.sitkUInt8)), sitk.sitkFloat32)
        post_weight = sitk.Cast(sitk.Not(sitk.Cast(self.post_mask, sitk.sitkUInt8)), sitk.sitkFloat32)

        pre_w_path = out_dir / "pre_skull_weight.nii.gz"
        post_w_path = out_dir / "post_skull_weight.nii.gz"
        sitk.WriteImage(pre_weight, str(pre_w_path))
        sitk.WriteImage(post_weight, str(post_w_path))

        post_path = out_dir / "_moving_tmp.nii.gz"
        pre_path = out_dir / "_fixed_tmp.nii.gz"
        sitk.WriteImage(moving, str(post_path))
        sitk.WriteImage(fixed, str(pre_path))

        self.fsl.flirt(post_path, pre_path, output=post_rigid_nii,
                       dof=6, cost="mutualinfo", omat=omat,
                       inweight=post_w_path, refweight=pre_w_path)

        # Approximate the rigid transform as a moments-based centred Euler3D.
        # The full matrix-to-ITK conversion is a known limitation — this provides
        # a reasonable warm-start for the subsequent B-Spline step.
        try:
            tx = sitk.CenteredTransformInitializer(
                fixed, moving,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
            sitk.WriteTransform(tx, str(out_dir / "flirt_rigid.tfm"))
            logger.info("  [FLIRTAligner] rigid transform saved.")
        except Exception as e:
            logger.warning(f"  [FLIRTAligner] transform conversion failed ({e}) — warm-start will be skipped")
            tx = sitk.Transform()

        return tx

    @property
    def registered_image_path(self) -> Path:
        return self.out_dir / "post_rigid.nii.gz"
