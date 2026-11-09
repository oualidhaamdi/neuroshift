"""
BrainExtractor implementation using FSL BET (via WSL2).
"""

from __future__ import annotations
import uuid
from pathlib import Path
import numpy as np
import SimpleITK as sitk

from src.backends.base import BrainExtractor
from src.core import setup_logger
from src.fsl_wrapper import FSLWrapper

logger = setup_logger(__name__)

class BETExtractor(BrainExtractor):
    """Brain extraction via FSL BET (Surface Model)."""

    def __init__(self, fsl: FSLWrapper, out_dir: Path, frac: float = 0.4):
        self.fsl = fsl
        self.out_dir = out_dir
        self.frac = frac

    def extract(self, image: sitk.Image) -> sitk.Image:
        """
        Runs BET. Uses a local tmp folder within the project to avoid 
        path spaces in Windows Temp directory.
        """
        # Create a local tmp dir inside the patient output folder (mapped in WSL without spaces)
        tmp_fsl_dir = self.out_dir / "tmp_fsl"
        tmp_fsl_dir.mkdir(parents=True, exist_ok=True)
        
        # Unique name per call: WSL2 caches the "deleted" state of a file after the
        # first BET's finally-unlink.  If the second call reuses the same name, WSL2
        # still reports the file as absent → _wait_for_file_wsl times out → BET sees
        # a non-existent file → "not valid".  A fresh name bypasses that stale cache.
        tmp_input = tmp_fsl_dir / f"bet_input_{uuid.uuid4().hex[:8]}.nii.gz"
        # In-memory images (e.g. sitk.Resample output) have no MetaDataDictionary,
        # so ITK computes qform/sform quaternions from the direction matrix.
        # Floating-point drift accumulated through DICOMOrient→write→read→Resample
        # can make that matrix slightly non-orthonormal → NaN quaternion → FSL rejects
        # the file as "not valid".  QR-orthonormalize before writing to guarantee a
        # clean NIfTI header regardless of the image's origin.
        image_out = sitk.Cast(image, sitk.sitkFloat32)
        d = np.array(image_out.GetDirection()).reshape(3, 3)
        q, _ = np.linalg.qr(d)
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        image_out.SetDirection(q.flatten().tolist())
        sitk.WriteImage(image_out, str(tmp_input))

        prefix = self.out_dir / "pre_bet"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.fsl.bet(tmp_input, prefix, frac=self.frac, mask=True)
            
            mask_path = Path(str(prefix) + "_mask.nii.gz")
            if mask_path.exists():
                mask = sitk.Cast(sitk.ReadImage(str(mask_path)), sitk.sitkUInt8)
                # The mask inherits the QR-orthonormalized header we wrote for FSL,
                # which can differ from the input's direction (e.g. RAS diag(-1,-1,1)
                # → QR returns identity). BET never resamples, so the mask shares the
                # input's voxel grid exactly; restore the original geometry so callers
                # that mask images keeping the original direction stay index-aligned.
                if mask.GetSize() == image.GetSize():
                    mask.CopyInformation(image)
                logger.info(f"  [BETExtractor] mask generated: {sitk.GetArrayFromImage(mask).sum():,} voxels")
                return mask
            
            raise FileNotFoundError(f"FSL BET failed to produce mask at {mask_path}")
        finally:
            # Clean up local tmp file
            if tmp_input.exists():
                tmp_input.unlink()
