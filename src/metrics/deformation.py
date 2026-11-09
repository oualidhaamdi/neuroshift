"""
Deformation metrics (Jacobian).
"""

import numpy as np
import SimpleITK as sitk


def compute_jacobian_determinant(transform: sitk.Transform, input_image: sitk.Image) -> np.ndarray:
    """
    Compute Jacobian determinant field from a transform.
    """
    disp_field = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        input_image.GetSize(),
        input_image.GetOrigin(),
        input_image.GetSpacing(),
        input_image.GetDirection()
    )

    jacobian_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jacobian_image = jacobian_filter.Execute(disp_field)

    return sitk.GetArrayFromImage(jacobian_image)
