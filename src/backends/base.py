"""
Abstract base classes for the tissue-guided registration pipeline.

Each component (brain extraction, segmentation, rigid alignment, deformable
registration) is defined as a Protocol/ABC so that backends can be swapped
without touching the orchestration logic in register_tissue.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import SimpleITK as sitk


class BrainExtractor(ABC):
    """Extract a binary brain mask from a 3-D MRI volume."""

    @abstractmethod
    def extract(self, image: sitk.Image) -> sitk.Image:
        """
        Args:
            image: Input MRI volume (any pixel type).
        Returns:
            Binary mask (sitkUInt8) where 1 = brain, 0 = background.
        """


class TissueSegmenter(ABC):
    """Segment the brain into CSF / GM / WM."""

    LABELS = {0: "BG", 1: "CSF", 2: "GM", 3: "WM"}

    @abstractmethod
    def segment(self, brain_img: sitk.Image, mask: sitk.Image) -> sitk.Image:
        """
        Args:
            brain_img: Skull-stripped brain volume (float32).
            mask:      Binary brain mask (sitkUInt8).
        Returns:
            Label map (sitkUInt8): 0=BG, 1=CSF, 2=GM, 3=WM.
        """


class RigidAligner(ABC):
    """Compute a rigid (6-DOF) transform between two volumes."""

    @abstractmethod
    def align(self, fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
        """
        Args:
            fixed:  Reference volume.
            moving: Volume to be aligned.
        Returns:
            Rigid SimpleITK transform mapping moving → fixed space.
        """


class DeformableRegistrar(ABC):
    """Estimate a deformable transform from pre-extracted edge maps."""

    @abstractmethod
    def register(
        self,
        fixed_edge: sitk.Image,
        moving_edge: sitk.Image,
        mask: sitk.Image,
        config,
        rigid_tx: sitk.Transform | None = None,
    ) -> sitk.Transform:
        """
        Args:
            fixed_edge:  Edge map of the fixed (pre-op) volume.
            moving_edge: Edge map of the moving (post-op) volume.
            mask:        Brain mask used to restrict the optimisation.
            config:      Project config module (provides REGISTRATION_MODES).
            rigid_tx:    Optional rigid warm-start transform.
        Returns:
            Deformable SimpleITK transform mapping moving → fixed space.
        """
