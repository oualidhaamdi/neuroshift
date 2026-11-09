"""
Tissue-Guided Registration Pipeline.
Executes BET, FAST, FLIRT and B-Spline registration on tissue boundaries.
"""

import sys
import os
import time
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import SimpleITK as sitk

from src.core import setup_logger, load_config
from src.data_manager import DataManager
from src.fsl_wrapper import FSLWrapper, check_fsl_availability
from src.backends.bet_fsl import BETExtractor
from src.backends.bet_sitk import OtsuExtractor
from src.backends.segmenter_fast import FASTSegmenter
from src.backends.segmenter_sitk import KMeansSegmenter
from src.backends.rigid_flirt import FLIRTAligner
from src.backends.rigid_sitk import Euler3DAligner
from src.backends.deformable_bspline import BSplineDistanceMapRegistrar, extract_tissue_edges
from src.backends.deformable_bspline_intensity import BSplineIntensityRegistrar

logger = setup_logger(__name__)

TISSUE_NAMES = {0: 'BG', 1: 'CSF', 2: 'GM', 3: 'WM'}

# Deformable registrar selection.
#   Default: intensity-based B-Spline (per config.REGISTRAR_DEFAULT, set to "intensity").
#   Override: env var REGISTRAR={intensity,tissue} (per-run, used by sweep flags).
#   Backward-compat: BASELINE_REGISTRAR=intensity is honoured as a synonym for intensity.
#
# Rationale (see docs/technical_notes.md §5 and the cavity_demo experiments):
# intensity-based registration with Cost Function Masking outperformed the
# tissue-guided distance-map approach on every metric in our cavity-mismatch
# validation. Tissue-guided remains available as an ablation/comparison option.
try:
    from config import REGISTRAR_DEFAULT as _registrar_default
except Exception:
    _registrar_default = "intensity"

_registrar_env = os.environ.get("REGISTRAR", "").lower()
if not _registrar_env and os.environ.get("BASELINE_REGISTRAR", "").lower() == "intensity":
    _registrar_env = "intensity"
_active_registrar = _registrar_env or _registrar_default
USE_INTENSITY_BASELINE = (_active_registrar == "intensity")
logger.info(
    f"Deformable registrar: {'BSplineIntensityRegistrar (intensity, default)' if USE_INTENSITY_BASELINE else 'BSplineDistanceMapRegistrar (tissue-guided, opt-in)'}"
)

# Env var REUSE_PRE_CACHE=1 → skip BET / segmentation on the PRE image when the
# cached brain_mask / seg_pre files already exist in the case's tissue_registration
# directory. Used by the multi-brain sweep with --cache-source to avoid recomputing
# the same source-brain segmentation for every synthetic case.
REUSE_PRE_CACHE = os.environ.get("REUSE_PRE_CACHE", "").lower() in ("1", "true", "yes")
if REUSE_PRE_CACHE:
    logger.info("REUSE_PRE_CACHE=1 — will reuse existing brain_mask / seg_pre if present")

# Env var DISABLE_CFM=1 → do NOT apply Cost Function Masking even if a cavity
# mask is found. Used by the CFM-ablation experiment to show how much CFM
# contributes to registration accuracy independently of the registrar choice.
DISABLE_CFM = os.environ.get("DISABLE_CFM", "").lower() in ("1", "true", "yes")
if DISABLE_CFM:
    logger.warning("DISABLE_CFM=1 — Cost Function Masking is OFF (cavity included in metric)")

class TissueRegistrationPipeline:
    """Orchestrates tissue-guided registration using FSL or SimpleITK backends."""

    def __init__(self, config, paths: dict):
        self.config = config
        self.paths = paths
        self.out_dir = paths["out_dir"] / "tissue_registration"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Normalize any None paths to safe defaults inside out_dir.
        # This handles the case where resolve() in DataManager returns None
        # for files that don't exist yet (they will be created by this pipeline).
        self._normalize_paths()

        fsl_ok, fsl_msg = check_fsl_availability()
        # Strip optional brain-id prefix added by multi_brain_sweep --parallel
        # (e.g. "SourceA__MB_A050_S250_X_C" → core = "MB_A050_S250_X_C")
        _core_id = self.paths['id'].upper().split("__", 1)[-1]
        self.is_synthetic = any(_core_id.startswith(p) for p in
                                ("TEST", "PILOT_", "SWEEP_", "MB_", "CAVITYDEMO_"))
        # BET is safe for synthetic cases — skip only FLIRT/FAST (rigid causes 20mm+ artifacts)
        self.use_fsl_bet = fsl_ok
        self.use_fsl = fsl_ok and not self.is_synthetic
        self.fsl = FSLWrapper() if fsl_ok else None
        if self.is_synthetic:
            logger.info("Backend: BET=FSL | FAST=SimpleITK | FLIRT=Identity (synthetic case)")
        else:
            logger.info(f"Backend: {'FSL (WSL2)' if self.use_fsl else 'SimpleITK (fallback)'}")

    def _normalize_paths(self):
        """Replace None path values with safe default paths inside out_dir.

        DataManager.resolve() returns None for files that don't exist yet.
        The pipeline will create them, so we just need a valid write target.
        """
        defaults = {
            "brain_mask":       self.out_dir / "brain_mask.nii.gz",
            "post_rigid":       self.out_dir / "post_rigid.nii.gz",
            "transform_rigid":  self.out_dir / "rigid_brain_guided.tfm",
            "transform":        self.out_dir / "tissue_bspline.tfm",
            "segmentation":     self.out_dir / "seg_pre.nii.gz",
        }
        for key, default in defaults.items():
            if self.paths.get(key) is None:
                self.paths[key] = default


    def _get_backends(self, pre_mask, post_mask):
        bet = BETExtractor(self.fsl, self.out_dir) if self.use_fsl_bet else OtsuExtractor()
        seg = FASTSegmenter(self.fsl, self.out_dir) if self.use_fsl else KMeansSegmenter()
        rig = FLIRTAligner(self.fsl, self.out_dir, pre_mask, post_mask) if self.use_fsl else Euler3DAligner()
        deformable = BSplineIntensityRegistrar() if USE_INTENSITY_BASELINE else BSplineDistanceMapRegistrar()
        return bet, seg, rig, deformable

    def run(self):
        logger.section(f"Tissue-Guided Registration: {self.paths['id']}")
        
        pre_img = sitk.Cast(sitk.ReadImage(str(self.paths["pre_op"])), sitk.sitkFloat32)
        post_img = sitk.Cast(sitk.ReadImage(str(self.paths["post_op"])), sitk.sitkFloat32)
        
        # 1. Rigid Alignment
        logger.info("[1/4] Brain extraction & rigid alignment...")
        temp_ext = BETExtractor(self.fsl, self.out_dir) if self.use_fsl_bet else OtsuExtractor()

        cached_pre_mask = self.paths.get("brain_mask")
        if (REUSE_PRE_CACHE and cached_pre_mask is not None
                and cached_pre_mask.exists()):
            logger.info(f"  [CACHE] Loading cached pre brain_mask: {cached_pre_mask}")
            pre_mask = sitk.Cast(sitk.ReadImage(str(cached_pre_mask)), sitk.sitkUInt8)
        else:
            pre_mask = temp_ext.extract(pre_img)

        # BET on moving (post) is always recomputed: the moving image differs per case
        post_m_raw = temp_ext.extract(post_img)

        _, _, rigid_align, deformable_reg = self._get_backends(pre_mask, post_m_raw)

        rigid_tx_path = self.paths["transform_rigid"]

        # Synthetic data is already aligned by construction. Rigid registration here
        # often introduces erroneous 20mm+ global shifts trying to 'fix' a local bubble.
        if self.is_synthetic:
            logger.info(f"SYNTHETIC CASE DETECTED: Forcing identity rigid transform for {self.paths['id']}")
            rigid_tx = sitk.Transform(3, sitk.sitkIdentity)
        elif rigid_tx_path.exists():
            rigid_tx = sitk.ReadTransform(str(rigid_tx_path))
        else:
            # Pre-check: if images are already well-aligned (NCC > 0.97), FLIRT would
            # only introduce spurious global shifts trying to explain local differences.
            ncc = self._compute_ncc(pre_img, post_img, pre_mask)
            logger.info(f"  Pre-FLIRT NCC = {ncc:.4f}")
            if ncc > 0.97:
                logger.info(
                    f"  NCC={ncc:.4f} > 0.97 — images already well-aligned, "
                    f"skipping FLIRT (identity rigid)."
                )
                rigid_tx = sitk.Transform(3, sitk.sitkIdentity)
            else:
                rigid_tx = rigid_align.align(pre_img, post_img)

        if not rigid_tx_path.exists() and not self.is_synthetic:
            sitk.WriteTransform(rigid_tx, str(rigid_tx_path))

        # Rigid quality check: large translation may bias B-Spline initialisation
        if not self.is_synthetic:
            try:
                params_r = rigid_tx.GetParameters()
                if len(params_r) >= 3:
                    translation = np.array(params_r[-3:])
                    t_mag = float(np.linalg.norm(translation))
                    logger.info(f"  Rigid tx: |translation|={t_mag:.1f}mm  {np.round(translation, 1)}")
                    if t_mag > 15.0:
                        logger.warning(
                            f"  Large rigid shift ({t_mag:.0f}mm) — "
                            f"B-Spline starts from a biased position. "
                            f"Consider re-running rigid alignment."
                        )
            except Exception:
                pass

        post_res = sitk.Resample(post_img, pre_img, rigid_tx, sitk.sitkLinear, 0.0)
        sitk.WriteImage(post_res, str(self.paths["post_rigid"]))

        # 2. Segmentation
        logger.info("[2/4] Tissue segmentation...")
        if self.is_synthetic:
            # Synthetic cases use identity rigid tx → post_res ≈ post_img (same grid).
            # Reuse post_m_raw rather than running a second FSL BET on an in-memory
            # Resample output, which can produce an invalid NIfTI header for some
            # source brains (oblique, very large high-resolution inputs).
            post_mask = sitk.Cast(
                sitk.Resample(post_m_raw, post_res,
                              sitk.Transform(3, sitk.sitkIdentity),
                              sitk.sitkNearestNeighbor, 0),
                sitk.sitkUInt8,
            )
            logger.info("  [SYNTHETIC] Reusing post_m_raw as post_mask (no second BET)")
        else:
            post_mask = temp_ext.extract(post_res)
        if not (REUSE_PRE_CACHE and self.paths["brain_mask"].exists()):
            sitk.WriteImage(pre_mask, str(self.paths["brain_mask"]))

        _, segmenter, _, _ = self._get_backends(pre_mask, post_mask)

        cached_pre_seg = self.paths.get("segmentation")
        if (REUSE_PRE_CACHE and cached_pre_seg is not None
                and cached_pre_seg.exists()):
            logger.info(f"  [CACHE] Loading cached pre seg: {cached_pre_seg}")
            pre_seg = sitk.Cast(sitk.ReadImage(str(cached_pre_seg)), sitk.sitkUInt8)
        else:
            pre_seg = segmenter.segment(sitk.Mask(pre_img, pre_mask), pre_mask)
        # Segmentation on moving (post) is always recomputed
        post_seg = segmenter.segment(sitk.Mask(post_res, post_mask), post_mask)

        if not (REUSE_PRE_CACHE and self.paths["segmentation"].exists()):
            sitk.WriteImage(sitk.Cast(pre_seg, sitk.sitkUInt8), str(self.paths["segmentation"]))

        if USE_INTENSITY_BASELINE:
            logger.info("[3/4] Baseline mode: skipping edge extraction, using masked raw intensities")
            pre_edge  = sitk.Mask(pre_img, pre_mask)        # raw intensity inputs
            post_edge = sitk.Mask(post_res, post_mask)
        else:
            logger.info("[3/4] Edge extraction...")
            pre_edge  = extract_tissue_edges(pre_seg)
            post_edge = extract_tissue_edges(post_seg)
        
        registration_mask = pre_mask

        # Auto Cost Function Masking: if a manual / pre-existing cavity mask
        # is present, exclude its voxels from the registration metric.
        # Looked up in (in order):
        #   RunBatch_Input/<id>/mask_cavity_manual.nii.gz   (synthetic cases)
        #   out_dir/mask_cavity_manual.nii.gz                (user-provided)
        #   out_dir/mask_cavity_auto.nii.gz                  (analyzer-generated)
        cavity_candidates = [
            Path(self.config.PROJECT_ROOT) / "RunBatch_Input" / self.paths["id"] / "mask_cavity_manual.nii.gz",
            Path(self.paths["out_dir"]) / "mask_cavity_manual.nii.gz",
            Path(self.paths["out_dir"]) / "mask_cavity_auto.nii.gz",
        ]
        cavity_path = next((p for p in cavity_candidates if p.exists()), None)
        if cavity_path is not None and DISABLE_CFM:
            logger.info(f"[3.5/4] Skipping Cost Function Masking (DISABLE_CFM=1, cavity found at {cavity_path.name})")
            cavity_path = None
        if cavity_path is not None:
            try:
                logger.info(f"[3.5/4] Cost Function Masking via {cavity_path.name}")
                cavity_img = sitk.ReadImage(str(cavity_path))
                if cavity_img.GetSize() != pre_mask.GetSize():
                    cavity_img = sitk.Resample(
                        cavity_img, pre_mask, sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0,
                    )
                pre_mask_arr = sitk.GetArrayFromImage(pre_mask) > 0
                cavity_arr   = sitk.GetArrayFromImage(cavity_img) > 0
                healthy_arr  = pre_mask_arr & (~cavity_arr)
                healthy_img  = sitk.GetImageFromArray(healthy_arr.astype(np.uint8))
                healthy_img.CopyInformation(pre_mask)
                registration_mask = sitk.Cast(healthy_img, sitk.sitkUInt8)
                excluded = int(cavity_arr.sum())
                total    = int(pre_mask_arr.sum())
                logger.success(
                    f"   Excluded {excluded:,} cavity voxels "
                    f"({100*excluded/max(total,1):.2f}% of brain) from registration metric"
                )
            except Exception as e:
                logger.warning(f"   Failed to apply cavity mask: {e}")

        logger.info(f"[4/4] B-Spline registration on {'raw intensities (baseline)' if USE_INTENSITY_BASELINE else 'tissue edges'}...")
        
        override_grid_sp = None  # default: use config value (for real patients)
        
        if self.is_synthetic:
            gt_disp_path = self.paths.get("gt_displacement")
            if gt_disp_path and gt_disp_path.exists():
                gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_disp_path)))
                mag = np.linalg.norm(gt_arr, axis=-1)
                z_i, y_i, x_i = np.where(mag > 0.5)  # only significant displacement
                if len(z_i) > 0:
                    sp = pre_edge.GetSpacing()
                    # Margin in voxels from mm
                    margin_x = int(30.0 / sp[0])
                    margin_y = int(30.0 / sp[1])
                    margin_z = int(30.0 / sp[2])
                    min_sz_x = int(40.0 / sp[0])
                    min_sz_y = int(40.0 / sp[1])
                    min_sz_z = int(40.0 / sp[2])

                    idx = [int(max(0, x_i.min() - margin_x)),
                           int(max(0, y_i.min() - margin_y)),
                           int(max(0, z_i.min() - margin_z))]
                    sz = [int(min(mag.shape[2], x_i.max() + margin_x) - idx[0]),
                          int(min(mag.shape[1], y_i.max() + margin_y) - idx[1]),
                          int(min(mag.shape[0], z_i.max() + margin_z) - idx[2])]

                    # Enforce minimum ROI size so the grid has enough support
                    sz = [max(s, m) for s, m in zip(sz, [min_sz_x, min_sz_y, min_sz_z])]
                    idx = [max(0, min(idx[i], pre_edge.GetSize()[i] - sz[i])) for i in range(3)]

                    roi = sitk.RegionOfInterestImageFilter()
                    roi.SetSize(sz)
                    roi.SetIndex(idx)
                    pre_edge          = roi.Execute(pre_edge)
                    post_edge         = roi.Execute(post_edge)
                    registration_mask = roi.Execute(registration_mask)
                    
                    phys_sz = [sz[i] * sp[i] for i in range(3)]
                    logger.warning(
                        f"SYNTHETIC SPEEDUP: Cropped B-Spline ROI to "
                        f"{phys_sz[0]:.0f}x{phys_sz[1]:.0f}x{phys_sz[2]:.0f} mm "
                        f"at Index={idx}"
                    )
                    override_grid_sp = 10.0  # fine grid for small ROI
        
        bspline_tx = deformable_reg.register(
            pre_edge, post_edge, registration_mask,
            self.config,
            override_grid_sp=override_grid_sp
        )
        sitk.WriteTransform(bspline_tx, str(self.paths["transform"]))

        self._generate_results(pre_img, pre_seg, post_res, post_seg, bspline_tx, pre_mask)
        
        # 5. Cleanup redundant files
        self._cleanup()

    def _cleanup(self):
        """Removes intermediate and temporary files to save space."""
        logger.info("Cleaning up intermediate files...")
        
        # 1. Directory cleanup: FSL tmp folder
        tmp_fsl = self.out_dir / "tmp_fsl"
        if tmp_fsl.exists() and tmp_fsl.is_dir():
            try:
                shutil.rmtree(tmp_fsl)
                logger.info(f"  Removed directory: {tmp_fsl.name}")
            except Exception as e:
                logger.warning(f"  Could not remove {tmp_fsl.name}: {e}")

        # 2. Files cleanup based on patterns and specific names
        patterns_to_delete = [
            "*_tmp.nii.gz",            # Internal temp images
            "edge_*.nii.gz",           # Edge maps (pre/post)
            "fast_*.nii.gz",           # FSL FAST intermediates
            "flirt_rigid.mat",         # FSL format matrix (we have .tfm)
            "*_bet.nii.gz",            # Intermediate brain captures
            "*_bet_mask.nii.gz",       # Intermediate masks
            "*_raw_bet*.nii.gz",       # Raw intermediates
            "*_skull_weight.nii.gz",   # Skeletonization weights
            "seg_post_original.nii.gz" # Unregistered segmentation
        ]
        
        files_deleted = 0
        for pattern in patterns_to_delete:
            for f in self.out_dir.glob(pattern):
                try:
                    f.unlink()
                    files_deleted += 1
                except Exception:
                    pass
        
        if files_deleted > 0:
            logger.info(f"  Deleted {files_deleted} intermediate files.")
        logger.success("Cleanup completed.")

    def _generate_results(self, pre_img, pre_seg, post_res, post_seg, bspline_tx, pre_mask):
        post_reg_img = sitk.Resample(post_res, pre_img, bspline_tx, sitk.sitkLinear, 0.0)
        post_reg_seg = sitk.Resample(post_seg, pre_img, bspline_tx, sitk.sitkNearestNeighbor, 0.0)
        
        sitk.WriteImage(sitk.Cast(post_reg_seg, sitk.sitkUInt8), str(self.out_dir / "seg_post_registered.nii.gz"))

        pre_arr, post_o_arr, post_r_arr = [sitk.GetArrayFromImage(x) for x in [pre_seg, post_seg, post_reg_seg]]
        
        dice_results = {}
        logger.section("Dice Results")
        for lbl in [1, 2, 3]:
            d_b = self._dice(pre_arr == lbl, post_o_arr == lbl)
            d_a = self._dice(pre_arr == lbl, post_r_arr == lbl)
            dice_results[lbl] = (d_b, d_a)
            logger.info(f"{TISSUE_NAMES[lbl]}: {d_b:.3f} -> {d_a:.3f}")

        report_path = self.out_dir / "tissue_registration_report.png"
        self._save_report(pre_img, pre_seg, post_res, post_reg_img, post_reg_seg, dice_results, report_path)

    @staticmethod
    def _compute_ncc(img_a: sitk.Image, img_b: sitk.Image, mask: sitk.Image | None = None) -> float:
        """Normalized cross-correlation within the brain mask (or whole image if no mask)."""
        a = sitk.GetArrayFromImage(img_a).astype(np.float32)
        b = sitk.GetArrayFromImage(img_b).astype(np.float32)
        if mask is not None:
            m = sitk.GetArrayFromImage(mask) > 0
            a, b = a[m], b[m]
        a -= a.mean(); b -= b.mean()
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    @staticmethod
    def _dice(a, b):
        return 2.0 * np.sum(a & b) / (np.sum(a) + np.sum(b) + 1e-9)

    def _save_report(self, pre, pre_seg, post_orig, post_reg, post_reg_seg, dice, path):
        z = pre.GetSize()[2] // 2
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        def _norm(img):
            arr = sitk.GetArrayFromImage(img)[z]
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-9)

        axes[0,0].imshow(_norm(pre), cmap='gray'); axes[0,0].set_title("Pre-Op")
        axes[0,1].imshow(_norm(post_orig), cmap='gray'); axes[0,1].set_title("Post-Op (Original)")
        axes[0,2].imshow(_norm(post_reg), cmap='gray'); axes[0,2].set_title("Post-Op (Registered)")
        
        axes[1,0].imshow(sitk.GetArrayFromImage(pre_seg)[z], cmap='jet'); axes[1,0].set_title("Pre Seg")
        axes[1,1].imshow(sitk.GetArrayFromImage(post_reg_seg)[z], cmap='jet'); axes[1,1].set_title("Post Seg Reg")
        
        # Summary text
        txt = "\n".join([f"{TISSUE_NAMES[l]}: {dice[l][0]:.3f} -> {dice[l][1]:.3f}" for l in [1,2,3]])
        axes[1,2].text(0.5, 0.5, txt, ha='center', va='center', fontsize=12, weight='bold')
        axes[1,2].axis('off')
        
        for ax in axes.flatten()[:5]: ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.success(f"Report: {path.name}")

def main():
    config = load_config()
    dm = DataManager(config)
    patient_id = os.environ.get("BRAIN_SHIFT_ACTIVE_PATIENT", config.ACTIVE_PATIENT)
    paths = dm.get_patient_paths(patient_id)
    
    if not paths["pre_op"] or not paths["pre_op"].exists():
        logger.error(f"Missing input files for patient {patient_id}")
        return

    pipeline = TissueRegistrationPipeline(config, paths)
    pipeline.run()

if __name__ == "__main__":
    main()
