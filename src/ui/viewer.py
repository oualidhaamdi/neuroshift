"""
Multi-planar visualization with Jacobian analysis and displacement metrics.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import colors as mcolors
import SimpleITK as sitk

from src.core import load_config, setup_logger

logger = setup_logger(__name__)

# Import configuration
try:
    config = load_config()
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    sys.exit(1)


from src.data_manager import DataManager

class InteractiveBrainShiftViewer:
    """GUI for visualizing brain shift and displacement fields."""
    
    def __init__(self, data_dir=None):
        """Initialize viewer state."""
        self.dm = DataManager(config)
        self.colors_cfg = config.COLORS
        self.view_mode = 'axial'
        self.roi_size = 60
        self.clicked_point = None
        self.displacement_vector = None
        self.jacobian = None
        self.overlay_alpha = 0.5
        self.show_displacement_overlay = False
        self.show_jacobian_map = False
        self.show_vector_field = False
        self.show_diff_overlay = False
        self.show_registered = False   # NEW: toggle between post_rigid and post_registered
        self.show_tumor_mask = True
        self.vector_step_mm = 6.0  # Balanced density
        self.vector_scale = 1.0
        self.colorbar = None
        self.colorbar_ax = None
        self.colorbar_jac = None
        self.colorbar_jac_ax = None
        self._nav_idx = 0
        self._nav_filter = "all"
        self.pending_error = None
        
        # Scan for available patients using DataManager
        self.available_patients = self.dm.get_available_patients()
        self.patient_dirs = {p: self.dm.get_patient_paths(p)["out_dir"] for p in self.available_patients}
        
        # Determine initial patient
        if data_dir is not None:
            data_path = Path(data_dir)
            # If it's a full path to an output folder, use its name as patient ID
            if data_path.exists() and data_path.is_dir():
                initial_patient = data_path.name
            else:
                initial_patient = str(data_dir)
        else:
            initial_patient = config.ACTIVE_PATIENT

        logger.info(f"Viewer initializing with patient: {initial_patient}")
        
        # Safe defaults in case load_patient_data fails
        self.max_slices = 1
        self.current_slice = 0
        self.aspect_ratio = 1.0
        self.pre_arr = np.zeros((10, 10, 10))
        self.post_arr = np.zeros((10, 10, 10))
        self.mask_arr = np.zeros((10, 10, 10))
        self.mask_arr_eroded = np.zeros((10, 10, 10))
        self.displacement_magnitude = np.zeros((10, 10, 10))
        self.jacobian_field = np.zeros((10, 10, 10))
        self.pre_vmin = 0.0
        self.pre_vmax = 1.0
        self.post_vmin = 0.0
        self.post_vmax = 1.0
        self.spacing = (1.0, 1.0, 1.0)
        self.synth_roi_arr = None
        self.gt_displacement_img = None
        self.test_info = None
        self.post_op_rigid = None
        self.post_op_registered = None
        self.tumor_arr = None

        self.load_patient_data_by_id(initial_patient)

        # If the initial case has no data, try the other available ones
        if self.pending_error:
            for alt_patient in self.available_patients:
                if alt_patient == initial_patient:
                    continue
                logger.warning(f"Trying fallback patient: {alt_patient}")
                self.pending_error = None
                self.load_patient_data_by_id(alt_patient)
                if not self.pending_error:
                    logger.info(f"Loaded fallback patient: {alt_patient}")
                    break

    def load_patient_data_by_id(self, patient_id):
        """Helper to load data by patient ID."""
        paths = self.dm.get_patient_paths(patient_id)
        self.load_patient_data(paths["out_dir"])

    def load_patient_data(self, data_dir):
        """Loads NIfTI volumes dynamically using DataManager."""
        self.data_dir = Path(data_dir)
        patient_id = self.data_dir.name
        self.pending_error = None
        if hasattr(self, 'fig') and self.fig:
            self.fig.suptitle(" ")
            if getattr(self.fig, '_suptitle', None):
                self.fig._suptitle.set_visible(False)
            self.fig.canvas.draw_idle()
        logger.info(f"Loading data for patient: {patient_id}")
        
        try:
            paths = self.dm.get_patient_paths(patient_id)
            if not paths["pre_op"] or not paths["pre_op"].exists():
                raise FileNotFoundError(f"Original pre_op NIfTI not found for {patient_id}")
                
            self.pre_op = sitk.ReadImage(str(paths["pre_op"]))
            self.pre_op = sitk.DICOMOrient(self.pre_op, 'RAS') # Force consistency
            
            if not paths["brain_mask"] or not paths["brain_mask"].exists():
                raise FileNotFoundError(f"Brain mask not found for {patient_id}")
            self.brain_mask = sitk.ReadImage(str(paths["brain_mask"]))
            self.brain_mask = sitk.DICOMOrient(self.brain_mask, 'RAS')

            # --- NEW: Load Synthetic Test Info ---
            self.test_info = None
            info_path = paths["out_dir"].parent / patient_id / "test_info.txt"
            if not info_path.exists(): # Try input dir too
                for base in self.dm.input_bases:
                    if (base / patient_id / "test_info.txt").exists():
                        info_path = base / patient_id / "test_info.txt"
                        break
            
            if info_path.exists():
                self.test_info = {}
                with open(info_path, "r") as f:
                    for line in f:
                        try:
                            parts = line.strip().split("=")
                            if len(parts) == 2:
                                k, v = parts
                                if k in ["axis", "slice"]:
                                    self.test_info[k] = int(v)
                                else:
                                    self.test_info[k] = v
                        except Exception:
                            continue
                logger.info(f"Loaded synthetic test info: {self.test_info}")

            # --- Build composite transform ---
            # tissue_bspline.tfm contains ONLY the B-Spline component (rigid not included).
            # We must compose rigid_brain_guided.tfm + tissue_bspline.tfm so that the
            # viewer displacement/vector calculations are correct.
            out_dir = self.data_dir
            if (out_dir / "tissue_registration" / "tissue_bspline.tfm").exists():
                bspline = sitk.ReadTransform(str(out_dir / "tissue_registration" / "tissue_bspline.tfm"))
                # We ONLY use the B-Spline transform to visualize actual Brain Shift!
                # The Rigid transform represents the patient's head moving in the MRI bed,
                # which causes a massive uniform vector field that visually drowns the local tissue sink.
                self.transform = bspline
                logger.info(f"Pure B-Spline transform (Tissue Deformation isolated) loaded for {patient_id}")
            elif (out_dir / "flirt_rigid.tfm").exists():
                self.transform = sitk.ReadTransform(str(out_dir / "flirt_rigid.tfm"))
                logger.warning(f"B-Spline not found for {patient_id} — using Rigid only")
            else:
                self.transform = sitk.ReadTransform(str(paths["transform"]))
                logger.warning(f"Using default transform for {patient_id}")

            # --- POST panel: Support for both rigid and registered (B-Spline) ---
            self.post_op_rigid = None
            self.post_op_registered = None

            post_rigid_path = paths["out_dir"] / "tissue_registration" / "post_rigid.nii.gz"
            if post_rigid_path.exists():
                self.post_op_rigid = sitk.DICOMOrient(sitk.ReadImage(str(post_rigid_path)), 'RAS')
                logger.info(f"Loaded post_rigid for {patient_id}")

            reg_path = paths["out_dir"] / "tissue_registration" / "post_registered_tissue.nii.gz"
            if reg_path.exists():
                self.post_op_registered = sitk.DICOMOrient(sitk.ReadImage(str(reg_path)), 'RAS')
                logger.info(f"Loaded post_registered for {patient_id}")

            # Default view is post_rigid (to show the brain shift that needs correction)
            if self.post_op_rigid:
                self.post_op = self.post_op_rigid
            elif self.post_op_registered:
                self.post_op = self.post_op_registered
                self.show_registered = True
            else:
                # Last resort: resample
                raw_post = sitk.DICOMOrient(sitk.ReadImage(str(paths["post_op"])), 'RAS') if paths["post_op"] else self.pre_op
                self.post_op = sitk.Resample(raw_post, self.pre_op, self.transform,
                                             sitk.sitkLinear, 0.0, self.pre_op.GetPixelID())
                self.post_op_registered = self.post_op

        except Exception as e:
            msg = f"Incomplete data for {patient_id}: {e}"
            logger.error(msg)
            self.pending_error = f"ERROR: {patient_id} registration data missing"
            if hasattr(self, 'fig') and self.fig:
                self.fig.suptitle(self.pending_error, color='red', fontsize=12, fontweight='bold')
                self.fig.canvas.draw_idle()
            return

        self.pre_arr = sitk.GetArrayFromImage(self.pre_op)
        self.post_arr = sitk.GetArrayFromImage(self.post_op)
        self.mask_arr = sitk.GetArrayFromImage(self.brain_mask)
        self.spacing = self.pre_op.GetSpacing()
        
        tumor_path = self.data_dir / "tissue_registration" / "tumor_mask.nii.gz"
        if tumor_path.exists():
            self.tumor_arr = sitk.GetArrayFromImage(
                sitk.DICOMOrient(sitk.ReadImage(str(tumor_path)), 'RAS')) > 0
        else:
            self.tumor_arr = None

        # NEW: Load synthetic ROI center (for Test 3/4)
        synth_roi_path = self.data_dir / "synthetic_center.nii.gz"
        if synth_roi_path.exists():
            self.synth_roi_arr = sitk.GetArrayFromImage(
                sitk.DICOMOrient(sitk.ReadImage(str(synth_roi_path)), 'RAS'))
            logger.info(f"Loaded synthetic ROI center for {patient_id}")
        else:
            self.synth_roi_arr = None

        # NEW: Load Ground Truth displacement for synthetic tests (Test 6-9)
        self.gt_displacement_img = None
        for base in self.dm.input_bases:
            gt_path = base / patient_id / "gt_displacement.nii.gz"
            if gt_path.exists():
                self.gt_displacement_img = sitk.DICOMOrient(sitk.ReadImage(str(gt_path)), 'RAS')
                logger.info(f"Loaded Ground Truth displacement for {patient_id}")
                break

        self.current_slice = self.pre_arr.shape[0] // 2
        self.update_slice_limits()

        # Eroded mask used only for vector field display to exclude skull/boundary voxels.
        # Otsu brain masks often include skull (bright in T1), causing spurious edge vectors.
        # 2-voxel erosion strips those without affecting statistics which use mask_arr.
        eroded_sitk = sitk.BinaryErode(
            sitk.Cast(self.brain_mask > 0, sitk.sitkUInt8), [2, 2, 2]
        )
        self.mask_arr_eroded = sitk.GetArrayFromImage(eroded_sitk)

        self.displacement_magnitude, self.jacobian_field = self.compute_fields()

        if np.any(self.mask_arr > 0):
            mask_bool = self.mask_arr > 0
            self.pre_vmin, self.pre_vmax = np.percentile(self.pre_arr[mask_bool], [1, 99])
            self.post_vmin, self.post_vmax = np.percentile(self.post_arr[mask_bool], [1, 99])
        else:
            self.pre_vmin, self.pre_vmax = np.min(self.pre_arr), np.max(self.pre_arr)
            self.post_vmin, self.post_vmax = np.min(self.post_arr), np.max(self.post_arr)
        
    def compute_fields(self):
        """Load pre-computed fields if available, otherwise compute using SITK filters."""
        paths = self.dm.get_patient_paths(self.data_dir.name)
        out_dir = paths["out_dir"]
        
        mag_path = out_dir / "brainshift_magnitude.nii.gz"
        jac_path = out_dir / "brainshift_jacobian.nii.gz"
        
        if mag_path.exists() and jac_path.exists():
            logger.info("Fast load: Found pre-computed displacement and Jacobian fields.")
            mag_img = sitk.DICOMOrient(sitk.ReadImage(str(mag_path), sitk.sitkFloat32), 'RAS')
            jac_img = sitk.DICOMOrient(sitk.ReadImage(str(jac_path), sitk.sitkFloat32), 'RAS')
            return sitk.GetArrayFromImage(mag_img), sitk.GetArrayFromImage(jac_img)
            
        logger.info("Computing displacement and Jacobian fields via SITK filters...")
        
        # 1. Displacement Field
        disp_field = sitk.TransformToDisplacementField(
            self.transform, 
            sitk.sitkVectorFloat32,
            self.pre_op.GetSize(),
            self.pre_op.GetOrigin(),
            self.pre_op.GetSpacing(),
            self.pre_op.GetDirection()
        )
        
        # 2. Magnitude (mm)
        mag_img = sitk.VectorMagnitude(disp_field)
        mag_field = sitk.GetArrayFromImage(mag_img)
        
        # 3. Jacobian Determinant
        jac_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
        jac_img = jac_filter.Execute(disp_field)
        jac_field = sitk.GetArrayFromImage(jac_img)
        
        logger.info("Fields computed (native optimization).")
        return mag_field, jac_field
    
    def update_slice_limits(self):
        """Update slice limits based on current view mode."""
        if self.view_mode == 'axial':
            self.max_slices = self.pre_arr.shape[0] - 1
            self.aspect_ratio = self.spacing[1] / self.spacing[0]
        elif self.view_mode == 'sagittal':
            self.max_slices = self.pre_arr.shape[2] - 1
            self.aspect_ratio = self.spacing[2] / self.spacing[1]
        elif self.view_mode == 'coronal':
            self.max_slices = self.pre_arr.shape[1] - 1
            self.aspect_ratio = self.spacing[2] / self.spacing[0]
        
        self.current_slice = min(self.current_slice, self.max_slices)
    
    def get_slice_data(self, volume, slice_idx):
        """Get 2D slice based on current view mode, properly oriented."""
        if self.view_mode == 'axial':
            slice_data = volume[slice_idx, :, :]
        elif self.view_mode == 'sagittal':
            slice_data = volume[:, :, slice_idx]
        elif self.view_mode == 'coronal':
            slice_data = volume[:, slice_idx, :]
        
        return np.flipud(slice_data)
    
    def get_view_title(self):
        """Get formatted title for current view."""
        view_names = {
            'axial': 'Axial',
            'sagittal': 'Sagittal',
            'coronal': 'Coronal'
        }
        return view_names[self.view_mode]
    
    def compute_jacobian_at_point(self, x, y, z, delta=1.0):
        """Compute Jacobian determinant at a point using finite differences."""
        points = []
        for dx in [-delta, 0, delta]:
            for dy in [-delta, 0, delta]:
                for dz in [-delta, 0, delta]:
                    px = np.clip(x + dx, 0, self.pre_arr.shape[2] - 1)
                    py = np.clip(y + dy, 0, self.pre_arr.shape[1] - 1)
                    pz = np.clip(z + dz, 0, self.pre_arr.shape[0] - 1)
                    
                    point_phys = self.pre_op.TransformIndexToPhysicalPoint(
                        [int(px), int(py), int(pz)]
                    )
                    point_transformed = self.transform.TransformPoint(point_phys)
                    
                    disp = np.array([
                        point_transformed[0] - point_phys[0],
                        point_transformed[1] - point_phys[1],
                        point_transformed[2] - point_phys[2]
                    ])
                    points.append((px, py, pz, disp))
        
        try:
            center_idx = 13
            
            du_dx = (points[center_idx + 1][3][0] - points[center_idx - 1][3][0]) / (2 * delta * self.spacing[0])
            dv_dx = (points[center_idx + 1][3][1] - points[center_idx - 1][3][1]) / (2 * delta * self.spacing[0])
            dw_dx = (points[center_idx + 1][3][2] - points[center_idx - 1][3][2]) / (2 * delta * self.spacing[0])
            
            du_dy = (points[center_idx + 3][3][0] - points[center_idx - 3][3][0]) / (2 * delta * self.spacing[1])
            dv_dy = (points[center_idx + 3][3][1] - points[center_idx - 3][3][1]) / (2 * delta * self.spacing[1])
            dw_dy = (points[center_idx + 3][3][2] - points[center_idx - 3][3][2]) / (2 * delta * self.spacing[1])
            
            du_dz = (points[center_idx + 9][3][0] - points[center_idx - 9][3][0]) / (2 * delta * self.spacing[2])
            dv_dz = (points[center_idx + 9][3][1] - points[center_idx - 9][3][1]) / (2 * delta * self.spacing[2])
            dw_dz = (points[center_idx + 9][3][2] - points[center_idx - 9][3][2]) / (2 * delta * self.spacing[2])
            
            J = np.array([
                [1 + du_dx, du_dy, du_dz],
                [dv_dx, 1 + dv_dy, dv_dz],
                [dw_dx, dw_dy, 1 + dw_dz]
            ])
            
            det_J = np.linalg.det(J)
            return det_J, J
        except (IndexError, ValueError, np.linalg.LinAlgError):
            return None, None
    
    def compute_roi_statistics(self, roi_pre, roi_post):
        """Compute statistical metrics for ROI."""
        mean_pre = np.mean(roi_pre)
        std_pre = np.std(roi_pre)
        mean_post = np.mean(roi_post)
        std_post = np.std(roi_post)
        
        roi_pre_flat = roi_pre.flatten()
        roi_post_flat = roi_post.flatten()
        correlation = np.corrcoef(roi_pre_flat, roi_post_flat)[0, 1]
        
        mad = np.mean(np.abs(roi_pre - roi_post))
        
        return {
            'mean_pre': mean_pre,
            'std_pre': std_pre,
            'mean_post': mean_post,
            'std_post': std_post,
            'correlation': correlation,
            'mad': mad
        }
    
    def project_vector_to_2d(self, vector_3d):
        """Project 3D displacement to current 2D view."""
        if self.view_mode == 'axial':
            return vector_3d[0], vector_3d[1]
        elif self.view_mode == 'sagittal':
            return vector_3d[1], vector_3d[2]
        elif self.view_mode == 'coronal':
            return vector_3d[0], vector_3d[2]
    
    def extract_roi(self, volume, x, y, z, size):
        """Extract ROI around point keeping physical aspect roughly square."""
        phys_radius = (size / 2.0) * self.spacing[0]
        
        rx = max(1, int(phys_radius / self.spacing[0]))
        ry = max(1, int(phys_radius / self.spacing[1]))
        rz = max(1, int(phys_radius / self.spacing[2]))
        
        x_min = max(0, x - rx)
        x_max = min(volume.shape[2], x + rx)
        y_min = max(0, y - ry)
        y_max = min(volume.shape[1], y + ry)
        z_min = max(0, z - rz)
        z_max = min(volume.shape[0], z + rz)
        
        if self.view_mode == 'axial':
            roi = volume[z, y_min:y_max, x_min:x_max]
        elif self.view_mode == 'sagittal':
            roi = volume[z_min:z_max, y_min:y_max, x]
        elif self.view_mode == 'coronal':
            roi = volume[z_min:z_max, y, x_min:x_max]
            
        return np.flipud(roi), ()
    
    def on_scroll(self, event):
        """Mouse wheel handler."""
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self.max_slices)
        elif event.button == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
        
        self.slice_slider.set_val(self.current_slice)
    
    def on_key(self, event):
        """Keyboard shortcut handler."""
        if event.key == 'up':
            self.current_slice = min(self.current_slice + 1, self.max_slices)
            self.slice_slider.set_val(self.current_slice)
        elif event.key == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
            self.slice_slider.set_val(self.current_slice)
        elif event.key == 'r':
            self.on_reset(None)
        elif event.key == 's':
            self.on_save(None)
        elif event.key == 'e':
            self.on_export(None)
        elif event.key == 'h':
            self.show_help()
    
    def show_help(self):
        """Display help dialog in popup."""
        root = tk.Tk()
        root.withdraw()
        
        help_text = """KEYBOARD SHORTCUTS:

UP/DOWN Arrows    Navigate slices
R                 Reset view
S                 Save screenshot
E                 Export point data (JSON)
H                 Show this help

MOUSE CONTROLS:

Scroll            Navigate slices
Click             Select point for analysis

VISUALIZATION OPTIONS:

• Heatmap: Displacement magnitude overlay
• Jacobian: Tissue expansion/contraction map
  - Blue = contraction (J < 1)
  - White = preserved (J ≈ 1)
  - Red = expansion (J > 1)
• Vectors: Displacement field arrows
  - Color indicates 3D magnitude
  - Direction shows 2D projection

ANALYSIS FEATURES:

• 3D displacement vectors (X,Y,Z)
• Jacobian determinant analysis
• Tissue volume change metrics
• ROI statistical comparison
"""
        
        messagebox.showinfo("Brain Shift Viewer - Help", help_text)
        root.destroy()
    
    def on_slice_change(self, val):
        """Slider change handler."""
        self.current_slice = int(val)
        self.update_main_view()
    
    def on_roi_size_change(self, val):
        """ROI size slider handler."""
        self.roi_size = int(val)
        if self.clicked_point is not None:
            # Re-calculate fields and redraw both
            self.update_main_view()
            self.update_roi_views()

    def on_alpha_change(self, val):

        """Alpha blend slider handler."""
        self.overlay_alpha = val
        self.update_main_view()
    
    def toggle_heatmap(self, event):
        """Toggle displacement heatmap."""
        self.show_displacement_overlay = not self.show_displacement_overlay
        self._refresh_ui_state()
        self.update_main_view()
        logger.info(f"Displacement heatmap: {'ON' if self.show_displacement_overlay else 'OFF'}")

    def toggle_jacobian(self, event):
        """Toggle Jacobian map."""
        self.show_jacobian_map = not self.show_jacobian_map
        self._refresh_ui_state()
        self.update_main_view()
        logger.info(f"Jacobian map: {'ON' if self.show_jacobian_map else 'OFF'}")

    def toggle_vectors(self, event):
        """Toggle vector field."""
        self.show_vector_field = not self.show_vector_field
        self._refresh_ui_state()
        self.update_main_view()
        logger.info(f"Vector field: {'ON' if self.show_vector_field else 'OFF'}")

    def toggle_diff(self, event):
        """Toggle |PRE-POST| difference map."""
        self.show_diff_overlay = not self.show_diff_overlay
        self._refresh_ui_state()
        self.update_main_view()
        logger.info(f"Diff map: {'ON' if self.show_diff_overlay else 'OFF'}")

    def toggle_registered(self, event):
        """Toggle between post_rigid (brain shift visible) and post_registered (B-Spline corrected)."""
        if not self.post_op_registered:
            logger.warning("B-Spline registered output not available for this patient")
            return

        self.show_registered = not self.show_registered
        self.post_op = self.post_op_registered if self.show_registered else self.post_op_rigid
        self.post_arr = sitk.GetArrayFromImage(self.post_op)
        self._refresh_ui_state()
        self.update_main_view()
        logger.info(f"Post-Op view: {'Registered (B-Spline)' if self.show_registered else 'Rigid (brain shift visible)'}")

    def toggle_tumor_mask(self, event):
        """Toggle tumor mask contour."""
        if self.tumor_arr is None:
            return
        self.show_tumor_mask = not self.show_tumor_mask
        self._refresh_ui_state()
        self.update_main_view()
        logger.info(f"Tumor mask contour: {'ON' if self.show_tumor_mask else 'OFF'}")

    # ── UI state helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _apply_btn_state(btn, is_on: bool, on_color: str, off_color: str = "#f1f5f9",
                         on_label: str = None, off_label: str = None):
        """Update button facecolor and optionally label to reflect on/off state."""
        color = on_color if is_on else off_color
        btn.ax.set_facecolor(color)
        btn.color = color
        if on_label and off_label:
            btn.label.set_text(on_label if is_on else off_label)

    def _refresh_ui_state(self):
        """Redraw all toggle buttons to reflect current viewer state."""
        if not hasattr(self, 'btn_heat'):
            return

        self._apply_btn_state(self.btn_heat, self.show_displacement_overlay,
                               "#7dd3fc", on_label="Heatmap ●", off_label="Heatmap")
        self._apply_btn_state(self.btn_jac, self.show_jacobian_map,
                               "#f9a8d4", on_label="Jacobian ●", off_label="Jacobian")
        self._apply_btn_state(self.btn_vec, self.show_vector_field,
                               "#86efac", on_label="Vectors ●", off_label="Vectors")
        self._apply_btn_state(self.btn_diff, self.show_diff_overlay,
                               "#fcd34d", on_label="Diff Map ●", off_label="Diff Map")

        # Post-op mode button
        if hasattr(self, 'btn_reg'):
            self._apply_btn_state(self.btn_reg, self.show_registered, "#c4b5fd",
                                  on_label="▶ Reg. POST", off_label="▶ Rigid POST")

        # Tumor button: gray-out when no mask
        if hasattr(self, 'btn_tumor'):
            if self.tumor_arr is None:
                self.btn_tumor.ax.set_facecolor("#e5e7eb")
                self.btn_tumor.color = "#e5e7eb"
                self.btn_tumor.label.set_text("Tumor N/A")
                self.btn_tumor.label.set_color("#9ca3af")
            else:
                self._apply_btn_state(self.btn_tumor, self.show_tumor_mask,
                                       "#fca5a5", on_label="Tumor ●", off_label="Tumor")
                self.btn_tumor.label.set_color("#1e293b")

        # View mode buttons
        if hasattr(self, 'btn_axial'):
            for btn, mode in [(self.btn_axial, 'axial'),
                              (self.btn_sag, 'sagittal'),
                              (self.btn_cor, 'coronal')]:
                active = (self.view_mode == mode)
                btn.ax.set_facecolor("#1d4ed8" if active else "#e0f2fe")
                btn.color = "#1d4ed8" if active else "#e0f2fe"
                btn.label.set_color("white" if active else "#1e40af")
                btn.label.set_weight("bold" if active else "normal")

        self.fig.canvas.draw_idle()

    def on_vector_step_change(self, val):
        """Vector density slider handler."""
        self.vector_step_mm = float(val)
        if self.show_vector_field:
            self.update_main_view()

    def on_vector_scale_change(self, val):
        """Vector scale slider handler."""
        self.vector_scale = float(val)
        if self.show_vector_field:
            self.update_main_view()
    


    def on_view_change(self, label):
        """View mode change handler."""
        view_map = {
            'Axial': 'axial',
            'Sagittal': 'sagittal',
            'Coronal': 'coronal'
        }
        self.view_mode = view_map[label]
        self.update_slice_limits()
        self.current_slice = self.max_slices // 2
        
        # Update slider range and value
        self.slice_slider.valmax = self.max_slices
        self.slice_slider.ax.set_xlim(0, self.max_slices)
        self.slice_slider.set_val(self.current_slice)
        
        self.clicked_point = None
        self.displacement_vector = None
        self.jacobian = None
        
        self._refresh_ui_state()
        self.update_main_view()
        self.clear_roi_views()
        logger.info(f"Switched to {label} view")

    def on_reset(self, event):
        """Reset button handler."""
        self.current_slice = self.max_slices // 2
        self.roi_size = 60
        self.clicked_point = None
        self.displacement_vector = None
        self.jacobian = None
        self.show_displacement_overlay = False
        self.show_jacobian_map = False
        self.show_vector_field = False
        self.show_diff_overlay = False

        self.slice_slider.set_val(self.current_slice)
        self.roi_slider.set_val(self.roi_size)
        self.alpha_slider.set_val(0.5)

        self._refresh_ui_state()
        self.update_main_view()
        self.clear_roi_views()
        logger.info("View reset")
    
    def on_save(self, event):
        """Save screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"brain_shift_{self.view_mode}_{timestamp}.png"
        self.fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        logger.info(f"Screenshot saved: {filename}")
    
    def on_export(self, event):
        """Export current point data to JSON."""
        if self.clicked_point is None:
            logger.warning("No point selected. Click on a brain point first.")
            return
        
        x, y, z = self.clicked_point
        magnitude = np.linalg.norm(self.displacement_vector)
        
        data = {
            'view_mode': self.view_mode,
            'point_voxel': {'x': int(x), 'y': int(y), 'z': int(z)},
            'displacement_mm': {
                'dx': float(self.displacement_vector[0]),
                'dy': float(self.displacement_vector[1]),
                'dz': float(self.displacement_vector[2])
            },
            'magnitude_mm': float(magnitude),
            'jacobian_determinant': float(self.jacobian[0]) if self.jacobian and self.jacobian[0] else None,
            'spacing_mm': {'x': float(self.spacing[0]), 'y': float(self.spacing[1]), 'z': float(self.spacing[2])},
            'slice': int(self.current_slice),
            'timestamp': datetime.now().isoformat()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"displacement_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data exported: {filename}")
    
    def on_click(self, event):
        """Mouse click handler."""
        if event.inaxes != self.ax_main:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        click_x = int(event.xdata)
        click_y_flipped = int(event.ydata)
        
        if self.view_mode == 'axial':
            shape_y = self.pre_arr.shape[1]
            click_y = shape_y - 1 - click_y_flipped
            x, y, z = click_x, click_y, self.current_slice
        elif self.view_mode == 'sagittal':
            shape_z = self.pre_arr.shape[0]
            click_y = shape_z - 1 - click_y_flipped
            x, y, z = self.current_slice, click_x, click_y
        elif self.view_mode == 'coronal':
            shape_z = self.pre_arr.shape[0]
            click_y = shape_z - 1 - click_y_flipped
            x, y, z = click_x, self.current_slice, click_y
        
        if (x < 0 or x >= self.pre_arr.shape[2] or 
            y < 0 or y >= self.pre_arr.shape[1] or
            z < 0 or z >= self.pre_arr.shape[0]):
            return
        
        if self.mask_arr[z, y, x] == 0:
            logger.warning(f"Point ({x}, {y}, {z}) outside brain mask")
            return
        
        self.clicked_point = (x, y, z)
        
        point_phys = self.pre_op.TransformIndexToPhysicalPoint([int(x), int(y), int(z)])
        point_transformed = self.transform.TransformPoint(point_phys)
        
        # Displacement in voxel-grid space (same convention as TransformToDisplacementField).
        # With Fixed=Pre, Moving=Post: T(p_pre) ≈ p_post, so idx_trans − idx_phys = p_post − p_pre
        # = actual tissue displacement direction. This matches the field vectors in update_main_view.
        idx_phys = np.array(self.pre_op.TransformPhysicalPointToContinuousIndex(point_phys))
        idx_trans = np.array(self.pre_op.TransformPhysicalPointToContinuousIndex(point_transformed))
        disp_grid = idx_trans - idx_phys
        
        # Convert to mm in Grid Space
        disp_grid_mm = disp_grid * np.array(self.spacing)
        self.displacement_vector = disp_grid_mm
        
        magnitude = np.linalg.norm(np.array([
            point_transformed[0] - point_phys[0],
            point_transformed[1] - point_phys[1],
            point_transformed[2] - point_phys[2]
        ]))
        
        det_J, J_matrix = self.compute_jacobian_at_point(x, y, z)
        self.jacobian = (det_J, J_matrix)
        
        logger.info(f"Analysis: Point({x},{y},{z}) | Disp: {magnitude:.3f}mm | Jac: {det_J if det_J else 'N/A'}")
        
        self.update_main_view()
        self.update_roi_views()
    
    def update_main_view(self):
        """Update main window."""
        self.ax_main.clear()
        
        # FIX: Safely remove old colorbars
        for cbar, cbar_ax in [(self.colorbar, self.colorbar_ax), 
                               (self.colorbar_jac, self.colorbar_jac_ax)]:
            try:
                if cbar is not None:
                    cbar.remove()
            except (KeyError, ValueError, AttributeError):
                pass
            try:
                if cbar_ax is not None:
                    cbar_ax.remove()
            except (KeyError, ValueError, AttributeError):
                pass
        
        self.colorbar = None
        self.colorbar_ax = None
        self.colorbar_jac = None
        self.colorbar_jac_ax = None
        
        slice_pre = self.get_slice_data(self.pre_arr, self.current_slice)
        slice_post = self.get_slice_data(self.post_arr, self.current_slice)
        mask_slice = self.get_slice_data(self.mask_arr, self.current_slice)

        # Normalize using GLOBAL volume thresholds for consistent contrast across all slices
        def _norm_global(arr, vmin, vmax):
            if vmax > vmin:
                return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
            return np.zeros_like(arr)

        slice_pre_n  = _norm_global(slice_pre, self.pre_vmin, self.pre_vmax)
        slice_post_n = _norm_global(slice_post, self.post_vmin, self.post_vmax)

        # Blended view
        blended = (1 - self.overlay_alpha) * slice_pre_n + self.overlay_alpha * slice_post_n

        im = self.ax_main.imshow(blended,
                                 cmap='gray', vmin=0, vmax=1.0,
                                 aspect=self.aspect_ratio,
                                 interpolation='bilinear')
        
        # ── Visual Alignment Guide ──
        # Draw a thin green contour of the brain mask to check alignment
        if np.any(mask_slice > 0):
            try:
                self.ax_main.contour(mask_slice, levels=[0.5], colors='#059669',
                                    linewidths=0.8, alpha=0.7)
            except Exception:
                pass

        # NEW: Highlight synthetic ROI center (Test 3/4)
        if self.synth_roi_arr is not None:
            synth_slice = self.get_slice_data(self.synth_roi_arr, self.current_slice)
            if np.any(synth_slice > 0):
                try:
                    self.ax_main.contour(synth_slice, levels=[0.5], colors='#f87171',
                                        linewidths=1.5, linestyles='--')
                except Exception:
                    pass

        # ── Diff |PRE-POST| overlay ──
        if self.show_diff_overlay:
            diff_slice = np.abs(self.get_slice_data(self.pre_arr, self.current_slice).astype(float)
                                - self.get_slice_data(self.post_arr, self.current_slice).astype(float))
            diff_masked = np.ma.masked_where(mask_slice == 0, diff_slice)
            max_diff = np.percentile(diff_slice[mask_slice > 0], 99.5) if np.any(mask_slice > 0) else 1.0
            if max_diff < 1e-6:
                max_diff = 1.0
            self.ax_main.imshow(diff_masked,
                                cmap='hot', alpha=0.7,
                                vmin=0, vmax=max_diff,
                                aspect=self.aspect_ratio,
                                interpolation='bilinear')
            # Dashed red contour of the perturbed region (>30% of peak)
            try:
                hot_thresh = max_diff * 0.3
                self.ax_main.contour(diff_slice * (mask_slice > 0), levels=[hot_thresh],
                                     colors='#f87171', linewidths=1.2, linestyles='--', alpha=0.9)
            except Exception:
                pass
            self.ax_main.text(5, diff_slice.shape[0] - 5, 'DIFF |PRE-POST|',
                              color='#f87171', fontsize=7, fontweight='bold', va='bottom')

        # Displacement heatmap overlay
        if self.show_displacement_overlay:
            disp_slice = self.get_slice_data(self.displacement_magnitude, self.current_slice)
            disp_masked = np.ma.masked_where(mask_slice == 0, disp_slice)
            
            max_val = np.percentile(disp_slice[mask_slice>0], 95) if np.any(mask_slice>0) else 1
            
            im_disp = self.ax_main.imshow(disp_masked,
                                          cmap='jet', alpha=0.6, 
                                          vmin=0, vmax=max_val,
                                          aspect=self.aspect_ratio,
                                          interpolation='bilinear')
            
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                self.colorbar_ax = inset_axes(self.ax_main,
                                             width="2%",
                                             height="25%",
                                             loc='lower left',
                                             bbox_to_anchor=(-0.03, 0.05, 1, 1),
                                             bbox_transform=self.ax_main.transAxes,
                                             borderpad=0)
                self.colorbar = self.fig.colorbar(im_disp, cax=self.colorbar_ax)
                self.colorbar.ax.yaxis.set_ticks_position('left')
                self.colorbar.ax.yaxis.set_label_position('left')
                self.colorbar.set_label('Disp (mm)', rotation=90, labelpad=5, fontsize=7)
                self.colorbar.ax.tick_params(labelsize=6)
            except Exception as e:
                logger.warning(f"Could not create colorbar: {e}")
        
        # Jacobian map overlay
        if self.show_jacobian_map:
            jac_slice = self.get_slice_data(self.jacobian_field, self.current_slice)
            jac_masked = np.ma.masked_where(mask_slice == 0, jac_slice)
            
            # Custom colormap: blue (contraction) -> white (1.0) -> red (expansion)
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=0.85, vcenter=1.0, vmax=1.15)
            
            im_jac = self.ax_main.imshow(jac_masked,
                                         cmap='RdBu_r', norm=norm, alpha=0.6,
                                         aspect=self.aspect_ratio,
                                         interpolation='bilinear')
            
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                self.colorbar_jac_ax = inset_axes(self.ax_main,
                                                  width="2%",
                                                  height="25%",
                                                  loc='upper left',
                                                  bbox_to_anchor=(-0.03, 0, 1, 0.95),
                                                  bbox_transform=self.ax_main.transAxes,
                                                  borderpad=0)
                self.colorbar_jac = self.fig.colorbar(im_jac, cax=self.colorbar_jac_ax)
                self.colorbar_jac.ax.yaxis.set_ticks_position('left')
                self.colorbar_jac.ax.yaxis.set_label_position('left')
                self.colorbar_jac.set_label('Jacobian', rotation=90, labelpad=5, fontsize=7)
                self.colorbar_jac.ax.tick_params(labelsize=6)
            except Exception as e:
                logger.warning(f"Could not create jacobian colorbar: {e}")
        
        # VECTOR FIELD — Inline 2D projection (per-plane, coherent with heatmap)
        # Mathematics:
        #   ContinuousIndex displacement: disp_grid = [dx_vox, dy_vox, dz_vox] (SimpleITK order: X, Y, Z)
        #   Axial   (slice Z): screen_col=voxX, screen_row(flipud)=voxY  → project (dx, dy)
        #   Sagittal(slice X): screen_col=voxY, screen_row(flipud)=voxZ  → project (dy, dz)
        #   Coronal (slice Y): screen_col=voxX, screen_row(flipud)=voxZ  → project (dx, dz)
        if self.show_vector_field:
            step_x = max(1, int(self.vector_step_mm / self.spacing[0]))
            step_y = max(1, int(self.vector_step_mm / self.spacing[1]))
            step_z = max(1, int(self.vector_step_mm / self.spacing[2]))
            
            if self.view_mode == 'axial':
                step_i, step_j = step_y, step_x
            elif self.view_mode == 'sagittal':
                step_i, step_j = step_z, step_y
            else:  # coronal
                step_i, step_j = step_z, step_x
            
            scale = self.vector_scale
            
            for i in range(step_i//2, slice_pre.shape[0], step_i):
                for j in range(step_j//2, slice_pre.shape[1], step_j):
                    
                    # --- Map screen (j,i) → voxel (x,y,z), accounting for flipud ---
                    if self.view_mode == 'axial':
                        x = j
                        y = slice_pre.shape[0] - 1 - i
                        z = self.current_slice
                    elif self.view_mode == 'sagittal':
                        x = self.current_slice
                        y = j
                        z = slice_pre.shape[0] - 1 - i
                    else:  # coronal
                        x = j
                        y = self.current_slice
                        z = slice_pre.shape[0] - 1 - i
                    
                    if not (0 <= x < self.pre_arr.shape[2] and
                            0 <= y < self.pre_arr.shape[1] and
                            0 <= z < self.pre_arr.shape[0] and
                            self.mask_arr_eroded[z, y, x] > 0):
                        continue
                    
                    point_phys = self.pre_op.TransformIndexToPhysicalPoint([int(x), int(y), int(z)])
                    point_trans = self.transform.TransformPoint(point_phys)
                    
                    # 3D physical magnitude (for color coding)
                    disp_phys = np.array([point_trans[k] - point_phys[k] for k in range(3)])
                    mag_3d = float(np.linalg.norm(disp_phys))
                    
                    # Map displacement to continuous-index (voxel-grid) space.
                    # This bakes out the NIfTI Direction Matrix so directions match anatomy on screen.
                    idx_src   = np.array(self.pre_op.TransformPhysicalPointToContinuousIndex(point_phys))
                    idx_dst   = np.array(self.pre_op.TransformPhysicalPointToContinuousIndex(point_trans))
                    disp_vox  = idx_dst - idx_src   # [dx_vox, dy_vox, dz_vox] in SimpleITK order (X,Y,Z)
                    
                    # --- Per-plane 2D projection ---
                    # screen_dx → right on screen; screen_dy → upward *before* flipud
                    # We negate screen_dy because flipud already inverted Y.
                    if self.view_mode == 'axial':
                        screen_dx =  disp_vox[0]          # voxX  → screen right
                        screen_dy = -disp_vox[1]          # voxY  → screen up (inverted by flipud)
                        px_x, px_y = self.spacing[0], self.spacing[1]
                    elif self.view_mode == 'sagittal':
                        screen_dx =  disp_vox[1]          # voxY  → screen right
                        screen_dy = -disp_vox[2]          # voxZ  → screen up (inverted by flipud)
                        px_x, px_y = self.spacing[1], self.spacing[2]
                    else:  # coronal
                        screen_dx =  disp_vox[0]          # voxX  → screen right
                        screen_dy = -disp_vox[2]          # voxZ  → screen up (inverted by flipud)
                        px_x, px_y = self.spacing[0], self.spacing[2]
                    
                    # Scale: convert voxel displacement to pixel displacement on screen
                    dx = screen_dx * scale
                    dy = (screen_dy / self.aspect_ratio) * scale   # correct anisotropy
                    
                    mag_2d = np.sqrt(dx**2 + dy**2)
                    if mag_2d < 0.3:
                        continue
                    
                    if mag_3d < 0.8:
                        color = '#10b981'
                    elif mag_3d < 2.0:
                        color = '#f59e0b'
                    else:
                        color = '#ef4444'
                    
                    arrow_width = min(2.0, 0.6 + mag_3d * 0.35)
                    self.ax_main.arrow(j, i, dx, dy,
                                       head_width=arrow_width*1.5,
                                       head_length=arrow_width*1.2,
                                       fc=color, ec='white',
                                       alpha=0.85, linewidth=0.6, zorder=10)

        
        # Brain mask contour
        self.ax_main.contour(mask_slice, 
                            levels=[0.5], colors='#3b82f6', 
                            linewidths=2, alpha=0.7)
                            
        # Tumor mask contour (Phase 2)
        if self.show_tumor_mask and getattr(self, 'tumor_arr', None) is not None:
            tumor_slice = self.get_slice_data(self.tumor_arr, self.current_slice)
            if tumor_slice.any():
                self.ax_main.contour(tumor_slice, 
                                    levels=[0.5], colors='#ef4444', 
                                    linewidths=1.5, alpha=0.9, linestyles='dashed')
        
        # Title
        patient_name = getattr(self, 'data_dir', None)
        patient_name = patient_name.name if patient_name else '—'
        title_text = f'{patient_name}  ·  {self.get_view_title()}  ·  {self.current_slice}/{self.max_slices}'
        
        # --- GT Displacement Overlay (for synthetic cases) ---
        if getattr(self, 'gt_displacement_img', None) is not None:
            try:
                gt_arr_full = sitk.GetArrayFromImage(self.gt_displacement_img)  # (Z,Y,X,3)
                gt_mag_full = np.linalg.norm(gt_arr_full, axis=-1)              # (Z,Y,X)
                gt_slice = self.get_slice_data(gt_mag_full, self.current_slice)  # 2D, flipud applied
                
                peak = float(gt_mag_full.max())
                slice_peak = float(gt_slice.max())
                
                if slice_peak > 0.5:  # Only show when there's meaningful displacement on this slice
                    # Semi-transparent heatmap fill
                    self.ax_main.imshow(
                        gt_slice,
                        cmap='hot', alpha=0.25,
                        vmin=0, vmax=max(peak, 1.0),
                        aspect=self.aspect_ratio,
                        zorder=5
                    )
                    # Dashed iso-contours at 1mm, 3mm, 5mm
                    contour_levels = [lv for lv in [1.0, 3.0, 5.0] if lv < slice_peak]
                    if contour_levels:
                        self.ax_main.contour(
                            gt_slice,
                            levels=contour_levels,
                            colors=['#fb923c', '#f97316', '#ea580c'],
                            linewidths=[1.0, 1.5, 2.0],
                            linestyles='dashed', alpha=0.85, zorder=6
                        )
                    # Label with peak magnitude on this slice
                    self.ax_main.text(
                        0.02, 0.04,
                        f'GT peak: {slice_peak:.1f}mm',
                        transform=self.ax_main.transAxes,
                        fontsize=8, color='#fb923c', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5),
                        zorder=15
                    )
            except Exception:
                pass  # Never let overlay crash the viewer

        # --- Visual clues for synthetic test zone ---

        if self.test_info and 'axis' in self.test_info:
            is_correct_view = False
            if self.test_info['axis'] == 0 and self.view_mode == 'axial': is_correct_view = True
            elif self.test_info['axis'] == 1 and self.view_mode == 'sagittal': is_correct_view = True
            elif self.test_info['axis'] == 2 and self.view_mode == 'coronal': is_correct_view = True
            
            if is_correct_view:
                target_slice = self.test_info.get('slice', -1)
                # Apply golden border if we are on the exact slice of synthetic deformation
                if self.current_slice == target_slice:
                    for spine in self.ax_main.spines.values():
                        spine.set_edgecolor('#f59e0b')
                        spine.set_linewidth(3.0)
                        spine.set_visible(True)
                else:
                    for spine in self.ax_main.spines.values():
                        spine.set_edgecolor('none')
            else:
                for spine in self.ax_main.spines.values():
                    spine.set_edgecolor('none')

        self.ax_main.set_title(title_text, fontsize=12, weight='600', color='#1e293b', pad=8)
        self.ax_main.axis('off')
        
        # Mark clicked point
        if self.clicked_point:
            x, y, z = self.clicked_point
            
            if self.view_mode == 'axial' and z == self.current_slice:
                shape_y = self.pre_arr.shape[1]
                plot_x, plot_y = x, shape_y - 1 - y
            elif self.view_mode == 'sagittal' and x == self.current_slice:
                shape_z = self.pre_arr.shape[0]
                plot_x, plot_y = y, shape_z - 1 - z
            elif self.view_mode == 'coronal' and y == self.current_slice:
                shape_z = self.pre_arr.shape[0]
                plot_x, plot_y = x, shape_z - 1 - z
            else:
                plot_x, plot_y = None, None
            
            if plot_x is not None:
                self.ax_main.plot(plot_x, plot_y, '+', color='#ef4444', 
                                 markersize=20, markeredgewidth=3, zorder=20)
                self.ax_main.plot(plot_x, plot_y, 'o', color='#ef4444', 
                                 markersize=25, fillstyle='none', markeredgewidth=3, zorder=20)
        
        self.fig.canvas.draw_idle()
    
    def clear_roi_views(self):
        """Clear ROI windows."""
        for ax, title, accent in [
            (self.ax_roi_pre,    'PRE-op',  '#059669'),
            (self.ax_roi_post,   'POST-op', '#dc2626'),
            (self.ax_roi_vector, 'Vector',  '#f59e0b'),
        ]:
            ax.clear()
            ax.axis('off')
            try:
                ax.set_aspect(self.aspect_ratio)
            except ValueError:
                ax.set_aspect('equal')
            ax.set_facecolor('#f8fafc')
            ax.text(0.5, 0.58, title, ha='center', va='center',
                    fontsize=8, weight='600', color=accent, alpha=0.55,
                    transform=ax.transAxes)
            ax.text(0.5, 0.32, 'click to activate', ha='center', va='center',
                    fontsize=7, style='italic', color='#94a3b8',
                    transform=ax.transAxes)

        # Info card — styled idle hint
        self.ax_info.clear()
        self.ax_info.axis('off')
        for text, fs, fw, fc, y, fst in [
            ('── ANALYSIS ──',  6.5, 'normal', '#cbd5e1', 0.84, 'normal'),
            ('Click brain to',   7.0, 'normal', '#64748b', 0.66, 'normal'),
            ('inspect a point',  7.0, 'normal', '#64748b', 0.53, 'normal'),
            ('Disp · Jacobian',  6.5, 'normal', '#94a3b8', 0.36, 'normal'),
            ('ROI Stats · GT',   6.5, 'normal', '#94a3b8', 0.26, 'normal'),
            ('H = help',         6.0, 'normal', '#cbd5e1', 0.10, 'italic'),
        ]:
            self.ax_info.text(0.5, y, text, ha='center', va='center',
                              fontsize=fs, weight=fw, style=fst, color=fc,
                              transform=self.ax_info.transAxes)

        self.fig.canvas.draw_idle()
    
    def draw_xyz_reference(self, ax, box_size):
        """Draw X,Y,Z coordinate reference system."""
        margin = box_size * 0.06
        center_x = margin + box_size * 0.09
        center_y = margin + box_size * 0.09
        length = box_size * 0.09
        
        if self.view_mode == 'axial':
            ax.arrow(center_x - length*0.4, center_y, length*0.8, 0,
                    head_width=length*0.15, head_length=length*0.12,
                    fc='#dc2626', ec='#dc2626', alpha=0.85, linewidth=1.5, zorder=15)
            ax.text(center_x + length*0.6, center_y, '+X', fontsize=8, weight='600',
                   color='#dc2626', va='center', ha='left', alpha=0.9, zorder=15)
            
            ax.arrow(center_x, center_y - length*0.4, 0, length*0.8,
                    head_width=length*0.15, head_length=length*0.12,
                    fc='#16a34a', ec='#16a34a', alpha=0.85, linewidth=1.5, zorder=15)
            ax.text(center_x, center_y + length*0.6, '+Y', fontsize=8, weight='600',
                   color='#16a34a', ha='center', va='top', alpha=0.9, zorder=15)
            
        elif self.view_mode == 'sagittal':
            ax.arrow(center_x - length*0.4, center_y, length*0.8, 0,
                    head_width=length*0.15, head_length=length*0.12,
                    fc='#16a34a', ec='#16a34a', alpha=0.85, linewidth=1.5, zorder=15)
            ax.text(center_x + length*0.6, center_y, '+Y', fontsize=8, weight='600',
                   color='#16a34a', va='center', ha='left', alpha=0.9, zorder=15)
            
            ax.arrow(center_x, center_y - length*0.4, 0, length*0.8,
                    head_width=length*0.15, head_length=length*0.12,
                    fc='#3b82f6', ec='#3b82f6', alpha=0.85, linewidth=1.5, zorder=15)
            ax.text(center_x, center_y + length*0.6, '+Z', fontsize=8, weight='600',
                   color='#3b82f6', ha='center', va='top', alpha=0.9, zorder=15)
            
        elif self.view_mode == 'coronal':
            ax.arrow(center_x - length*0.4, center_y, length*0.8, 0,
                    head_width=length*0.15, head_length=length*0.12,
                    fc='#dc2626', ec='#dc2626', alpha=0.85, linewidth=1.5, zorder=15)
            ax.text(center_x + length*0.6, center_y, '+X', fontsize=8, weight='600',
                   color='#dc2626', va='center', ha='left', alpha=0.9, zorder=15)
            
            ax.arrow(center_x, center_y - length*0.4, 0, length*0.8,
                    head_width=length*0.15, head_length=length*0.12,
                    fc='#3b82f6', ec='#3b82f6', alpha=0.85, linewidth=1.5, zorder=15)
            ax.text(center_x, center_y + length*0.6, '+Z', fontsize=8, weight='600',
                   color='#3b82f6', ha='center', va='top', alpha=0.9, zorder=15)
    
    def update_roi_views(self):
        """Update ROI windows."""
        if not self.clicked_point:
            return
        
        x, y, z = self.clicked_point
        
        roi_pre, _ = self.extract_roi(self.pre_arr, x, y, z, self.roi_size)
        roi_post, _ = self.extract_roi(self.post_arr, x, y, z, self.roi_size)
        
        stats = self.compute_roi_statistics(roi_pre, roi_post)
        
        self.ax_roi_pre.clear()
        self.ax_roi_post.clear()
        self.ax_roi_vector.clear()
        
        shape_y, shape_x = roi_pre.shape
        extent = [0, shape_x, shape_y, 0]
        aspect = self.aspect_ratio
        
        def _norm01_roi(arr):
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
            if hi > lo:
                return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            return np.zeros_like(arr)

        roi_pre_n  = _norm01_roi(roi_pre)
        roi_post_n = _norm01_roi(roi_post)

        # PRE-op
        self.ax_roi_pre.imshow(roi_pre_n, cmap='gray', vmin=0, vmax=1.0,
                               aspect=aspect, extent=extent)
        cross_x = shape_x / 2.0
        cross_y = shape_y / 2.0
        self.ax_roi_pre.plot(cross_x, cross_y, '+', color='#059669',
                            markersize=12, markeredgewidth=2.5)
        self.ax_roi_pre.text(0.5, 0.97, f'PRE  {self.roi_size}px',
                             ha='center', va='top', transform=self.ax_roi_pre.transAxes,
                             fontsize=8, weight='600', color='white', zorder=20,
                             bbox=dict(boxstyle='round,pad=0.18', facecolor='#059669', alpha=0.75))
        self.ax_roi_pre.set_xlim(0, shape_x)
        self.ax_roi_pre.set_ylim(shape_y, 0)
        self.ax_roi_pre.axis('off')
        
        rect = Rectangle((0, 0), shape_x, shape_y,
                        linewidth=2, edgecolor='#059669', facecolor='none')
        self.ax_roi_pre.add_patch(rect)
        
        # POST-op
        self.ax_roi_post.imshow(roi_post_n, cmap='gray', vmin=0, vmax=1.0,
                                aspect=aspect, extent=extent)
        self.ax_roi_post.plot(cross_x, cross_y, '+', color='#dc2626',
                             markersize=12, markeredgewidth=2.5)
        self.ax_roi_post.text(0.5, 0.97, f'POST  {self.roi_size}px',
                              ha='center', va='top', transform=self.ax_roi_post.transAxes,
                              fontsize=8, weight='600', color='white', zorder=20,
                              bbox=dict(boxstyle='round,pad=0.18', facecolor='#dc2626', alpha=0.75))
        self.ax_roi_post.set_xlim(0, shape_x)
        self.ax_roi_post.set_ylim(shape_y, 0)
        self.ax_roi_post.axis('off')
        
        rect2 = Rectangle((0, 0), shape_x, shape_y,
                         linewidth=2, edgecolor='#dc2626', facecolor='none')
        self.ax_roi_post.add_patch(rect2)
        
        # Displacement vector (proportional to actual displacement)
        self.ax_roi_vector.imshow(roi_pre_n, cmap='gray', vmin=0, vmax=1.0,
                                  aspect=aspect, extent=extent)

        disp_2d = self.project_vector_to_2d(self.displacement_vector)

        if self.view_mode == 'axial':
            dx_px = disp_2d[0] / self.spacing[0]
            dy_px = -(disp_2d[1] / self.spacing[1])
        elif self.view_mode == 'sagittal':
            dx_px = disp_2d[0] / self.spacing[1]
            dy_px = -(disp_2d[1] / self.spacing[2])
        else: # coronal
            dx_px = disp_2d[0] / self.spacing[0]
            dy_px = -(disp_2d[1] / self.spacing[2])
        
        visual_scale = 3.0
        dx_vis = dx_px * visual_scale
        dy_vis = dy_px * visual_scale
        
        # Cap the visual arrow length to 40% of the ROI size so it doesn't leak out
        max_len_px = min(shape_x, shape_y) * 0.4
        curr_len_px = np.sqrt(dx_vis**2 + dy_vis**2)
        
        if curr_len_px > max_len_px:
            scale_down = max_len_px / curr_len_px
            dx_vis *= scale_down
            dy_vis *= scale_down
            arrow_note = " (max)"
        else:
            arrow_note = f" (x{visual_scale})"
            
        # ─── GROUND TRUTH ARROW (If available) ───
        gt_note = ""
        if self.gt_displacement_img:
            try:
                gt_vec = np.array(self.gt_displacement_img.GetPixel(int(x), int(y), int(z)))
                gt_2d = self.project_vector_to_2d(gt_vec)

                if self.view_mode == 'axial':
                    gtvx, gtvy = gt_2d[0] / self.spacing[0], -(gt_2d[1] / self.spacing[1])
                elif self.view_mode == 'sagittal':
                    gtvx, gtvy = gt_2d[0] / self.spacing[1], -(gt_2d[1] / self.spacing[2])
                else:
                    gtvx, gtvy = gt_2d[0] / self.spacing[0], -(gt_2d[1] / self.spacing[2])

                gtmag = np.linalg.norm(gt_vec)  # magnitude is sign-invariant
                gt_note = f" (GT:{gtmag:.1f}mm)"
                
                # Draw Blue Shadow Shadow (GT)
                if gtmag > 0.1:
                    gt_arrow = FancyArrow(cross_x, cross_y, gtvx*visual_scale, gtvy*visual_scale,
                                         width=0.8, head_width=3.5, head_length=3.5,
                                         fc='#3b82f6', ec='#1e3a8a', alpha=0.5, 
                                         linewidth=0.8, zorder=4)
                    self.ax_roi_vector.add_patch(gt_arrow)
            except (IndexError, RuntimeError):
                pass

        if abs(dx_vis) > 0.2 or abs(dy_vis) > 0.2:
            # Draw a beautiful bold red arrow
            arrow = FancyArrow(cross_x, cross_y, dx_vis, dy_vis,
                              width=1.2, head_width=4.5, head_length=4.5,
                              fc='#ef4444', ec='#7f1d1d', alpha=0.95, 
                              linewidth=1.0, zorder=5)
            self.ax_roi_vector.add_patch(arrow)
            # Dot at the base of the arrow
            self.ax_roi_vector.plot(cross_x, cross_y, 'o', color='#7f1d1d', 
                                   markersize=5, zorder=6)
        
        # Central cross (subtle)
        self.ax_roi_vector.plot(cross_x, cross_y, '+', color='#f1f5f9', 
                               markersize=14, markeredgewidth=1.0, alpha=0.5, zorder=4)
        
        mag_2d = np.sqrt(disp_2d[0]**2 + disp_2d[1]**2)
        self.ax_roi_vector.text(0.5, 0.97, f'Disp {mag_2d:.2f}mm{gt_note}',
                                ha='center', va='top', transform=self.ax_roi_vector.transAxes,
                                fontsize=8, weight='600', color='white', zorder=20,
                                bbox=dict(boxstyle='round,pad=0.18', facecolor='#b45309', alpha=0.75))
        self.ax_roi_vector.set_xlim(0, shape_x)
        self.ax_roi_vector.set_ylim(shape_y, 0)
        self.ax_roi_vector.axis('off')
        
        rect3 = Rectangle((0, 0), shape_x, shape_y,
                         linewidth=2, edgecolor='#f59e0b', facecolor='none')
        self.ax_roi_vector.add_patch(rect3)
        
        self.draw_xyz_reference(self.ax_roi_vector, shape_x)
        
        # Info panel — structured display
        mag_3d = float(np.sqrt(np.sum(self.displacement_vector**2)))

        self.ax_info.clear()
        self.ax_info.axis('off')

        def _row(y, label, value, vcol='#1e293b', fs=6.8):
            self.ax_info.text(0.06, y, label, ha='left', va='center',
                              fontsize=fs, color='#64748b',
                              transform=self.ax_info.transAxes)
            self.ax_info.text(0.94, y, value, ha='right', va='center',
                              fontsize=fs, weight='600', color=vcol,
                              transform=self.ax_info.transAxes)

        def _divider(y):
            self.ax_info.plot([0.04, 0.96], [y, y], color='#e2e8f0',
                              linewidth=0.7, transform=self.ax_info.transAxes,
                              clip_on=False)

        _row(0.88, 'Voxel', f'({x},{y},{z})', vcol='#475569', fs=6.3)

        mag_col = '#ef4444' if mag_3d > 3 else '#f59e0b' if mag_3d > 1 else '#10b981'
        _row(0.73, '|d| 3D', f'{mag_3d:.2f} mm', vcol=mag_col)

        _divider(0.62)

        if self.jacobian and self.jacobian[0]:
            det_J = self.jacobian[0]
            delta = (det_J - 1) * 100
            jcol = '#ef4444' if abs(det_J - 1) > 0.10 else '#f59e0b' if abs(det_J - 1) > 0.05 else '#10b981'
            _row(0.52, 'Jacobian', f'{det_J:.3f} ({delta:+.0f}%)', vcol=jcol)
        else:
            _row(0.52, 'Jacobian', 'N/A')

        _divider(0.41)

        corr = stats["correlation"]
        corr_col = '#10b981' if corr > 0.9 else '#f59e0b' if corr > 0.7 else '#ef4444'
        _row(0.31, 'Corr r', f'{corr:.2f}', vcol=corr_col)
        _row(0.18, 'MAD', f'{stats["mad"]:.0f}')
        _row(0.06, 'Pre/Post', f'{stats["mean_pre"]:.0f} / {stats["mean_post"]:.0f}', fs=6.3)
        
        self.fig.canvas.draw_idle()
    
    def run(self):
        """Start viewer."""
        logger.info("Starting Interactive Brain Shift Viewer...")
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#f8fafc')
        self.fig.canvas.manager.set_window_title('Brain Shift Analysis')
        
        self._setup_layout()
        self._setup_widgets()
        
        # If init left a pending error, show it now
        if self.pending_error:
            self.fig.suptitle(self.pending_error,
                             color='red', fontsize=12, fontweight='bold')
            self.ax_main.set_facecolor('#f1f5f9')
            self.ax_main.text(
                0.5, 0.5,
                "No registration data found.\n\n"
                "Run first:\n  python main.py --register-tissue",
                ha='center', va='center', transform=self.ax_main.transAxes,
                fontsize=13, color='#64748b', style='italic',
                bbox=dict(boxstyle='round,pad=1.2', facecolor='white',
                          edgecolor='#cbd5e1', linewidth=2, alpha=0.9)
            )
            self.ax_main.axis('off')
            self.clear_roi_views()
        else:
            self.update_main_view()
            self.clear_roi_views()
        
        self._connect_events()
        
        logger.info("System ready. Displaying interface.")
        plt.show()

    def _setup_layout(self):
        """Define GridSpec layout."""
        # 3-column layout: left sidebar (22%) | main view (52%) | controls (26%)
        gs = GridSpec(1, 3, figure=self.fig,
                      width_ratios=[0.22, 0.52, 0.26],
                      wspace=0.10,
                      left=0.01, right=0.99, top=0.93, bottom=0.05)

        # Col 0: Left sidebar — 3 ROI panels + info card at bottom
        gs_left = GridSpecFromSubplotSpec(
            4, 1, subplot_spec=gs[0],
            height_ratios=[0.29, 0.29, 0.29, 0.13],
            hspace=0.08,
        )
        self.ax_roi_pre    = self.fig.add_subplot(gs_left[0])
        self.ax_roi_post   = self.fig.add_subplot(gs_left[1])
        self.ax_roi_vector = self.fig.add_subplot(gs_left[2])
        self.ax_info       = self.fig.add_subplot(gs_left[3])
        self.ax_info.axis('off')

        # Col 1: Main MRI view
        self.ax_main = self.fig.add_subplot(gs[1])

        # Col 2: Controls panel — header | buttons | controls | navigator | footer
        self.gs_right = GridSpecFromSubplotSpec(
            5, 1, subplot_spec=gs[2],
            height_ratios=[0.045, 0.115, 0.455, 0.315, 0.070],
            hspace=0.18,
        )

    def _setup_widgets(self):
        """Create and position interactive widgets."""

        # ── 1. Branded header bar ──────────────────────────────────────────────
        ax_header = self.fig.add_subplot(self.gs_right[0])
        ax_header.set_facecolor('#1e293b')
        for spine in ax_header.spines.values():
            spine.set_visible(False)
        ax_header.set_xticks([]); ax_header.set_yticks([])
        ax_header.text(0.5, 0.62, 'BRAIN SHIFT VIEWER',
                       ha='center', va='center',
                       fontsize=8.5, weight='bold', color='white',
                       transform=ax_header.transAxes)
        patient_lbl = getattr(self, 'data_dir', None)
        patient_lbl = patient_lbl.name if patient_lbl else '—'
        ax_header.text(0.5, 0.20, patient_lbl,
                       ha='center', va='center',
                       fontsize=6.5, color='#94a3b8', style='italic',
                       transform=ax_header.transAxes)
        self._ax_header = ax_header  # keep ref to update on patient switch

        # ── 2. View + Action buttons — single 2×3 block ───────────────────────
        gs_buttons = GridSpecFromSubplotSpec(
            2, 3, subplot_spec=self.gs_right[1], wspace=0.05, hspace=0.12
        )
        # Row 0: view mode
        self.btn_axial = Button(self.fig.add_subplot(gs_buttons[0, 0]), 'Axial',
                                color='#e0f2fe', hovercolor='#bae6fd')
        self.btn_axial.label.set_fontsize(7.5)
        self.btn_axial.on_clicked(lambda x: self.on_view_change('Axial'))

        self.btn_sag = Button(self.fig.add_subplot(gs_buttons[0, 1]), 'Sagittal',
                              color='#f0f9ff', hovercolor='#bae6fd')
        self.btn_sag.label.set_fontsize(7.5)
        self.btn_sag.on_clicked(lambda x: self.on_view_change('Sagittal'))

        self.btn_cor = Button(self.fig.add_subplot(gs_buttons[0, 2]), 'Coronal',
                              color='#f0f9ff', hovercolor='#bae6fd')
        self.btn_cor.label.set_fontsize(7.5)
        self.btn_cor.on_clicked(lambda x: self.on_view_change('Coronal'))

        # Row 1: actions
        self.btn_reset = Button(self.fig.add_subplot(gs_buttons[1, 0]), 'Reset',
                                color='#f1f5f9', hovercolor='#e2e8f0')
        self.btn_reset.label.set_fontsize(7.5)
        self.btn_reset.on_clicked(self.on_reset)

        self.btn_save = Button(self.fig.add_subplot(gs_buttons[1, 1]), 'Save',
                               color='#f1f5f9', hovercolor='#e2e8f0')
        self.btn_save.label.set_fontsize(7.5)
        self.btn_save.on_clicked(self.on_save)

        self.btn_export = Button(self.fig.add_subplot(gs_buttons[1, 2]), 'Export',
                                 color='#f1f5f9', hovercolor='#e2e8f0')
        self.btn_export.label.set_fontsize(7.5)
        self.btn_export.on_clicked(self.on_export)

        # ── 3. Sliders + Overlay toggles ──────────────────────────────────────
        # Sub-grid: 5 sliders (rows 0-4) + header + 2×3 toggle grid (row 5)
        gs_controls = GridSpecFromSubplotSpec(
            6, 1, subplot_spec=self.gs_right[2],
            height_ratios=[0.11, 0.11, 0.11, 0.11, 0.11, 0.45],
            hspace=0.58,
        )

        def _style_slider(slider):
            # Move label above the bar so it never bleeds left onto the main image
            slider.label.set_color('#475569')
            slider.label.set_fontsize(7.5)
            slider.label.set_position((0.0, 1.4))
            slider.label.set_va('bottom')
            slider.label.set_ha('left')
            slider.valtext.set_color('#3b82f6')
            slider.valtext.set_fontsize(7.5)
            slider.valtext.set_position((1.0, 1.4))
            slider.valtext.set_va('bottom')
            slider.valtext.set_ha('right')

        ax_slice = self.fig.add_subplot(gs_controls[0])
        self.slice_slider = Slider(ax_slice, 'Slice', 0, self.max_slices,
                                   valinit=self.current_slice, valstep=1,
                                   color='#3b82f6', alpha=0.8)
        _style_slider(self.slice_slider)
        self.slice_slider.on_changed(self.on_slice_change)

        ax_alpha = self.fig.add_subplot(gs_controls[1])
        self.alpha_slider = Slider(ax_alpha, 'Overlay', 0, 1, valinit=0.5,
                                   color='#8b5cf6', alpha=0.8)
        _style_slider(self.alpha_slider)
        self.alpha_slider.on_changed(self.on_alpha_change)

        ax_roi = self.fig.add_subplot(gs_controls[2])
        self.roi_slider = Slider(ax_roi, 'ROI Size', 30, 120, valinit=self.roi_size,
                                 valstep=5, color='#10b981', alpha=0.8)
        _style_slider(self.roi_slider)
        self.roi_slider.on_changed(self.on_roi_size_change)

        ax_vec_step = self.fig.add_subplot(gs_controls[3])
        self.vec_step_slider = Slider(ax_vec_step, 'Vec Step', 2, 20,
                                      valinit=int(self.vector_step_mm), valstep=1,
                                      color='#d97706', alpha=0.8)
        _style_slider(self.vec_step_slider)
        self.vec_step_slider.on_changed(self.on_vector_step_change)

        ax_vec_scale = self.fig.add_subplot(gs_controls[4])
        self.vec_scale_slider = Slider(ax_vec_scale, 'Vec Scale', 0.5, 10.0,
                                       valinit=self.vector_scale, valstep=0.5,
                                       color='#d97706', alpha=0.8)
        _style_slider(self.vec_scale_slider)
        self.vec_scale_slider.on_changed(self.on_vector_scale_change)

        # Overlay toggle section: tiny label row + 2×3 grid
        gs_ov_outer = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_controls[5],
            height_ratios=[0.13, 0.87], hspace=0.05,
        )
        ax_ov_lbl = self.fig.add_subplot(gs_ov_outer[0])
        ax_ov_lbl.axis('off')
        ax_ov_lbl.text(0.5, 0.5, '─── OVERLAYS ───',
                       ha='center', va='center', fontsize=7, color='#94a3b8',
                       transform=ax_ov_lbl.transAxes)

        gs_toggles = GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs_ov_outer[1], wspace=0.08, hspace=0.22
        )
        self.btn_heat = Button(self.fig.add_subplot(gs_toggles[0, 0]), 'Heatmap',
                               color='#e0f2fe', hovercolor='#bae6fd')
        self.btn_heat.label.set_fontsize(7.5)
        self.btn_heat.on_clicked(self.toggle_heatmap)

        self.btn_jac = Button(self.fig.add_subplot(gs_toggles[0, 1]), 'Jacobian',
                              color='#fce7f3', hovercolor='#fbcfe8')
        self.btn_jac.label.set_fontsize(7.5)
        self.btn_jac.on_clicked(self.toggle_jacobian)

        self.btn_vec = Button(self.fig.add_subplot(gs_toggles[0, 2]), 'Vectors',
                              color='#dcfce7', hovercolor='#bbf7d0')
        self.btn_vec.label.set_fontsize(7.5)
        self.btn_vec.on_clicked(self.toggle_vectors)

        self.btn_diff = Button(self.fig.add_subplot(gs_toggles[1, 0]), 'Diff Map',
                               color='#fef3c7', hovercolor='#fde68a')
        self.btn_diff.label.set_fontsize(7.5)
        self.btn_diff.on_clicked(self.toggle_diff)

        self.btn_reg = Button(self.fig.add_subplot(gs_toggles[1, 1]), '▶ Rigid POST',
                              color='#f1f5f9', hovercolor='#ddd6fe')
        self.btn_reg.label.set_fontsize(7)
        self.btn_reg.label.set_weight('bold')
        self.btn_reg.on_clicked(self.toggle_registered)

        self.btn_tumor = Button(self.fig.add_subplot(gs_toggles[1, 2]), 'Tumor',
                                color='#fee2e2', hovercolor='#fecaca')
        self.btn_tumor.label.set_fontsize(7.5)
        self.btn_tumor.on_clicked(self.toggle_tumor_mask)

        # ── 4. Patient Navigator ───────────────────────────────────────────────
        gs_nav = GridSpecFromSubplotSpec(
            4, 1, subplot_spec=self.gs_right[3],
            height_ratios=[0.13, 0.23, 0.36, 0.28], hspace=0.08,
        )
        ax_nav_hdr = self.fig.add_subplot(gs_nav[0])
        ax_nav_hdr.set_facecolor('#f1f5f9')
        for spine in ax_nav_hdr.spines.values():
            spine.set_visible(False)
        ax_nav_hdr.set_xticks([]); ax_nav_hdr.set_yticks([])
        ax_nav_hdr.text(0.5, 0.5, 'PATIENT NAVIGATOR',
                        ha='center', va='center',
                        fontsize=7.5, weight='bold', color='#475569',
                        transform=ax_nav_hdr.transAxes)

        if getattr(self, 'available_patients', []):
            current_name = self.data_dir.name
            pts = self._get_filtered_patients()
            self._nav_idx = pts.index(current_name) if current_name in pts else 0

            gs_flt = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_nav[1], wspace=0.05)
            filter_defs = [
                ("All",   "all",   "#dbeafe"),
                ("Real",  "real",  "#dcfce7"),
                ("Pilot", "pilot", "#fef9c3"),
                ("Sweep", "sweep", "#f3e8ff"),
            ]
            self._filter_btns = {}
            for col, (label, fname, active_col) in enumerate(filter_defs):
                is_active = (fname == self._nav_filter)
                btn = Button(self.fig.add_subplot(gs_flt[col]), label,
                             color=active_col if is_active else "#f1f5f9",
                             hovercolor=active_col)
                btn.label.set_fontsize(7)
                btn.on_clicked(lambda e, n=fname: self._set_nav_filter(n))
                self._filter_btns[fname] = btn

            gs_nav_sel = GridSpecFromSubplotSpec(
                1, 3, subplot_spec=gs_nav[2],
                width_ratios=[0.2, 0.6, 0.2], wspace=0.05,
            )
            self.btn_nav_prev = Button(self.fig.add_subplot(gs_nav_sel[0]), '‹',
                                       color='#f1f5f9', hovercolor='#e2e8f0')
            self.btn_nav_prev.label.set_fontsize(11)
            self.btn_nav_prev.on_clicked(self._nav_prev)

            self.ax_nav_name = self.fig.add_subplot(gs_nav_sel[1])
            self.ax_nav_name.axis('off')

            self.btn_nav_next = Button(self.fig.add_subplot(gs_nav_sel[2]), '›',
                                       color='#f1f5f9', hovercolor='#e2e8f0')
            self.btn_nav_next.label.set_fontsize(11)
            self.btn_nav_next.on_clicked(self._nav_next)

            self.btn_nav_load = Button(self.fig.add_subplot(gs_nav[3]), 'Load Selected',
                                       color='#dbeafe', hovercolor='#bfdbfe')
            self.btn_nav_load.label.set_fontsize(8)
            self.btn_nav_load.label.set_weight('bold')
            self.btn_nav_load.on_clicked(self._nav_load)

            self._update_nav_display()
        else:
            ax_nav_empty = self.fig.add_subplot(gs_nav[2])
            ax_nav_empty.axis('off')
            ax_nav_empty.text(0.5, 0.5, "No patients found",
                              ha='center', va='center', fontsize=8, color='#94a3b8')

        # ── 5. Footer ──────────────────────────────────────────────────────────
        ax_footer = self.fig.add_subplot(self.gs_right[4])
        ax_footer.axis('off')
        ax_footer.text(0.5, 0.65, 'Scroll · Click to inspect  |  R reset  S save',
                       ha='center', va='center', fontsize=6, color='#94a3b8',
                       transform=ax_footer.transAxes)
        ax_footer.text(0.5, 0.25, 'H = help  |  E = export JSON',
                       ha='center', va='center', fontsize=6, color='#94a3b8',
                       transform=ax_footer.transAxes)

        # Apply initial button state
        self._refresh_ui_state()

    def _update_nav_display(self):
        """Update the patient name text in the navigator."""
        if not hasattr(self, 'ax_nav_name') or not getattr(self, 'available_patients', []):
            return

        pts = self._get_filtered_patients()
        self.ax_nav_name.clear()
        self.ax_nav_name.axis('off')

        if not pts:
            self.ax_nav_name.text(0.5, 0.5, "(no matches)", ha='center', va='center',
                                  fontsize=8, color='#94a3b8')
            self.fig.canvas.draw_idle()
            return

        self._nav_idx = self._nav_idx % len(pts)
        name = pts[self._nav_idx]
        is_active = (name == self.data_dir.name)
        counter = f"{self._nav_idx + 1}/{len(pts)}"

        bg_color = '#dbeafe' if is_active else '#f8fafc'
        text_color = '#1e40af' if is_active else '#334155'
        weight = 'bold' if is_active else 'normal'

        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle='round,pad=0.05',
                             linewidth=1.2, edgecolor='#cbd5e1', facecolor=bg_color)
        self.ax_nav_name.add_patch(box)
        self.ax_nav_name.text(0.5, 0.62, name, ha='center', va='center',
                              fontsize=8, weight=weight, color=text_color,
                              fontfamily='monospace', wrap=True)
        self.ax_nav_name.text(0.5, 0.18, counter, ha='center', va='center',
                              fontsize=7, color='#64748b')
        self.fig.canvas.draw_idle()

    def _get_filtered_patients(self) -> list:
        pts = getattr(self, 'available_patients', [])
        f = self._nav_filter
        if f == "real":
            return [p for p in pts if not any(p.upper().startswith(x)
                    for x in ("PILOT_", "SWEEP_", "TEST"))]
        if f == "pilot":
            return [p for p in pts if p.upper().startswith("PILOT_")]
        if f == "sweep":
            return [p for p in pts if p.upper().startswith("SWEEP_")]
        return pts

    def _set_nav_filter(self, name: str):
        self._nav_filter = name
        self._nav_idx = 0
        # Update button colours to highlight active filter
        colors = {"all": "#dbeafe", "real": "#dcfce7", "pilot": "#fef9c3", "sweep": "#f3e8ff"}
        inactive = "#f1f5f9"
        for fname, btn in getattr(self, '_filter_btns', {}).items():
            btn.color = colors[fname] if fname == name else inactive
            btn.hovercolor = colors[fname] if fname == name else "#e2e8f0"
            btn.ax.set_facecolor(colors[fname] if fname == name else inactive)
        self._update_nav_display()
        self.fig.canvas.draw_idle()

    def _nav_prev(self, event):
        pts = self._get_filtered_patients()
        if not pts: return
        self._nav_idx = (self._nav_idx - 1) % len(pts)
        self._update_nav_display()

    def _nav_next(self, event):
        pts = self._get_filtered_patients()
        if not pts: return
        self._nav_idx = (self._nav_idx + 1) % len(pts)
        self._update_nav_display()

    def _nav_load(self, event):
        pts = self._get_filtered_patients()
        if not pts: return
        self.on_patient_switch(pts[self._nav_idx])

    def on_patient_switch(self, label):
        if label not in getattr(self, 'patient_dirs', {}):
            return
        if label == self.data_dir.name:
            return 
            
        logger.info(f"Switching patient to {label}...")

        # Central graphic loading banner
        if hasattr(self, 'ax_main'):
            self.ax_main.clear()
            self.ax_main.set_facecolor('#f1f5f9')
            self.ax_main.text(0.5, 0.5, f"LOADING PATIENT\n\n{label}\n\nPlease wait...", 
                              ha='center', va='center', transform=self.ax_main.transAxes,
                              fontsize=14, weight='bold', color='#1e40af',
                              bbox=dict(boxstyle='round,pad=2', facecolor='white', 
                                       edgecolor='#3b82f6', alpha=0.9, linewidth=2))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        try:
            self.load_patient_data(self.patient_dirs[label])
        except Exception as e:
            self.pending_error = f"FATAL ERROR: {label} failed to load."
            logger.error(f"{self.pending_error}: {e}")

        # Refresh interface
        if hasattr(self, 'slice_slider') and hasattr(self, 'max_slices'):
            self.slice_slider.valmax = self.max_slices
            self.slice_slider.set_val(self.current_slice)
            self.slice_slider.ax.set_xlim(0, self.max_slices)

        self.clicked_point = None
        self._refresh_ui_state()
        self._update_nav_display()
        # Refresh branded header with new patient name
        if hasattr(self, '_ax_header') and len(self._ax_header.texts) > 1:
            self._ax_header.texts[1].set_text(self.data_dir.name)
        self.clear_roi_views()
        self.update_main_view()
        self.fig.canvas.draw_idle()

    def _connect_events(self):
        """Bind input events."""
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)


def main():
    """Main entry point."""
    try:
        viewer = InteractiveBrainShiftViewer()
        viewer.run()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
