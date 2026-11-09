"""
Global Configuration for Brain Shift Analysis.
Focuses on registration parameters and project-wide settings.
"""

import os
from pathlib import Path

# ── Project Structure ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.absolute()
# Default case ID used when --patient / BRAIN_SHIFT_ACTIVE_PATIENT are not given.
# Any folder name under RunBatch_Input/ works (e.g. a public RHUH-GBM case ID).
ACTIVE_PATIENT = os.environ.get("BRAIN_SHIFT_ACTIVE_PATIENT", "RHUH-0002")

# ── Reporting Defaults (Patient-specific figures) ───────────────────────────
DIR_OUTPUT_FIGURES      = PROJECT_ROOT / "figures"

# ── Feature Flags ───────────────────────────────────────────────────────────
# Default deformable registrar for the pipeline.
#   "intensity" — B-Spline on raw intensities + Cost Function Masking
#                 (current default; proven more accurate on our cavity-mismatch tests)
#   "tissue"    — B-Spline on Danielsson distance maps of tissue-edge segmentations
#                 (kept as an ablation option; see docs/technical_notes.md §5)
# Override per-run via env var REGISTRAR={intensity,tissue} or CLI flag --tissue-guided.
REGISTRAR_DEFAULT = "intensity"

# ── Registration Profiles ───────────────────────────────────────────────────
SELECTED_REGISTRATION_MODE = 'fast'

REGISTRATION_MODES = {
    'fast': {
        'metric_bins': 32,
        'learning_rate': 1.0,
        'max_iterations': 30,
        'convergence_min_val': 1e-5,
        'convergence_window': 10,
        'shrink_factors': [4, 2, 1],
        'smoothing_sigmas': [2.0, 1.0, 0.0],
        'sampling_percentage': 0.03,
        'grid_spacing': [40.0],
        'distance_map_ds_factor': 4,
    },
    'balanced': {
        'metric_bins': 50,
        'learning_rate': 1.0,
        'max_iterations': 50,
        'convergence_min_val': 1e-6,
        'convergence_window': 20,
        'shrink_factors': [4, 2, 1],
        'smoothing_sigmas': [2.0, 1.0, 0.0],
        'sampling_percentage': 0.05,
        'grid_spacing': [30.0],
        'distance_map_ds_factor': 4,
    },
    'enhanced': {
        'metric_bins': 64,
        'learning_rate': 0.5,
        'max_iterations': 150,
        'convergence_min_val': 1e-7,
        'convergence_window': 30,
        'shrink_factors': [8, 4, 2, 1],
        'smoothing_sigmas': [3.0, 2.0, 1.0, 0.0],
        'sampling_percentage': 0.10,
        'grid_spacing': [20.0],
        'distance_map_ds_factor': 2,
    },
    # Calibrated on the synthetic cases (QC sweep, mean EPE 0.73 mm).
    # Coarse: grid_sp*1.5=15mm → fine: 10mm. ~2-3× slower than 'enhanced'.
    # Real cases (full brain): LBFGS2 — learning_rate ignored, sampling×10 (→100%).
    # Synthetic cases (small ROI): GradientDescent — learning_rate and sampling×2 (→60%) active.
    'precise': {
        'metric_bins': 64,
        'learning_rate': 0.2,
        'max_iterations': 200,
        'convergence_min_val': 1e-7,
        'convergence_window': 30,
        'shrink_factors': [4, 2, 1],
        'smoothing_sigmas': [2.0, 1.0, 0.0],
        'sampling_percentage': 0.30,
        'grid_spacing': [10.0],
        'distance_map_ds_factor': 2,
    },
}

# ── Aesthetics ──────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#2563eb',
    'pre':     '#059669',
    'post':    '#dc2626',
    'vector':  '#d97706',
    'text':    '#1e293b',
}
