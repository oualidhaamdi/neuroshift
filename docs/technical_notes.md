# Technical notes

Methods, conventions and design decisions behind NeuroShift. For usage see the
[README](../README.md).

## 1. Overview

Input: one pre-operative and one post-operative T1-weighted MRI of the same
subject. Output: a dense displacement field **u(x)** in millimetres defined on the
pre-operative grid, its magnitude ‖u‖, the Jacobian determinant det J of the
transform, per-tissue Dice scores and regional statistics.

Orchestration lives in `src/register_tissue.py` (registration) and
`src/brain_shift_analysis.py` (quantification); every algorithmic stage is a
backend implementing an abstract interface in `src/backends/base.py`.

## 2. Preprocessing

| Step | FSL backend | SimpleITK fallback |
|---|---|---|
| Brain extraction | `bet` (f = 0.4) — `bet_fsl.py` | Gaussian smoothing → Otsu → closing → largest component → hole filling — `bet_sitk.py` |
| Tissue segmentation | `fast`, 3 classes (CSF/GM/WM) — `segmenter_fast.py` | multi-Otsu thresholds on intensity-normalised brain voxels — `segmenter_sitk.py` |

Images are reoriented to RAS before processing. Backends are chosen at run time
by `check_fsl_availability()` (`src/fsl_wrapper.py`). The FSL bridge targets
WSL2 on Windows: Windows paths are converted to `/mnt/<drive>/…` and, to avoid
spaces that FSL rejects, routed through a symlink at `/tmp/NeuroShift_Workspace`.

## 3. Rigid alignment

The post-op image is aligned to the pre-op image with a 6-DOF rigid transform:
FSL `flirt` with mutual information (`rigid_flirt.py`) or, as fallback, a
SimpleITK `Euler3DTransform` with Mattes mutual information, geometry-centred
initialisation and a 3-level pyramid (`rigid_sitk.py`). A rigid model is used on
purpose: the skull does not change shape between the two scans, so any residual
non-rigid motion is attributed to brain shift.

## 4. Deformable registration

`BSplineIntensityRegistrar` (`deformable_bspline_intensity.py`):

- **Transform**: cubic B-Spline free-form deformation.
- **Metric**: mean squares on intensities normalised to the [1, 99] percentile
  range inside the brain mask, evaluated only inside the registration mask.
- **Optimiser**: L-BFGS (`LBFGS2`) for full-brain volumes; gradient descent with
  per-iteration learning-rate estimation for small fields of view (≤ 200 mm), such
  as the synthetic validation crops.
- **Coarse-to-fine**: two stages — coarse control-point grid (1.5–2× the target
  spacing), then the fine grid — each with a multi-resolution pyramid. The two
  B-Splines are combined into a single composite transform.
- **Profiles**: grid spacing, iterations, sampling and pyramid are set in
  `config.py` (`fast`, `balanced`, `enhanced`, `precise`). An optional
  `random_seed` makes metric sampling reproducible.

## 5. Registrar choice and Cost Function Masking

**Cost Function Masking (CFM).** The registration mask is the brain mask minus the
resection cavity. Voxels inside the cavity therefore contribute nothing to the
metric: the optimiser is not rewarded for pulling healthy tissue into the dark
void, and the displacement inside the cavity is obtained by the smooth B-Spline
interpolation of its surroundings. The cavity mask is taken, in order, from
`RunBatch_Input/<id>/mask_cavity_manual.nii.gz`, `<out>/mask_cavity_manual.nii.gz`
or `<out>/mask_cavity_auto.nii.gz` (written by the analysis step, which thresholds
the pre/post difference inside the brain at mean + 2.5 SD and removes small
components). `DISABLE_CFM=1` turns CFM off for ablation studies.

**Why intensity + CFM is the default.** Two deformable registrars were compared:

| Registrar | Input to the metric | Module |
|---|---|---|
| Intensity (default) | normalised T1 intensities + CFM | `deformable_bspline_intensity.py` |
| Tissue-guided (ablation) | Danielsson distance maps of CSF/GM/WM edges | `deformable_bspline.py` |

The tissue-guided variant was designed to be insensitive to the cavity's intensity
discontinuity. In the cavity-mismatch experiments (`src/qc/cavity_demo.py`) and in
the synthetic sweeps, however, intensity-based registration **with CFM** was more
accurate: masking removes the cavity problem at its source while keeping the full
intensity information that distance maps throw away. The tissue-guided registrar is
kept as an ablation (`REGISTRAR=tissue`).

| Synthetic cases with a cavity | Without CFM | With CFM |
|---|---|---|
| Mean error inside the cavity | 0.96 mm | 0.43 mm |
| Jacobian folding | 0.207 % | 0.002 % |

On cases without a cavity, CFM produced no significant change.

## 6. Synthetic ground truth

Real post-operative data has no ground-truth displacement, so accuracy is measured
on synthetic deformations (`src/qc/create_synthetic_patients.py`):

    d(x) = A · exp(−‖x − c‖² / 2σ²) · n̂,   inside the brain mask, 0 outside

with amplitude A ∈ {3, 5, 8, 15} mm, width σ ∈ {15, 25, 40} mm, direction
n̂ ∈ {X, Y, Z, XYZ} and centre c at the brain centre or at one of four off-centre
locations (anterior, posterior, lateral, superior).

- **Core grid** 4 × 3 × 4 = 48 cases + **location grid** 8 cases = 56 cases.
- **PRE variant**: fixed = pre-op, moving = pre-op warped by d (no cavity).
- **POST variant**: fixed = post-op (rigidly aligned), moving = post-op warped by d
  (a real cavity is present in both). PRE/POST pairs isolate the effect of the cavity.
- Total: **112 cases** (`src/qc/run_synthetic_sweep.py`).
- **Multi-brain cross-validation** (`src/qc/multi_brain_sweep.py`): 10 cases per
  source brain on several brains from two scanners (N = 50), to test generalisation
  across anatomies and acquisition protocols.
- **Cavity-mismatch demo** (`src/qc/cavity_demo.py`): a spherical dark cavity is
  added to the moving image only, to stress the metric with an intensity
  discontinuity that has no counterpart in the fixed image.

Sources with an in-plane spacing finer than 0.5 mm are first resampled to 1 mm
isotropic; on very fine anisotropic grids the mean-squares gradient becomes too
weak and the optimiser stalls at identity.

## 7. Conventions and metrics

- The stored ground truth `gt_displacement.nii.gz` is a **pull-back** field: the
  synthetic image is sampled at x + d(x), so tissue moves by −d. A perfect
  registration returns **u = −d**.
- **End-point error** EPE(x) = ‖u(x) + d(x)‖, evaluated on brain voxels with
  ‖d‖ > 0.5 mm. Reported as mean/median, 95th percentile ("worst 5 %") and max.
- **Angular error**: angle between u and −d (degrees); also reported weighted by ‖d‖
  and restricted to ‖d‖ > 2 mm, where direction is meaningful.
- **Recovery (dot ratio)**: (u · −d) / ‖d‖²; 1 = perfect, 0 = no recovery,
  < 0 = wrong direction.
- **Folding**: percentage of brain voxels with det J ≤ 0 (locally non-invertible).
- Arrays follow the SimpleITK/NumPy convention (z, y, x) with vector components
  ordered (x, y, z) in millimetres.

## 8. Outputs

`RunBatch_Output/<id>/`:

| File | Content |
|---|---|
| `tissue_registration/post_rigid.nii.gz` | post-op after rigid alignment |
| `tissue_registration/tissue_bspline.tfm` | deformable transform |
| `tissue_registration/brain_mask.nii.gz`, `seg_*.nii.gz` | masks and tissue maps |
| `brainshift_displacement.nii.gz` | displacement field u (vector, mm) |
| `brainshift_magnitude.nii.gz` | ‖u‖ (mm) |
| `brainshift_jacobian.nii.gz` | det J |
| `mask_cavity_auto.nii.gz` | automatically detected cavity |
| `validation/` | optional QC figures (`--validate-*` flags) |
