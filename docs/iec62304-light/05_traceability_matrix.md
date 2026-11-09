# 5. Requirements and traceability matrix

> Educational exercise — not regulatory documentation. See [README](README.md).

## 5.1 Software requirements

| ID | Requirement | Type | Risk |
|---|---|---|---|
| REQ-PRE-01 | The system shall extract a brain mask from a T1 MRI, using FSL `bet` when available and an Otsu-based SimpleITK method otherwise. | Functional | R5, R6 |
| REQ-PRE-02 | The system shall segment the brain into three tissue classes (CSF, GM, WM). | Functional | R5 |
| REQ-PRE-03 | The system shall rigidly align (6 DOF) the post-op image to the pre-op image. | Functional | R5 |
| REQ-PRE-04 | The system shall reorient all inputs to RAS and operate in physical (mm) space. | Safety | R4 |
| REQ-REG-01 | The system shall estimate a dense displacement field with a B-Spline FFD registration, recovering a known smooth deformation with a mean EPE below 50 % of the deformation itself. | Functional / performance | R1 |
| REQ-REG-02 | The system shall compute the Jacobian determinant and report the fraction of folded voxels (det J ≤ 0). | Safety | R2 |
| REQ-CFM-01 | When a cavity mask is available, the system shall exclude it from the registration metric (CFM) unless explicitly disabled, and accuracy near the cavity shall not degrade. | Safety | R1 |
| REQ-MET-01 | The system shall compute image similarity metrics (NCC, MI). | Functional | R3 |
| REQ-MET-02 | The Jacobian computation shall return 1 for the identity and s³ for an isotropic scaling s. | Functional | R2 |
| REQ-VAL-01 | The validation framework shall generate synthetic Gaussian displacements with known amplitude, width and direction, confined to the brain mask. | Verification | R7 |
| REQ-VAL-02 | Validation metrics shall return zero error for a perfect recovery and the ground-truth magnitude for no recovery, using the pull-back convention. | Verification | R7 |
| REQ-UI-01 | The viewer shall display displacement magnitude, Jacobian and vector overlays on the anatomy for review. | Functional | R1, R2 |
| REQ-SYS-01 | The system shall run without FSL, selecting fallbacks automatically and without raising errors when FSL is missing. | Functional | R5 |
| REQ-SYS-02 | Every output, log and export shall carry the case ID. | Safety | R3 |
| REQ-SYS-03 | The software shall be buildable reproducibly from pinned dependencies (requirements files, Docker). | Non-functional | R9 |

## 5.2 Traceability: requirement → implementation → verification

| Requirement | Implementation | Verification | Type |
|---|---|---|---|
| REQ-PRE-01 | `src/backends/bet_fsl.py`, `bet_sitk.py` | `tests/test_backends.py::test_otsu_brain_extraction_matches_phantom` | Automated |
| REQ-PRE-02 | `src/backends/segmenter_fast.py`, `segmenter_sitk.py` | `tests/test_backends.py::test_kmeans_segmentation_produces_three_classes_inside_mask` | Automated |
| REQ-PRE-03 | `src/backends/rigid_flirt.py`, `rigid_sitk.py` | `tests/test_backends.py::test_euler3d_recovers_small_translation` | Automated |
| REQ-PRE-04 | `src/register_tissue.py`, `src/qc/create_synthetic_patients.py` (`DICOMOrient`) | Code review; `--validate-preprocessing` QC figures | Manual |
| REQ-REG-01 | `src/backends/deformable_bspline_intensity.py` | `tests/test_registration.py::test_bspline_recovers_known_gaussian_shift`; 112-case sweep and multi-brain cross-validation (`src/qc/`) | Automated + system |
| REQ-REG-02 | `src/brain_shift_analysis.py`, `src/metrics/deformation.py` | `tests/test_registration.py::test_bspline_field_is_diffeomorphic_on_phantom` | Automated |
| REQ-CFM-01 | `src/register_tissue.py` (CFM block), `DISABLE_CFM` switch | `tests/test_registration.py::test_cost_function_masking_near_cavity`; `src/qc/cavity_demo.py`; CFM ablation sweep | Automated + system |
| REQ-MET-01 | `src/metrics/similarity.py` | `tests/test_metrics.py::test_ncc_*`, `test_mi_self_higher_than_noise` | Automated |
| REQ-MET-02 | `src/metrics/deformation.py` | `tests/test_metrics.py::test_jacobian_identity_is_one`, `test_jacobian_uniform_scaling` | Automated |
| REQ-VAL-01 | `src/qc/create_synthetic_patients.py::apply_gauss3d_shift` | `tests/test_synthetic_ground_truth.py::test_gaussian_field_peak_direction_and_support`, `test_gaussian_shift_changes_image` | Automated |
| REQ-VAL-02 | `src/qc/run_synthetic_sweep.py::compute_metrics` | `tests/test_synthetic_ground_truth.py::test_metrics_perfect_recovery_is_zero`, `test_metrics_no_recovery_equals_gt_magnitude` | Automated |
| REQ-UI-01 | `src/ui/viewer.py` | Manual review checklist (planes, overlays, case selector) | Manual |
| REQ-SYS-01 | `src/fsl_wrapper.py::check_fsl_availability`, backend selection in `register_tissue.py` | `tests/test_backends.py::test_fsl_check_never_raises`; whole test-suite runs in CI without FSL | Automated |
| REQ-SYS-02 | `src/core.py` logger, `src/data_manager.py`, `export_metrics.py` | Code review | Manual |
| REQ-SYS-03 | `requirements*.txt`, `Dockerfile`, `.github/workflows/tests.yml` | CI job `docker` builds the image and runs the tests inside it | Automated |

## 5.3 Coverage summary

- 15 requirements; 12 verified by automated tests (11 by pytest, 1 by the CI Docker
  job), 3 by manual review.
- Every software risk control in [04](04_risk_analysis.md) traces to at least one
  requirement.
