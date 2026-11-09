# 2. SOUP list (Software Of Unknown Provenance)

> Educational exercise — not regulatory documentation. See [README](README.md).

| SOUP | Version | Purpose in NeuroShift | Licence | Required? |
|---|---|---|---|---|
| Python | 3.10–3.12 (3.11 in Docker) | Runtime | PSF License | Yes |
| SimpleITK | 2.4.1 | Image I/O, resampling, B-Spline / rigid registration, fallback brain extraction and segmentation, Jacobian | Apache-2.0 | Yes |
| NumPy | 1.26.4 | Array computation, metrics | BSD-3-Clause | Yes |
| SciPy | 1.13.1 | Statistical tests (Wilcoxon) in validation reports | BSD-3-Clause | Yes |
| Matplotlib | 3.9.2 | Interactive viewer, QC and report figures | Matplotlib License (PSF-based, BSD-compatible) | Yes |
| pandas | 2.2.3 | Tabular statistics and export | BSD-3-Clause | Yes |
| openpyxl | 3.1.5 | Excel export | MIT | Yes |
| pytest | 8.3.3 | Test framework (development only) | MIT | Dev only |
| **FSL** (`bet`, `flirt`, `fast`) | 6.0.x via the FSL conda channel (not pinned by this project) | Preferred backend for brain extraction, rigid alignment, tissue segmentation | **FSL licence — non-commercial use only.** Commercial use requires a licence from Oxford University Innovation | Optional (SimpleITK fallback) |
| WSL2 + Ubuntu | Host-dependent | Runs FSL on Windows | Microsoft / various open-source | Optional |
| HD-BET | Not pinned | Optional neural brain extraction backend (`bet_hdbet.py`), not installed by default | Apache-2.0 | Optional |

Not used: pydicom (DICOM data must be converted to NIfTI beforehand, e.g. with
`dcm2niix`).

## Requirements on SOUP (§5.3.3) and anomaly evaluation (§7.1.3)

| SOUP | Functional / performance requirement relied upon | Known anomalies to monitor |
|---|---|---|
| SimpleITK | Correct B-Spline transform evaluation, `TransformToDisplacementField`, `DisplacementFieldJacobianDeterminant`; deterministic results when a sampling seed is given | Multi-threaded registration is not bit-wise reproducible across machines; NIfTI direction-cosine handling with oblique scans |
| FSL | `bet` produces a brain mask; `flirt` 6-DOF returns a valid rigid matrix; `fast` returns 3 tissue classes | Header issues with very large / oblique inputs (mitigated by reorientation and fallbacks) |
| NumPy / SciPy / pandas | Numerical correctness of reductions and statistical tests | Behaviour changes between major versions (mitigated by pinning) |

Each SOUP release is reviewed for published anomalies before the pinned version is
updated; the update is a change under configuration management and re-runs the full
test-suite.

## Licence compatibility

The NeuroShift code is Apache-2.0. FSL is **invoked as an external program and not
redistributed**, so its licence does not apply to the NeuroShift source, but it
does restrict how *users* may run FSL: **commercial use of the FSL backend requires
a commercial FSL licence**. The SimpleITK-only configuration (e.g. the Docker image)
contains no non-commercial components.
