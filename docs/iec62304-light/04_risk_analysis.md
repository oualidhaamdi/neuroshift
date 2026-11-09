# 4. Risk analysis (ISO 14971 style)

> Educational exercise — not regulatory documentation. See [README](README.md).

Under the hypothetical intended use defined in [03](03_safety_classification.md).

## 4.1 Scales

**Severity (S)**: 1 negligible · 2 minor (no injury, inconvenience) · 3 serious
(non-serious injury, additional procedure) · 4 critical (serious injury) ·
5 catastrophic (death).

**Probability (P)** of harm: 1 improbable · 2 remote · 3 occasional · 4 probable ·
5 frequent.

**Risk acceptability**: S × P ≤ 4 acceptable · 5–9 reduce as far as possible ·
≥ 10 unacceptable.

## 4.2 Hazard analysis

Sequence: **hazard → hazardous situation → harm**, then risk control. P is
estimated before (P₀) and after (P₁) controls.

| ID | Hazard (cause in software) | Hazardous situation | Harm | S | P₀ | Risk control measures | P₁ | Residual |
|---|---|---|---|---|---|---|---|---|
| R1 | Registration underestimates displacement near the resection cavity (metric fooled by the intensity mismatch) | Clinician believes the anatomy moved less than it did | Treatment planning based on wrong anatomy; additional imaging or procedure | 3 | 3 | **CFM** excludes the cavity from the metric (REQ-CFM-01); validated on 112 synthetic cases; cavity mask saved and shown in the viewer for review (REQ-UI-01) | 2 | 6 — ALARP, disclosed in labelling |
| R2 | Physically implausible deformation (folding, det J ≤ 0) | Displacement map shows non-anatomical patterns that are interpreted as real | Misinterpretation, unnecessary follow-up | 2 | 3 | Folding fraction computed and reported for every case (REQ-REG-02); multi-level B-Spline regularises the field; Jacobian map in the viewer | 1 | 2 — acceptable |
| R3 | Pre-op and post-op of **different patients** or mis-assigned folders | Displacement computed between unrelated brains | Grossly wrong assessment | 3 | 2 | Input structure fixed per case (`<id>/0`, `<id>/1`); case ID printed in every log, figure and export; rigid NCC reported for sanity check (REQ-SYS-02) | 1 | 3 — acceptable |
| R4 | Wrong orientation / left–right flip when reading NIfTI | Shift reported on the wrong hemisphere | Wrong-side assessment | 4 | 2 | All images reoriented to RAS with SimpleITK `DICOMOrient`; physical-space resampling only; viewer labels axes (REQ-PRE-04) | 1 | 4 — acceptable |
| R5 | FSL silently unavailable or failing, fallback backend less accurate | Lower-quality skull stripping/segmentation without the user knowing | Reduced accuracy (as R1) | 3 | 3 | Backend selection logged explicitly; SimpleITK fallbacks tested (REQ-PRE-01..03, REQ-SYS-01) | 2 | 6 — ALARP, disclosed |
| R6 | Brain extraction leaves skull / removes cortex | Displacement at the brain surface (where shift is largest) is wrong | As R1 | 3 | 3 | Brain mask overlay in QC figures (`--validate-preprocessing`); Dice of fallback against phantom tested (REQ-PRE-01) | 2 | 6 — ALARP |
| R7 | Units or sign convention error (voxels vs mm, pull-back vs push-forward) | Magnitudes or directions systematically wrong | As R1 | 3 | 2 | Single documented convention (`docs/technical_notes.md` §7); tests on known ground truth check sign and magnitude (REQ-VAL-01, REQ-VAL-02) | 1 | 3 — acceptable |
| R8 | Software used outside its validated scope (non-T1 sequences, paediatric, intra-operative) | Output trusted where accuracy is unknown | As R1 / serious injury if used intra-operatively | 4 | 2 | Intended use and limitations in README; prominent "not a medical device" disclaimer; T1 file priority in input discovery | 2 | 8 — disclosed; not acceptable for clinical use |
| R9 | SOUP update changes numerical behaviour | Results differ silently between installations | As R1 | 3 | 2 | Dependencies pinned; CI on every change; Docker image (REQ-SYS-03) | 1 | 3 — acceptable |

## 4.3 Overall residual risk

All risks except R8 are acceptable or reduced as far as possible under the
hypothetical intended use. R8 is the reason the software is released **only as
research software, not for clinical use**: residual risks R1, R5, R6 and R8 are
disclosed in the README (Limitations) and in the disclaimer.

## 4.4 Risk control verification

Every software risk control above maps to a requirement and a test in the
[traceability matrix](05_traceability_matrix.md).
