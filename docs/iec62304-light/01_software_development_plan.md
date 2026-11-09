# 1. Software development plan (synthetic)

> Educational exercise — not regulatory documentation. See [README](README.md).

## 1.1 Scope

Software item: **NeuroShift**, a Python pipeline that estimates brain shift between
a pre-operative and a post-operative T1-weighted MRI (displacement field, Jacobian,
regional statistics) and an interactive viewer for the results.

Hypothetical safety class: **B** (see [03](03_safety_classification.md)).

## 1.2 Life-cycle model

Iterative development on a single main branch with short-lived feature branches
and pull requests. Each iteration goes through requirements → design →
implementation → verification; the activities below are repeated for every change.

| IEC 62304 activity | How it is done in this project |
|---|---|
| §5.2 Requirements analysis | Requirements with unique IDs in [05](05_traceability_matrix.md) |
| §5.3 Architectural design | Layered design: CLI → orchestration (`register_tissue.py`, `brain_shift_analysis.py`) → backends behind abstract interfaces (`src/backends/base.py`); documented in `docs/technical_notes.md` |
| §5.4 Detailed design | Docstrings and module-level design notes |
| §5.5 Unit implementation and verification | Code review on pull requests; pytest unit tests |
| §5.6 Integration and integration testing | End-to-end tests on a synthetic phantom (`tests/test_registration.py`) |
| §5.7 System testing | Synthetic validation: 112-case sweep, multi-brain cross-validation (N = 50), cavity-mismatch stress test (`src/qc/`) |
| §5.8 Release | Tagged release, pinned dependencies, Docker image, release notes |

## 1.3 Standards, methods and tools

- Language: Python 3.10–3.12. Dependencies pinned in `requirements*.txt`.
- Version control: Git; commits follow Conventional Commits.
- Continuous integration: GitHub Actions runs the test-suite on Python 3.10,
  3.11 and 3.12 and builds the Docker image on every push and pull request.
- Reproducible environment: `Dockerfile` (SimpleITK backends; FSL excluded).

## 1.4 Configuration management (§8)

- Every configuration item (source, tests, documentation, `Dockerfile`,
  `requirements*.txt`) is under version control.
- SOUP is identified with name, version and licence in [02](02_soup_list.md).
- Registration parameters live in `config.py`; runtime overrides (`REGISTRAR`,
  `DISABLE_CFM`) are environment variables recorded in the run log.

## 1.5 Problem resolution (§9)

Defects are tracked as GitHub issues with label, severity and the affected
requirement ID. A fix is merged only with a test that reproduces the defect.

## 1.6 Risk management (§7)

Risk analysis follows ISO 14971 ([04](04_risk_analysis.md)). Every risk control
implemented in software is linked to a requirement and to its verification in the
traceability matrix.

## 1.7 Verification acceptance criteria

| Level | Criterion |
|---|---|
| Unit / integration | All pytest tests pass in CI |
| System (synthetic) | Aggregate typical EPE < 1 mm; Jacobian folding < 0.5 % of brain voxels |
| Release | No open defect affecting a class-B risk control |
