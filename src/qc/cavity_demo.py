"""
Cavity Mismatch Demo — stress test for post-surgical intensity discontinuities.

Generates TWO synthetic cases with identical content (pre-op + Gaussian shift +
spherical dark cavity in the moving image only). Runs:
  • CavityDemo_TISSUE   →  tissue-guided B-Spline (distance maps)
  • CavityDemo_BASELINE →  intensity-baseline B-Spline (raw intensities)

Same Fixed, same Moving, same GT. The only difference is the registrar.
Hypotheses under test:
  baseline -> MSE on raw intensities may be fooled by the dark cavity and drag
              neighbouring tissue into the void. Cost Function Masking (CFM)
              excludes the cavity from the metric; use --no-cfm to see the
              unmasked behaviour.
  tissue   -> Danielsson distance maps of tissue edges are less sensitive to
              the dark hole, but discard fine intensity detail.
In the thesis experiments intensity + CFM outperformed the tissue-guided
registrar, which is why it is the pipeline default.

Usage
-----
python -m src.qc.cavity_demo --source-brain RHUH-0002
python -m src.qc.cavity_demo --radius 20 --A 8         # bigger cavity, larger shift

After it finishes, view the displacement fields side-by-side with:
    python main.py --view
and select 'CavityDemo_TISSUE' or 'CavityDemo_BASELINE' from the dropdown.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import load_config, setup_logger
from src.data_manager import DataManager
from src.qc.create_synthetic_patients import generate_cavity_mismatch_test
from src.qc.run_synthetic_sweep import compute_metrics

logger = setup_logger("cavity_demo")


def _run_pipeline(case_id: str, registrar: str, no_cfm: bool, project_root: Path) -> bool:
    """Invoke main.py --register-tissue --analyze with the chosen registrar + CFM setting."""
    env = os.environ.copy()
    env["REGISTRAR"] = registrar    # "intensity" (default) | "tissue" (ablation)
    env.pop("BASELINE_REGISTRAR", None)
    if no_cfm:
        env["DISABLE_CFM"] = "1"
    else:
        env.pop("DISABLE_CFM", None)
    cmd = [sys.executable, "main.py", "--register-tissue", "--analyze",
           "--patient", case_id]
    logger.info(f"  $ {' '.join(cmd)}    (REGISTRAR={registrar}, CFM={'OFF' if no_cfm else 'ON'})")
    return subprocess.run(cmd, cwd=project_root, env=env).returncode == 0


def _print_summary(name_t: str, m_t: dict, name_b: str, m_b: dict) -> None:
    if not m_t or not m_b:
        logger.warning(
            "Cannot print comparison table: registration outputs are missing.\n"
            f"  tissue   metrics: {'OK' if m_t else 'MISSING — run without --skip-tissue'}\n"
            f"  baseline metrics: {'OK' if m_b else 'MISSING — run without --skip-baseline'}\n"
            "Hint: launch `python -m src.qc.cavity_demo` (no flags) for the full demo."
        )
        return
    print(f"\n{'='*82}")
    print(f"Cavity-Mismatch Demo  --  tissue-guided vs intensity-baseline")
    print(f"{'='*82}")
    print(f"  {'Metric':<32s}  {name_t:>14s}  {name_b:>14s}  {'D (T-B)':>10s}")
    print("  " + "-" * 76)
    rows = [
        ("EPE mean (mm)",                    m_t["epe"]["mean"],         m_b["epe"]["mean"],         True),
        ("EPE P95  (mm)",                    m_t["epe"]["p95"],          m_b["epe"]["p95"],          True),
        ("Angular mean (deg)",               m_t["angular_err"]["mean"], m_b["angular_err"]["mean"], True),
        ("Angular central (|GT|>2mm)",       m_t["angular_err"]["mean_central"],
                                              m_b["angular_err"]["mean_central"],                    True),
        ("Angular weighted (deg*mm/mm)",     m_t["angular_err"]["weighted_mean"],
                                              m_b["angular_err"]["weighted_mean"],                   True),
        ("Dot ratio (recovery)",             m_t["dot_ratio"]["mean"],   m_b["dot_ratio"]["mean"],   False),
        ("Jacobian folding (%)",             m_t["jacobian"]["pct_fold"], m_b["jacobian"]["pct_fold"], True),
    ]
    for label, tv, bv, lower_better in rows:
        delta = tv - bv
        if lower_better:
            verdict = "tissue wins" if delta < 0 else "baseline wins" if delta > 0 else "tie"
        else:
            verdict = "tissue wins" if delta > 0 else "baseline wins" if delta < 0 else "tie"
        print(f"  {label:<32s}  {tv:>14.4f}  {bv:>14.4f}  {delta:>+10.4f}   {verdict}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Cavity-mismatch demo: tissue vs baseline")
    parser.add_argument("--source-brain",   default="RHUH-0002",
                        help="Case ID under RunBatch_Input/ used as source anatomy")
    parser.add_argument("--A",       type=float, default=5.0,  help="Gaussian amplitude (mm)")
    parser.add_argument("--sigma",   type=float, default=25.0, help="Gaussian σ (mm)")
    parser.add_argument("--radius",  type=float, default=15.0, help="Cavity radius (mm)")
    parser.add_argument("--intensity-factor", type=float, default=0.05,
                        help="Cavity intensity multiplier (0=black, 1=normal)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Reuse existing CavityDemo_* input dirs (don't regenerate)")
    parser.add_argument("--skip-tissue",   action="store_true",
                        help="Skip the tissue-guided (ablation) pipeline run")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the intensity (default) pipeline run")
    parser.add_argument("--no-cfm",        action="store_true",
                        help="Disable Cost Function Masking in BOTH runs (CFM-ablation study)")
    args = parser.parse_args()

    config       = load_config()
    project_root = Path(config.PROJECT_ROOT)
    dm           = DataManager(config)

    paths = dm.get_patient_paths(args.source_brain)
    if not paths["pre_op"] or not paths["pre_op"].exists():
        logger.error(f"Source brain {args.source_brain} not found")
        return

    # Load source
    pre_img  = sitk.DICOMOrient(sitk.ReadImage(str(paths["pre_op"])), "RAS")
    if paths["brain_mask"] and paths["brain_mask"].exists():
        mask_img = sitk.DICOMOrient(sitk.ReadImage(str(paths["brain_mask"])), "RAS")
    else:
        from src.backends.bet_sitk import OtsuExtractor
        mask_img = OtsuExtractor().extract(sitk.Cast(pre_img, sitk.sitkFloat32))

    name_t = "CavityDemo_TISSUE"
    name_b = "CavityDemo_BASELINE"

    if not args.skip_generate:
        for case_id in (name_t, name_b):
            logger.section(f"Generating {case_id}")
            generate_cavity_mismatch_test(
                pre_img, mask_img,
                test_name=case_id,
                direction_vec=[1.0, 0.0, 0.0],
                A_mm=args.A, sigma_mm=args.sigma,
                center_frac=(0.5, 0.5, 0.5),
                cavity_radius_mm=args.radius,
                cavity_intensity_factor=args.intensity_factor,
            )

    # Run tissue-guided (ablation)
    if not args.skip_tissue:
        logger.section(f"[1/2] Tissue-guided ablation pipeline → {name_t}")
        if not _run_pipeline(name_t, registrar="tissue", no_cfm=args.no_cfm,
                             project_root=project_root):
            logger.error(f"  Tissue-guided pipeline failed on {name_t}")
            return

    # Run intensity (default)
    if not args.skip_baseline:
        logger.section(f"[2/2] Intensity (default) pipeline → {name_b}")
        if not _run_pipeline(name_b, registrar="intensity", no_cfm=args.no_cfm,
                             project_root=project_root):
            logger.error(f"  Intensity pipeline failed on {name_b}")
            return

    # Compute metrics — DataManager handles both RunBatch_Output/ and outputs/
    def _metrics_for(case_id: str):
        case_paths = dm.get_patient_paths(case_id)
        out_dir = case_paths.get("out_dir")
        pred = (out_dir / "brainshift_displacement.nii.gz") if out_dir else None
        gt   = project_root / "RunBatch_Input"  / case_id / "gt_displacement.nii.gz"
        mask = case_paths.get("brain_mask")
        if not (pred and pred.exists() and gt.exists() and mask and mask.exists()):
            logger.warning(
                f"  Missing output for {case_id}:\n"
                f"    pred: {pred} {'OK' if pred and pred.exists() else 'MISSING'}\n"
                f"    gt:   {gt} {'OK' if gt.exists() else 'MISSING'}\n"
                f"    mask: {mask} {'OK' if mask and mask.exists() else 'MISSING'}"
            )
            return None
        return compute_metrics(pred, gt, mask)

    m_t = _metrics_for(name_t)
    m_b = _metrics_for(name_b)
    _print_summary(name_t, m_t, name_b, m_b)

    print("View the displacement overlays:")
    print(f"  python main.py --view")
    print(f"  Then select '{name_t}' or '{name_b}' from the dropdown")
    print()
    print("Direct file comparison (with any NIfTI viewer):")
    print(f"  Fixed:        RunBatch_Input/{name_t}/0/t1.nii.gz")
    print(f"  Moving:       RunBatch_Input/{name_t}/1/t1.nii.gz   (has the dark cavity)")
    print(f"  GT field:     RunBatch_Input/{name_t}/gt_displacement.nii.gz")
    print(f"  Tissue pred:  RunBatch_Output/{name_t}/brainshift_displacement.nii.gz")
    print(f"  Baseline pred: RunBatch_Output/{name_b}/brainshift_displacement.nii.gz")
    print(f"  Cavity mask:  RunBatch_Input/{name_t}/mask_cavity_manual.nii.gz")


if __name__ == "__main__":
    main()
