"""
Brain Shift Analysis - Central Launcher.
"""

import sys
import argparse
import os
import SimpleITK as sitk
from pathlib import Path
from src.core import setup_logger, load_config
from src.data_manager import DataManager

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Brain Shift Analysis Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Core pipeline
    parser.add_argument("--register-tissue", action="store_true", help="Run preprocessing + rigid + deformable registration (FSL or SimpleITK)")
    parser.add_argument("--analyze", action="store_true", help="Run Deformable Analysis & Quantification")

    # Utilities
    parser.add_argument("--view", action="store_true", help="Launch interactive matplotlib viewer")
    parser.add_argument("--patient", type=str, help="Specify patient ID (e.g. RHUH-0001)")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    # Validations & Extra
    parser.add_argument("--validate-preprocessing", action="store_true")
    parser.add_argument("--validate-rigid", action="store_true")
    parser.add_argument("--validate-deformable", action="store_true")
    parser.add_argument("--validate-segmentation", action="store_true")
    parser.add_argument("--validate-robustness", action="store_true")
    parser.add_argument("--capture-range", action="store_true")
    parser.add_argument("--thesis-figures", action="store_true", help="Generate final publication-ready figures")
    parser.add_argument("--export", action="store_true")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 1. Configuration & Path Resolution
    config = load_config()
    dm = DataManager(config)
    
    patient_id = args.patient or os.environ.get("BRAIN_SHIFT_ACTIVE_PATIENT", config.ACTIVE_PATIENT)
    paths = dm.get_patient_paths(patient_id)
    
    # Ensure output directories exist
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    logger.setup_file_logging(paths["out_dir"])
    logger.section(f"Pipeline: {patient_id}")

    # 2. Execution Flow
    if args.register_tissue or args.all:
        from src.register_tissue import TissueRegistrationPipeline
        TissueRegistrationPipeline(config, paths).run()

    if args.analyze or args.all:
        from src.brain_shift_analysis import BrainShiftAnalyzer
        
        # Determine best available inputs for analysis
        pre_op_path = paths["pre_op"]
        post_op_path = paths["post_rigid"] if paths["post_rigid"].exists() else paths["post_op"]
        
        # Check for transform (bspline preferred over rigid)
        transform_path = paths["transform"] if paths["transform"].exists() else paths["transform_rigid"]
        
        if not pre_op_path or not pre_op_path.exists():
            logger.error(f"Pre-Op file missing for {patient_id}. Run registration first?")
        elif not transform_path or not transform_path.exists():
            logger.error(f"No registration transform found for {patient_id}.")
        else:
            fixed = sitk.ReadImage(str(pre_op_path), sitk.sitkFloat32)
            moving_affine = sitk.ReadImage(str(post_op_path), sitk.sitkFloat32)
            transform = sitk.ReadTransform(str(transform_path))
            
            analyzer = BrainShiftAnalyzer(config, paths)
            brain_mask, cavity_mask = analyzer.run_segmentation(fixed, moving_affine)
            analyzer.quantify(fixed, brain_mask, cavity_mask, transform)

    # 3. Viewers
    if args.view:
        from src.ui.viewer import InteractiveBrainShiftViewer
        InteractiveBrainShiftViewer(data_dir=paths["out_dir"]).run()

    # 4. Validations & Export
    if args.validate_preprocessing:
        from src.qc.validate_preprocessing import validate_preprocessing
        validate_preprocessing(config, paths)
    
    if args.validate_rigid:
        from src.qc.validate_rigid import validate_rigid
        validate_rigid(config, paths)

    if args.validate_deformable:
        from src.qc.validate_deformable import validate_deformable
        validate_deformable(config, paths)

    if args.validate_segmentation:
        from src.qc.validate_segmentation import validate_segmentation
        validate_segmentation(config, paths)

    if args.validate_robustness:
        from src.qc.validate_robustness import validate_robustness
        validate_robustness(config, paths)

    if args.capture_range:
        from src.qc.analyze_capture_range import analyze_capture_range
        analyze_capture_range(config, paths)

    if args.thesis_figures:
        from src.qc.report_thesis import generate_thesis_report
        generate_thesis_report(config, paths)

    if args.export:
        import subprocess
        subprocess.run([sys.executable, "export_metrics.py"], check=True)

    logger.success("Main pipeline execution finished.")

if __name__ == "__main__":
    main()
