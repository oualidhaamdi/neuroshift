"""
Batch Runner for processing multiple patients sequentially.
Uses DataManager for robust path resolution.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.core import setup_logger, load_config
from src.data_manager import DataManager

logger = setup_logger("batch_runner")

def main():
    parser = argparse.ArgumentParser(description="Batch Runner for sequential patient processing.")
    parser.add_argument("--input-dir", type=str, default="RunBatch_Input", help="Input directory with patient subfolders.")
    parser.add_argument("--output-dir", type=str, default="RunBatch_Output", help="Output directory for results.")
    parser.add_argument("--completed-dir", type=str, default="RunBatch_Completed", help="Directory to move successfully processed patients.")
    parser.add_argument("--args", type=str, default="--register-tissue --analyze --thesis-figures", 
                        help="Arguments for main.py (default: registration + analysis + all figures).")
    args = parser.parse_args()

    try:
        config = load_config()
    except Exception as e:
        print(f"Config error: {e}")
        sys.exit(1)

    dm = DataManager(config)
    input_base = Path(args.input_dir).resolve()
    output_base = Path(args.output_dir).resolve()
    completed_base = Path(args.completed_dir).resolve()

    if not input_base.exists():
        logger.error(f"Input directory {input_base} does not exist.")
        return

    completed_base.mkdir(parents=True, exist_ok=True)
    output_base.mkdir(parents=True, exist_ok=True)

    patients = [d for d in input_base.iterdir() if d.is_dir()]
    if not patients:
        logger.info(f"No patient folders found in {input_base}.")
        return

    logger.section(f"BATCH RUNNER: {len(patients)} patients")

    success_count = 0
    fail_count = 0

    for patient_dir in patients:
        name = patient_dir.name
        logger.info(f"---> Processing: {name}")
        
        paths = dm.get_patient_paths(name)
        if not paths["pre_op"] or not paths["post_op"]:
            logger.error(f"  Could not find input NIfTIs for {name}. Skipping.")
            fail_count += 1
            continue
            
        env = os.environ.copy()
        env["BRAIN_SHIFT_ACTIVE_PATIENT"] = name
        
        cmd = [sys.executable, "main.py"] + args.args.split()
        
        try:
            subprocess.run(cmd, env=env, check=True)
            logger.success(f"Successfully processed {name}.")
            
            # Move to completed
            dest = completed_base / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(patient_dir), str(dest))
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline failed for {name} (code {e.returncode}).")
            fail_count += 1
            
    logger.section("BATCH RUNNER FINISHED")
    logger.info(f"Success: {success_count}/{len(patients)}")
    logger.info(f"Failed:  {fail_count}/{len(patients)}")

if __name__ == "__main__":
    main()
