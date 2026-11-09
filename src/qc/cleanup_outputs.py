import shutil
import argparse
from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import load_config, setup_logger
from src.data_manager import DataManager

logger = setup_logger("cleanup_utility")

PATTERNS_TO_DELETE = [
    "*_tmp.nii.gz",            # Internal temp images
    "edge_*.nii.gz",           # Edge maps (pre/post)
    "fast_*.nii.gz",           # FSL FAST intermediates
    "flirt_rigid.mat",         # FSL format matrix (we have .tfm)
    "*_bet.nii.gz",            # Intermediate brain captures
    "*_bet_mask.nii.gz",       # Intermediate masks
    "*_raw_bet*.nii.gz",       # Raw intermediates
    "*_skull_weight.nii.gz",   # Skeletonization weights
    "seg_post_original.nii.gz" # Unregistered segmentation
]

def cleanup_patient(out_dir: Path, dry_run: bool = False):
    """Cleans up a single patient's output directory."""
    if not out_dir.exists():
        return

    label = "[DRY-RUN] " if dry_run else ""
    logger.info(f"{label}Processing: {out_dir.parent.name}")

    # 1. Directory cleanup: FSL tmp folder
    tmp_fsl = out_dir / "tmp_fsl"
    if tmp_fsl.exists() and tmp_fsl.is_dir():
        if dry_run:
            logger.info(f"  [DRY] Would remove directory: {tmp_fsl.name}")
        else:
            try:
                shutil.rmtree(tmp_fsl)
                logger.info(f"  [RM DIR] {tmp_fsl.name}")
            except Exception as e:
                logger.warning(f"  [ERR] Could not remove {tmp_fsl.name}: {e}")

    # 2. Files cleanup
    files_found = []
    for pattern in PATTERNS_TO_DELETE:
        files_found.extend(out_dir.glob(pattern))

    if dry_run:
        for f in files_found:
            logger.info(f"  [DRY] Would delete: {f.name}")
    else:
        deleted = 0
        for f in files_found:
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass
        if deleted > 0:
            logger.info(f"  [DEL] {deleted} intermediate files removed.")

def main():
    parser = argparse.ArgumentParser(description="Cleanup intermediate pipeline output files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview files to be deleted without actually deleting them.")
    args = parser.parse_args()

    config = load_config()

    logger.section("Mass Storage Cleanup" + (" [DRY-RUN MODE]" if args.dry_run else ""))

    batch_dir = config.PROJECT_ROOT / "RunBatch_Output"
    if not batch_dir.exists():
        logger.error(f"Directory {batch_dir} not found.")
        return

    reg_dirs = list(batch_dir.glob("**/tissue_registration"))

    if not reg_dirs:
        logger.warning("No tissue_registration directories found to clean.")
        return

    logger.info(f"Found {len(reg_dirs)} registration folders to process.")

    for reg_dir in reg_dirs:
        cleanup_patient(reg_dir, dry_run=args.dry_run)

    if args.dry_run:
        logger.success("\nDry-run complete. No files were deleted.")
    else:
        logger.success("\nMass cleanup completed successfully.")

if __name__ == "__main__":
    main()
