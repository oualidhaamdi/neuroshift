"""
Multi-Brain Synthetic Sweep Orchestrator.

Picks brains from RunBatch_Input/ (or RunBatch_Completed/) and runs the
10-case multibrain sweep on each, aggregating into one combined JSON for
cross-brain analysis.

The single-brain machinery in run_synthetic_sweep.py is reused via subprocess
with --source-brain / --out-json / --cases-set multibrain — no code duplicated.

Test matrix per brain (MULTIBRAIN_CASES, 10 base cases × {PRE, POST}):
    4 directions      @ A=5mm  σ=25mm  center
    3 amplitudes      @ X      σ=25mm  center  (A = 3, 8, 15)
    2 sigmas          @ X      A=5mm   center  (σ = 15, 40)
    1 anterior offset @ X      A=5mm   σ=25mm

Usage
-----
python -m src.qc.multi_brain_sweep                     # auto-discover all brains
python -m src.qc.multi_brain_sweep --brains BrainA,BrainB
python -m src.qc.multi_brain_sweep --pre               # PRE only (default)
python -m src.qc.multi_brain_sweep --pre --post        # both passes
python -m src.qc.multi_brain_sweep --dry-run           # preview, no execution
python -m src.qc.multi_brain_sweep --resume            # skip cases already OK
python -m src.qc.multi_brain_sweep --no-cleanup        # keep intermediate files
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import load_config, setup_logger
from src.qc.run_synthetic_sweep import MULTIBRAIN_CASES
from src.qc import sanitize_for_json, restore_nan_from_json

logger = setup_logger("multi_brain")


SYNTHETIC_PREFIXES = ("Sweep_", "Pilot_", "MB_", "TEST_", "CavityDemo_")


def _is_synthetic(name: str) -> bool:
    """Skip leftover synthetic patient folders from previous sweeps."""
    upper = name.upper()
    return any(upper.startswith(p.upper()) for p in SYNTHETIC_PREFIXES)


def discover_brains(config) -> list[str]:
    """Find real brain IDs with both pre-op (0/) and post-op (1/) NIfTI files."""
    seen: set[str] = set()
    found: list[str] = []
    for base in [config.PROJECT_ROOT / "RunBatch_Input",
                 config.PROJECT_ROOT / "RunBatch_Completed"]:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or d.name in seen:
                continue
            if _is_synthetic(d.name):
                continue
            pre  = d / "0"
            post = d / "1"
            has_pre  = pre.exists()  and any(pre.glob("*.nii*"))
            has_post = post.exists() and any(post.glob("*.nii*"))
            if has_pre and has_post:
                seen.add(d.name)
                found.append(d.name)
    return found


def run_single_brain(brain_id: str, args, project_root: Path) -> Path:
    """Invoke run_synthetic_sweep.py for a single brain. Returns the output JSON path."""
    # Suffix: which results variant this is (tissue-guided or CFM-off ablation).
    if args.tissue_guided:
        suffix = "_tissue"
    elif args.no_cfm:
        suffix = "_nocfm"
    else:
        suffix = ""
    out_json = project_root / "RunBatch_Output" / "multi_brain" / f"sweep_{brain_id}{suffix}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.qc.run_synthetic_sweep",
        "--source-brain", brain_id,
        "--cases-set",    "multibrain",
        "--out-json",     str(out_json),
    ]
    if args.pre:           cmd.append("--pre")
    if args.post:          cmd.append("--post")
    if args.resume:        cmd.append("--resume")
    if args.no_cleanup:    cmd.append("--no-cleanup")
    if args.tissue_guided: cmd.append("--tissue-guided")
    if args.no_cfm:        cmd.append("--no-cfm")
    if args.cache_source:  cmd.append("--cache-source")
    # Always namespace output dirs by brain ID so parallel sweeps don't collide
    # on the same RunBatch_Output/MB_* paths. JSON keys remain unprefixed.
    cmd.extend(["--pid-prefix", f"{brain_id}__"])

    logger.section(f"[{brain_id}] Starting sweep ({len(MULTIBRAIN_CASES)} cases)")
    logger.info(f"  → {out_json}")
    t0 = time.monotonic()

    result = subprocess.run(cmd, cwd=project_root)
    elapsed = (time.monotonic() - t0) / 60.0

    if result.returncode != 0:
        logger.error(f"[{brain_id}] sweep exited with code {result.returncode}  "
                     f"after {elapsed:.1f} min")
    else:
        logger.success(f"[{brain_id}] sweep finished in {elapsed:.1f} min")

    return out_json


def merge_results(brain_jsons: list[tuple[str, Path]], combined_path: Path) -> dict:
    """Merge per-brain JSONs into a single combined JSON. Re-keys with brain prefix."""
    combined: dict = {}
    for brain_id, jpath in brain_jsons:
        if not jpath.exists():
            logger.warning(f"  Missing JSON for {brain_id}: {jpath}")
            continue
        with open(jpath) as f:
            data = restore_nan_from_json(json.load(f))
        for case_name, case_data in data.items():
            new_key = f"{brain_id}__{case_name}"
            if isinstance(case_data, dict):
                case_data = dict(case_data)
                case_data["brain_id"] = brain_id
            combined[new_key] = case_data

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(sanitize_for_json(combined), f, indent=2, allow_nan=False)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Multi-brain synthetic sweep orchestrator"
    )
    parser.add_argument("--brains", type=str, default="all",
                        help="Comma-separated brain IDs, or 'all' to auto-discover")
    parser.add_argument("--pre",  action="store_true", default=False,
                        help="Run PRE pass (synthetic on clean pre-op)")
    parser.add_argument("--post", action="store_true", default=False,
                        help="Run POST pass (synthetic on real post-op with cavity)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Preview brain × case list without running")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip cases already marked OK")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep intermediate files per brain")
    parser.add_argument("--tissue-guided", action="store_true",
                        help="Run the tissue-guided B-Spline registrar (distance maps) as ABLATION. "
                             "Default is intensity-based. Combined results saved to "
                             "multi_brain_results_tissue.json.")
    parser.add_argument("--no-cfm", action="store_true",
                        help="Disable Cost Function Masking globally. Used for CFM-ablation studies.")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Run N brain sweeps concurrently via ThreadPoolExecutor. "
                             "Default 1 (sequential). Each brain ≈ 2-3 GB RAM at peak; "
                             "with 16 GB total, N=3 is the practical max.")
    parser.add_argument("--cache-source", action="store_true",
                        help="Pre-compute BET+FAST on each source brain (RAS) ONCE and reuse "
                             "for all of its synthetic cases. ~30-40% speedup, opt-in.")
    args = parser.parse_args()
    if args.parallel < 1:
        args.parallel = 1

    # Default to PRE-only (per multi-brain validation plan)
    if not args.pre and not args.post:
        args.pre = True

    config = load_config()
    project_root = Path(config.PROJECT_ROOT)

    if args.brains == "all":
        brains = discover_brains(config)
    else:
        brains = [b.strip() for b in args.brains.split(",") if b.strip()]

    if not brains:
        logger.error("No brains available. Place pre/post-op pairs in "
                     "RunBatch_Input/<ID>/{0,1}/ (or RunBatch_Completed/).")
        return

    pass_label = ("PRE+POST" if (args.pre and args.post)
                  else "PRE" if args.pre else "POST")
    n_cases_per_brain = len(MULTIBRAIN_CASES) * (
        2 if (args.pre and args.post) else 1
    )
    total = len(brains) * n_cases_per_brain

    logger.section(
        f"Multi-brain sweep  —  brains={len(brains)}  cases/brain={n_cases_per_brain}  "
        f"total={total}  [{pass_label}]"
    )
    for b in brains:
        logger.info(f"  · {b}")

    if args.dry_run:
        logger.success(f"Dry-run: {total} total runs would execute.")
        return

    t_total = time.monotonic()
    brain_jsons: list[tuple[str, Path]] = []

    if args.parallel > 1:
        logger.info(f"Parallel mode: {args.parallel} concurrent brain sweeps")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            future_to_brain = {
                pool.submit(run_single_brain, b, args, project_root): b
                for b in brains
            }
            for fut in concurrent.futures.as_completed(future_to_brain):
                b = future_to_brain[fut]
                try:
                    out_json = fut.result()
                    brain_jsons.append((b, out_json))
                    logger.success(f"[{b}] completed (parallel)")
                except Exception as e:
                    logger.error(f"[{b}] failed in parallel pool: {e}")
    else:
        for i, brain_id in enumerate(brains):
            logger.section(f"[BRAIN {i+1}/{len(brains)}]")
            out_json = run_single_brain(brain_id, args, project_root)
            brain_jsons.append((brain_id, out_json))

    if args.tissue_guided:
        suffix = "_tissue"
    elif args.no_cfm:
        suffix = "_nocfm"
    else:
        suffix = ""
    combined_path = project_root / "RunBatch_Output" / f"multi_brain_results{suffix}.json"
    combined = merge_results(brain_jsons, combined_path)

    n_ok = sum(1 for v in combined.values()
               if isinstance(v, dict) and v.get("status") == "OK")
    elapsed = (time.monotonic() - t_total) / 60.0
    logger.success(
        f"Multi-brain sweep complete  —  {n_ok}/{len(combined)} OK  "
        f"in {elapsed:.1f} min total"
    )
    logger.success(f"  → {combined_path}")


if __name__ == "__main__":
    main()
