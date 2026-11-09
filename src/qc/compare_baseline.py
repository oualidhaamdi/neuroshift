"""
Baseline Comparison Analysis.

Loads paired results from the intensity sweep (default) and the tissue-guided
sweep (ablation), then produces:
  - paired Wilcoxon p-values per metric (stdout)
  - side-by-side box plots with paired-case connecting lines
  - per-direction breakdown for the EPE metric

Designed to work with as few as 1 paired case (sanity test) and to scale up
to the full multi-brain × 10 cases run. Works on both single-brain JSONs
(sweep_results.json) and merged multi-brain JSONs (multi_brain_results.json).

Usage
-----
python -m src.qc.compare_baseline                              # auto-detect both JSONs
python -m src.qc.compare_baseline --tissue PATH --baseline PATH
python -m src.qc.compare_baseline --out-dir PATH               # custom output dir
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import load_config, setup_logger

logger = setup_logger("compare_baseline")

DIR_COLORS = {"X": "#3b82f6", "Y": "#10b981", "Z": "#f59e0b", "XYZ": "#ef4444"}

# (json_key, display_label, lower_is_better)
METRICS = [
    ("epe_mean",          "EPE mean (mm)",                  True),
    ("epe_p95",           "EPE P95 (mm)",                   True),
    ("epe_mean_central",  "EPE central (mm, ‖GT‖>2mm)",     True),
    ("ang_mean",          "Angular mean (°, raw)",          True),
    ("ang_mean_central",  "Angular central (°, ‖GT‖>2mm)",  True),
    ("ang_weighted_mean", "Angular weighted (°·mm/mm)",     True),
    ("dot_mean",          "Dot ratio (recovery)",           False),
    ("jac_pct_fold",      "Jacobian folding (%)",           True),
]


def _load(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    from src.qc import restore_nan_from_json
    with open(path) as f:
        return restore_nan_from_json(json.load(f))


def _auto_detect(project_root: Path, ablation: bool) -> Path | None:
    """Find newest JSON among the expected candidate locations.

    ablation=False → default pipeline results (intensity-based, the current default)
    ablation=True  → tissue-guided ablation results
    """
    if ablation:
        candidates = [
            project_root / "RunBatch_Output" / "multi_brain_results_tissue.json",
            project_root / "RunBatch_Output" / "sweep_results_tissue.json",
            # legacy / backward-compat: in the old naming "tissue" was the default
            project_root / "RunBatch_Output" / "multi_brain_results_legacy_tissue.json",
        ]
    else:
        candidates = [
            project_root / "RunBatch_Output" / "multi_brain_results.json",
            project_root / "RunBatch_Output" / "sweep_results.json",
            # legacy: when intensity was opt-in and stored separately
            project_root / "RunBatch_Output" / "multi_brain_results_baseline.json",
            project_root / "RunBatch_Output" / "sweep_results_baseline.json",
        ]
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return None
    # Newest by mtime
    return max(existing, key=lambda p: p.stat().st_mtime)


def find_pairs(tissue: dict, baseline: dict) -> list[tuple[str, dict, dict]]:
    """For each case OK in both JSONs, return (key, tissue_rec, baseline_rec).

    Handles both naming schemes:
      single-brain → key is the case name (e.g. MB_A050_S250_X_C)
      multi-brain  → key is "<brain>__<case>"  (tries fallback to just case)
    """
    pairs = []
    for name, t in tissue.items():
        if not isinstance(t, dict) or t.get("status") != "OK":
            continue
        b = baseline.get(name)
        if b is None and "__" in name:
            b = baseline.get(name.split("__", 1)[1])
        if b is None:
            # Reverse: maybe baseline has multi-brain keys, tissue has bare names
            for bk, bv in baseline.items():
                if isinstance(bv, dict) and bv.get("status") == "OK":
                    if bk == name or bk.endswith(f"__{name}"):
                        b = bv
                        break
        if not isinstance(b, dict) or b.get("status") != "OK":
            continue
        pairs.append((name, t, b))
    return pairs


def _vals(pairs, key, side):
    """Extract metric values for one side ('t' or 'b') of each pair."""
    idx = 1 if side == "t" else 2
    out = np.array([p[idx].get(key, np.nan) for p in pairs], dtype=float)
    return out


def print_summary_table(pairs: list, tissue_path: Path, baseline_path: Path) -> None:
    print(f"\n{'='*92}")
    print(f"Registrar Ablation — {len(pairs)} paired case(s)")
    print(f"  Default  (intensity B-Spline) : {baseline_path.name}")
    print(f"  Ablation (tissue-guided)      : {tissue_path.name}")
    print(f"{'='*92}")
    print(f"  {'Metric':<36s} {'Tissue (med)':>13s} {'Default (med)':>15s} "
          f"{'D (T-D)':>10s} {'p (paired)':>12s}")
    print("  " + "-" * 90)

    for key, label, lower_is_better in METRICS:
        t = _vals(pairs, key, "t"); b = _vals(pairs, key, "b")
        mask = ~(np.isnan(t) | np.isnan(b))
        if not mask.any():
            continue
        t, b = t[mask], b[mask]
        t_med, b_med = float(np.median(t)), float(np.median(b))
        delta = t_med - b_med

        if len(t) >= 5:
            try:
                _, p = wilcoxon(t, b)
                tag = ("***" if p < 0.001 else "**" if p < 0.01
                       else "*" if p < 0.05 else "ns")
                p_str = f"{p:.4f}{tag}"
            except Exception:
                p_str = "—"
        else:
            p_str = f"N={len(t)} (skip)"

        arrow = ""
        if lower_is_better:
            if delta < 0: arrow = "  (ablation better)"
            elif delta > 0: arrow = "  (default  better)"
        else:
            if delta > 0: arrow = "  (ablation better)"
            elif delta < 0: arrow = "  (default  better)"

        print(f"  {label:<36s} {t_med:>13.4f} {b_med:>15.4f} {delta:>+10.4f} "
              f"{p_str:>12s}{arrow}")
    print()


def plot_comparison(pairs: list, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="white")
    rng = np.random.default_rng(7)

    for i, (key, label, lower_is_better) in enumerate(METRICS):
        ax = axes[i // 4, i % 4]
        t = _vals(pairs, key, "t"); b = _vals(pairs, key, "b")
        mask = ~(np.isnan(t) | np.isnan(b))
        if not mask.any():
            ax.axis("off"); continue
        t, b = t[mask], b[mask]

        bp = ax.boxplot(
            [t, b], tick_labels=["Tissue\n(ablation)", "Intensity\n(default)"],
            widths=0.55, patch_artist=True,
            medianprops=dict(color="#1e293b", lw=2.0),
            whiskerprops=dict(color="#94a3b8", lw=1.0),
            capprops=dict(color="#94a3b8", lw=1.0),
            flierprops=dict(marker="", alpha=0),
        )
        bp["boxes"][0].set_facecolor("#3b82f6"); bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][1].set_facecolor("#ea580c"); bp["boxes"][1].set_alpha(0.55)

        # Paired connecting lines + points
        x_t = np.full(len(t), 1) + rng.uniform(-0.12, 0.12, len(t))
        x_b = np.full(len(b), 2) + rng.uniform(-0.12, 0.12, len(b))
        for xt_, xb_, yt_, yb_ in zip(x_t, x_b, t, b):
            ax.plot([xt_, xb_], [yt_, yb_], color="#94a3b8",
                    lw=0.5, alpha=0.45, zorder=1)
        ax.scatter(x_t, t, color="#3b82f6", s=28, alpha=0.85,
                   zorder=3, edgecolors="white", lw=0.4)
        ax.scatter(x_b, b, color="#ea580c", s=28, alpha=0.85,
                   zorder=3, edgecolors="white", lw=0.4)

        if len(t) >= 5:
            try:
                _, p = wilcoxon(t, b)
                sig = ("***" if p < 0.001 else "**" if p < 0.01
                       else "*" if p < 0.05 else "ns")
                ax.text(0.5, 0.96, f"Wilcoxon p={p:.3f} ({sig})",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=8, color="#475569",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#cbd5e1", alpha=0.85))
            except Exception:
                pass
        else:
            ax.text(0.5, 0.96, f"N={len(t)} (no stat test)",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="#94a3b8")

        arrow = "↓ better" if lower_is_better else "↑ better"
        ax.set_title(f"{label}\n({arrow})", fontsize=9.5, pad=6)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Registrar ablation: Intensity (default) vs Tissue-Guided  —  N={len(pairs)} paired cases\n"
        "Blue = tissue-guided (ablation, distance maps)  -  Orange = intensity (default)  -  "
        "Gray lines = matched cases",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved: {out_path}")


def compare_baselines(tissue_path: Path, baseline_path: Path,
                      out_dir: Path) -> int:
    """Programmatic entry point. Returns the number of paired cases found."""
    tissue   = _load(tissue_path)
    baseline = _load(baseline_path)
    if not tissue:
        logger.error(f"Tissue JSON empty or missing: {tissue_path}")
        return 0
    if not baseline:
        logger.error(f"Baseline JSON empty or missing: {baseline_path}")
        return 0

    pairs = find_pairs(tissue, baseline)
    if not pairs:
        logger.error("No paired OK cases between the two JSONs. "
                     "Make sure case names match.")
        # Show what each side has for debugging
        t_keys = sorted(k for k, v in tissue.items()
                        if isinstance(v, dict) and v.get("status") == "OK")
        b_keys = sorted(k for k, v in baseline.items()
                        if isinstance(v, dict) and v.get("status") == "OK")
        logger.info(f"  Tissue OK keys  ({len(t_keys)}): {t_keys[:5]}{'...' if len(t_keys) > 5 else ''}")
        logger.info(f"  Baseline OK keys({len(b_keys)}): {b_keys[:5]}{'...' if len(b_keys) > 5 else ''}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    print_summary_table(pairs, tissue_path, baseline_path)
    plot_comparison(pairs, out_dir / "baseline_comparison.png")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Compare tissue-guided vs intensity-baseline B-Spline runs"
    )
    parser.add_argument("--tissue",   type=str, default=None,
                        help="Path to tissue-guided JSON (default: auto-detect)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline JSON (default: auto-detect)")
    parser.add_argument("--out-dir",  type=str, default=None,
                        help="Output directory for figures "
                             "(default: RunBatch_Output/sweep_figures)")
    args = parser.parse_args()

    config = load_config()
    project_root = Path(config.PROJECT_ROOT)

    # New semantics: "tissue" CLI arg → ablation JSON, "baseline" CLI arg → default JSON.
    # Backward-compat names retained (paths can still be passed via the original flags).
    tissue_path   = Path(args.tissue)   if args.tissue   else _auto_detect(project_root, ablation=True)
    baseline_path = Path(args.baseline) if args.baseline else _auto_detect(project_root, ablation=False)

    if not tissue_path or not tissue_path.exists():
        logger.error("Tissue JSON not found. Use --tissue PATH or run the sweep first.")
        return
    if not baseline_path or not baseline_path.exists():
        logger.error("Baseline JSON not found. Run with --baseline first, "
                     "or pass --baseline PATH explicitly.")
        return

    out_dir = (Path(args.out_dir) if args.out_dir
               else project_root / "RunBatch_Output" / "sweep_figures")

    logger.info(f"Tissue   JSON: {tissue_path}")
    logger.info(f"Baseline JSON: {baseline_path}")
    n = compare_baselines(tissue_path, baseline_path, out_dir)
    if n > 0:
        logger.success(f"Comparison done: {n} paired cases.")


if __name__ == "__main__":
    main()
