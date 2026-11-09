"""
Sweep Results Analysis & Visualization.
Enhanced High-Quality Version for Thesis/Publication.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from scipy.stats import wilcoxon as scipy_wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import load_config, setup_logger
from src.qc.run_synthetic_sweep import GT_SIG_THRESHOLD_MM

logger = setup_logger("sweep_analysis")

# ── Palette & constants ────────────────────────────────────────────────────────
AMP_VALUES   = [3.0,  5.0,  8.0,  15.0]
SIGMA_VALUES = [15.0, 25.0, 40.0]
DIRECTIONS   = ["X", "Y", "Z", "XYZ"]
DIR_COLORS   = {"X": "#3b82f6", "Y": "#10b981", "Z": "#f59e0b", "XYZ": "#ef4444"}
FACE_COLOR   = "#f8fafc"
TEXT_COLOR   = "#1e293b"
GRID_COLOR   = "#cbd5e1"

# Acceptance thresholds
EPE_OK   = 1.5    # mm  — good
EPE_WARN = 3.0    # mm  — acceptable upper limit
ANG_OK   = 10.0   # °   — excellent direction accuracy
ANG_WARN = 30.0   # °   — acceptable limit
DOT_WARN = 0.50   # ratio — minimum acceptable recovery
DOT_OK   = 0.80   # ratio — good recovery

def setup_plot_style():
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.size':         10,
        'axes.labelcolor':   TEXT_COLOR,
        'axes.edgecolor':    GRID_COLOR,
        'axes.titleweight':  'bold',
        'axes.titlesize':    11,
        'axes.titlecolor':   TEXT_COLOR,
        'figure.facecolor':  FACE_COLOR,
        'axes.facecolor':    FACE_COLOR,
        'grid.alpha':        0.35,
        'grid.linestyle':    '--',
        'grid.color':        GRID_COLOR,
        'xtick.color':       TEXT_COLOR,
        'ytick.color':       TEXT_COLOR,
        'legend.framealpha': 0.9,
        'legend.edgecolor':  GRID_COLOR,
    })

setup_plot_style()

# ── Utilities ──────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    from src.qc import restore_nan_from_json
    with open(path) as f:
        return restore_nan_from_json(json.load(f))

def _case_name(a, s, dir_tag, loc="center"):
    loc_tag = loc[:3].upper() if loc != "center" else "C"
    return f"Sweep_A{int(a*10):03d}_S{int(s*10):03d}_{dir_tag}_{loc_tag}"

def _pilot_name(a, s, dir_tag, loc="center"):
    loc_tag = loc[:3].upper() if loc != "center" else "C"
    return f"Pilot_A{int(a*10):03d}_S{int(s*10):03d}_{dir_tag}_{loc_tag}"

def _get(results, name, key, default=np.nan):
    r = results.get(name)
    if r and r.get("status") == "OK":
        return r.get(key, default)
    return default

def _get_any(results, a, s, dir_tag, metric_key, loc="center"):
    val = _get(results, _case_name(a, s, dir_tag, loc), metric_key)
    if np.isnan(val):
        val = _get(results, _pilot_name(a, s, dir_tag, loc), metric_key)
    return val

def _get_any_post(results, a, s, dir_tag, metric_key, loc="center"):
    """Like _get_any but for POST variants (name + '_POST')."""
    for name_fn in [_case_name, _pilot_name]:
        val = _get(results, name_fn(a, s, dir_tag, loc) + "_POST", metric_key)
        if not np.isnan(val):
            return val
    return np.nan

def build_matrix(results, dir_tag, metric_key):
    mat = np.full((len(AMP_VALUES), len(SIGMA_VALUES)), np.nan)
    for i, a in enumerate(AMP_VALUES):
        for j, s in enumerate(SIGMA_VALUES):
            mat[i, j] = _get_any(results, a, s, dir_tag, metric_key)
    return mat

def _ok_cases(results, source_type: str = "pre") -> list:
    """Return OK cases of the given source_type (default 'pre' for backward compat)."""
    return [v for v in results.values()
            if isinstance(v, dict) and v.get("status") == "OK" and "epe_mean" in v
            and v.get("source_type", "pre") == source_type]


def _pair_post_cases(results: dict) -> list[tuple[dict, dict]]:
    """Find (pre_result, post_result) pairs by matching base name to <name>_POST."""
    pairs = []
    for name, r in results.items():
        if not isinstance(r, dict): continue
        if r.get("source_type", "pre") != "post": continue
        if r.get("status") != "OK": continue
        if not name.endswith("_POST"): continue
        pre_name = name[:-5]
        pre_r = results.get(pre_name)
        if (isinstance(pre_r, dict) and pre_r.get("status") == "OK"
                and "epe_mean" in pre_r and "epe_mean" in r):
            pairs.append((pre_r, r))
    return sorted(pairs, key=lambda t: t[0].get("name", ""))

def _verdict(dot_val, epe_p95=np.nan):
    if dot_val > 0.95 and (np.isnan(epe_p95) or epe_p95 < 1.0):
        return "EXCELLENT"
    if dot_val > DOT_OK:  return "GOOD"
    if dot_val > 0.15:    return "ACCEPTABLE"
    return "POOR"

VERDICT_COLOR = {
    "EXCELLENT": "#16a34a", "GOOD": "#3b82f6",
    "ACCEPTABLE": "#f59e0b", "POOR": "#dc2626",
}

_METRIC_GLOSSARY = (
    "EPE (End-Point Error): residual displacement magnitude after registration [mm] — 0 mm = perfect recovery\n"
    "Angular Error [°]: angle between predicted and ground-truth displacement vectors\n"
    "Dot Ratio: recovery factor = (pred·GT)/(‖pred‖·‖GT‖), range −1→+1  —  1.0 = perfect,  0 = no correction,  <0 = worsened\n"
    "Jacobian Folding [%]: fraction of voxels with det(J) < 0  —  physically implausible deformation, should be near 0%"
)

def _add_metric_glossary(fig, y: float = -0.12) -> None:
    """Attach a compact metric-definition box at the bottom of a figure."""
    fig.text(
        0.5, y, _METRIC_GLOSSARY,
        ha="center", fontsize=7.5, color="#475569", style="italic",
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f8fafc",
                  edgecolor="#cbd5e1", alpha=0.95),
    )

def _annotate_heatmap(ax, matrix, fmt, vmin, vmax):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=6.5, color="#94a3b8")
                continue
            norm = (val - vmin) / (vmax - vmin + 1e-9)
            color = "white" if (norm < 0.2 or norm > 0.75) else TEXT_COLOR
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)

# ── 1. Per-test report ─────────────────────────────────────────────────────────

def generate_per_test_report(name: str, r: dict, output_dir: Path,
                             all_results: dict = None) -> None:
    fig = plt.figure(figsize=(13, 8), facecolor=FACE_COLOR)
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.2], hspace=0.58, wspace=0.36)
    dot_val = r.get("dot_mean", 0) or 0
    epe_p95 = r.get("epe_p95",  0) or 0
    ang_mean = r.get("ang_mean", 0) or 0
    pct_gt90 = r.get("ang_pct_gt90", np.nan)
    verdict  = _verdict(dot_val, epe_p95)
    vc       = VERDICT_COLOR[verdict]

    # Subtitle line: params + PRE/POST tag
    loc_str  = r.get("location", "center")
    src_type = r.get("source_type", "pre").upper()
    src_color = "#ea580c" if src_type == "POST" else "#3b82f6"
    sub_line = (f"A={r.get('A_mm','?')}mm  ·  σ={r.get('sigma_mm','?')}mm  ·  "
                f"dir={r.get('direction','?')}  ·  loc={loc_str}  ·  [{src_type}]")
    plt.suptitle(f"VALIDATION REPORT  ·  {name}", fontsize=15,
                 fontweight="bold", color=TEXT_COLOR, y=0.98)
    fig.text(0.5, 0.945, sub_line, ha="center", fontsize=9, color=src_color)

    from matplotlib.patches import Rectangle as Rect
    _gy0, _gh = 4.2, 1.6   # shared gauge geometry (bar bottom y, height)

    # ── A: Magnitude ──────────────────────────────────────────────────────────
    ax_mag = fig.add_subplot(gs[0, 0])
    epe_mean_v   = r.get("epe_mean",   0) or 0
    epe_median_v = r.get("epe_median", 0) or 0
    gt_peak_v    = r.get("gt_peak_mm", 0) or 0
    labels = ["EPE Median", "EPE Mean", "EPE P95"]
    vals   = [epe_median_v, epe_mean_v, epe_p95]
    y_pos  = np.arange(len(labels))
    mx     = max(list(vals) + [gt_peak_v, 0.1])
    ax_mag.barh(y_pos, vals, color="#3b82f6", alpha=0.75, height=0.52)
    ax_mag.axvline(gt_peak_v, color="#64748b", lw=1.8, alpha=0.50, zorder=5)
    ax_mag.text(gt_peak_v, len(labels) - 0.08,
                f"GT {gt_peak_v:.1f}mm", fontsize=7.5, color="#64748b",
                ha="center", va="bottom")
    ax_mag.axvline(EPE_OK,   color="#16a34a", lw=0.9, ls=":", alpha=0.55)
    ax_mag.axvline(EPE_WARN, color="#f59e0b", lw=0.9, ls=":", alpha=0.55)
    ax_mag.set_yticks(y_pos)
    ax_mag.set_yticklabels(labels, fontsize=9, color="#475569")
    ax_mag.invert_yaxis()
    for i, v in enumerate(vals):
        inside = v > mx * 0.58
        ax_mag.text(v - mx * 0.02 if inside else v + mx * 0.02, i,
                    f"{v:.2f}", va="center",
                    ha="right" if inside else "left",
                    fontsize=8.5, fontweight="bold",
                    color="white" if inside else "#3b82f6")
    ax_mag.set_xlim(0, mx * 1.22)
    ax_mag.spines[["top", "right", "left"]].set_visible(False)
    ax_mag.tick_params(axis="y", length=0)
    ax_mag.set_xlabel("mm", fontsize=8)
    ax_mag.set_title("Magnitude (mm)", pad=10)

    # ── B: Directional error — binned bar chart by ‖GT‖ magnitude ────────────
    ax_ang = fig.add_subplot(gs[0, 1])
    bin_specs = [
        ("lt1",  "0.5–1\nmm"),
        ("1to2", "1–2\nmm"),
        ("2to5", "2–5\nmm"),
        ("gt5",  ">5\nmm"),
    ]
    bin_angs = [r.get(f"ang_{tag}_mean", np.nan) for tag, _ in bin_specs]
    bin_ns   = [r.get(f"n_{tag}",        0)      for tag, _ in bin_specs]
    bin_pct  = [r.get(f"pct_gt90_{tag}", np.nan) for tag, _ in bin_specs]

    has_binned = any(not np.isnan(v) for v in bin_angs)

    if has_binned:
        ax_ang.set_title("Directional Error by ‖GT‖ magnitude", pad=10)
        xs = np.arange(len(bin_specs))
        bar_colors = []
        for a in bin_angs:
            if np.isnan(a):       bar_colors.append("#e2e8f0")
            elif a < ANG_OK:      bar_colors.append("#16a34a")
            elif a < ANG_WARN:    bar_colors.append("#f59e0b")
            else:                 bar_colors.append("#ef4444")
        heights = [0 if np.isnan(a) else a for a in bin_angs]
        ax_ang.bar(xs, heights, color=bar_colors, alpha=0.78,
                   edgecolor="white", linewidth=1.2, width=0.72)

        # Annotations: angle on top, n + wrong-dir % under bar
        for x, a, n, p in zip(xs, bin_angs, bin_ns, bin_pct):
            if np.isnan(a):
                ax_ang.text(x, 3, "no\nvoxels", ha="center", va="bottom",
                            fontsize=7, color="#94a3b8", style="italic")
                continue
            ax_ang.text(x, a + 2.0, f"{a:.0f}°", ha="center",
                        fontsize=10, fontweight="bold", color=TEXT_COLOR)
            extra = (f"n={n:,}" if n < 1000 else f"n={n/1000:.0f}k")
            if not np.isnan(p) and p >= 0.05:
                extra += f"  ·  >90°: {p:.1f}%"
            ax_ang.text(x, -6, extra, ha="center", va="top",
                        fontsize=6.5, color="#64748b")

        # Acceptance lines
        ax_ang.axhline(ANG_OK,   color="#16a34a", lw=0.8, ls=":", alpha=0.55)
        ax_ang.axhline(ANG_WARN, color="#f59e0b", lw=0.8, ls=":", alpha=0.55)
        ax_ang.text(len(xs) - 0.5, ANG_OK + 1.2, f"OK<{ANG_OK:.0f}°",
                    fontsize=6.5, color="#16a34a", ha="right")
        ax_ang.text(len(xs) - 0.5, ANG_WARN + 1.2, f"Warn<{ANG_WARN:.0f}°",
                    fontsize=6.5, color="#f59e0b", ha="right")

        ax_ang.set_xticks(xs)
        ax_ang.set_xticklabels([lbl for _, lbl in bin_specs], fontsize=7.5)
        ax_ang.set_ylabel("Angular error (°)", fontsize=8)
        ax_ang.set_ylim(0, 95)
        ax_ang.set_yticks([0, 15, 30, 60, 90])
        ax_ang.tick_params(axis="y", labelsize=7)
        ax_ang.spines[["top", "right"]].set_visible(False)
        ax_ang.grid(axis="y", alpha=0.25)

        # Aggregate annotation top-right (mean + weighted)
        ang_central  = r.get("ang_mean_central",  np.nan)
        ang_weighted = r.get("ang_weighted_mean", np.nan)
        agg_lines = []
        if not np.isnan(ang_central):
            agg_lines.append(f"central (‖GT‖>2mm): {ang_central:.1f}°")
        if not np.isnan(ang_weighted):
            agg_lines.append(f"weighted (× ‖GT‖): {ang_weighted:.1f}°")
        if agg_lines:
            ax_ang.text(0.98, 0.97, "\n".join(agg_lines),
                        transform=ax_ang.transAxes, ha="right", va="top",
                        fontsize=7.5, color="#475569",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white",
                                  edgecolor="#cbd5e1", alpha=0.9))
    else:
        # ── Fallback: legacy gauge (old sweep_results.json without binned fields) ──
        ax_ang.set_xlim(-3, 93)
        ax_ang.set_ylim(0, 10)
        ax_ang.axis("off")
        ax_ang.set_title("Directional Error", pad=10)
        ang_col = ("#16a34a" if ang_mean < ANG_OK
                   else "#f59e0b" if ang_mean < ANG_WARN else "#ef4444")
        for z0, z1, zc in [(0, ANG_OK, "#16a34a"),
                            (ANG_OK, ANG_WARN, "#f59e0b"),
                            (ANG_WARN, 90, "#ef4444")]:
            ax_ang.add_patch(Rect((z0, _gy0), z1 - z0, _gh,
                                   color=zc, alpha=0.20, zorder=2))
            ax_ang.add_patch(Rect((z0, _gy0), z1 - z0, _gh,
                                   fill=False, edgecolor="#e2e8f0", lw=0.6, zorder=3))
        for tx, tlbl in [(0, "0°"), (ANG_OK, f"{ANG_OK:.0f}°"),
                         (ANG_WARN, f"{ANG_WARN:.0f}°"), (90, "90°")]:
            ax_ang.plot([tx, tx], [_gy0 - 0.2, _gy0], color="#94a3b8", lw=0.9, zorder=4)
            ax_ang.text(tx, _gy0 - 0.55, tlbl, ha="center", fontsize=7, color="#94a3b8")
        nx_ang = min(90, ang_mean)
        ax_ang.plot([nx_ang, nx_ang], [_gy0 - 0.1, _gy0 + _gh + 0.1],
                    color=TEXT_COLOR, lw=2.5, zorder=6)
        ax_ang.scatter([nx_ang], [_gy0 - 0.18], s=70,
                       color=TEXT_COLOR, marker="v", zorder=7)
        ax_ang.text(45, _gy0 + _gh + 2.0, f"{ang_mean:.1f}°",
                    ha="center", fontsize=18, fontweight="bold", color=ang_col)
        if not np.isnan(pct_gt90):
            pct_col = "#ef4444" if pct_gt90 > 20 else "#f59e0b" if pct_gt90 > 5 else "#16a34a"
            ax_ang.text(45, _gy0 - 3.0, f"Wrong direction: {pct_gt90:.1f}%",
                        ha="center", fontsize=8, color=pct_col)

    # ── C: Recovery gauge ─────────────────────────────────────────────────────
    ax_rec = fig.add_subplot(gs[0, 2])
    ax_rec.set_xlim(-1.18, 1.18)
    ax_rec.set_ylim(0, 10)
    ax_rec.axis("off")
    ax_rec.set_title("Recovery", pad=10)
    for z0, z1, zc in [(-1.0, 0.0, "#ef4444"),
                        (0.0, DOT_WARN, "#f59e0b"),
                        (DOT_WARN, 1.0, "#16a34a")]:
        ax_rec.add_patch(Rect((z0, _gy0), z1 - z0, _gh,
                               color=zc, alpha=0.20, zorder=2))
        ax_rec.add_patch(Rect((z0, _gy0), z1 - z0, _gh,
                               fill=False, edgecolor="#e2e8f0", lw=0.6, zorder=3))
    for tx, tlbl in [(-1, "-1"), (0, "0"), (DOT_WARN, f"+{DOT_WARN}"), (1, "+1")]:
        ax_rec.plot([tx, tx], [_gy0 - 0.2, _gy0], color="#94a3b8", lw=0.9, zorder=4)
        ax_rec.text(tx, _gy0 - 0.55, tlbl, ha="center", fontsize=7, color="#94a3b8")
    for zx, zlbl, zc in [(-0.5, "Worsened", "#ef4444"),
                          (DOT_WARN / 2, "Marginal", "#f59e0b"),
                          ((DOT_WARN + 1) / 2, "Good", "#16a34a")]:
        ax_rec.text(zx, _gy0 - 1.7, zlbl, ha="center", fontsize=7.5, color=zc)
    needle_x = max(-1.0, min(1.0, dot_val))
    ax_rec.plot([needle_x, needle_x], [_gy0 - 0.1, _gy0 + _gh + 0.1],
                color=TEXT_COLOR, lw=2.5, zorder=6)
    ax_rec.scatter([needle_x], [_gy0 - 0.18], s=70,
                   color=TEXT_COLOR, marker="v", zorder=7)
    ax_rec.text(0, _gy0 + _gh + 2.0, f"{dot_val:+.2f}",
                ha="center", fontsize=18, fontweight="bold", color=vc)
    ax_rec.text(0, _gy0 + _gh + 3.5, verdict,
                ha="center", fontsize=9, fontweight="bold", color=vc)
    ax_rec.text(0, _gy0 - 3.0,
                "1.0 = perfect  |  0 = no effect  |  <0 = worsened",
                ha="center", fontsize=7, color="#94a3b8", style="italic")

    # ── D: Component error breakdown ──────────────────────────────────────────
    ax_comp = fig.add_subplot(gs[1, 0])
    comp_cfg = [("X", "err_x_mean", DIR_COLORS["X"]),
                ("Y", "err_y_mean", DIR_COLORS["Y"]),
                ("Z", "err_z_mean", DIR_COLORS["Z"])]
    valid = [(l, r.get(k, np.nan), c) for l, k, c in comp_cfg
             if not np.isnan(r.get(k, np.nan))]
    if valid:
        cl, cv, cc = zip(*valid)
        bars2 = ax_comp.bar(cl, cv, color=cc, alpha=0.82, width=0.5)
        ymax_c = max(cv)
        for bar, v in zip(bars2, cv):
            ax_comp.text(bar.get_x() + bar.get_width() / 2,
                         v + ymax_c * 0.02,
                         f"{v:.2f}", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")
    ax_comp.set_ylabel("Mean error (mm)", fontsize=9)
    ax_comp.set_title("Error by Component", pad=10)
    ax_comp.spines[["top", "right"]].set_visible(False)
    ax_comp.grid(axis="y", alpha=0.3)

    # ── E: Metrics table ──────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 1:])
    ax_tbl.axis("off")
    pct_str = f"{pct_gt90:.1f}%" if not np.isnan(pct_gt90) else "N/A"

    def _st(ok, warn=None, ok_s="✓ OK", warn_s="⚠", fail_s="✗"):
        return ok_s if ok else (warn_s if (warn is not None and warn) else fail_s)

    rows = [
        ["Metric",        "Value",                        "Target",             "Status"],
        ["EPE Mean",      f"{epe_mean_v:.3f} mm",         f"< {EPE_OK} mm",
         _st(epe_mean_v < EPE_OK, epe_mean_v < EPE_WARN, warn_s="⚠ Marginal")],
        ["EPE Median",    f"{epe_median_v:.3f} mm",       "—",                  "—"],
        ["EPE P95",       f"{epe_p95:.2f} mm",            f"< {EPE_WARN} mm",
         _st(epe_p95 < EPE_WARN, fail_s="⚠ High tail")],
        ["Angular error", f"{ang_mean:.1f}°",             f"< {ANG_WARN:.0f}°",
         _st(ang_mean < ANG_OK, ang_mean < ANG_WARN, warn_s="⚠ Divergent", fail_s="✗ Divergent")],
        ["Wrong dir",     pct_str,                        "< 5%",
         _st(not np.isnan(pct_gt90) and pct_gt90 < 5, fail_s="⚠")],
        ["Recovery",      f"{dot_val:+.2f}",              f"> {DOT_WARN:.2f}",  verdict],
    ]
    tbl = ax_tbl.table(cellText=rows, loc="center", cellLoc="left",
                       colWidths=[0.28, 0.24, 0.24, 0.24])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 2.2)
    STATUS_COLOR = {"✓": "#16a34a", "⚠": "#d97706", "✗": "#dc2626",
                    "EXCELLENT": "#16a34a", "GOOD": "#3b82f6",
                    "ACCEPTABLE": "#d97706", "POOR": "#dc2626"}
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row == 0:
            cell.set_facecolor(TEXT_COLOR)
            cell.set_text_props(weight="bold", color="white")
        elif col == 3 and row > 0:
            txt = cell.get_text().get_text()
            for key, col_val in STATUS_COLOR.items():
                if txt.startswith(key):
                    cell.set_text_props(color=col_val, weight="bold")
                    break

    # Verdict banner
    plt.figtext(0.5, 0.03, f"OVERALL VERDICT: {verdict}", ha="center",
                fontsize=13, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.45", facecolor=vc, edgecolor="none"))

    _add_metric_glossary(fig, y=-0.11)
    out_path = output_dir / "per_test" / f"{name}.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# ── 2. Unified heatmap dashboard (3 metrics × 4 directions in one figure) ─────

def build_matrix_post(results: dict, dir_tag: str, metric_key: str) -> np.ndarray:
    """Like build_matrix but looks up POST variants (<name>_POST)."""
    mat = np.full((len(AMP_VALUES), len(SIGMA_VALUES)), np.nan)
    for i, a in enumerate(AMP_VALUES):
        for j, s in enumerate(SIGMA_VALUES):
            for name_fn in (_pilot_name, _case_name):
                key = name_fn(a, s, dir_tag) + "_POST"
                r = results.get(key)
                if r and r.get("status") == "OK":
                    v = r.get(metric_key, np.nan)
                    if not np.isnan(float(v if v is not None else np.nan)):
                        mat[i, j] = v
                        break
    return mat


def plot_heatmap_dashboard(results: dict, output_dir: Path) -> None:
    col_labels = [f"σ={v:.0f}mm" for v in SIGMA_VALUES]
    row_labels  = [f"A={v:.0f}mm" for v in AMP_VALUES]

    metrics_cfg = [
        ("epe_mean", "Mean EPE",          ".2f", "mm", 0, EPE_WARN, "RdYlGn_r"),
        ("ang_mean", "Mean Angular Error", ".1f", "°",  0, 45.0,    "RdYlGn_r"),
        ("dot_mean", "Recovery Factor",   ".2f", "",   0, 1.0,      "RdYlGn"),
    ]

    fig = plt.figure(figsize=(22, 15), facecolor=FACE_COLOR)
    fig.suptitle("Sweep Validation — Complete Metric Dashboard",
                 fontsize=18, fontweight="bold", color=TEXT_COLOR, y=1.0)
    outer_gs = GridSpec(3, 1, figure=fig, hspace=0.5)

    for row_idx, (metric_key, metric_title, fmt, unit, _, vmax_ref, cmap) in enumerate(metrics_cfg):
        inner_gs = GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer_gs[row_idx],
            width_ratios=[1, 1, 1, 1, 0.07], wspace=0.16,
        )
        matrices = [build_matrix(results, d, metric_key) for d in DIRECTIONS]
        all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices]) if any(m[~np.isnan(m)].size > 0 for m in matrices) else np.array([0, vmax_ref])
        vmin = float(np.nanmin(all_vals)) if len(all_vals) else 0
        vmax = max(float(np.nanmax(all_vals)), vmax_ref) if len(all_vals) else vmax_ref

        last_im = None
        for col_idx, (d, mat) in enumerate(zip(DIRECTIONS, matrices)):
            ax = fig.add_subplot(inner_gs[col_idx])
            masked = np.ma.masked_invalid(mat)
            im = ax.imshow(masked, cmap=cmap, aspect="auto",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            last_im = im

            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=8)
            if col_idx == 0:
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=8)
                ax.set_ylabel(
                    f"{metric_title}" + (f" ({unit})" if unit else ""),
                    fontsize=9, labelpad=6,
                )
            else:
                ax.set_yticks([])
            ax.set_xlabel("Gaussian Width σ", fontsize=8)
            ax.set_title(f"Axis: {d}", fontsize=10, fontweight="bold",
                         color=DIR_COLORS[d], pad=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(DIR_COLORS[d])
                spine.set_linewidth(1.8)
            ax.tick_params(which="both", length=0)
            _annotate_heatmap(ax, mat, fmt, vmin, vmax)

        # Shared colorbar
        ax_cb = fig.add_subplot(inner_gs[4])
        cb = plt.colorbar(last_im, cax=ax_cb)
        cb.ax.tick_params(labelsize=8)
        if unit:
            cb.set_label(unit, fontsize=8)

    _add_metric_glossary(fig, y=-0.04)
    plt.savefig(output_dir / "heatmap_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: heatmap_dashboard.png")


# ── 2b. PRE vs POST heatmap comparison ────────────────────────────────────────

def plot_heatmap_prepost(results: dict, output_dir: Path) -> None:
    """
    Figure: heatmap_prepost.png

    Same A×σ grid as heatmap_dashboard, but each direction panel is split into
    two rows — PRE (blue border) on top, POST (orange border) below — sharing
    the same colorscale so you can directly compare the two conditions.
    """
    ok_post = _ok_cases(results, "post")
    if len(ok_post) < 2:
        logger.info("Not enough POST cases — skipping heatmap_prepost.png")
        return

    col_labels = [f"σ={v:.0f}mm" for v in SIGMA_VALUES]
    row_labels  = [f"A={v:.0f}mm" for v in AMP_VALUES]

    # Only EPE — the most interpretable metric for a PRE/POST comparison.
    # Showing all 3 metrics produced 24 tiny sub-heatmaps unreadable at print size.
    metric_key   = "epe_mean"
    metric_title = "EPE Mean (mm)"
    fmt          = ".2f"
    cmap         = "RdYlGn_r"
    vmax_ref     = EPE_WARN

    mats_pre  = [build_matrix(results, d, metric_key)      for d in DIRECTIONS]
    mats_post = [build_matrix_post(results, d, metric_key) for d in DIRECTIONS]

    all_finite = np.concatenate([
        m[~np.isnan(m)] for m in mats_pre + mats_post
    ]) if any(m[~np.isnan(m)].size > 0 for m in mats_pre + mats_post) else np.array([0.0, vmax_ref])
    vmin = float(np.nanmin(all_finite))
    vmax = max(float(np.nanmax(all_finite)), vmax_ref)

    # Layout: 2 rows (PRE / POST) × 5 cols (4 directions + colorbar)
    fig = plt.figure(figsize=(18, 8), facecolor=FACE_COLOR)
    fig.suptitle(
        f"PRE vs POST — {metric_title}",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    gs = GridSpec(2, 5, figure=fig,
                  height_ratios=[1, 1],
                  width_ratios=[1, 1, 1, 1, 0.06],
                  wspace=0.10, hspace=0.18)

    last_im = None
    for col_idx, d in enumerate(DIRECTIONS):
        for src_row, (mat, label, border) in enumerate([
            (mats_pre[col_idx],  "PRE",  "#3b82f6"),
            (mats_post[col_idx], "POST", "#ea580c"),
        ]):
            ax = fig.add_subplot(gs[src_row, col_idx])
            masked = np.ma.masked_invalid(mat)
            im = ax.imshow(masked, cmap=cmap, aspect="auto",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            last_im = im

            if col_idx == 0:
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=8.5)
                ax.set_ylabel(label, fontsize=10, fontweight="bold",
                              color=border, labelpad=5)
            else:
                ax.set_yticks([])

            if src_row == 1:
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, fontsize=8.5)
                ax.set_xlabel("Gaussian Width σ", fontsize=9)
            else:
                ax.set_xticks([])
                ax.set_title(f"Axis: {d}", fontsize=11, fontweight="bold",
                             color=DIR_COLORS[d], pad=7)

            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2.5)
            ax.tick_params(which="both", length=0)
            _annotate_heatmap(ax, mat, fmt, vmin, vmax)

    # Shared colorbar
    ax_cb = fig.add_subplot(gs[:, 4])
    cb = plt.colorbar(last_im, cax=ax_cb)
    cb.ax.tick_params(labelsize=9)
    cb.set_label(metric_title, fontsize=9)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="#dbeafe", edgecolor="#3b82f6", linewidth=2, label="PRE  — blue border"),
            Patch(facecolor="#ffedd5", edgecolor="#ea580c", linewidth=2, label="POST — orange border"),
        ],
        loc="upper right", fontsize=9, bbox_to_anchor=(0.98, 0.98),
    )

    _add_metric_glossary(fig, y=-0.06)
    plt.savefig(output_dir / "heatmap_prepost.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: heatmap_prepost.png")


# ── 3. Recovery curves: EPE (top) + Angular (bottom), 2 rows × 4 dirs ─────────

def plot_recovery_curves(results: dict, output_dir: Path) -> None:
    sigma_styles = [
        ("#3b82f6", "-",  "o"),
        ("#f59e0b", "--", "s"),
        ("#10b981", "-.", "^"),
    ]
    fig = plt.figure(figsize=(22, 10), facecolor=FACE_COLOR)
    gs  = GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.22,
                   height_ratios=[1.4, 1])
    fig.suptitle("Recovery Performance vs Deformation Scale",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02)

    for col_idx, dir_tag in enumerate(DIRECTIONS):
        ax_epe = fig.add_subplot(gs[0, col_idx])
        ax_ang = fig.add_subplot(gs[1, col_idx])
        dc = DIR_COLORS[dir_tag]

        # Count total valid points across all sigmas for this direction
        total_valid_epe = sum(
            1 for a in AMP_VALUES for s in SIGMA_VALUES
            if not np.isnan(_get_any(results, a, s, dir_tag, "epe_mean"))
        )

        for (sc, ss, sm), s in zip(sigma_styles, SIGMA_VALUES):
            ys_mean = [_get_any(results, a, s, dir_tag, "epe_mean") for a in AMP_VALUES]
            ys_p95  = [_get_any(results, a, s, dir_tag, "epe_p95")  for a in AMP_VALUES]
            ys_ang  = [_get_any(results, a, s, dir_tag, "ang_mean")  for a in AMP_VALUES]

            vi = [i for i, y in enumerate(ys_mean) if not np.isnan(y)]
            if not vi:
                continue
            xs = [AMP_VALUES[i] for i in vi]

            if len(vi) >= 2:
                # Enough points — draw line + fill
                ax_epe.plot(xs, [ys_mean[i] for i in vi],
                            marker=sm, color=sc, ls=ss, lw=2.2, markersize=7,
                            label=f"σ={s:.0f}mm")
                fill_top = [ys_p95[i] if not np.isnan(ys_p95[i]) else ys_mean[i] for i in vi]
                ax_epe.fill_between(xs, [ys_mean[i] for i in vi], fill_top,
                                    color=sc, alpha=0.09)
            else:
                # Single point — scatter + value label
                ax_epe.scatter(xs, [ys_mean[i] for i in vi],
                               s=90, color=sc, marker=sm, zorder=5,
                               label=f"σ={s:.0f}mm")
                ax_epe.annotate(f"{ys_mean[vi[0]]:.2f}mm",
                                (xs[0], ys_mean[vi[0]]),
                                textcoords="offset points", xytext=(6, 4),
                                fontsize=7.5, color=sc, fontweight="bold")

            vai = [i for i, y in enumerate(ys_ang) if not np.isnan(y)]
            if vai:
                if len(vai) >= 2:
                    ax_ang.plot([AMP_VALUES[i] for i in vai],
                                [ys_ang[i]  for i in vai],
                                marker=sm, color=sc, ls=ss, lw=1.8, markersize=5)
                else:
                    ax_ang.scatter([AMP_VALUES[vai[0]]], [ys_ang[vai[0]]],
                                   s=60, color=sc, marker=sm, zorder=5)
                    ax_ang.annotate(f"{ys_ang[vai[0]]:.1f}°",
                                    (AMP_VALUES[vai[0]], ys_ang[vai[0]]),
                                    textcoords="offset points", xytext=(6, 3),
                                    fontsize=7, color=sc)

        # Sparse data watermark
        if total_valid_epe < 3:
            for _ax in (ax_epe, ax_ang):
                _ax.text(0.5, 0.52, f"sparse data\n({total_valid_epe} pt{'s' if total_valid_epe != 1 else ''})",
                         ha="center", va="center", transform=_ax.transAxes,
                         fontsize=9, color="#cbd5e1", style="italic",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                   edgecolor="#e2e8f0", alpha=0.7), zorder=0)

        # EPE thresholds & styling
        ax_epe.axhline(EPE_OK,   color="#16a34a", lw=1.3, ls=":",
                       alpha=0.9, label=f"OK < {EPE_OK}mm")
        ax_epe.axhline(EPE_WARN, color="#ef4444", lw=1.3, ls=":",
                       alpha=0.9, label=f"Limit {EPE_WARN}mm")
        ax_epe.set_title(f"Axis: {dir_tag}", fontsize=12,
                         fontweight="bold", color=dc, pad=8)
        ax_epe.set_ylabel("Mean EPE (mm)", fontsize=9)
        ax_epe.legend(fontsize=7.5, loc="upper left")
        ax_epe.grid(True, alpha=0.3)
        ax_epe.set_xticklabels([])
        for sp in ax_epe.spines.values():
            sp.set_edgecolor(dc); sp.set_linewidth(1.5)

        # Angular thresholds & styling
        ax_ang.axhline(ANG_OK,   color="#16a34a", lw=1.3, ls=":", alpha=0.9)
        ax_ang.axhline(ANG_WARN, color="#ef4444", lw=1.3, ls=":", alpha=0.9)
        ax_ang.set_xlabel("Displacement Amplitude (mm)", fontsize=9)
        ax_ang.set_ylabel("Angular Error (°)", fontsize=9)
        ax_ang.grid(True, alpha=0.3)
        for sp in ax_ang.spines.values():
            sp.set_edgecolor(dc); sp.set_linewidth(1.5)

        # Annotate the angular noise-floor phenomenon: at A=3mm direction is
        # near-noise, so angular error is highest at the smallest amplitude.
        y_3mm = _get_any(results, 3.0, 25.0, dir_tag, "ang_mean")
        y_8mm = _get_any(results, 8.0, 25.0, dir_tag, "ang_mean")
        if not np.isnan(y_3mm) and not np.isnan(y_8mm) and y_3mm > y_8mm:
            ax_ang.annotate(
                "Dir. uncertainty\nat small shifts",
                xy=(3.0, y_3mm), xytext=(5.5, y_3mm + 2.5),
                fontsize=7, color="#64748b",
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8),
            )

    _add_metric_glossary(fig, y=-0.06)
    plt.savefig(output_dir / "recovery_curves.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: recovery_curves.png")


# ── 4. Location bias: 3 panels (EPE / Angular / Recovery) ─────────────────────

def plot_location_bias(results: dict, output_dir: Path) -> None:
    all_locations  = ["center", "anterior", "posterior", "lateral_R", "superior"]
    all_loc_labels = ["Center", "Anterior", "Posterior", "Lateral R", "Superior"]
    dir_pair       = ["X", "Y"]
    metrics_cfg    = [
        ("epe_mean", "Mean EPE (mm)",     EPE_OK),
        ("ang_mean", "Angular Error (°)", ANG_WARN),
        ("dot_mean", "Recovery Factor",   DOT_WARN),
    ]

    # Only show locations that have at least one tested case
    has_data = [
        any(not np.isnan(_get_any(results, 5.0, 25.0, d, "epe_mean", loc))
            for d in dir_pair)
        for loc in all_locations
    ]
    locations  = [l  for l,  h in zip(all_locations,  has_data) if h]
    loc_labels = [ll for ll, h in zip(all_loc_labels, has_data) if h]
    n_missing  = len(all_locations) - len(locations)

    w = 0.28          # bar width
    gap = 0.06        # gap between the two bars in a group
    # Offsets so the two bars are centred on each x tick
    offsets = np.array([-(w + gap) / 2, (w + gap) / 2])

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(locations) * 3.5 + 1), 6),
                             facecolor=FACE_COLOR)
    fig.suptitle("Anatomical Location Bias  (A=5mm, σ=25mm)",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    if n_missing:
        fig.text(0.5, -0.03,
                 f"{n_missing} location(s) not yet tested — run full sweep for complete results",
                 ha="center", fontsize=8.5, color="#94a3b8", style="italic")

    for ax, (metric, ylabel, threshold) in zip(axes, metrics_cfg):
        x = np.arange(len(locations))

        # ── Collect values and compute ymax before drawing anything ───────────
        all_vals = {}   # (k, loc_idx) -> float|nan
        for k, dir_tag in enumerate(dir_pair):
            for li, loc in enumerate(locations):
                all_vals[(k, li)] = _get_any(results, 5.0, 25.0, dir_tag, metric, loc)

        valid_vals = [v for v in all_vals.values() if not np.isnan(v)]
        ymax = max(valid_vals + [threshold]) * 1.28

        # ── Draw bars ─────────────────────────────────────────────────────────
        for k, dir_tag in enumerate(dir_pair):
            bar_x  = x + offsets[k]
            heights = [all_vals[(k, li)] for li in range(len(locations))]

            for li, (bx, v) in enumerate(zip(bar_x, heights)):
                if np.isnan(v):
                    # Gray stub to make the slot visible
                    ax.bar(bx, ymax * 0.03, w, color="#e2e8f0",
                           edgecolor="#cbd5e1", lw=0.8, zorder=2)
                    ax.text(bx + w / 2, ymax * 0.05, "N/T",
                            ha="center", va="bottom",
                            fontsize=6.5, color="#94a3b8")
                else:
                    ax.bar(bx, v, w, color=DIR_COLORS[dir_tag],
                           alpha=0.82, edgecolor="white", lw=0.8, zorder=2,
                           label=f"Dir {dir_tag}" if li == 0 else None)
                    ax.text(bx + w / 2, v + ymax * 0.015,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=8, fontweight="bold",
                            color=DIR_COLORS[dir_tag])

        # ── Reference lines with anti-overlap labels ───────────────────────────
        center_vals = [_get_any(results, 5.0, 25.0, d, metric, "center")
                       for d in dir_pair]
        center_mean = np.nanmean(center_vals)

        right_x = len(locations) - 0.5 + offsets[1] + w * 0.55

        if not np.isnan(center_mean):
            ax.axhline(center_mean, color="#64748b", lw=1.4, ls="--", alpha=0.8)
        ax.axhline(threshold, color="#ef4444", lw=1.3, ls=":", alpha=0.85)

        # Compute label y-positions; separate them if too close
        y_ctr = center_mean if not np.isnan(center_mean) else None
        y_thr = threshold
        min_sep = ymax * 0.07   # minimum 7% of y-range between labels

        if y_ctr is not None and abs(y_ctr - y_thr) < min_sep:
            mid = (y_ctr + y_thr) / 2
            y_ctr = mid + min_sep / 2
            y_thr = mid - min_sep / 2

        ax.text(right_x, y_thr, f" thr {threshold}",
                va="center", fontsize=7.5, color="#ef4444")
        if y_ctr is not None:
            ax.text(right_x, y_ctr, f" ctr {center_mean:.2f}",
                    va="center", fontsize=7.5, color="#64748b")

        # ── Axes styling ───────────────────────────────────────────────────────
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(loc_labels, fontsize=10)
        ax.set_xlim(-0.55, len(locations) - 0.45)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, pad=12)
        ax.legend(fontsize=8, loc="upper right",
                  framealpha=0.92, edgecolor=GRID_COLOR)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _add_metric_glossary(fig, y=-0.09)
    plt.savefig(output_dir / "location_bias.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: location_bias.png")


# ── 5. Isotropy radar: multi-amplitude, 2 metrics ─────────────────────────────

def plot_isotropy(results: dict, output_dir: Path) -> None:
    """
    Figure: isotropy.png

    Grouped bar chart showing EPE (left) and Angular Error (right) for each
    shift direction (X / Y / Z / XYZ) at sigma=25mm, one bar group per
    direction, four bars per group (A = 3 / 5 / 8 / 15 mm).
    Replaces radar chart — bars are immediately readable without radar conventions.
    """
    sigma = 25.0
    amp_palette = {3.0: "#94a3b8", 5.0: "#3b82f6", 8.0: "#f59e0b", 15.0: "#ef4444"}

    metrics_cfg = [
        ("epe_mean", f"Mean EPE (mm)  [σ={sigma:.0f}mm]",        EPE_OK,   EPE_WARN),
        ("ang_mean", f"Angular Error (°)  [σ={sigma:.0f}mm]",    ANG_OK,   ANG_WARN),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=FACE_COLOR)
    fig.suptitle(f"Performance by Shift Direction — σ={sigma:.0f}mm, all amplitudes",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)

    n_dirs = len(DIRECTIONS)
    n_amps = len(AMP_VALUES)
    total_width = 0.72
    bar_w = total_width / n_amps
    offsets = np.linspace(-total_width / 2 + bar_w / 2,
                           total_width / 2 - bar_w / 2, n_amps)
    x_pos = np.arange(n_dirs)

    for ax, (key, ylabel, thr_ok, thr_warn) in zip(axes, metrics_cfg):
        for amp_idx, a in enumerate(AMP_VALUES):
            vals = [_get_any(results, a, sigma, d, key) for d in DIRECTIONS]
            bar_colors = [amp_palette[a]] * n_dirs
            ax.bar(x_pos + offsets[amp_idx], vals, width=bar_w,
                   color=bar_colors, alpha=0.85, edgecolor="white",
                   linewidth=0.6, label=f"A = {a:.0f} mm", zorder=3)

        ax.axhline(thr_ok,   color="#16a34a", lw=1.4, ls=":",  alpha=0.9,
                   label=f"Good  < {thr_ok}")
        ax.axhline(thr_warn, color="#ef4444", lw=1.4, ls="--", alpha=0.9,
                   label=f"Limit = {thr_warn}")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"Shift\n{d}" for d in DIRECTIONS],
            fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, pad=8)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    _add_metric_glossary(fig, y=-0.09)
    plt.savefig(output_dir / "isotropy.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: isotropy.png")


# ── 6. System summary: 5 panels + pass/fail table ─────────────────────────────

def plot_system_summary(results: dict, output_dir: Path) -> None:
    ok = _ok_cases(results)
    if len(ok) < 2:
        return

    # Expected total: infer pilot vs full sweep from case names
    is_pilot = all(v.get("name", v.get("direction", "")).startswith("Pilot") or
                   next(iter(k for k in results if results[k] is v), "").startswith("Pilot")
                   for v in ok) if ok else False
    is_pilot = any(k.startswith("Pilot_") for k in results if isinstance(results[k], dict))
    EXPECTED_N = 6 if is_pilot else 56
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(21, 15), facecolor=FACE_COLOR)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.32)
    fig.suptitle(
        f"System Performance Summary  —  {len(ok)}/{EXPECTED_N} PRE cases",
        fontsize=18, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    # ── A: EPE distribution per axis (box + jittered points) ──────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    data_epe = [
        [v["epe_mean"] for v in ok if v.get("direction") == d and "epe_mean" in v]
        for d in DIRECTIONS
    ]
    bp = ax_a.boxplot(
        data_epe, patch_artist=True, widths=0.45,
        medianprops=dict(color="white", lw=2.2),
        whiskerprops=dict(color=GRID_COLOR, lw=1.2),
        capprops=dict(color=GRID_COLOR, lw=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.4),
    )
    for patch, d in zip(bp["boxes"], DIRECTIONS):
        patch.set_facecolor(DIR_COLORS[d]); patch.set_alpha(0.72)
    for i, (pts, d) in enumerate(zip(data_epe, DIRECTIONS)):
        jitter = rng.uniform(-0.13, 0.13, len(pts))
        ax_a.scatter(np.full(len(pts), i + 1) + jitter, pts,
                     color=DIR_COLORS[d], s=22, alpha=0.85, zorder=4)
    ax_a.axhline(EPE_OK,   color="#16a34a", lw=1.3, ls="--", alpha=0.85)
    ax_a.axhline(EPE_WARN, color="#ef4444", lw=1.3, ls="--", alpha=0.85)
    ax_a.set_xticks(range(1, 5)); ax_a.set_xticklabels(DIRECTIONS)
    ax_a.set_ylabel("Mean EPE (mm)"); ax_a.set_title("EPE by Axis")
    ax_a.grid(axis="y", alpha=0.3); ax_a.spines[["top", "right"]].set_visible(False)

    # ── B: Recovery factor distribution per axis ───────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    data_dot = [
        [v["dot_mean"] for v in ok if v.get("direction") == d and "dot_mean" in v]
        for d in DIRECTIONS
    ]
    bp2 = ax_b.boxplot(
        data_dot, patch_artist=True, widths=0.45,
        medianprops=dict(color="white", lw=2.2),
        whiskerprops=dict(color=GRID_COLOR, lw=1.2),
        capprops=dict(color=GRID_COLOR, lw=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.4),
    )
    for patch, d in zip(bp2["boxes"], DIRECTIONS):
        patch.set_facecolor(DIR_COLORS[d]); patch.set_alpha(0.72)
    for i, (pts, d) in enumerate(zip(data_dot, DIRECTIONS)):
        jitter = rng.uniform(-0.13, 0.13, len(pts))
        ax_b.scatter(np.full(len(pts), i + 1) + jitter, pts,
                     color=DIR_COLORS[d], s=22, alpha=0.85, zorder=4)
    ax_b.axhline(DOT_OK,   color="#16a34a", lw=1.3, ls="--", alpha=0.85)
    ax_b.axhline(DOT_WARN, color="#f59e0b", lw=1.3, ls="--", alpha=0.85)
    ax_b.set_xticks(range(1, 5)); ax_b.set_xticklabels(DIRECTIONS)
    ax_b.set_ylabel("Recovery Factor"); ax_b.set_title("Recovery by Axis")
    ax_b.grid(axis="y", alpha=0.3); ax_b.spines[["top", "right"]].set_visible(False)

    # ── C: Quality map (EPE vs Recovery), size ~ amplitude ────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    for d in DIRECTIONS:
        pts = [v for v in ok if v.get("direction") == d]
        if not pts:
            continue
        ax_c.scatter(
            [v.get("dot_mean", np.nan) for v in pts],
            [v.get("epe_mean", np.nan) for v in pts],
            s=[max(20, (v.get("A_mm") or 5) * 16) for v in pts],
            color=DIR_COLORS[d], label=f"Axis {d}",
            alpha=0.75, edgecolors="white", lw=0.8,
        )
    ax_c.axvline(DOT_OK,   color="#16a34a", lw=1.3, ls="--", alpha=0.75)
    ax_c.axvline(DOT_WARN, color="#f59e0b", lw=1.3, ls="--", alpha=0.75)
    ax_c.axhline(EPE_OK,   color="#16a34a", lw=1.3, ls="--", alpha=0.75)
    ax_c.axhline(EPE_WARN, color="#ef4444", lw=1.3, ls="--", alpha=0.75)
    ax_c.axvspan(DOT_OK, 1.05, alpha=0.05, color="#16a34a", zorder=0)
    ax_c.set_xlim(-0.05, 1.05); ax_c.set_ylim(bottom=0)
    ax_c.set_xlabel("Recovery Factor  (higher = better)")
    ax_c.set_ylabel("Mean EPE mm  (lower = better)")
    ax_c.set_title("Quality Map  (size ∝ amplitude)")
    ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.3)
    ax_c.spines[["top", "right"]].set_visible(False)

    # ── D: Amplitude sensitivity (EPE, σ=25mm, center) ────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    for d in DIRECTIONS:
        vals = [_get_any(results, a, 25.0, d, "epe_mean") for a in AMP_VALUES]
        vi   = [i for i, v in enumerate(vals) if not np.isnan(v)]
        if vi:
            ax_d.plot([AMP_VALUES[i] for i in vi], [vals[i] for i in vi],
                      marker="o", color=DIR_COLORS[d], lw=2.2, markersize=7, label=d)
    ax_d.axhline(EPE_OK,   color="#16a34a", lw=1.3, ls="--", alpha=0.85)
    ax_d.axhline(EPE_WARN, color="#ef4444", lw=1.3, ls="--", alpha=0.85)
    ax_d.set_xlabel("Displacement Amplitude (mm)")
    ax_d.set_ylabel("Mean EPE (mm)")
    ax_d.set_title("Amplitude Sensitivity  (σ=25mm)")
    ax_d.legend(fontsize=9); ax_d.grid(True, alpha=0.3)
    ax_d.spines[["top", "right"]].set_visible(False)

    # ── E: Angular error vs amplitude ─────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    for d in DIRECTIONS:
        vals = [_get_any(results, a, 25.0, d, "ang_mean") for a in AMP_VALUES]
        vi   = [i for i, v in enumerate(vals) if not np.isnan(v)]
        if vi:
            ax_e.plot([AMP_VALUES[i] for i in vi], [vals[i] for i in vi],
                      marker="s", color=DIR_COLORS[d], lw=2.2, markersize=6, label=d)
    ax_e.axhline(ANG_OK,   color="#16a34a", lw=1.3, ls="--", alpha=0.85)
    ax_e.axhline(ANG_WARN, color="#ef4444", lw=1.3, ls="--", alpha=0.85)
    ax_e.set_xlabel("Displacement Amplitude (mm)")
    ax_e.set_ylabel("Mean Angular Error (°)")
    ax_e.set_title("Angular Sensitivity  (σ=25mm)")
    ax_e.legend(fontsize=9); ax_e.grid(True, alpha=0.3)
    ax_e.spines[["top", "right"]].set_visible(False)

    # ── F: Per-direction global averages table ─────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    tbl_data = [["Axis", "EPE μ", "EPE P95", "Ang μ", "Recovery"]]
    for d in DIRECTIONS:
        pts = [v for v in ok if v.get("direction") == d]
        if not pts:
            tbl_data.append([d, "N/A", "N/A", "N/A", "N/A"]); continue
        tbl_data.append([
            d,
            f"{np.nanmean([v.get('epe_mean', np.nan) for v in pts]):.2f}mm",
            f"{np.nanmean([v.get('epe_p95',  np.nan) for v in pts]):.2f}mm",
            f"{np.nanmean([v.get('ang_mean', np.nan) for v in pts]):.1f}°",
            f"{np.nanmean([v.get('dot_mean', np.nan) for v in pts])*100:.0f}%",
        ])
    all_epe = [v.get("epe_mean", np.nan) for v in ok]
    all_ang = [v.get("ang_mean", np.nan) for v in ok]
    all_dot = [v.get("dot_mean", np.nan) for v in ok]
    tbl_data.append([
        "ALL",
        f"{np.nanmean(all_epe):.2f}mm",
        f"{np.nanpercentile(all_epe, 95):.2f}mm",
        f"{np.nanmean(all_ang):.1f}°",
        f"{np.nanmean(all_dot)*100:.0f}%",
    ])
    tbl = ax_f.table(cellText=tbl_data, loc="center", cellLoc="center",
                     colWidths=[0.14, 0.22, 0.22, 0.22, 0.22])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.15)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row == 0:
            cell.set_facecolor(TEXT_COLOR)
            cell.set_text_props(weight="bold", color="white")
        elif col == 0:
            d = tbl_data[row][0]
            cell.set_facecolor(DIR_COLORS.get(d, TEXT_COLOR))
            cell.set_text_props(weight="bold", color="white")
    ax_f.set_title("Direction Averages", fontsize=11, fontweight="bold", pad=12)

    # ── G: Pass/fail grid (axis × amplitude, σ=25mm) ──────────────────────────
    ax_g = fig.add_subplot(gs[2, :])
    ax_g.axis("off")
    pf_cols = [f"A={a:.0f}mm" for a in AMP_VALUES]
    pf_rows = []
    for d in DIRECTIONS:
        row = [d]
        for a in AMP_VALUES:
            dot = _get_any(results, a, 25.0, d, "dot_mean")
            epe = _get_any(results, a, 25.0, d, "epe_p95")
            row.append("N/A" if np.isnan(dot) else _verdict(dot, epe))
        pf_rows.append(row)
    pf_data = [["Axis"] + pf_cols] + pf_rows
    pf_tbl  = ax_g.table(
        cellText=pf_data, loc="center", cellLoc="center",
        colWidths=[0.09] + [0.16] * len(AMP_VALUES),
    )
    pf_tbl.auto_set_font_size(False); pf_tbl.set_fontsize(11); pf_tbl.scale(1, 3.2)
    CELL_FC = {
        "EXCELLENT": ("#d1fae5", "#065f46"),
        "GOOD":      ("#dbeafe", "#1e3a8a"),
        "ACCEPTABLE":("#fef3c7", "#92400e"),
        "POOR":      ("#fee2e2", "#7f1d1d"),
    }
    for (row, col), cell in pf_tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row == 0:
            cell.set_facecolor(TEXT_COLOR)
            cell.set_text_props(weight="bold", color="white")
        elif col == 0:
            d = pf_data[row][0]
            cell.set_facecolor(DIR_COLORS.get(d, TEXT_COLOR))
            cell.set_text_props(weight="bold", color="white")
        else:
            txt = cell.get_text().get_text()
            if txt in CELL_FC:
                fc, tc = CELL_FC[txt]
                cell.set_facecolor(fc)
                cell.set_text_props(weight="bold", color=tc)
            elif txt == "N/A":
                cell.set_facecolor("#f1f5f9")
                cell.get_text().set_text("not tested")
                cell.set_text_props(color="#94a3b8", style="italic")
    ax_g.set_title("Pass/Fail Grid  (σ=25mm, center — amplitudes × axes)",
                   fontsize=11, fontweight="bold", pad=14)

    _add_metric_glossary(fig, y=-0.04)
    plt.savefig(output_dir / "system_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.success("Saved: system_summary.png")


# ── Summary & CSV ──────────────────────────────────────────────────────────────

def print_summary(results: dict):
    ok = _ok_cases(results)
    if not ok:
        return
    logger.section(f"Global Summary — {len(ok)} Cases")
    for label, key in [
        ("EPE mean (mm)",   "epe_mean"),
        ("Angular error (°)", "ang_mean"),
        ("Recovery Index",  "dot_mean"),
    ]:
        data = [v[key] for v in ok if key in v]
        if data:
            logger.info(
                f"  {label:<22} mean={np.mean(data):.3f}  "
                f"median={np.median(data):.3f}  "
                f"P95={np.percentile(data, 95):.3f}"
            )

def _mode_continuous(arr: np.ndarray, bins: int = 10) -> float:
    """Modal class centre for continuous data (peak of the histogram)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float(arr[0]) if len(arr) == 1 else np.nan
    n_bins = min(bins, max(3, len(arr) // 2))
    counts, edges = np.histogram(arr, bins=n_bins)
    peak = int(np.argmax(counts))
    return float((edges[peak] + edges[peak + 1]) / 2)


def plot_statistical_summary(results: dict, output_dir: Path) -> None:
    """
    Figure: statistical_summary.png

    Aggregate statistics (mean / median / mode ± std) for PRE and POST cases.
    Top: styled table with PRE and POST columns.
    Bottom: violin panels with PRE (blue) and POST (orange) distributions overlaid.
    """
    ok_pre  = _ok_cases(results, "pre")
    ok_post = _ok_cases(results, "post")
    n_pre, n_post = len(ok_pre), len(ok_post)

    if n_pre < 2 and n_post < 2:
        logger.info("Not enough cases for statistical_summary.png — skipping")
        return

    has_post = n_post >= 2

    # ── Metric catalogue ──────────────────────────────────────────────────────
    METRICS = [
        ("epe_mean",     "EPE Mean",       "mm", ".3f"),
        ("epe_p95",      "EPE P95",        "mm", ".3f"),
        ("ang_mean",     "Angular Error",  "°",  ".2f"),
        ("ang_pct_gt90", "Wrong Direction","%",  ".2f"),
        ("dot_mean",     "Dot Ratio",      "—",  ".3f"),
        ("err_x_mean",   "Comp. Error X",  "mm", ".3f"),
        ("err_y_mean",   "Comp. Error Y",  "mm", ".3f"),
        ("err_z_mean",   "Comp. Error Z",  "mm", ".3f"),
    ]
    if any("jac_pct_fold" in v for v in ok_pre + ok_post):
        METRICS.append(("jac_pct_fold", "Jac. Folding", "%", ".2f"))

    # ── Compute statistics ────────────────────────────────────────────────────
    def _stats(ok, key):
        vals = np.array([v.get(key, np.nan) for v in ok], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return dict(mean=np.nan, median=np.nan, mode=np.nan, std=np.nan, vals=vals)
        return dict(
            mean   = float(np.mean(vals)),
            median = float(np.median(vals)),
            mode   = _mode_continuous(vals),
            std    = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            vals   = vals,
        )

    stats_pre  = {key: _stats(ok_pre,  key) for key, *_ in METRICS}
    stats_post = {key: _stats(ok_post, key) for key, *_ in METRICS} if has_post else {}

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor=FACE_COLOR)
    gs  = GridSpec(2, 3, figure=fig,
                   height_ratios=[1.05, 1.15],
                   hspace=0.54, wspace=0.30)

    if has_post:
        title = f"Statistical Summary — {n_pre} PRE  +  {n_post} POST test cases"
    else:
        title = f"Statistical Summary — {n_pre} PRE test cases"
    fig.suptitle(title, fontsize=17, fontweight="bold", color=TEXT_COLOR, y=0.99)

    small_n = min(n_pre, n_post) if has_post else n_pre
    if small_n < 20:
        fig.text(0.5, 0.965,
                 f"Note: n={small_n} is small — mode estimate based on histogram bins and may be unstable.",
                 ha="center", fontsize=8.5, color="#f59e0b", style="italic")

    # ── Row 0: full-width styled table ───────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.axis("off")
    ax_tbl.set_title(
        "All metrics  ·  each cell = statistic ± std dev  ·  computed across all n cases",
        fontsize=10, pad=8, color="#475569",
    )

    if has_post:
        col_labels = [
            "Metric", "Unit",
            "PRE Mean±Std", "PRE Median±Std", "PRE Mode±Std",
            "POST Mean±Std", "POST Median±Std", "POST Mode±Std",
        ]
        col_widths = [0.13, 0.05, 0.145, 0.145, 0.145, 0.145, 0.145, 0.145]
        HDR_COLORS = {
            "PRE Mean±Std":    "#3b82f6",
            "PRE Median±Std":  "#10b981",
            "PRE Mode±Std":    "#f59e0b",
            "POST Mean±Std":   "#ea580c",
            "POST Median±Std": "#7c3aed",
            "POST Mode±Std":   "#dc2626",
        }
    else:
        col_labels = ["Metric", "Unit", "Mean ± Std", "Median ± Std", "Mode ± Std"]
        col_widths  = [0.17, 0.06, 0.23, 0.23, 0.23]
        HDR_COLORS = {
            "Mean ± Std":   "#3b82f6",
            "Median ± Std": "#10b981",
            "Mode ± Std":   "#f59e0b",
        }

    rows = []
    for key, label, unit, fmt in METRICS:
        sp  = stats_pre[key]
        spo = stats_post.get(key, {}) if has_post else {}

        def _fmt_pre(stat, s=sp, f=fmt):
            v, sd = s.get(stat, np.nan), s.get("std", np.nan)
            if np.isnan(v) or np.isnan(sd): return "N/A"
            return f"{v:{f}} ± {sd:{f}}"

        if has_post:
            def _fmt_post(stat, s=spo, f=fmt):
                v, sd = s.get(stat, np.nan), s.get("std", np.nan)
                if np.isnan(v) or np.isnan(sd): return "N/A"
                return f"{v:{f}} ± {sd:{f}}"
            rows.append([label, unit,
                         _fmt_pre("mean"), _fmt_pre("median"), _fmt_pre("mode"),
                         _fmt_post("mean"), _fmt_post("median"), _fmt_post("mode")])
        else:
            rows.append([label, unit, _fmt_pre("mean"), _fmt_pre("median"), _fmt_pre("mode")])

    tbl = ax_tbl.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5 if has_post else 9.5)
    tbl.scale(1, 2.15)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row == 0:
            hdr = col_labels[col] if col < len(col_labels) else ""
            fc  = HDR_COLORS.get(hdr, TEXT_COLOR)
            cell.set_facecolor(fc)
            cell.set_text_props(weight="bold", color="white")
        elif col == 0:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(weight="bold", color=TEXT_COLOR)
        elif has_post and col >= 5:
            if row % 2 == 0:
                cell.set_facecolor("#fff7ed")  # light orange tint for POST columns
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")

    # ── Wilcoxon paired test (computed once, injected into violin panels) ────
    wilcox_pvals: dict[str, tuple[float, str]] = {}
    if has_post:
        wilcox_pairs = _pair_post_cases(results)
        wilcox_cfg = [
            ("epe_mean", "greater"),   # H1: PRE > POST  → POST better
            ("ang_mean", "greater"),   # H1: PRE > POST  → POST better
            ("dot_mean", "less"),      # H1: PRE < POST  → POST better
        ]
        for wkey, walt in wilcox_cfg:
            if len(wilcox_pairs) < 5:
                continue
            wv_pre  = [p[0].get(wkey, np.nan) for p in wilcox_pairs]
            wv_post = [p[1].get(wkey, np.nan) for p in wilcox_pairs]
            mask    = [not (np.isnan(a) or np.isnan(b)) for a, b in zip(wv_pre, wv_post)]
            wv_pre  = [v for v, m in zip(wv_pre,  mask) if m]
            wv_post = [v for v, m in zip(wv_post, mask) if m]
            if len(wv_pre) < 5:
                continue
            try:
                _, p_val = scipy_wilcoxon(wv_pre, wv_post, alternative=walt)
                sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                       else "*" if p_val < 0.05 else "ns")
                wilcox_pvals[wkey] = (p_val, sig)
            except Exception:
                pass

    # ── Row 1: distribution panels (violin + stats markers) ──────────────────
    PANELS = [
        (gs[1, 0], "epe_mean",  "EPE Mean (mm)",     EPE_WARN, True),
        (gs[1, 1], "ang_mean",  "Angular Error (°)", ANG_WARN, True),
        (gs[1, 2], "dot_mean",  "Dot Ratio",         DOT_WARN, False),
    ]
    STAT_STYLES_PRE = [
        ("mean",   "PRE Mean",   "#3b82f6", "-",  2.2),
        ("median", "PRE Median", "#10b981", "--", 2.0),
        ("mode",   "PRE Mode",   "#f59e0b", ":",  2.0),
    ]
    STAT_STYLES_POST = [
        ("mean",   "POST Mean",   "#ea580c", "-",  2.2),
        ("median", "POST Median", "#7c3aed", "--", 2.0),
        ("mode",   "POST Mode",   "#dc2626", ":",  2.0),
    ]

    for subplot_spec, key, ylabel, threshold, lower_is_better in PANELS:
        ax       = fig.add_subplot(subplot_spec)
        sp       = stats_pre[key]
        spo      = stats_post.get(key, {"vals": np.array([]), "mean": np.nan,
                                        "median": np.nan, "mode": np.nan, "std": np.nan})
        pre_vals  = sp["vals"]
        post_vals = spo.get("vals", np.array([])) if has_post else np.array([])

        if len(pre_vals) == 0 and len(post_vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#94a3b8")
            ax.set_title(ylabel, fontsize=10, pad=8)
            continue

        pos_pre = -0.25 if has_post else 0
        width_v  = 0.40 if has_post else 0.70
        rng = np.random.default_rng(42)

        # PRE violin
        if len(pre_vals) >= 2:
            vp = ax.violinplot(pre_vals, positions=[pos_pre], widths=width_v,
                               showmedians=False, showextrema=True)
            for pc in vp["bodies"]:
                pc.set_facecolor("#93c5fd"); pc.set_alpha(0.55)
            for part in ("cmins", "cmaxes", "cbars"):
                if part in vp:
                    vp[part].set_color("#3b82f6"); vp[part].set_linewidth(1.0)
            jitter = rng.uniform(-0.06, 0.06, len(pre_vals))
            ax.scatter(np.full(len(pre_vals), pos_pre) + jitter, pre_vals,
                       s=22, alpha=0.75, color="#3b82f6",
                       edgecolors="white", linewidth=0.5, zorder=4)

        # POST violin
        if has_post and len(post_vals) >= 2:
            vp2 = ax.violinplot(post_vals, positions=[0.25], widths=0.40,
                                showmedians=False, showextrema=True)
            for pc in vp2["bodies"]:
                pc.set_facecolor("#fdba74"); pc.set_alpha(0.55)
            for part in ("cmins", "cmaxes", "cbars"):
                if part in vp2:
                    vp2[part].set_color("#ea580c"); vp2[part].set_linewidth(1.0)
            rng2 = np.random.default_rng(99)
            jitter2 = rng2.uniform(-0.06, 0.06, len(post_vals))
            ax.scatter(np.full(len(post_vals), 0.25) + jitter2, post_vals,
                       s=22, alpha=0.75, color="#ea580c",
                       edgecolors="white", linewidth=0.5, zorder=4)

        # ±1 std band (PRE only when no POST — avoids cluttered overlap)
        if not has_post and not np.isnan(sp["mean"]) and not np.isnan(sp["std"]):
            ax.axhspan(sp["mean"] - sp["std"], sp["mean"] + sp["std"],
                       color="#3b82f6", alpha=0.10, zorder=2,
                       label=f"±1 std ({sp['std']:.3f})")

        # PRE stat lines
        x_l_pre = (pos_pre - 0.20) if has_post else -0.42
        x_r_pre = (pos_pre + 0.20) if has_post else  0.42
        for stat_key, stat_label, color, ls, lw in STAT_STYLES_PRE:
            v = sp.get(stat_key, np.nan)
            if np.isnan(v): continue
            ax.hlines(v, x_l_pre, x_r_pre, colors=color, linestyles=ls,
                      linewidth=lw, zorder=5, label=f"{stat_label} = {v:.3f}")
            if not has_post:
                ax.text(x_r_pre + 0.04, v, f"{v:.3f}",
                        va="center", ha="left", fontsize=8,
                        color=color, fontweight="bold")

        # POST stat lines
        if has_post:
            for stat_key, stat_label, color, ls, lw in STAT_STYLES_POST:
                v = spo.get(stat_key, np.nan)
                if np.isnan(v): continue
                ax.hlines(v, 0.05, 0.45, colors=color, linestyles=ls,
                          linewidth=lw, zorder=5, label=f"{stat_label} = {v:.3f}")

        # Threshold reference
        ref_color = "#ef4444" if lower_is_better else "#16a34a"
        ref_label = f"{'Limit' if lower_is_better else 'Min'} {threshold}"
        ax.axhline(threshold, color=ref_color, lw=1.2, ls="-.", alpha=0.7,
                   label=ref_label, zorder=3)

        if has_post:
            ax.set_xticks([-0.25, 0.25])
            ax.set_xticklabels(["PRE", "POST"], fontsize=9)
            ax.set_xlim(-0.55, 0.60)
        else:
            ax.set_xticks([])
            ax.set_xlim(-0.52, 0.70)

        ax.set_ylabel(ylabel, fontsize=10)

        # Wilcoxon p-value embedded in title
        wp = wilcox_pvals.get(key)
        if wp:
            p_val, sig = wp
            ax.set_title(f"{ylabel}\nWilcoxon p={p_val:.3f} ({sig})", fontsize=9, pad=6)
        else:
            ax.set_title(ylabel, fontsize=10, pad=8)

        ax.legend(fontsize=7 if has_post else 7.5, loc="upper right",
                  framealpha=0.92, edgecolor=GRID_COLOR)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right", "bottom"]].set_visible(False)

    _add_metric_glossary(fig, y=-0.10)
    plt.savefig(output_dir / "statistical_summary.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: statistical_summary.png")


def _draw_epe_scatter(ax, records: list, pairs: list) -> None:
    """4th panel of cavity_effect: scatter EPE_POST vs EPE_PRE with identity diagonal.

    Each dot = one test case. Color = direction (X/Y/Z/XYZ). Points below the
    diagonal mean POST ≤ PRE (cavity did not hurt, or even helped).
    """
    pre_epe_vals  = [pre.get("epe_mean", np.nan) for pre, _ in pairs]
    post_epe_vals = [post.get("epe_mean", np.nan) for _, post in pairs]
    dirs          = [pre.get("direction", "X") for pre, _ in pairs]

    for x_v, y_v, d in zip(pre_epe_vals, post_epe_vals, dirs):
        if np.isnan(x_v) or np.isnan(y_v):
            continue
        ax.scatter(x_v, y_v, s=55, color=DIR_COLORS.get(d, "#64748b"),
                   alpha=0.78, edgecolors="white", linewidth=0.5, zorder=4)

    finite_xy = [(x, y) for x, y in zip(pre_epe_vals, post_epe_vals)
                 if not (np.isnan(x) or np.isnan(y))]
    if finite_xy:
        lim = max(max(v for v, _ in finite_xy), max(v for _, v in finite_xy)) * 1.12
        ax.plot([0, lim], [0, lim], "--", color="#94a3b8", linewidth=1.4)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        # Zone labels
        ax.text(lim * 0.55, lim * 0.30, "POST better\n(below line)",
                fontsize=8, color="#16a34a", style="italic", ha="center")
        ax.text(lim * 0.30, lim * 0.72, "PRE better\n(above line)",
                fontsize=8, color="#dc2626", style="italic", ha="center")

    ax.set_xlabel("EPE with clean brain — PRE (mm)", fontsize=8.5)
    ax.set_ylabel("EPE with surgical cavity — POST (mm)", fontsize=8.5)
    ax.set_title("EPE: POST vs PRE\n(each dot = one test case)", fontsize=10, pad=8)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    dir_handles = [plt.scatter([], [], s=55, color=DIR_COLORS[d],
                               edgecolors="white", label=f"Shift dir. {d}") for d in DIRECTIONS]
    ax.legend(handles=dir_handles, fontsize=7.5, loc="upper left",
              title="Shift direction", title_fontsize=7.5)


def plot_cavity_effect(results: dict, output_dir: Path) -> None:
    """
    Figure: cavity_effect.png  (Δ = POST − PRE per matched pair).

    n ≤ 12  → bar chart (one bar per pair, readable for pilot subset).
    n  > 12 → scatter grouped by direction + box overlay (full sweep, 56 pairs).
    """
    pairs = _pair_post_cases(results)
    if not pairs:
        logger.info("No PRE/POST pairs found — skipping cavity_effect.png")
        return

    n = len(pairs)

    records = []
    for pre, post in pairs:
        pre_jac  = pre.get("jac_pct_fold", np.nan)
        post_jac = post.get("jac_pct_fold", np.nan)
        d_jac = (post_jac - pre_jac) if not (np.isnan(pre_jac) or np.isnan(post_jac)) else np.nan
        records.append({
            "name":      pre["name"],
            "direction": pre.get("direction", "?"),
            "A_mm":      pre.get("A_mm", 5.0),
            "sigma_mm":  pre.get("sigma_mm", 25.0),
            "location":  pre.get("location", "center"),
            "d_epe": post["epe_mean"] - pre["epe_mean"],
            "d_ang": post["ang_mean"] - pre["ang_mean"],
            "d_dot": post["dot_mean"] - pre["dot_mean"],
            "d_jac": d_jac,
        })

    d_epe_all = [r["d_epe"] for r in records]
    n_neg_epe = sum(1 for v in d_epe_all if not np.isnan(v) and v < 0)

    # 3 delta panels + 1 scatter EPE_POST vs EPE_PRE (ΔJacobian removed — near-zero, uninformative)
    panels_cfg = [
        ("d_epe", "ΔEPE Mean (mm)",        False),
        ("d_ang", "ΔAngular Error (°)",     False),
        ("d_dot", "ΔDot Ratio (recovery)",  True),
    ]

    # ── Shared note ────────────────────────────────────────────────────────────
    def _build_note():
        if n_neg_epe > 0:
            return (
                f"Negative ΔEPE (green, {n_neg_epe}/{n} pairs): POST outperforms PRE — "
                "surgical cavity creates sharper tissue edges in distance maps, "
                "improving B-Spline guidance (counterintuitive but physically plausible).  "
                "Positive Δ = cavity degrades quality."
            ), "#1e40af"
        return "Positive Delta = surgical cavity degrades registration quality.", "#64748b"

    # ══════════════════════════════════════════════════════════════════════════
    # MODE A — bar chart (pilot, n ≤ 12)
    # ══════════════════════════════════════════════════════════════════════════
    if n <= 12:
        labels = [r["name"] for r in records]
        x = np.arange(n)

        fig, axes = plt.subplots(1, 4, figsize=(max(14, n * 2.4 + 2), 6),
                                 facecolor=FACE_COLOR)
        # axes[3] is reserved for EPE scatter (built after the bar loop)
        fig.suptitle("Cavity Effect on Registration Accuracy  (Δ = POST − PRE)",
                     fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)

        note_txt, note_col = _build_note()
        fig.text(0.5, -0.04, note_txt, ha="center", fontsize=8, color=note_col, style="italic")

        def _safe_max_abs(vals):
            finite = [abs(v) for v in vals if not np.isnan(v)]
            return max(finite) if finite else 1.0

        for ax, (metric_key, ylabel, pos_good) in zip(axes[:3], panels_cfg):
            deltas = [r[metric_key] for r in records]
            all_nan = all(np.isnan(v) for v in deltas)
            safe_d  = [v if not np.isnan(v) else 0.0 for v in deltas]
            colors  = ["#e2e8f0" if np.isnan(orig) else
                       ("#16a34a" if (pos_good and v > 0) or (not pos_good and v <= 0) else "#dc2626")
                       for v, orig in zip(safe_d, deltas)]

            bars = ax.bar(x, safe_d, color=colors, alpha=0.82, edgecolor="white", linewidth=0.8)
            ax.axhline(0, color=TEXT_COLOR, lw=1.4, ls="--", alpha=0.6, zorder=5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(ylabel, fontsize=10, pad=8)
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            if all_nan:
                ax.text(0.5, 0.5, "Not available\n(re-run with --analyze)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="#94a3b8", style="italic")
                continue
            mxa = _safe_max_abs(safe_d)
            for bar, v, orig in zip(bars, safe_d, deltas):
                if np.isnan(orig): continue
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + (mxa * 0.04 if v >= 0 else -mxa * 0.04),
                        f"{v:+.3f}", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7, fontweight="bold", color="#1e293b")

        # 4th panel: EPE scatter POST vs PRE
        _draw_epe_scatter(axes[3], records, pairs)

        _add_metric_glossary(fig, y=-0.22)
        plt.tight_layout()
        plt.savefig(output_dir / "cavity_effect.png", dpi=180, bbox_inches="tight")
        plt.close()
        logger.success("Saved: cavity_effect.png")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODE B — scatter grouped by direction (full sweep, n > 12)
    # ══════════════════════════════════════════════════════════════════════════
    DIR_ORDER = ["X", "Y", "Z", "XYZ"]
    has_loc = {d: any(r["direction"] == d and r["location"] != "center" for r in records)
               for d in ["X", "Y"]}
    groups = list(DIR_ORDER)
    if has_loc.get("X"): groups.append("X-loc")
    if has_loc.get("Y"): groups.append("Y-loc")
    gpos = {g: i for i, g in enumerate(groups)}

    def _group(r):
        return f"{r['direction']}-loc" if r["location"] != "center" else r["direction"]

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=FACE_COLOR)
    axes = axes.flatten()
    fig.suptitle("Cavity Effect on Registration Accuracy  (Δ = POST − PRE)",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.01)

    for ax, (metric_key, ylabel, pos_good) in zip(axes[:3], panels_cfg):
        # Box plot per group (IQR overlay)
        gdata = {g: [] for g in groups}
        for r in records:
            g = _group(r)
            v = r[metric_key]
            if g in gdata and not np.isnan(v):
                gdata[g].append(v)

        nonempty = [(g, gdata[g]) for g in groups if gdata[g]]
        if nonempty:
            bp = ax.boxplot([d for _, d in nonempty],
                            positions=[gpos[g] for g, _ in nonempty],
                            widths=0.45, patch_artist=True, zorder=2,
                            medianprops=dict(color=TEXT_COLOR, lw=2.0),
                            whiskerprops=dict(color="#94a3b8", lw=1.0),
                            capprops=dict(color="#94a3b8", lw=1.0),
                            flierprops=dict(marker="", alpha=0))
            for patch in bp["boxes"]:
                patch.set_facecolor("#f1f5f9"); patch.set_alpha(0.5)

        # Individual dots: green = cavity improves, red = cavity degrades
        all_nan = True
        for r in records:
            g = _group(r)
            if g not in gpos: continue
            v = r[metric_key]
            if np.isnan(v): continue
            all_nan = False
            good = (pos_good and v > 0) or (not pos_good and v <= 0)
            color = "#16a34a" if good else "#dc2626"
            jit = rng.uniform(-0.18, 0.18)
            ax.scatter(gpos[g] + jit, v, s=45,
                       color=color, alpha=0.80,
                       edgecolors="white", linewidth=0.4, zorder=4)

        ax.axhline(0, color=TEXT_COLOR, lw=1.4, ls="--", alpha=0.55, zorder=3)

        # Zone labels — tell reader what above/below zero means
        ylims = ax.get_ylim()
        yrange = ylims[1] - ylims[0] if ylims[1] != ylims[0] else 1.0
        y_top = ylims[1] - yrange * 0.06
        y_bot = ylims[0] + yrange * 0.06
        n_groups = len(groups)
        ax.text(n_groups / 2 - 0.5, y_top, "cavity degrades  ▲",
                ha="center", fontsize=7.5, color="#dc2626", style="italic", va="top")
        ax.text(n_groups / 2 - 0.5, y_bot, "▼  cavity improves",
                ha="center", fontsize=7.5, color="#16a34a", style="italic", va="bottom")

        ax.set_xticks(list(gpos.values()))
        ax.set_xticklabels(
            [f"Shift {g}" if "-" not in g else f"Shift {g}\n(off-centre)" for g in groups],
            fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10, pad=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        if all_nan:
            ax.text(0.5, 0.5, "Not available\n(re-run with --analyze)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#94a3b8", style="italic")

    # 4th panel: EPE scatter POST vs PRE
    _draw_epe_scatter(axes[3], records, pairs)

    note_txt, note_col = _build_note()
    fig.text(0.5, -0.04,
             "Green dot = cavity improves that metric  ·  Red dot = cavity degrades  ·  "
             "X-loc / Y-loc = shift centred away from brain centre\n" + note_txt,
             ha="center", fontsize=8, color=note_col, style="italic")

    _add_metric_glossary(fig, y=-0.18)
    plt.tight_layout()
    plt.savefig(output_dir / "cavity_effect.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: cavity_effect.png")


def save_csv(results: dict, output_dir: Path):
    base_fields = [
        "name", "source_type", "direction", "A_mm", "sigma_mm", "location",
        "epe_mean", "epe_median", "epe_p95",
        "ang_mean", "ang_pct_gt90",
        "dot_mean", "gt_peak_mm",
        "jac_min", "jac_mean", "jac_pct_fold",
        "elapsed_min",
    ]
    delta_fields = ["delta_epe_mean", "delta_ang_mean", "delta_dot_mean", "delta_jac_pct_fold"]
    all_fields   = base_fields + delta_fields

    # Build delta lookup from matched pairs
    pairs = _pair_post_cases(results)
    delta_lookup: dict[str, dict] = {}
    for pre, post in pairs:
        delta = {
            "delta_epe_mean":    post["epe_mean"] - pre["epe_mean"],
            "delta_ang_mean":    post["ang_mean"] - pre["ang_mean"],
            "delta_dot_mean":    post["dot_mean"] - pre["dot_mean"],
            "delta_jac_pct_fold": (
                (post.get("jac_pct_fold", np.nan) - pre.get("jac_pct_fold", np.nan))
                if not (np.isnan(post.get("jac_pct_fold", np.nan))
                        or np.isnan(pre.get("jac_pct_fold", np.nan)))
                else ""
            ),
        }
        delta_lookup[pre.get("name",  "")] = delta
        delta_lookup[post.get("name", "")] = delta

    path = output_dir / "sweep_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for name, r in results.items():
            if isinstance(r, dict) and r.get("status") == "OK":
                row = {k: r.get(k) for k in base_fields}
                row["source_type"] = r.get("source_type", "pre")
                deltas = delta_lookup.get(name, {})
                for df in delta_fields:
                    row[df] = deltas.get(df, "")
                writer.writerow(row)

def plot_angular_weighted_analysis(results: dict, output_dir: Path, config=None) -> None:
    """
    Figure: angular_weighted_analysis.png

    Row 1 — PRE  heatmaps (4 directions): amplitude-weighted angular error
    Row 2 — POST heatmaps (4 directions): same metric for POST variants
    Row 3 — Left:  box plot ΔEPE = EPE_POST − EPE_PRE, grouped by shift direction
             Right: per-voxel EPE triplanar map on Pilot_A050_S250_X_C

    Weighting: ang_mean × (gt_peak_mm / global_max_gt_peak)
    Rationale: at A=3mm the displacement direction is near-random noise; weighting
    by the actual GT peak magnitude suppresses that noise and focuses the colourscale
    on cases where directional accuracy is clinically relevant.
    """
    import SimpleITK as sitk

    ok_pre  = _ok_cases(results, "pre")
    ok_post = _ok_cases(results, "post")
    pairs   = _pair_post_cases(results)

    if not ok_pre:
        logger.info("No PRE cases — skipping angular_weighted_analysis.png")
        return

    # ── Weighted angular matrices: ang_mean × epe_mean (°·mm) ────────────────
    # Rationale: EPE = ‖pred + gt‖ is the residual magnitude after recovery.
    # Weighting angle by EPE means voxels/cases where both direction AND magnitude
    # recovery failed contribute most. Cases well-recovered (small EPE) barely
    # contribute even if angular error is high (small EPE → direction irrelevant).
    def _wmat(getter):
        mats = {}
        for d in DIRECTIONS:
            mat = np.full((len(AMP_VALUES), len(SIGMA_VALUES)), np.nan)
            for i, a in enumerate(AMP_VALUES):
                for j, s in enumerate(SIGMA_VALUES):
                    ang = getter(results, a, s, d, "ang_mean")
                    epe = getter(results, a, s, d, "epe_mean")
                    if not (np.isnan(ang) or np.isnan(epe)):
                        mat[i, j] = ang * epe
            mats[d] = mat
        return mats

    pre_mats  = _wmat(_get_any)
    post_mats = _wmat(_get_any_post)

    all_finite = np.concatenate([
        m.ravel() for m in list(pre_mats.values()) + list(post_mats.values())
    ])
    all_finite = all_finite[~np.isnan(all_finite)]
    vmin_h = 0.0
    vmax_h = float(np.percentile(all_finite, 97)) if len(all_finite) else 1.0

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 17), facecolor=FACE_COLOR)
    gs_main = GridSpec(3, 1, figure=fig,
                       height_ratios=[1.05, 1.05, 1.6], hspace=0.46)

    gs_pre  = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0], wspace=0.06)
    gs_post = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1], wspace=0.06)
    gs_bot  = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_main[2],
                                      width_ratios=[2.4, 1, 1, 1, 0.08], wspace=0.08)

    ax_box = fig.add_subplot(gs_bot[0, 0])
    ax_axl = fig.add_subplot(gs_bot[0, 1])
    ax_cor = fig.add_subplot(gs_bot[0, 2])
    ax_sag = fig.add_subplot(gs_bot[0, 3])
    ax_cb  = fig.add_subplot(gs_bot[0, 4])

    cmap_h = "YlOrRd"

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    for row_idx, (mats, label, lcolor) in enumerate([
        (pre_mats,  "PRE  — Weighted Angular Error  (ang × EPE,  °·mm)",  "#3b82f6"),
        (post_mats, "POST — Weighted Angular Error  (ang × EPE,  °·mm)",  "#ea580c"),
    ]):
        gs_row = gs_pre if row_idx == 0 else gs_post
        for col, d in enumerate(DIRECTIONS):
            ax = fig.add_subplot(gs_row[0, col])
            mat = mats[d]
            ax.imshow(mat, cmap=cmap_h, vmin=vmin_h, vmax=vmax_h,
                      aspect="auto", origin="upper")
            _annotate_heatmap(ax, mat, ".1f", vmin_h, vmax_h)
            ax.set_xticks(range(len(SIGMA_VALUES)))
            ax.set_xticklabels([f"σ={s:.0f}" for s in SIGMA_VALUES], fontsize=8)
            if col == 0:
                ax.set_yticks(range(len(AMP_VALUES)))
                ax.set_yticklabels([f"A={a:.0f}mm" for a in AMP_VALUES], fontsize=8)
                ax.set_ylabel(label, fontsize=8.5, color=lcolor, fontweight="bold")
            else:
                ax.set_yticks([])
            ax.set_title(f"Shift {d}", fontsize=10, fontweight="bold",
                         color=DIR_COLORS[d], pad=5)
            for sp in ax.spines.values():
                sp.set_edgecolor(DIR_COLORS[d]); sp.set_linewidth(1.8)

    # Shared colorbar for heatmaps — placed as figure-level text annotation
    sm_h = plt.cm.ScalarMappable(cmap=cmap_h, norm=plt.Normalize(vmin_h, vmax_h))
    sm_h.set_array([])
    cbar_ax = fig.add_axes([0.915, 0.42, 0.008, 0.38])
    cb_h = fig.colorbar(sm_h, cax=cbar_ax)
    cb_h.set_label("ang × EPE  (°·mm)", fontsize=8, rotation=90, labelpad=6)
    cb_h.ax.tick_params(labelsize=7)

    # ── ΔEPE box plot ─────────────────────────────────────────────────────────
    if pairs:
        by_dir: dict[str, list] = {d: [] for d in DIRECTIONS}
        for pre_r, post_r in pairs:
            d     = pre_r.get("direction", "X")
            delta = post_r.get("epe_mean", np.nan) - pre_r.get("epe_mean", np.nan)
            if d in by_dir and not np.isnan(delta):
                by_dir[d].append(delta)

        rng   = np.random.default_rng(42)
        x_pos = np.arange(len(DIRECTIONS))
        nonempty = [(i, d, by_dir[d]) for i, d in enumerate(DIRECTIONS) if by_dir[d]]

        if nonempty:
            bp = ax_box.boxplot(
                [v for _, _, v in nonempty],
                positions=[x_pos[i] for i, _, _ in nonempty],
                widths=0.46, patch_artist=True,
                medianprops=dict(color=TEXT_COLOR, lw=2.2),
                whiskerprops=dict(color="#94a3b8", lw=1.2),
                capprops=dict(color="#94a3b8", lw=1.2),
                flierprops=dict(marker="", alpha=0),
            )
            for patch, (_, d, _) in zip(bp["boxes"], nonempty):
                patch.set_facecolor(DIR_COLORS[d]); patch.set_alpha(0.50)
            for i, d, vals in nonempty:
                jitter = rng.uniform(-0.14, 0.14, len(vals))
                colors_s = ["#16a34a" if v <= 0 else "#dc2626" for v in vals]
                ax_box.scatter(np.full(len(vals), x_pos[i]) + jitter, vals,
                               c=colors_s, s=44, alpha=0.85,
                               edgecolors="white", linewidth=0.5, zorder=4)

        ax_box.axhline(0, color=TEXT_COLOR, lw=1.6, ls="--", alpha=0.55, zorder=3)

        ylims  = ax_box.get_ylim()
        yr     = max(ylims[1] - ylims[0], 0.1)
        ax_box.text(len(DIRECTIONS) / 2 - 0.5, ylims[1] - yr * 0.06,
                    "cavity degrades  ▲", ha="center", fontsize=9,
                    color="#dc2626", style="italic", va="top")
        ax_box.text(len(DIRECTIONS) / 2 - 0.5, ylims[0] + yr * 0.06,
                    "▼  cavity improves", ha="center", fontsize=9,
                    color="#16a34a", style="italic", va="bottom")

        all_deltas = [
            post_r.get("epe_mean", np.nan) - pre_r.get("epe_mean", np.nan)
            for pre_r, post_r in pairs
            if not np.isnan(post_r.get("epe_mean", np.nan))
            and not np.isnan(pre_r.get("epe_mean", np.nan))
        ]
        if len(all_deltas) >= 5:
            try:
                _, p_val = scipy_wilcoxon(all_deltas, alternative="greater")
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax_box.text(0.98, 0.97,
                            f"Wilcoxon p={p_val:.3f} ({sig})\n(H₁: POST > PRE)",
                            transform=ax_box.transAxes, ha="right", va="top",
                            fontsize=8, color="#64748b",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      edgecolor="#cbd5e1", alpha=0.9))
            except Exception:
                pass

        ax_box.set_xticks(x_pos)
        ax_box.set_xticklabels([f"Shift {d}" for d in DIRECTIONS], fontsize=10)
        ax_box.set_ylabel("ΔEPE = EPE_POST − EPE_PRE  (mm)", fontsize=10)
        ax_box.set_title("Cavity Effect on Recovery Accuracy\n"
                         "(green = cavity did not degrade, red = degraded)",
                         fontsize=10, pad=8)
        ax_box.grid(axis="y", alpha=0.3)
        ax_box.spines[["top", "right", "bottom"]].set_visible(False)
    else:
        ax_box.text(0.5, 0.5, "No PRE/POST pairs found",
                    ha="center", va="center", transform=ax_box.transAxes,
                    fontsize=11, color="#94a3b8", style="italic")
        ax_box.axis("off")

    # ── Brain EPE map ─────────────────────────────────────────────────────────
    PILOT_ID  = "Pilot_A050_S250_X_C"
    brain_ok  = False

    if config is not None:
        # Check both standard output locations (DataManager searches these in order)
        _out_candidates = [
            config.PROJECT_ROOT / "RunBatch_Output" / PILOT_ID,
            config.PROJECT_ROOT / "outputs" / PILOT_ID,
        ]
        _out_dir = next((d for d in _out_candidates if d.exists()), None)
        pred_p = (_out_dir / "brainshift_displacement.nii.gz") if _out_dir else Path("/nonexistent")
        gt_p   = config.PROJECT_ROOT / "RunBatch_Input"  / PILOT_ID / "gt_displacement.nii.gz"
        pre_p  = config.PROJECT_ROOT / "RunBatch_Input"  / PILOT_ID / "0" / "t1.nii.gz"
        msk_p  = config.PROJECT_ROOT / "RunBatch_Input"  / PILOT_ID / "brain_mask.nii.gz"

        if pred_p.exists() and gt_p.exists():
            try:
                pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_p))).astype(np.float32)
                gt_arr   = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_p))).astype(np.float32)
                epe_arr  = np.linalg.norm(pred_arr + gt_arr, axis=-1)  # (Z,Y,X)

                pre_arr = None
                if pre_p.exists():
                    pre_arr = sitk.GetArrayFromImage(
                        sitk.Cast(sitk.ReadImage(str(pre_p)), sitk.sitkFloat32)
                    )

                mask_arr = None
                if msk_p.exists():
                    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_p))) > 0
                    if mask_arr.shape == epe_arr.shape:
                        epe_arr[~mask_arr] = np.nan

                # Locate peak GT displacement → centre slices there
                gt_mag = np.linalg.norm(gt_arr, axis=-1)
                gt_mag_m = gt_mag * (mask_arr.astype(float) if mask_arr is not None and mask_arr.shape == gt_mag.shape else 1)
                peak_idx = np.unravel_index(int(np.nanargmax(gt_mag_m)), gt_mag.shape)
                sz, sy, sx = peak_idx

                # ── Per-voxel angular error ─────────────────────────────────
                eps_v = 1e-9
                pred_mag_v = np.linalg.norm(pred_arr, axis=-1) + eps_v
                gt_mag_arr = np.linalg.norm(gt_arr,   axis=-1).astype(np.float32)
                gt_mag_v   = gt_mag_arr + eps_v

                dot_prod = np.sum(pred_arr * (-gt_arr), axis=-1)  # pred · (-gt)
                cos_ang  = np.clip(dot_prod / (pred_mag_v * gt_mag_v), -1.0, 1.0)
                ang_arr  = np.degrees(np.arccos(cos_ang)).astype(np.float32)

                # Weighted per voxel: ang × EPE  (°·mm)
                weighted_arr = ang_arr * epe_arr

                # Mask: brain + GT > threshold (suppress peripheral near-zero noise)
                valid = (
                    (mask_arr.astype(bool) if mask_arr is not None and mask_arr.shape == gt_mag_arr.shape
                     else np.ones_like(gt_mag_arr, dtype=bool))
                    & (gt_mag_arr > GT_SIG_THRESHOLD_MM)
                )
                gt_m  = np.where(valid, gt_mag_arr,   np.nan)
                ang_m = np.where(valid, ang_arr,       np.nan)
                w_m   = np.where(valid, weighted_arr,  np.nan)

                gt_vmax  = float(np.nanpercentile(gt_m,  99)) if np.any(~np.isnan(gt_m))  else 8.0
                ang_vmax = float(np.nanpercentile(ang_m, 99)) if np.any(~np.isnan(ang_m)) else 90.0
                w_vmax   = float(np.nanpercentile(w_m,   99)) if np.any(~np.isnan(w_m))   else 30.0

                def _bg(arr2d):
                    if arr2d is None: return None
                    lo, hi = np.percentile(arr2d[arr2d > 0], [1, 99]) if np.any(arr2d > 0) else (0, 1)
                    return np.clip((arr2d - lo) / (hi - lo + eps_v), 0, 1)

                bg_sl = _bg(pre_arr[sz, :, :] if pre_arr is not None else None)

                def _panel(ax, ov2d, cmap, vmax, title):
                    if bg_sl is not None:
                        ax.imshow(np.flipud(bg_sl), cmap="gray", vmin=0, vmax=1,
                                  interpolation="bilinear")
                    ax.imshow(np.flipud(ov2d), cmap=cmap, vmin=0, vmax=vmax,
                              alpha=0.82, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title(title, fontsize=8.5, pad=4, color=TEXT_COLOR)

                _panel(ax_axl, gt_m[sz, :, :],  "viridis", gt_vmax,
                       f"‖GT‖  (mm)\n{PILOT_ID} · z={sz}")
                _panel(ax_cor, ang_m[sz, :, :], "YlOrRd",  ang_vmax,
                       f"Angular Error  (°)\n(angle pred vs −GT)")
                _panel(ax_sag, w_m[sz, :, :],   "hot",     w_vmax,
                       f"ang × EPE  (°·mm)\n(weighted importance)")

                sm_b = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=0, vmax=w_vmax))
                sm_b.set_array([])
                cb_b = fig.colorbar(sm_b, cax=ax_cb)
                cb_b.set_label("ang × EPE  (°·mm)", fontsize=8, rotation=90, labelpad=6)
                cb_b.ax.tick_params(labelsize=7)

                brain_ok = True
            except Exception as e:
                logger.warning(f"Brain EPE map failed: {e}")

    if not brain_ok:
        for ax_s in (ax_axl, ax_cor, ax_sag, ax_cb):
            ax_s.axis("off")
        ax_cor.text(
            0.5, 0.5,
            "Brain EPE map not available.\nRe-run pilot case first:\n\n"
            "python src/qc/create_synthetic_patients.py\n"
            "python main.py --register-tissue --analyze \\\n"
            f"       --patient {PILOT_ID}",
            ha="center", va="center", transform=ax_cor.transAxes,
            fontsize=9, color="#64748b", style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc",
                      edgecolor="#cbd5e1", alpha=0.9),
        )

    # ── Global title & footer ─────────────────────────────────────────────────
    fig.suptitle(
        "Angular Error Analysis — Weighted by EPE + Cavity Effect + Per-Voxel Brain Map",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.005,
    )
    fig.text(
        0.5, 0.985,
        "Heatmap weight: ang_mean × epe_mean  (°·mm)  —  EPE = ‖pred + gt‖, the residual magnitude; "
        "well-recovered cases (small EPE) contribute near-zero even when angular error is high",
        ha="center", fontsize=9, color="#64748b", style="italic",
    )
    _add_metric_glossary(fig, y=-0.04)
    plt.savefig(output_dir / "angular_weighted_analysis.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: angular_weighted_analysis.png")


def plot_angular_voxelwise_analysis(results: dict, output_dir: Path, config=None) -> None:
    """
    Figure: angular_voxelwise_analysis.png

    Per-voxel scatter + box plots from Pilot_A050_S250_X_C showing:
      Row 1 — Raw angular error vs ‖GT‖: noise floor visible at small displacements
      Row 2 — Weighted metric (ang × ‖GT‖): noise suppressed, relevant error highlighted

    Justifies why raw ang_mean cannot be reported as-is in the thesis.
    """
    import SimpleITK as sitk

    if config is None:
        logger.info("No config — skipping angular_voxelwise_analysis.png")
        return

    PILOT_ID = "Pilot_A050_S250_X_C"
    _out_candidates = [
        config.PROJECT_ROOT / "RunBatch_Output" / PILOT_ID,
        config.PROJECT_ROOT / "outputs" / PILOT_ID,
    ]
    _out_dir = next((d for d in _out_candidates if d.exists()), None)
    pred_p = (_out_dir / "brainshift_displacement.nii.gz") if _out_dir else Path("/nonexistent")
    gt_p   = config.PROJECT_ROOT / "RunBatch_Input" / PILOT_ID / "gt_displacement.nii.gz"
    msk_p  = config.PROJECT_ROOT / "RunBatch_Input" / PILOT_ID / "brain_mask.nii.gz"
    pre_dir = config.PROJECT_ROOT / "RunBatch_Input" / PILOT_ID / "0"
    pre_niftis = sorted(pre_dir.glob("*.nii*")) if pre_dir.exists() else []

    if not pred_p.exists() or not gt_p.exists():
        logger.warning("angular_voxelwise_analysis: pilot displacement files not found — skipping.\n"
                       f"  Run: python main.py --register-tissue --analyze --patient {PILOT_ID}")
        return

    try:
        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_p))).astype(np.float32)
        gt_arr   = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_p))).astype(np.float32)
        mask_arr = (sitk.GetArrayFromImage(sitk.ReadImage(str(msk_p))) > 0
                    if msk_p.exists() else None)
        pre_arr  = (sitk.GetArrayFromImage(
                        sitk.Cast(sitk.ReadImage(str(pre_niftis[0])), sitk.sitkFloat32))
                    if pre_niftis else None)
    except Exception as e:
        logger.warning(f"angular_voxelwise_analysis: load error: {e}")
        return

    eps = 1e-9

    gt_mag   = np.linalg.norm(gt_arr,   axis=-1)          # (Z,Y,X)
    pred_mag = np.linalg.norm(pred_arr, axis=-1) + eps

    # Angular error: angle between pred and −GT (0° = perfect recovery)
    dot_prod = np.sum(pred_arr * (-gt_arr), axis=-1)
    cos_ang  = np.clip(dot_prod / (pred_mag * (gt_mag + eps)), -1.0, 1.0)
    ang      = np.degrees(np.arccos(cos_ang))              # (Z,Y,X)

    # Valid mask: inside brain AND GT above noise threshold
    valid = (
        (mask_arr if mask_arr is not None and mask_arr.shape == gt_mag.shape
         else np.ones_like(gt_mag, dtype=bool))
        & (gt_mag > GT_SIG_THRESHOLD_MM)
    )

    # Full spatial arrays (NaN outside valid region) — used for brain maps
    ang_full = np.where(valid, ang,          np.nan)
    w1_full  = np.where(valid, ang * gt_mag, np.nan)   # ang × ‖GT‖  (°·mm)

    # 1D arrays for scatter / box plots
    gt_v  = gt_mag[valid]
    ang_v = ang[valid]
    w1_v  = (ang * gt_mag)[valid]

    if len(gt_v) == 0:
        logger.warning("angular_voxelwise_analysis: no valid voxels found — skipping")
        return

    # GT magnitude bins for box plots
    gt_max  = float(np.percentile(gt_v, 99))
    bins    = [GT_SIG_THRESHOLD_MM, 1.0, 2.5, 5.0, max(gt_max * 1.01, 5.1)]
    blabels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    bidx    = np.clip(np.digitize(gt_v, bins) - 1, 0, len(bins) - 2)
    n_bins  = len(bins) - 1

    grouped_ang = [ang_v[bidx == k] for k in range(n_bins)]
    grouped_w1  = [w1_v[bidx == k]  for k in range(n_bins)]

    # Axial slice at peak GT displacement
    sz = int(np.unravel_index(int(np.where(valid, gt_mag, 0.0).argmax()), gt_mag.shape)[0])

    ang_vmax = float(np.nanpercentile(ang_full, 99)) if np.any(~np.isnan(ang_full)) else 90.0
    w1_vmax  = float(np.nanpercentile(w1_full,  99)) if np.any(~np.isnan(w1_full))  else 30.0

    BIN_COLORS = ["#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]

    # ── Layout ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor=FACE_COLOR)
    gs  = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.50,
                   width_ratios=[1.7, 1.4, 1.0])

    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    def _bg_norm(arr2d):
        if arr2d is None:
            return None
        lo, hi = (np.percentile(arr2d[arr2d > 0], [1, 99])
                  if np.any(arr2d > 0) else (0, 1))
        return np.clip((arr2d - lo) / (hi - lo + eps), 0, 1)

    bg_sl = _bg_norm(pre_arr[sz, :, :] if pre_arr is not None else None)

    def _scatter(ax, yv, ylabel, ylim, title):
        hb = ax.hexbin(gt_v, yv, gridsize=55, cmap="YlOrRd",
                       mincnt=1, bins="log", linewidths=0.2)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("log₁₀(voxel count)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax.set_xlabel("‖GT‖  (mm)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, ylim)
        ax.set_title(title, fontsize=10, pad=6)
        for xv in [1.0, 2.5, 5.0]:
            if xv < gt_v.max():
                ax.axvline(xv, color="#94a3b8", lw=0.8, ls="--", alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    def _boxplot(ax, groups, ylabel, title):
        nonempty = [(l, g) for l, g in zip(blabels, groups) if len(g) > 1]
        if not nonempty:
            ax.axis("off"); return
        bps = ax.boxplot(
            [g for _, g in nonempty],
            tick_labels=[l for l, _ in nonempty],
            patch_artist=True, widths=0.55,
            medianprops=dict(color=TEXT_COLOR, lw=2.2),
            whiskerprops=dict(color="#94a3b8", lw=1.1),
            capprops=dict(color="#94a3b8", lw=1.1),
            flierprops=dict(marker="o", markersize=2, alpha=0.25,
                            markerfacecolor="#94a3b8", markeredgecolor="none"),
        )
        for patch, c in zip(bps["boxes"], BIN_COLORS[:len(nonempty)]):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        for tick, (_, g) in enumerate(nonempty):
            ax.text(tick + 1, 0.97, f"n={len(g):,}",
                    ha="center", va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=6.5, color="#64748b")
        ax.set_xlabel("‖GT‖ bin (mm)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    def _brain(ax, sl2d, cmap, vmax, title):
        if bg_sl is not None:
            ax.imshow(np.flipud(bg_sl), cmap="gray", vmin=0, vmax=1,
                      interpolation="bilinear")
        ax.imshow(np.flipud(sl2d), cmap=cmap, vmin=0, vmax=vmax,
                  alpha=0.83, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=9, pad=4)

    # ── Row 1: raw angular error ──────────────────────────────────────────────
    _scatter(axes[0][0], ang_v, "Angular error (°)", ang_vmax * 1.08,
             "Raw ang° vs ‖GT‖\n(all valid voxels inside brain mask)")
    _boxplot(axes[0][1], grouped_ang, "Angular error (°)",
             "ang° distribution per ‖GT‖ bin\n(unweighted)")
    _brain(axes[0][2], ang_full[sz, :, :], "YlOrRd", ang_vmax,
           f"Raw ang°  axial z={sz}")

    # ── Row 2: weighted ang × ‖GT‖ ───────────────────────────────────────────
    _scatter(axes[1][0], w1_v, "ang × ‖GT‖  (°·mm)", w1_vmax * 1.08,
             "Weighted ang×‖GT‖ vs ‖GT‖\n(peripheral noise suppressed)")
    _boxplot(axes[1][1], grouped_w1, "ang × ‖GT‖  (°·mm)",
             "ang×‖GT‖ distribution per ‖GT‖ bin\n(weighted by displacement magnitude)")
    _brain(axes[1][2], w1_full[sz, :, :], "hot", w1_vmax,
           f"Weighted ang×‖GT‖  axial z={sz}")

    # Row labels
    for row, label, color in [
        (0, "Raw angular error (°)", "#dc2626"),
        (1, "Weighted  ang × ‖GT‖  (°·mm)", "#16a34a"),
    ]:
        axes[row][0].annotate(
            label, xy=(-0.22, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", color=color,
            rotation=90, ha="center", va="center",
        )

    fig.suptitle(
        f"Per-Voxel Angular Error Analysis — {PILOT_ID}  (A=5mm, σ=25mm, shift X)\n"
        "Row 1: Raw ang° inflated by near-zero GT voxels at brain periphery  •  "
        "Row 2: ang × ‖GT‖ suppresses that noise — only large & misdirected shifts stand out",
        fontsize=10.5, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    _add_metric_glossary(fig, y=-0.05)
    plt.savefig(output_dir / "angular_voxelwise_analysis.png",
                dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: angular_voxelwise_analysis.png")


def plot_angular_regional_categorical(results: dict, output_dir: Path, config=None) -> None:
    """
    Figure: angular_regional_categorical.png

    Cross-cases categorical analysis: angular error pooled across all sweep cases,
    grouped by shift direction × source_type (PRE/POST).

    Row 1 — Raw ang_mean (°)         per direction × PRE/POST
    Row 2 — Weighted ang_mean × A_mm per direction × PRE/POST

    Demonstrates:
      - Whether the registration handles all directions equally
      - Whether the cavity (POST) degrades certain directions more
      - That weighting by shift amplitude properly suppresses small-shift noise
        across categories (angular error weighted by displacement magnitude)

    When new patients (e.g. healthy brain) are added to the sweep, this figure
    automatically pools their cases — becoming a true cross-brain analysis.
    """
    pre_cases  = _ok_cases(results, "pre")
    post_cases = _ok_cases(results, "post")

    if not pre_cases:
        logger.info("No PRE cases — skipping angular_regional_categorical.png")
        return

    def _group_by_dir(cases):
        out = {d: [] for d in DIRECTIONS}
        for c in cases:
            d = c.get("direction", "X")
            if d in out:
                out[d].append(c)
        return out

    pre_g  = _group_by_dir(pre_cases)
    post_g = _group_by_dir(post_cases)
    pairs  = _pair_post_cases(results)
    pair_g = {d: [] for d in DIRECTIONS}
    for pre_r, post_r in pairs:
        d = pre_r.get("direction", "X")
        if d in pair_g:
            pair_g[d].append((pre_r, post_r))

    # Detect distinct base brains (count distinct prefixes before _A###)
    base_brains = set()
    for c in pre_cases + post_cases:
        nm = c.get("name", "")
        if "_A" in nm:
            base_brains.add(nm.split("_A")[0])
    n_brains = max(len(base_brains), 1)

    fig, axes = plt.subplots(2, 4, figsize=(17, 9.5),
                             facecolor=FACE_COLOR, sharey="row")
    rng = np.random.default_rng(7)

    metrics = [
        ("raw",      "ang_mean (°)",          "#3b82f6", "#ea580c"),
        ("weighted", "ang × A_mm  (°·mm)",    "#3b82f6", "#ea580c"),
    ]

    def _extract(case_list, mode):
        vals = []
        for c in case_list:
            a = c.get("ang_mean", np.nan)
            if mode == "weighted":
                a = a * c.get("A_mm", 1.0)
            if not np.isnan(a):
                vals.append(a)
        return np.array(vals)

    for row, (mode, ylabel, c_pre, c_post) in enumerate(metrics):
        for col, d in enumerate(DIRECTIONS):
            ax = axes[row, col]

            v_pre  = _extract(pre_g[d],  mode)
            v_post = _extract(post_g[d], mode)

            data, labels, colors = [], [], []
            if len(v_pre) > 0:
                data.append(v_pre);  labels.append("PRE");  colors.append(c_pre)
            if len(v_post) > 0:
                data.append(v_post); labels.append("POST"); colors.append(c_post)
            if not data:
                ax.axis("off"); continue

            bp = ax.boxplot(
                data, tick_labels=labels, widths=0.55,
                patch_artist=True,
                medianprops=dict(color=TEXT_COLOR, lw=2.0),
                whiskerprops=dict(color="#94a3b8", lw=1.0),
                capprops=dict(color="#94a3b8", lw=1.0),
                flierprops=dict(marker="", alpha=0),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.55)

            for k, (vals, c) in enumerate(zip(data, colors)):
                jitter = rng.uniform(-0.16, 0.16, len(vals))
                ax.scatter(np.full(len(vals), k + 1) + jitter, vals,
                           c=c, s=28, alpha=0.78,
                           edgecolors="white", linewidth=0.4, zorder=4)

            for k, vals in enumerate(data):
                ax.text(k + 1, 0.97, f"n={len(vals)}",
                        ha="center", va="top",
                        transform=ax.get_xaxis_transform(),
                        fontsize=7.5, color="#64748b")

            # Paired Wilcoxon PRE vs POST per direction (uses paired cases)
            if len(pair_g[d]) >= 5:
                v1 = np.array([_extract([pr], mode)[0] if len(_extract([pr], mode)) else np.nan
                               for pr, _ in pair_g[d]])
                v2 = np.array([_extract([po], mode)[0] if len(_extract([po], mode)) else np.nan
                               for _, po in pair_g[d]])
                msk = ~(np.isnan(v1) | np.isnan(v2))
                if msk.sum() >= 5:
                    try:
                        _, p_val = scipy_wilcoxon(v2[msk], v1[msk], alternative="two-sided")
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        ax.text(0.5, 1.04, f"Wilcoxon (paired) p={p_val:.3f} ({sig})",
                                transform=ax.transAxes, ha="center", va="bottom",
                                fontsize=7.5, color="#475569")
                    except Exception:
                        pass

            ax.set_title(f"Shift {d}", fontsize=11, fontweight="bold",
                         color=DIR_COLORS[d], pad=18)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

    for row, label, color in [
        (0, "Raw  ang°",                      "#dc2626"),
        (1, "Weighted  ang × A_mm  (°·mm)",   "#16a34a"),
    ]:
        axes[row][0].annotate(
            label, xy=(-0.28, 0.5), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color=color,
            rotation=90, ha="center", va="center",
        )

    n_total = len(pre_cases) + len(post_cases)
    brains_label = (f"{n_brains} brain" if n_brains == 1
                    else f"{n_brains} brains  —  cross-brain")
    fig.suptitle(
        f"Cross-Cases Categorical Analysis — {n_total} sweep cases  "
        f"(PRE={len(pre_cases)}, POST={len(post_cases)}, {brains_label})\n"
        "Top: raw angular error per shift direction  •  "
        "Bottom: weighted by shift amplitude (captures clinical relevance)",
        fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )
    _add_metric_glossary(fig, y=-0.03)
    plt.savefig(output_dir / "angular_regional_categorical.png",
                dpi=180, bbox_inches="tight")
    plt.close()
    logger.success("Saved: angular_regional_categorical.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("--no-per-test", action="store_true",
                        help="Skip per-test individual reports")
    args = parser.parse_args()

    config       = load_config()
    results_path = config.PROJECT_ROOT / "RunBatch_Output" / "sweep_results.json"
    if not results_path.exists():
        logger.error(f"sweep_results.json not found: {results_path}")
        return

    output_dir = config.PROJECT_ROOT / "RunBatch_Output" / "sweep_figures"
    output_dir.mkdir(exist_ok=True)
    results = load_results(results_path)

    if not args.no_per_test:
        logger.info("Generating per-test reports...")
        for name, r in results.items():
            if isinstance(r, dict) and r.get("status") == "OK":
                generate_per_test_report(name, r, output_dir, all_results=results)

    # Remove stale aggregate_overview.png from old code version
    stale = output_dir / "aggregate_overview.png"
    if stale.exists():
        stale.unlink()
        logger.info("Removed stale aggregate_overview.png")

    logger.info("Generating global figures...")
    plot_heatmap_dashboard(results, output_dir)
    plot_heatmap_prepost(results, output_dir)
    plot_recovery_curves(results, output_dir)
    plot_location_bias(results, output_dir)
    plot_isotropy(results, output_dir)
    plot_system_summary(results, output_dir)
    plot_statistical_summary(results, output_dir)
    plot_cavity_effect(results, output_dir)
    plot_angular_weighted_analysis(results, output_dir, config=config)
    plot_angular_voxelwise_analysis(results, output_dir, config=config)
    plot_angular_regional_categorical(results, output_dir, config=config)
    save_csv(results, output_dir)
    print_summary(results)
    logger.success(f"All figures saved to: {output_dir}")

if __name__ == "__main__":
    main()
