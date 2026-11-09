"""
Regenerate the summary figures used in the README.

The numbers are the aggregate results reported in the thesis (no imaging data
is read or shown). Run from the repository root:

    python docs/scripts/make_readme_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "images"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def _bars(ax, labels, values, colors, fmt):
    bars = ax.bar(labels, values, color=colors, width=0.62,
                  edgecolor=SURFACE, linewidth=2)
    top = max(values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03, fmt.format(v),
                ha="center", va="bottom", color=INK, fontsize=9.5)
    ax.set_ylim(0, top * 1.22)
    _style(ax)


def cfm_effect():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=150)
    labels = ["Without CFM", "With CFM"]
    _bars(axes[0], labels, [0.96, 0.43], SERIES[:2], "{:.2f} mm")
    axes[0].set_title("Mean error inside the cavity")
    axes[0].set_ylabel("mm")
    _bars(axes[1], labels, [0.207, 0.002], SERIES[:2], "{:.3f} %")
    axes[1].set_title("Jacobian folding (det J ≤ 0)")
    axes[1].set_ylabel("% of brain voxels")
    fig.suptitle("Cost Function Masking on synthetic cases with a surgical cavity",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "cfm_effect.png")
    plt.close(fig)


def multibrain():
    groups = ["Site A\n(N=20)", "RHUH-GBM\n(N=30)", "Aggregate\n(N=50)"]
    panels = [
        ("Typical EPE", "mm", [0.46, 0.92, 0.74], "{:.2f}"),
        ("EPE, worst 5 %", "mm", [1.00, 3.09, 2.25], "{:.2f}"),
        ("Angular error", "degrees", [11.3, 18.1, 15.4], "{:.1f}°"),
        ("Jacobian folding", "% voxels", [0.001, 0.17, 0.10], "{:.3f} %"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), dpi=150)
    for ax, (title, unit, vals, fmt) in zip(axes, panels):
        _bars(ax, groups, vals, SERIES, fmt)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Multi-brain cross-validation — N = 50 synthetic cases, 2 scanners",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "multibrain_validation.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cfm_effect()
    multibrain()
    print(f"Figures written to {OUT}")
