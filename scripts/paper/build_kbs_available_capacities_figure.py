"""Build an available-capacity replay figure from the available-capacities trend CSV.

This is a separate, explicitly-named companion to
``build_kbs_policy_trend_artifacts.py``. It does NOT touch Figure 2
(``figure2_main_performance_comparison``) or Figure 3
(``figure3_improvement_vs_lru``), which remain gated on the single
capacity-blind canonical CSV (``analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv``)
that does not exist yet. Those figures average across whatever rows they
are given and have no capacity axis, so reusing them here would silently
average over capacities and hide the cap128 anomaly this figure exists to
show.

Reads:
    analysis/kbs_policy_trend_available_capacities.csv
    (built by build_kbs_policy_trend_artifacts.py)

Writes (capacity-explicit; never the canonical figure2/figure3):
    figures/manuscript/figure_available_capacities_trend.pdf
    figures/manuscript/figure_available_capacities_trend.png
    manuscript_source/figures/figure_available_capacities_trend.png
    reports/manuscript_artifacts/latex_snippets/figure_available_capacities_trend_snippet.tex
"""

from __future__ import annotations

import csv
from pathlib import Path
import shutil

import matplotlib.pyplot as plt

from manuscript_figure_common import apply_manuscript_matplotlib_style, save_figure_pdf_png

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = REPO_ROOT / "analysis" / "kbs_policy_trend_available_capacities.csv"
FIGURES_DIR = REPO_ROOT / "figures" / "manuscript"
SNIPPET_PATH = (
    REPO_ROOT
    / "reports"
    / "manuscript_artifacts"
    / "latex_snippets"
    / "figure_available_capacities_trend_snippet.tex"
)
MANUSCRIPT_FIGURE_PATH = (
    REPO_ROOT / "manuscript_source" / "figures" / "figure_available_capacities_trend.png"
)
STEM = "figure_available_capacities_trend"

POLICY_ORDER = [
    "lru",
    "sieve",
    "fifo_reinsertion",
    "predictive_marker",
    "blind_oracle_lru_combiner",
    "trust_and_doubt",
    "rest_v1",
    "evict_value_v1",
]
HIGHLIGHT_POLICY = "evict_value_v1"
GAP_BASELINES = ["lru", "sieve", "fifo_reinsertion"]


def load_rows() -> list[dict[str, str]]:
    if not IN_CSV.exists():
        raise SystemExit(
            f"{IN_CSV} not found — run build_kbs_policy_trend_artifacts.py first."
        )
    with IN_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    rows = load_rows()
    capacities = sorted({r["capacity"] for r in rows}, key=int)

    mean_misses: dict[tuple[str, str], float] = {}
    gap_vs_lru: dict[tuple[str, str], float] = {}
    for r in rows:
        key = (r["capacity"], r["policy"])
        mean_misses[key] = float(r["mean_misses"])
        if r["rel_gap_vs_lru_pct"] != "":
            gap_vs_lru[key] = float(r["rel_gap_vs_lru_pct"])

    policies = [p for p in POLICY_ORDER if any((c, p) in mean_misses for c in capacities)]

    apply_manuscript_matplotlib_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.7))

    x = [int(c) for c in capacities]
    grays = ["#1a1a1a", "#3d3d3d", "#5c5c5c", "#7a7a7a", "#999999", "#b5b5b5", "#0b3d91"]
    for i, pol in enumerate(policies):
        ys = [mean_misses.get((c, pol)) for c in capacities]
        is_highlight = pol == HIGHLIGHT_POLICY
        ax0.plot(
            x,
            ys,
            marker="o",
            markersize=5,
            linewidth=2.0 if is_highlight else 1.2,
            color="#0b3d91" if is_highlight else grays[i % len(grays)],
            label=pol.replace("_", " "),
            zorder=5 if is_highlight else 3,
        )
    ax0.set_xticks(x)
    ax0.set_xlabel("Cache capacity (slots)")
    ax0.set_ylabel("Mean replay misses across 7 trace families\n(lower is better)")
    ax0.set_title("(a) Mean misses by capacity", fontsize=10.5, pad=8)
    ax0.grid(True, linestyle=":", linewidth=0.6, alpha=0.85)
    ax0.legend(fontsize=6.6, ncol=2, frameon=False, loc="upper right")

    # evict_value_v1 gap vs each baseline, computed directly from mean_misses
    # (the trend CSV's rel_gap_vs_lru_pct column only covers the LRU baseline).
    for baseline in GAP_BASELINES:
        ys = []
        for c in capacities:
            num = mean_misses.get((c, HIGHLIGHT_POLICY))
            den = mean_misses.get((c, baseline))
            ys.append(100.0 * (num - den) / den if (num is not None and den) else None)
        ax1.plot(
            x,
            ys,
            marker="s",
            markersize=5,
            linewidth=1.8,
            label=f"vs {baseline.replace('_', ' ')}",
        )
    ax1.axhline(0.0, color="0.2", linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xlabel("Cache capacity (slots)")
    ax1.set_ylabel("evict\\_value\\_v1 gap (%)\n(positive = more misses, i.e.\\ worse)")
    ax1.set_title("(b) evict_value_v1 gap vs baselines", fontsize=10.5, pad=8)
    ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.85)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.suptitle(
        "Available-capacity replay only (capacities 32, 64, 128; cap256 not evaluated)",
        fontsize=8.5,
        color="#8a1f1f",
        y=1.02,
    )
    fig.tight_layout()

    pdf_path, png_path = save_figure_pdf_png(fig, FIGURES_DIR, STEM)
    MANUSCRIPT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png_path, MANUSCRIPT_FIGURE_PATH)

    SNIPPET_PATH.parent.mkdir(parents=True, exist_ok=True)
    snippet = (
        "% Available-capacity replay snippet -- capacities evaluated: "
        + ", ".join(capacities)
        + ". "
        "cap256 NOT run; do not cite this figure as covering cap256.\n"
        "% This is a capacity-explicit companion figure, not Figure 2 / Figure 3.\n"
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{figures/manuscript/{STEM}.pdf}}\n"
        "  \\caption{Capacity-trend comparison across the available-capacity replay "
        "(capacities 32, 64, and 128; cap256 not evaluated in this revision). "
        "(a) Mean replay misses per policy at each capacity, averaged across the 7 trace "
        "families. (b) \\texttt{evict\\_value\\_v1}'s relative miss gap against LRU, SIEVE, "
        "and FIFO-Reinsertion at each capacity; positive values indicate more misses than "
        "the baseline. The non-monotonic widening at capacity 128 is discussed in the "
        "Limitations section.}\n"
        f"  \\label{{fig:available-capacities-trend}}\n"
        "\\end{figure}\n"
    )
    SNIPPET_PATH.write_text(snippet)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {MANUSCRIPT_FIGURE_PATH}")
    print(f"Wrote {SNIPPET_PATH}")
    print(f"Capacities plotted: {capacities} (cap256 NOT included)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
