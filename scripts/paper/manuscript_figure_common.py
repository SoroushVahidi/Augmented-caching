"""Shared matplotlib figures for KBS manuscript (vector PDF + PNG).

Used by build_kbs_main_manuscript_artifacts.py and regenerate_evidence_aligned_manuscript_figures.py.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def apply_manuscript_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "sans-serif",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _box_axes(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    txt: str,
    *,
    fs: float = 7.0,
    face: str = "#f2f2f2",
    edge: str = "#222222",
    wrap: int = 26,
) -> None:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(p)
    display = textwrap.fill(txt, wrap)
    ax.text(
        x + w / 2,
        y + h / 2,
        display,
        ha="center",
        va="center",
        fontsize=fs,
        color="#111111",
        transform=ax.transAxes,
    )


def _arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.05,
            color="#222222",
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def _stack_boxes_vertical(
    ax: plt.Axes,
    labels: Tuple[str, ...],
    *,
    x: float,
    w: float,
    y_lo: float,
    y_hi: float,
    gap: float,
    fs: float,
    wrap: int,
    face: str = "#f2f2f2",
) -> None:
    """Equal-height boxes with vertical arrows (transAxes coordinates)."""
    n = len(labels)
    if n == 0:
        return
    total_gap = gap * max(0, n - 1)
    bh = (y_hi - y_lo - total_gap) / n
    y = y_lo
    xc = x + w / 2
    for i, lab in enumerate(labels):
        _box_axes(ax, x, y, w, bh, lab, fs=fs, face=face, wrap=wrap)
        if i < n - 1:
            _arrow(ax, xc, y + bh, xc, y + bh + gap)
        y += bh + gap


def make_method_overview_two_panel_figure() -> plt.Figure:
    """Two-panel method figure: offline supervised construction + online guarded deployment.

    Short panel titles only; full wording belongs in the LaTeX caption. Box labels are minimal schematic text.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.2, 5.45), constrained_layout=False)
    for ax in (ax_a, ax_b):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Short titles — manuscript describes panels fully in the caption.
    title_fs, box_fs, wrap = 10.0, 10.25, 22
    ax_a.set_title("(a) Offline training", fontsize=title_fs, fontweight="bold", pad=8, color="#111111")
    ax_b.set_title("(b) Online deployment", fontsize=title_fs, fontweight="bold", pad=8, color="#111111")

    bx, bw, gap = 0.10, 0.80, 0.024
    _stack_boxes_vertical(
        ax_a,
        (
            "Trace replay",
            "Full-cache miss labeling",
            "Candidate feature extraction",
            "Counterfactual target construction",
            "Fit eviction-value model",
        ),
        x=bx,
        w=bw,
        y_lo=0.08,
        y_hi=0.91,
        gap=gap,
        fs=box_fs,
        wrap=wrap,
    )

    _stack_boxes_vertical(
        ax_b,
        (
            "Request arrives",
            "Hit: update state",
            "Miss, not full: insert",
            "Miss, full: score candidates",
            "Guarded fallback if unsafe",
            "Evict and update",
        ),
        x=bx,
        w=bw,
        y_lo=0.07,
        y_hi=0.91,
        gap=gap,
        fs=box_fs,
        wrap=wrap,
        face="#ededed",
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.06, wspace=0.28)
    return fig


# Main manuscript comparison policies — same roster and order as Table~3 in build_kbs_main_manuscript_artifacts.py.
MAIN_PERF_POLICIES: Tuple[str, ...] = (
    "lru",
    "predictive_marker",
    "trust_and_doubt",
    "blind_oracle_lru_combiner",
    "rest_v1",
    "evict_value_v1",
)

DELTA_POLICIES: Tuple[str, ...] = (
    "evict_value_v1",
    "predictive_marker",
    "trust_and_doubt",
    "blind_oracle_lru_combiner",
    "rest_v1",
)


def _family_policy_means(policy_rows: List[Dict[str, str]], policies: Sequence[str]) -> Tuple[List[str], Dict[Tuple[str, str], float]]:
    raw: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in policy_rows:
        p = str(r["policy"])
        if p not in policies:
            continue
        fam = str(r["trace_family"])
        raw[(fam, p)].append(float(r["misses"]))
    fams = sorted({k[0] for k in raw})
    means: Dict[Tuple[str, str], float] = {}
    for k, vs in raw.items():
        means[k] = mean(vs)
    return fams, means


def make_main_performance_comparison_figure(policy_rows: List[Dict[str, str]]) -> plt.Figure:
    """Grouped bars by trace family; mean replay misses (lower better)."""
    fams, means = _family_policy_means(policy_rows, MAIN_PERF_POLICIES)
    n_f = len(fams)
    n_p = len(MAIN_PERF_POLICIES)
    x = np.arange(n_f, dtype=float)
    width = min(0.85 / max(n_p, 1), 0.13)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    grays = ["#1a1a1a", "#3d3d3d", "#5c5c5c", "#7a7a7a", "#999999", "#b5b5b5"]

    for i, pol in enumerate(MAIN_PERF_POLICIES):
        ys = [means.get((f, pol), np.nan) for f in fams]
        offset = (i - (n_p - 1) / 2.0) * width
        ec = "0.15" if pol != "evict_value_v1" else "#0b3d91"
        lw = 0.9 if pol != "evict_value_v1" else 1.5
        ax.bar(
            x + offset,
            ys,
            width=width * 0.92,
            label=pol.replace("_", " "),
            color=grays[i % len(grays)],
            edgecolor=ec,
            linewidth=lw,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ") for f in fams], fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Mean replay misses (lower is better)")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", linestyle=":", linewidth=0.65, alpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def make_improvement_vs_lru_figure(policy_rows: List[Dict[str, str]]) -> plt.Figure:
    """Δ mean misses = mean(policy) − mean(LRU) per family (negative ⇒ fewer misses than LRU)."""
    fams, means = _family_policy_means(policy_rows, set(MAIN_PERF_POLICIES))
    lru = {f: means[(f, "lru")] for f in fams}
    n_f = len(fams)
    n_p = len(DELTA_POLICIES)
    x = np.arange(n_f, dtype=float)
    width = min(0.88 / max(n_p, 1), 0.15)
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    grays = ["#252525", "#4d4d4d", "#737373", "#9e9e9e", "#c4c4c4"]
    for i, pol in enumerate(DELTA_POLICIES):
        deltas = [means[(f, pol)] - lru[f] for f in fams]
        offset = (i - (n_p - 1) / 2.0) * width
        ec = "0.2"
        lw = 0.85
        if pol == "evict_value_v1":
            ec = "#0b3d91"
            lw = 1.55
        ax.bar(
            x + offset,
            deltas,
            width=width * 0.9,
            label=pol.replace("_", " "),
            color=grays[i % len(grays)],
            edgecolor=ec,
            linewidth=lw,
        )
    ax.axhline(0.0, color="0.2", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ") for f in fams], fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Δ mean misses vs LRU (negative ⇒ fewer misses than LRU)")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    ax.grid(axis="y", linestyle=":", linewidth=0.65, alpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return fig


def make_offline_ablation_figure(train_rows: List[Dict[str, str]]) -> plt.Figure:
    """Two panels: validation vs test mean regret vs horizon (canonical model_comparison rows; lower better)."""
    by_val: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    by_test: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    ref: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for r in train_rows:
        h = int(r["horizon"])
        m = str(r["model"])
        vr = float(r["val_mean_regret"])
        tr = float(r["test_mean_regret"])
        by_val[m].append((h, vr))
        by_test[m].append((h, tr))
        ref[(h, m)] = (vr, tr)
    for m in by_val:
        by_val[m] = sorted(by_val[m], key=lambda t: t[0])
    for m in by_test:
        by_test[m] = sorted(by_test[m], key=lambda t: t[0])

    for m, pts in by_val.items():
        for h, v in pts:
            if not np.isclose(v, ref[(h, m)][0], rtol=0.0, atol=1e-12):
                raise ValueError(f"internal val_mean_regret mismatch for {m=} {h=}")
    for m, pts in by_test.items():
        for h, v in pts:
            if not np.isclose(v, ref[(h, m)][1], rtol=0.0, atol=1e-12):
                raise ValueError(f"internal test_mean_regret mismatch for {m=} {h=}")

    # Distinct line styles + markers for grayscale print; white marker edges improve separation.
    styles = {"ridge": ("-", "o"), "random_forest": ("--", "s"), "hist_gb": ("-.", "^")}
    colors = {"ridge": "#000000", "random_forest": "#555555", "hist_gb": "#222222"}
    zorder = {"ridge": 2, "random_forest": 3, "hist_gb": 4}
    plot_order = ("ridge", "random_forest", "hist_gb")
    model_labels = {"ridge": "Ridge", "random_forest": "Random forest", "hist_gb": r"Hist.\ GB"}

    # Shared legend below panels only — no figure suptitle; captions live in LaTeX.
    fig = plt.figure(figsize=(11.2, 4.7))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.24],
        hspace=0.22,
        left=0.085,
        right=0.985,
        top=0.93,
        bottom=0.11,
    )
    gs_panels = gs[0].subgridspec(1, 2, wspace=0.36)
    ax0 = fig.add_subplot(gs_panels[0, 0])
    ax1 = fig.add_subplot(gs_panels[0, 1], sharex=ax0)
    axes = (ax0, ax1)

    def _panel_label(ax: plt.Axes, tag: str) -> None:
        ax.text(
            0.03,
            0.97,
            tag,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="top",
            ha="left",
            color="#111111",
            zorder=15,
        )

    for ax, data, title, tag in (
        (axes[0], by_val, "Validation regret", "(a)"),
        (axes[1], by_test, "Test regret", "(b)"),
    ):
        for m in plot_order:
            if m not in data:
                continue
            xs = [a for a, _ in data[m]]
            ys = [b for _, b in data[m]]
            ls, mk = styles[m]
            ax.plot(
                xs,
                ys,
                linestyle=ls,
                marker=mk,
                color=colors[m],
                label=model_labels[m],
                linewidth=1.65,
                markersize=6.5,
                markeredgecolor="0.95",
                markeredgewidth=0.6,
                zorder=zorder[m],
                clip_on=False,
            )
        ax.set_xticks([4, 8, 16])
        ax.set_xlabel(r"Horizon $H$", fontsize=10)
        ax.set_ylabel("Mean regret vs.\ oracle (lower is better)", fontsize=10)
        ax.set_title(title, fontsize=10, pad=6)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.65, alpha=0.88, color="0.45")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.45, alpha=0.5, color="0.75")
        ax.set_axisbelow(True)
        _panel_label(ax, tag)

    handles, labels = ax0.get_legend_handles_labels()
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.set_axis_off()
    ax_leg.patch.set_alpha(0)
    leg = ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.45",
        facecolor="0.98",
        fontsize=9,
        title="Model family",
        title_fontsize=9,
        columnspacing=1.5,
        handlelength=2.8,
        handletextpad=0.55,
        borderpad=0.5,
        labelspacing=0.4,
    )
    leg.get_frame().set_linewidth(0.55)

    return fig


def make_offline_top1_ablation_figure(train_rows: List[Dict[str, str]]) -> plt.Figure:
    """Two panels: validation vs test Top-1 error vs horizon (same CSV as regret ablation; lower better)."""
    by_val: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    by_test: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    ref: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for r in train_rows:
        h = int(r["horizon"])
        m = str(r["model"])
        v1 = float(r["val_top1"])
        t1 = float(r["test_top1"])
        by_val[m].append((h, v1))
        by_test[m].append((h, t1))
        ref[(h, m)] = (v1, t1)
    for m in by_val:
        by_val[m] = sorted(by_val[m], key=lambda t: t[0])
    for m in by_test:
        by_test[m] = sorted(by_test[m], key=lambda t: t[0])

    for m, pts in by_val.items():
        for h, v in pts:
            if not np.isclose(v, ref[(h, m)][0], rtol=0.0, atol=1e-12):
                raise ValueError(f"internal val_top1 mismatch for {m=} {h=}")
    for m, pts in by_test.items():
        for h, v in pts:
            if not np.isclose(v, ref[(h, m)][1], rtol=0.0, atol=1e-12):
                raise ValueError(f"internal test_top1 mismatch for {m=} {h=}")

    styles = {"ridge": ("-", "o"), "random_forest": ("--", "s"), "hist_gb": ("-.", "^")}
    colors = {"ridge": "#000000", "random_forest": "#555555", "hist_gb": "#222222"}
    zorder = {"ridge": 2, "random_forest": 3, "hist_gb": 4}
    plot_order = ("ridge", "random_forest", "hist_gb")
    model_labels = {"ridge": "Ridge", "random_forest": "Random forest", "hist_gb": r"Hist.\ GB"}

    fig = plt.figure(figsize=(11.2, 4.7))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.24],
        hspace=0.22,
        left=0.085,
        right=0.985,
        top=0.93,
        bottom=0.11,
    )
    gs_panels = gs[0].subgridspec(1, 2, wspace=0.36)
    ax0 = fig.add_subplot(gs_panels[0, 0])
    ax1 = fig.add_subplot(gs_panels[0, 1], sharex=ax0)
    axes = (ax0, ax1)

    def _panel_label(ax: plt.Axes, tag: str) -> None:
        ax.text(
            0.03,
            0.97,
            tag,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="top",
            ha="left",
            color="#111111",
            zorder=15,
        )

    for ax, data, title, tag in (
        (axes[0], by_val, "Validation Top-1", "(a)"),
        (axes[1], by_test, "Test Top-1", "(b)"),
    ):
        for m in plot_order:
            if m not in data:
                continue
            xs = [a for a, _ in data[m]]
            ys = [b for _, b in data[m]]
            ls, mk = styles[m]
            ax.plot(
                xs,
                ys,
                linestyle=ls,
                marker=mk,
                color=colors[m],
                label=model_labels[m],
                linewidth=1.65,
                markersize=6.5,
                markeredgecolor="0.95",
                markeredgewidth=0.6,
                zorder=zorder[m],
                clip_on=False,
            )
        ax.set_xticks([4, 8, 16])
        ax.set_xlabel(r"Horizon $H$", fontsize=10)
        ax.set_ylabel("Top-1 error (lower is better)", fontsize=10)
        ax.set_title(title, fontsize=10, pad=6)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.65, alpha=0.88, color="0.45")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.45, alpha=0.5, color="0.75")
        ax.set_axisbelow(True)
        _panel_label(ax, tag)

    handles, labels = ax0.get_legend_handles_labels()
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.set_axis_off()
    ax_leg.patch.set_alpha(0)
    leg = ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.45",
        facecolor="0.98",
        fontsize=9,
        title="Model family",
        title_fontsize=9,
        columnspacing=1.5,
        handlelength=2.8,
        handletextpad=0.55,
        borderpad=0.5,
        labelspacing=0.4,
    )
    leg.get_frame().set_linewidth(0.55)

    return fig


def save_figure_pdf_png(fig: plt.Figure, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png
