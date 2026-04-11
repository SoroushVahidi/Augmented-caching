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
) -> None:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        textwrap.fill(txt, 28),
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
            mutation_scale=11,
            linewidth=1.0,
            color="#222222",
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def make_method_overview_two_panel_figure() -> plt.Figure:
    """Two-panel method figure: offline supervised construction + online guarded deployment."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.2, 6.4), constrained_layout=False)
    for ax in (ax_a, ax_b):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax_a.set_title("A. Offline training / supervised target construction", fontsize=11, fontweight="bold", pad=12)
    # Vertical pipeline (top → bottom)
    boxes_a = [
        (0.12, 0.82, 0.76, 0.09, "Trace replay (fixed request sequence)"),
        (0.12, 0.69, 0.76, 0.09, "Label full-cache misses; enumerate eviction candidates"),
        (0.12, 0.56, 0.76, 0.09, "Extract candidate features (predicted caches, ranks, horizons)"),
        (0.12, 0.43, 0.76, 0.11, "Horizon-H counterfactual replay vs oracle to build eviction-value targets"),
        (0.12, 0.27, 0.76, 0.11, "Fit supervised model (ridge / RF / HistGB) for eviction-value prediction"),
    ]
    for x, y, w, h, t in boxes_a:
        _box_axes(ax_a, x, y, w, h, t, fs=7.2)
    for y0, y1 in [(0.82, 0.78), (0.69, 0.65), (0.56, 0.52), (0.43, 0.39)]:
        _arrow(ax_a, 0.5, y0, 0.5, y1)
    ax_a.text(
        0.5,
        0.08,
        "Selection uses validation mean regret vs oracle on heavy_r1 shards (canonical train_metrics / best_config).",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
        transform=ax_a.transAxes,
    )

    ax_b.set_title("B. Online deployment (guarded eviction-value policy)", fontsize=11, fontweight="bold", pad=12)
    _box_axes(ax_b, 0.08, 0.86, 0.84, 0.07, "Request arrives", fs=7.2)
    _arrow(ax_b, 0.5, 0.86, 0.5, 0.80)
    _box_axes(ax_b, 0.08, 0.71, 0.84, 0.07, "Hit? → standard recency / metadata update", fs=7.0)
    _box_axes(ax_b, 0.08, 0.61, 0.84, 0.07, "Miss & cache not full → insert (no eviction)", fs=7.0)
    _arrow(ax_b, 0.5, 0.71, 0.5, 0.68)
    _arrow(ax_b, 0.5, 0.61, 0.5, 0.56)
    _box_axes(ax_b, 0.08, 0.44, 0.84, 0.10, "Miss & cache full → build candidate features → predict losses → choose min predicted-loss victim", fs=6.8)
    _arrow(ax_b, 0.5, 0.56, 0.5, 0.54)
    _arrow(ax_b, 0.5, 0.44, 0.5, 0.40)
    _box_axes(
        ax_b,
        0.08,
        0.26,
        0.84,
        0.11,
        "Guard: if repeated early-return mistakes are suspicious → temporary fallback mode (e.g., LRU-like) until trust recovers",
        fs=6.8,
        face="#e8e8e8",
    )
    _arrow(ax_b, 0.5, 0.40, 0.5, 0.37)
    _arrow(ax_b, 0.5, 0.26, 0.5, 0.22)
    _box_axes(ax_b, 0.08, 0.10, 0.84, 0.08, "Evict chosen victim; update cache state", fs=7.2)
    _arrow(ax_b, 0.5, 0.22, 0.5, 0.18)
    ax_b.text(
        0.5,
        0.03,
        "Grayscale-safe schematic; not quantitative results.",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
        transform=ax_b.transAxes,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.06, wspace=0.22)
    return fig


# Main manuscript comparison policies (canonical heavy_r1 eval roster; no exploratory extras)
MAIN_PERF_POLICIES: Tuple[str, ...] = (
    "evict_value_v1",
    "lru",
    "predictive_marker",
    "trust_and_doubt",
    "blind_oracle_lru_combiner",
    "rest_v1",
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
    degenerate: List[str] = []
    for fi, fam in enumerate(fams):
        vals = [means.get((fam, p), np.nan) for p in MAIN_PERF_POLICIES]
        clean = [v for v in vals if v == v]
        if len(clean) >= 2 and (max(clean) - min(clean)) < 1e-6:
            degenerate.append(fam)

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
    ttl = "Main end-to-end comparison by trace family (heavy_r1 policy comparison)"
    if degenerate:
        ttl += "\nNote: " + ", ".join(degenerate[:4]) + ("…" if len(degenerate) > 4 else "") + " show near-identical means across policies (tie/saturated regime)."
    ax.set_title(ttl, fontsize=10)
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
    ax.set_title("Relative performance vs LRU by trace family (heavy_r1)", fontsize=10)
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

    styles = {"ridge": ("-", "o"), "random_forest": ("--", "s"), "hist_gb": (":", "^")}
    colors = {"ridge": "#1a1a1a", "random_forest": "#5a5a5a", "hist_gb": "#2a2a2a"}
    zorder = {"ridge": 2, "random_forest": 3, "hist_gb": 4}
    # Plot order: back to front so hist_gb reads on top
    plot_order = ("ridge", "random_forest", "hist_gb")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.35), sharex=True, sharey=False)

    def _panel_label(ax: plt.Axes, tag: str) -> None:
        ax.text(
            0.03,
            0.97,
            tag,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            color="#111111",
            zorder=15,
        )

    for ax, data, title, tag in (
        (axes[0], by_val, "Validation mean regret (offline)", "(a)"),
        (axes[1], by_test, "Test mean regret (offline)", "(b)"),
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
                label=m,
                linewidth=1.55,
                markersize=6.0,
                zorder=zorder[m],
                clip_on=False,
            )
        ax.set_xticks([4, 8, 16])
        ax.set_xlabel(r"Horizon $H$")
        ax.set_ylabel("Mean regret vs.\ oracle (lower is better)")
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.65, alpha=0.88, color="0.45")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.45, alpha=0.5, color="0.75")
        ax.set_axisbelow(True)
        _panel_label(ax, tag)

    leg = axes[1].legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="0.55",
        facecolor="0.98",
        borderpad=0.6,
        labelspacing=0.35,
        handlelength=2.6,
        fontsize=9,
        title="Model",
        title_fontsize=9,
    )
    leg.set_zorder(20)

    fig.suptitle("Offline eviction-value training ablation (heavy_r1 shards)", fontsize=10.8, y=1.01)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94], pad=1.4, w_pad=2.0)
    return fig


def save_figure_pdf_png(fig: plt.Figure, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png
