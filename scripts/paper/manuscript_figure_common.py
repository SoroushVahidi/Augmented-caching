"""Shared matplotlib figures for KBS manuscript (vector PDF + PNG).

Used by build_kbs_main_manuscript_artifacts.py and build_kbs_manuscript_pre_eval_artifacts.py.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def apply_manuscript_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "sans-serif",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_method_overview_figure(
    *,
    subtitle: str = "Offline-trained eviction-value model (HistGradientBoosting, heavy_r1 selection) — schematic, not online policy metrics",
) -> plt.Figure:
    """Non-results pipeline diagram for evict_value_v1."""
    fig, ax = plt.subplots(figsize=(10.2, 3.55))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x: float, y: float, w: float, h: float, txt: str) -> None:
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025",
            linewidth=1.35,
            edgecolor="#222222",
            facecolor="#fafafa",
        )
        ax.add_patch(p)
        ax.text(
            x + w / 2,
            y + h / 2,
            textwrap.fill(txt, 22),
            ha="center",
            va="center",
            fontsize=10.5,
            color="#111111",
        )

    box(0.02, 0.36, 0.16, 0.32, "Request arrives")
    box(0.24, 0.36, 0.20, 0.32, "Candidate feature builder")
    box(0.50, 0.36, 0.20, 0.32, "Eviction-value predictor")
    box(0.76, 0.36, 0.20, 0.32, "Evict minimum predicted loss")
    z = 3
    ax.add_patch(
        FancyArrowPatch(
            (0.18, 0.52),
            (0.24, 0.52),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.35,
            color="#222222",
            zorder=z,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.44, 0.52),
            (0.50, 0.52),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.35,
            color="#222222",
            zorder=z,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.70, 0.52),
            (0.76, 0.52),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.35,
            color="#222222",
            zorder=z,
        )
    )
    ax.text(0.50, 0.14, textwrap.fill(subtitle, 88), ha="center", va="center", fontsize=9.5, color="#333333")
    fig.tight_layout()
    return fig


def make_offline_ablation_figure(train_rows: List[Dict[str, str]]) -> plt.Figure:
    """Mean regret vs horizon by model family from model_comparison CSV rows."""
    by_model_val: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    by_model_test: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in train_rows:
        h = int(r["horizon"])
        m = str(r["model"])
        by_model_val[m].append((h, float(r["val_mean_regret"])))
        by_model_test[m].append((h, float(r["test_mean_regret"])))

    styles = {
        "ridge": ("-", "o"),
        "random_forest": ("--", "s"),
        "hist_gb": (":", "^"),
    }
    colors = {"ridge": "#1b1b1b", "random_forest": "#444444", "hist_gb": "#0066aa"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.95), sharex=True)
    for m in sorted(by_model_val):
        xs = [x for x, _ in sorted(by_model_val[m])]
        ys = [y for _, y in sorted(by_model_val[m])]
        ls, mk = styles.get(m, ("-", "o"))
        c = colors.get(m, "#000000")
        axes[0].plot(xs, ys, linestyle=ls, marker=mk, color=c, label=m, linewidth=1.35, markersize=5.5)
    for m in sorted(by_model_test):
        xs = [x for x, _ in sorted(by_model_test[m])]
        ys = [y for _, y in sorted(by_model_test[m])]
        ls, mk = styles.get(m, ("-", "o"))
        c = colors.get(m, "#000000")
        axes[1].plot(xs, ys, linestyle=ls, marker=mk, color=c, label=m, linewidth=1.35, markersize=5.5)

    axes[0].set_title("Validation mean regret (offline)")
    axes[1].set_title("Test mean regret (offline)")
    for ax in axes:
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean regret")
        ax.grid(True, linestyle=":", linewidth=0.65, alpha=0.85)
    axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Offline eviction-value training ablation (heavy_r1 shards)", fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def save_figure_pdf_png(fig: plt.Figure, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png
