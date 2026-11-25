"""Render report figures from committed tables of prior measurements."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


REPORT = Path(__file__).resolve().parents[1]
TABLES = REPORT / "aggregate-results"
FIGURES = REPORT / "figures"
COLORS = ("#166d6b", "#ad562d", "#485691")
METADATA = {"Creator": "Report figures.py", "CreationDate": None, "ModDate": None}


def rows(filename):
    with (TABLES / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def benchmark():
    elapsed = {
        (int(row["capacity"]), int(row["horizon"]), row["method"]):
        float(row["seconds_per_decision"])
        for row in rows("phase2b-labeler-benchmark.csv")
    }
    methods = {
        "november_style_exhaustive_compiled": "Compiled exhaustive",
        "phase2a_coalescence_compiled": "Compiled paired",
        "repository_simulator": "Repository simulator",
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.9, 3.0), sharey=True)
    for ax, capacity in zip(axes, (32, 128, 1024)):
        for (method, label), color, marker in zip(methods.items(), COLORS, ("o", "^", "s")):
            series = sorted(
                (horizon, seconds / elapsed[capacity, horizon, "shared_ordered_dict"])
                for (cap, horizon, measured), seconds in elapsed.items()
                if cap == capacity and measured == method
            )
            x, y = zip(*series)
            ax.plot(x, y, marker=marker, markersize=3, color=color, label=label)
        ax.axhline(1, linestyle="--", color="0.4", linewidth=0.8)
        ax.set(xscale="log", yscale="log", title=f"{capacity:,} cache slots")
        ax.set_xticks([4, 16, 64, 256, 1024])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Comparator time / shared time")
    fig.supxlabel("Future requests")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.87))
    fig.savefig(FIGURES / "labeler-benchmarks.pdf", metadata=METADATA)
    plt.close(fig)


def tradeoffs():
    labels = {
        "D_lru_profiled": ("LRU", (5, 6)),
        "D_hgb_finalist": ("Boosting finalist", (5, 11)),
        "D_hgb_preliminary": ("Preliminary abstention", (-62, 13)),
        "D_hgb_approx": ("Approximate tie", (5, -6)),
        "B_F2": ("Compact history", (-86, -13)),
        "C_two_part": ("Benefit / risk", (7, -12)),
        "C_full_scoring": ("Full scoring", (-56, 10)),
        "D_ridge_finalist": ("Ridge finalist", (-59, 10)),
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.3), layout="constrained")
    ax.axhline(0, color="0.65", linewidth=0.8)
    ax.axvline(5, linestyle="--", color="0.45", linewidth=0.9)
    ax.text(5.2, -3.5, "5× CPU limit", color="0.35", fontsize=9)
    for row in rows("phase2cb-policies.csv"):
        if row["portfolio"] not in labels:
            continue
        name, offset = labels[row["portfolio"]]
        x = float(row["cpu_ratio"])
        y = 100 * float(row["relative_miss_reduction"])
        ax.scatter(x, y, color=COLORS[0] if y >= 0 else COLORS[1], zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offset, fontsize=9)
    ax.set(xscale="log", xlim=(0.9, 54), ylim=(-3.8, 1.28),
           xlabel="Whole-request CPU relative to LRU",
           ylabel="Relative miss reduction (%)")
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIGURES / "finalist-tradeoffs.pdf", metadata=METADATA)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    benchmark()
    tradeoffs()
