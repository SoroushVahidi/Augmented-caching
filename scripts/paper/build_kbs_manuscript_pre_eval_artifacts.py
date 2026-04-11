#!/usr/bin/env python3
"""Emit manuscript tables/figures safe before canonical heavy_r1 *online* policy comparison exists.

Uses only heavy_r1 training/dataset artifacts (see docs/evict_value_v1_kbs_canonical_artifacts.md).
Does not read analysis/evict_value_wulver_v1_policy_comparison*.csv (unsuffixed or heavy_r1).
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables" / "manuscript" / "pre_eval"
FIGURES = ROOT / "figures" / "manuscript" / "pre_eval"
REPORT = ROOT / "reports" / "manuscript_artifacts" / "pre_eval"

DEFAULT_MANIFEST = ROOT / "data" / "derived" / "evict_value_v1_wulver_heavy_r1" / "manifest.json"
DEFAULT_SPLIT = ROOT / "data" / "derived" / "evict_value_v1_wulver_heavy_r1" / "split_summary.csv"
DEFAULT_TRAIN_CMP = ROOT / "analysis" / "evict_value_wulver_v1_model_comparison_heavy_r1.csv"


def _ensure_dirs() -> None:
    for p in (TABLES, FIGURES, REPORT):
        p.mkdir(parents=True, exist_ok=True)


def _latex_texttt_path(rel: Path) -> str:
    s = str(rel).replace("_", r"\_")
    return f"\\texttt{{{s}}}"


def _latex_model_name(name: str) -> str:
    return name.replace("_", r"\_")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_method_overview_figure() -> Tuple[Path, Path]:
    """Schematic only; no empirical data."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, txt: str) -> None:
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.2, edgecolor="black", facecolor="white")
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, textwrap.fill(txt, 22), ha="center", va="center", fontsize=10)

    box(0.02, 0.35, 0.16, 0.32, "Request arrives")
    box(0.24, 0.35, 0.20, 0.32, "Candidate feature builder")
    box(0.50, 0.35, 0.20, 0.32, "Eviction-value predictor")
    box(0.76, 0.35, 0.20, 0.32, "Evict min predicted loss")
    ax.add_patch(FancyArrowPatch((0.18, 0.51), (0.24, 0.51), arrowstyle="->", mutation_scale=12, linewidth=1.2))
    ax.add_patch(FancyArrowPatch((0.44, 0.51), (0.50, 0.51), arrowstyle="->", mutation_scale=12, linewidth=1.2))
    ax.add_patch(FancyArrowPatch((0.70, 0.51), (0.76, 0.51), arrowstyle="->", mutation_scale=12, linewidth=1.2))
    ax.text(
        0.50,
        0.16,
        "Trained model: HistGradientBoosting (heavy_r1 offline selection)",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout()
    stem = "fig_method_overview_evict_value_v1_pre_eval"
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    snippet = REPORT / "snippet_fig_method_overview_pre_eval.tex"
    snippet.write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=0.95\\textwidth]{figures/manuscript/pre_eval/"
        + stem
        + ".pdf}\n"
        "\\caption{Eviction-value \\texttt{evict\\_value\\_v1} decision pipeline (method overview; not performance data).}\n"
        "\\label{fig:pre-eval-method-overview}\n"
        "\\end{figure*}\n",
        encoding="utf-8",
    )
    return pdf, png


def build_dataset_table(manifest_path: Path, split_summary_path: Path) -> Tuple[Path, Path]:
    rows_in = list(csv.DictReader(split_summary_path.open(encoding="utf-8")))
    for r in rows_in:
        r["row_count"] = int(r["row_count"])
        r["decision_count"] = int(r["decision_count"])
        r["capacity"] = int(r["capacity"])
        r["horizon"] = int(r["horizon"])

    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta_short = {
        "split_mode": meta.get("split_mode"),
        "chunk_size": meta.get("chunk_size"),
        "trace_count": meta.get("trace_count"),
        "shard_count": meta.get("shard_count"),
    }

    out_rows: List[Dict[str, object]] = []
    for r in sorted(rows_in, key=lambda x: (x["split"], x["trace_family"], x["capacity"], x["horizon"])):
        out_rows.append(
            {
                "split": r["split"],
                "trace_family": r["trace_family"],
                "capacity": r["capacity"],
                "horizon": r["horizon"],
                "row_count": r["row_count"],
                "decision_count": r["decision_count"],
            }
        )

    csv_path = TABLES / "tab_dataset_characterization_heavy_r1_pre_eval.csv"
    tex_path = TABLES / "tab_dataset_characterization_heavy_r1_pre_eval.tex"
    _write_csv(csv_path, out_rows)

    # TeX: one row per (split, family, capacity); row/decision counts match all horizons in split_summary.
    dedup_key = {}
    for r in rows_in:
        k = (r["split"], r["trace_family"], r["capacity"])
        dedup_key[k] = (r["row_count"], r["decision_count"])
    dedup_rows = sorted(dedup_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))

    rel_manifest = manifest_path.relative_to(ROOT)
    sm = str(meta_short.get("split_mode", "")).replace("_", r"\_")
    cap_tex = (
        f"Heavy\\_r1 supervised shards derived from { _latex_texttt_path(rel_manifest) }: "
        f"split mode \\texttt{{{sm}}}, chunk size {meta_short.get('chunk_size')}, "
        f"{meta_short.get('trace_count')} traces, {meta_short.get('shard_count')} shards. "
        "Row and decision counts are identical across prediction horizons $H \\in \\{4,8,16\\}$ in "
        f"\\texttt{{split\\_summary.csv}}; this table lists one row per (split, family, capacity). "
        "Offline training/eval splits only; not online policy outcomes."
    )

    lines = [
        "% Requires \\usepackage{booktabs,longtable}",
        "\\begin{longtable}{llrrr}",
        "\\caption{"
        + cap_tex
        + "}\\label{tab:pre-eval-dataset-heavy-r1}\\\\",
        "\\toprule",
        "Split & Family & Cap & Rows & Decisions \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{5}{c}{{\\tablename\\ \\thetable{} --- continued from previous page}} \\\\",
        "\\toprule",
        "Split & Family & Cap & Rows & Decisions \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{5}{r}{{Continued on next page}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for (spl, fam, cap), (nrow, ndec) in dedup_rows:
        lines.append(f"{spl} & {fam} & {cap} & {nrow:,} & {ndec:,} \\\\")
    lines.append("\\end{longtable}")

    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_tex = REPORT / "snippet_tab_dataset_pre_eval.tex"
    table_tex.write_text(
        "% longtable cannot be placed inside table*; \\input from main text (after booktabs/longtable in preamble).\n"
        "\\input{tables/manuscript/pre_eval/tab_dataset_characterization_heavy_r1_pre_eval.tex}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def build_training_ablation(train_csv: Path) -> Tuple[Path, Path, Path, Path]:
    train_rows = list(csv.DictReader(train_csv.open(encoding="utf-8")))
    for r in train_rows:
        r["horizon"] = int(r["horizon"])
        r["val_mean_regret"] = float(r["val_mean_regret"])
        r["test_mean_regret"] = float(r["test_mean_regret"])
        r["val_top1"] = float(r["val_top1"])
        r["test_top1"] = float(r["test_top1"])

    tab_rows = sorted(train_rows, key=lambda x: (x["horizon"], x["model"]))
    csv_path = TABLES / "tab_training_ablation_heavy_r1_pre_eval.csv"
    tex_path = TABLES / "tab_training_ablation_heavy_r1_pre_eval.tex"
    _write_csv(csv_path, tab_rows)

    lines = [
        "\\begin{tabular}{r l r r r r}",
        "\\toprule",
        "Horizon & Model & Val regret & Test regret & Val top1 & Test top1 \\\\",
        "\\midrule",
    ]
    for r in tab_rows:
        lines.append(
            f"{r['horizon']} & {_latex_model_name(str(r['model']))} & {r['val_mean_regret']:.4f} & {r['test_mean_regret']:.4f} & "
            f"{r['val_top1']:.4f} & {r['test_top1']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharex=True)
    for m in sorted(by_model_val):
        xs = [x for x, _ in sorted(by_model_val[m])]
        ys = [y for _, y in sorted(by_model_val[m])]
        ls, mk = styles.get(m, ("-", "o"))
        axes[0].plot(xs, ys, linestyle=ls, marker=mk, color="black", label=m, linewidth=1.2, markersize=5)
    for m in sorted(by_model_test):
        xs = [x for x, _ in sorted(by_model_test[m])]
        ys = [y for _, y in sorted(by_model_test[m])]
        ls, mk = styles.get(m, ("-", "o"))
        axes[1].plot(xs, ys, linestyle=ls, marker=mk, color="black", label=m, linewidth=1.2, markersize=5)
    axes[0].set_title("Validation mean regret (offline)")
    axes[1].set_title("Test mean regret (offline)")
    for ax in axes:
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean regret")
        ax.grid(True, linestyle=":", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    stem = "fig_training_ablation_regret_heavy_r1_pre_eval"
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    (REPORT / "snippet_tab_training_ablation_pre_eval.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Offline training ablation for \\texttt{evict\\_value\\_v1} on heavy\\_r1 shards "
        "(model family $\\times$ horizon; not online misses).}\n"
        "\\label{tab:pre-eval-training-ablation}\n"
        "\\input{tables/manuscript/pre_eval/tab_training_ablation_heavy_r1_pre_eval.tex}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )
    (REPORT / "snippet_fig_training_ablation_pre_eval.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=0.9\\textwidth]{figures/manuscript/pre_eval/"
        + stem
        + ".pdf}\n"
        "\\caption{Offline mean regret vs horizon by model family (heavy\\_r1 training selection; not simulator policy comparison).}\n"
        "\\label{fig:pre-eval-training-ablation}\n"
        "\\end{figure*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path, pdf, png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--split-summary", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--train-model-comparison", type=Path, default=DEFAULT_TRAIN_CMP)
    args = ap.parse_args()

    _ensure_dirs()
    missing = [p for p in (args.manifest, args.split_summary, args.train_model_comparison) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    build_method_overview_figure()
    build_dataset_table(args.manifest, args.split_summary)
    build_training_ablation(args.train_model_comparison)
    print("Wrote pre-eval manuscript assets under tables/manuscript/pre_eval, figures/manuscript/pre_eval, reports/manuscript_artifacts/pre_eval")


if __name__ == "__main__":
    main()
