from __future__ import annotations

import csv
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from lafc.evict_value_wulver_v1 import load_trace_from_any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from manuscript_figure_common import (
    apply_manuscript_matplotlib_style,
    make_method_overview_figure,
    make_offline_ablation_figure,
    save_figure_pdf_png,
)


ROOT = Path(".")
ANALYSIS = ROOT / "analysis"
TABLES = ROOT / "tables" / "manuscript"
FIGURES = ROOT / "figures" / "manuscript"
REPORTS = ROOT / "reports" / "manuscript_artifacts"
LATEX = REPORTS / "latex_snippets"


# Main Wulver quantitative evidence for KBS: heavy_r1 pipeline only (see docs/wulver_heavy_evict_value_experiment.md).
# Policy comparison must come from slurm/evict_value_v1_wulver_heavy_eval.sbatch (or equivalent run with EXP_TAG=heavy_r1).
EVIDENCE_FILES = {
    "policy_comparison": ANALYSIS / "evict_value_wulver_v1_policy_comparison_heavy_r1.csv",
    "policy_comparison_md": ANALYSIS / "evict_value_wulver_v1_policy_comparison_heavy_r1.md",
    "dataset_summary": ANALYSIS / "evict_value_v1_wulver_dataset_summary_heavy_r1.md",
    "train_model_comparison": ANALYSIS / "evict_value_wulver_v1_model_comparison_heavy_r1.csv",
    "train_metrics": ANALYSIS / "evict_value_wulver_v1_train_metrics_heavy_r1.json",
    "best_config": ANALYSIS / "evict_value_wulver_v1_best_config_heavy_r1.json",
    "trace_manifest": ANALYSIS / "wulver_trace_manifest_full.csv",
    "baseline_doc": ROOT / "docs" / "baselines.md",
}


def _ensure_dirs() -> None:
    for p in [TABLES, FIGURES, REPORTS, LATEX]:
        p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _fmt_cell(v: float, best: float, second: float) -> str:
    txt = f"{v:.2f}"
    if abs(v - best) < 1e-9:
        return f"\\textbf{{{txt}}}"
    if abs(v - second) < 1e-9:
        return f"\\underline{{{txt}}}"
    return txt


def _build_table1_dataset_summary(policy_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    traces: Dict[str, Dict[str, object]] = {}
    for r in policy_rows:
        t = str(r["trace_name"])
        traces.setdefault(
            t,
            {
                "trace_name": t,
                "family": str(r["trace_family"]),
                "path": str(r["path"]),
                "capacities": set(),
            },
        )
        traces[t]["capacities"].add(int(float(r["capacity"])))
    out_rows: List[Dict[str, object]] = []
    for t, rec in sorted(traces.items()):
        reqs, _pages, _src = load_trace_from_any(str(rec["path"]))
        out_rows.append(
            {
                "trace_name": t,
                "family": rec["family"],
                "request_count": len(reqs),
                "capacities_used": ",".join(str(c) for c in sorted(rec["capacities"])),
                "hint_type": "next-arrival-derived metadata",
                "role": "main Wulver evaluation",
            }
        )
    csv_path = TABLES / "table1_dataset_summary.csv"
    tex_path = TABLES / "table1_dataset_summary.tex"
    _write_csv(csv_path, out_rows)
    lines = [
        "\\begin{tabular}{l l r l l l}",
        "\\toprule",
        "Trace & Family & Requests & Capacities & Hint type & Role \\\\",
        "\\midrule",
    ]
    for r in out_rows:
        lines.append(
            f"{r['trace_name']} & {r['family']} & {r['request_count']} & {r['capacities_used']} & {r['hint_type']} & {r['role']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LATEX.joinpath("table1_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Dataset/trace summary used in the main Wulver evaluation.}\n"
        "\\input{tables/manuscript/table1_dataset_summary.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _build_table2_policy_roster() -> Tuple[Path, Path]:
    rows = [
        {"policy": "lru", "category": "classical baseline", "description": "recency eviction baseline", "role": "reference", "faithfulness_note": "canonical"},
        {"policy": "blind_oracle", "category": "predictor baseline", "description": "evict furthest predicted reuse", "role": "stress comparator", "faithfulness_note": "interpreted per docs"},
        {"policy": "predictive_marker", "category": "robust baseline", "description": "marker with prediction-aware tie bias", "role": "strong robust baseline", "faithfulness_note": "canonical"},
        {"policy": "trust_and_doubt", "category": "robust baseline", "description": "adaptive trust/doubt with predicted caches", "role": "strong robust baseline", "faithfulness_note": "paper-aligned with documented interpretation"},
        {"policy": "blind_oracle_lru_combiner", "category": "robust combiner", "description": "shadow BO/LRU follow-the-leader combiner", "role": "strong robust baseline", "faithfulness_note": "interpreted details documented"},
        {"policy": "rest_v1", "category": "robust heuristic", "description": "selective trust policy with fallback behavior", "role": "strong robust baseline", "faithfulness_note": "repo method"},
        {"policy": "evict_value_v1", "category": "learned method", "description": "predict eviction regret/value and evict minimum-score candidate", "role": "proposed main method", "faithfulness_note": "artifact-backed trained model"},
    ]
    csv_path = TABLES / "table2_policy_roster.csv"
    tex_path = TABLES / "table2_policy_roster.tex"
    _write_csv(csv_path, rows)
    lines = [
        "\\begin{tabular}{l l p{4.5cm} l p{3.4cm}}",
        "\\toprule",
        "Policy & Category & Description & Role & Note \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['policy']} & {r['category']} & {r['description']} & {r['role']} & {r['faithfulness_note']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LATEX.joinpath("table2_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Policy roster used in the main comparison.}\n"
        "\\input{tables/manuscript/table2_policy_roster.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _aggregate_main(policy_rows: List[Dict[str, str]]) -> Tuple[List[str], List[Dict[str, object]]]:
    use_policies = [
        "evict_value_v1",
        "lru",
        "blind_oracle",
        "predictive_marker",
        "trust_and_doubt",
        "blind_oracle_lru_combiner",
        "rest_v1",
    ]
    fam_pol: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in policy_rows:
        p = str(r["policy"])
        if p not in use_policies:
            continue
        fam = str(r["trace_family"])
        fam_pol[(fam, p)].append(float(r["misses"]))
    families = sorted({k[0] for k in fam_pol.keys()})
    rows: List[Dict[str, object]] = []
    for p in use_policies:
        row: Dict[str, object] = {"policy": p}
        vals = []
        for fam in families:
            v = mean(fam_pol[(fam, p)]) if (fam, p) in fam_pol else float("nan")
            row[fam] = v
            if v == v:
                vals.append(v)
        row["overall_mean"] = mean(vals) if vals else float("nan")
        rows.append(row)
    return families, rows


def _build_table3_main_comparison(policy_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    families, rows = _aggregate_main(policy_rows)
    csv_path = TABLES / "table3_main_quantitative_comparison.csv"
    tex_path = TABLES / "table3_main_quantitative_comparison.tex"
    _write_csv(csv_path, rows)
    cols = families + ["overall_mean"]
    col_spec = "l" + "r" * len(cols)
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule", "Policy & " + " & ".join([*families, "Overall"]) + " \\\\", "\\midrule"]
    col_best = {c: min(float(r[c]) for r in rows if float(r[c]) == float(r[c])) for c in cols}
    col_second = {}
    for c in cols:
        uniq = sorted({float(r[c]) for r in rows if float(r[c]) == float(r[c])})
        col_second[c] = uniq[1] if len(uniq) > 1 else uniq[0]
    for r in rows:
        vals = [_fmt_cell(float(r[c]), col_best[c], col_second[c]) for c in cols]
        lines.append(f"{r['policy']} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LATEX.joinpath("table3_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Main quantitative comparison (mean misses; lower is better). Bold: best, underline: second-best per column.}\n"
        "\\input{tables/manuscript/table3_main_quantitative_comparison.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _build_table4_ablation(train_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    rows = []
    for r in train_rows:
        rows.append(
            {
                "horizon": int(r["horizon"]),
                "model": r["model"],
                "val_mean_regret": float(r["val_mean_regret"]),
                "test_mean_regret": float(r["test_mean_regret"]),
                "val_top1": float(r["val_top1"]),
                "test_top1": float(r["test_top1"]),
            }
        )
    rows = sorted(rows, key=lambda x: (x["horizon"], x["model"]))
    csv_path = TABLES / "table4_main_ablation.csv"
    tex_path = TABLES / "table4_main_ablation.tex"
    _write_csv(csv_path, rows)
    lines = [
        "\\begin{tabular}{r l r r r r}",
        "\\toprule",
        "Horizon & Model & Val regret & Test regret & Val top1 & Test top1 \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['horizon']} & {r['model']} & {r['val_mean_regret']:.4f} & {r['test_mean_regret']:.4f} & {r['val_top1']:.4f} & {r['test_top1']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LATEX.joinpath("table4_snippet.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Main ablation of eviction-value model family/horizon choices.}\n"
        "\\input{tables/manuscript/table4_main_ablation.tex}\n\\end{table}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _save_fig(fig: plt.Figure, stem: str) -> Tuple[Path, Path]:
    return save_figure_pdf_png(fig, FIGURES, stem)


def _figure1_method_overview() -> Tuple[Path, Path]:
    apply_manuscript_matplotlib_style()
    fig = make_method_overview_figure()
    LATEX.joinpath("figure1_snippet.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.95\\textwidth]{figures/manuscript/figure1_method_overview.pdf}\n"
        "\\caption{Eviction-value prediction pipeline for \\texttt{evict\\_value\\_v1} (schematic only; not online policy metrics).}\n"
        "\\label{fig:method-evict-value-pipeline}\n\\end{figure*}\n",
        encoding="utf-8",
    )
    return _save_fig(fig, "figure1_method_overview")


def _figure2_main_comparison(policy_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    focus = ["evict_value_v1", "lru", "predictive_marker", "trust_and_doubt", "rest_v1"]
    fam_pol: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in policy_rows:
        p = str(r["policy"])
        if p not in focus:
            continue
        fam = str(r["trace_family"])
        fam_pol[(fam, p)].append(float(r["misses"]))
    fams = sorted({k[0] for k in fam_pol})
    x = np.arange(len(fams))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 4.4))
    colors = ["#111111", "#444444", "#777777", "#999999", "#bbbbbb"]
    for i, p in enumerate(focus):
        vals = [mean(fam_pol[(f, p)]) for f in fams]
        ax.bar(x + (i - 2) * width, vals, width=width, label=p, color=colors[i], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(f, 10) for f in fams], fontsize=9)
    ax.set_ylabel("Mean misses")
    ax.set_title("Main performance comparison by trace family")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    LATEX.joinpath("figure2_snippet.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.95\\textwidth]{figures/manuscript/figure2_main_performance.pdf}\n"
        "\\caption{Main policy comparison on Wulver trace families (mean misses across capacities).}\n"
        "\\label{fig:main-policy-comparison}\n\\end{figure*}\n",
        encoding="utf-8",
    )
    return _save_fig(fig, "figure2_main_performance")


def _figure3_aggregate_improvement(policy_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    policies = ["evict_value_v1", "predictive_marker", "trust_and_doubt", "rest_v1", "blind_oracle_lru_combiner"]
    by_pol: Dict[str, List[float]] = defaultdict(list)
    for r in policy_rows:
        p = str(r["policy"])
        if p in policies or p == "lru":
            by_pol[p].append(float(r["misses"]))
    lru_mean = mean(by_pol["lru"])
    rows = []
    for p in policies:
        pm = mean(by_pol[p])
        rel = (lru_mean - pm) / lru_mean * 100.0
        rows.append((p, rel))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color="white", edgecolor="black", hatch="//", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Relative miss reduction vs LRU (%)")
    ax.set_title("Aggregate improvement vs LRU")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.grid(axis="x", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    LATEX.joinpath("figure3_snippet.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{figures/manuscript/figure3_aggregate_improvement.pdf}\n"
        "\\caption{Aggregate relative miss reduction versus LRU.}\n"
        "\\label{fig:aggregate-vs-lru}\n\\end{figure}\n",
        encoding="utf-8",
    )
    return _save_fig(fig, "figure3_aggregate_improvement")


def _figure4_ablation(train_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    apply_manuscript_matplotlib_style()
    fig = make_offline_ablation_figure(train_rows)
    LATEX.joinpath("figure4_snippet.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.9\\textwidth]{figures/manuscript/figure4_ablation.pdf}\n"
        "\\caption{Offline ablation of model family and horizon for eviction-value training (heavy\\_r1 shards; not online misses).}\n"
        "\\label{fig:offline-training-ablation}\n\\end{figure*}\n",
        encoding="utf-8",
    )
    return _save_fig(fig, "figure4_ablation")


def main() -> None:
    _ensure_dirs()
    missing = [k for k, p in EVIDENCE_FILES.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required evidence files: {missing}")

    policy_rows = _read_csv(EVIDENCE_FILES["policy_comparison"])
    train_rows = _read_csv(EVIDENCE_FILES["train_model_comparison"])

    apply_manuscript_matplotlib_style()
    plt.rcParams.update({"xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8})

    created: Dict[str, List[str]] = {"tables": [], "figures": [], "latex_snippets": []}
    skipped: List[Dict[str, str]] = []

    t1 = _build_table1_dataset_summary(policy_rows)
    t2 = _build_table2_policy_roster()
    t3 = _build_table3_main_comparison(policy_rows)
    t4 = _build_table4_ablation(train_rows)
    for pair in [t1, t2, t3, t4]:
        created["tables"] += [str(pair[0]), str(pair[1])]

    f1 = _figure1_method_overview()
    f2 = _figure2_main_comparison(policy_rows)
    f3 = _figure3_aggregate_improvement(policy_rows)
    f4 = _figure4_ablation(train_rows)
    for pair in [f1, f2, f3, f4]:
        created["figures"] += [str(pair[0]), str(pair[1])]

    for p in sorted(LATEX.glob("*.tex")):
        created["latex_snippets"].append(str(p))

    manifest = {
        "inputs": {k: str(v) for k, v in EVIDENCE_FILES.items()},
        "outputs": created,
        "notes": [
            "All paths in inputs are heavy_r1 artifacts from heavy_train + heavy_eval (docs/wulver_heavy_evict_value_experiment.md).",
            "Unsuffixed analysis/evict_value_wulver_v1_policy_comparison.csv (if present) is not used; it may include extra policies (e.g. ml_gate, atlas_v3) from multi_phase or ad hoc runs.",
            "Guarded-variant ablation on the same heavy Wulver pool is not available as a dedicated artifact; Table/Figure 4 therefore uses model-family ablation within evict_value_v1.",
        ],
    }
    (REPORTS / "manuscript_artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# Manuscript artifact generation report",
        "",
        "## Strongest manuscript-safe basis selected",
        "- Main comparison: `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (from heavy eval; baseline set per heavy runbook).",
        "- Main training/ablation: `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` and `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json`.",
        "- Dataset coverage: `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` and `analysis/wulver_trace_manifest_full.csv`.",
        "",
        "## Created tables",
        "- Table 1: dataset/trace summary.",
        "- Table 2: policy roster.",
        "- Table 3: main quantitative comparison (bold best, underline second-best).",
        "- Table 4: model-family/horizon ablation for evict_value_v1.",
        "",
        "## Created figures",
        "- Figure 1: method overview schematic.",
        "- Figure 2: family-level main performance comparison.",
        "- Figure 3: aggregate improvement vs LRU.",
        "- Figure 4: ablation plot (val/test mean regret).",
        "",
        "## Skipped or constrained items",
        "- Guarded/fallback ablation specifically on the same heavy Wulver artifact pool was not found as a dedicated canonical artifact; main ablation uses in-pool model-family/horizon evidence instead.",
        "",
        "## Output roots",
        f"- Tables: `{TABLES}`",
        f"- Figures: `{FIGURES}`",
        f"- Manifest/report/LaTeX snippets: `{REPORTS}`",
        "",
        "## Evidence completeness",
        "This run succeeded only if every path listed under `inputs` in `manuscript_artifact_manifest.json` existed.",
        "Committed tables/figures from an earlier run may not match canonical `heavy_r1` inputs if those inputs were added or renamed afterward; re-run this script after heavy eval produces `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`.",
    ]
    if skipped:
        report_lines.append("")
        report_lines.append("### Detailed skipped list")
        for s in skipped:
            report_lines.append(f"- {s['item']}: {s['reason']}")
    (REPORTS / "manuscript_artifact_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
