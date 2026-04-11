from __future__ import annotations

import csv
import json
import sys
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
    make_improvement_vs_lru_figure,
    make_main_performance_comparison_figure,
    make_method_overview_two_panel_figure,
    make_offline_ablation_figure,
    save_figure_pdf_png,
)


ROOT = Path(".")
ANALYSIS = ROOT / "analysis"
TABLES = ROOT / "tables" / "manuscript"
FIGURES = ROOT / "figures" / "manuscript"
REPORTS = ROOT / "reports" / "manuscript_artifacts"
LATEX = REPORTS / "latex_snippets"

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

# Main quantitative comparison (canonical heavy_r1 eval; excludes exploratory-only policies)
TABLE3_POLICIES: Tuple[str, ...] = (
    "lru",
    "evict_value_v1",
    "predictive_marker",
    "trust_and_doubt",
    "blind_oracle_lru_combiner",
    "rest_v1",
)

SHORT_POLICY_LABEL = {
    "lru": "LRU",
    "evict_value_v1": "EV",
    "predictive_marker": "PredMk",
    "trust_and_doubt": "T\\&D",
    "blind_oracle_lru_combiner": "BO/LRU",
    "rest_v1": "REST",
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


def _fmt_cell_lower_better(v: float, best: float, second: float) -> str:
    return _fmt_cell(v, best, second)


def _latex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def _core_evidence_ok() -> List[str]:
    """Paths required for any partial build."""
    keys = [
        "train_model_comparison",
        "dataset_summary",
        "trace_manifest",
        "baseline_doc",
        "train_metrics",
        "best_config",
    ]
    return [k for k in keys if not EVIDENCE_FILES[k].exists()]


def _policy_evidence_ok() -> bool:
    return EVIDENCE_FILES["policy_comparison"].exists()


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
    manifest_by_name = {r["trace_name"]: r for r in _read_csv(EVIDENCE_FILES["trace_manifest"])}
    out_rows: List[Dict[str, object]] = []
    for t, rec in sorted(traces.items()):
        reqs, pages, _src = load_trace_from_any(str(rec["path"]))
        man = manifest_by_name.get(t, {})
        src = str(man.get("dataset_source", "")) or "—"
        caps = ",".join(str(c) for c in sorted(rec["capacities"]))
        out_rows.append(
            {
                "trace": t,
                "family": rec["family"],
                "requests": len(reqs),
                "unique_pages": len(pages),
                "capacities_in_comparison": caps,
                "hint_metadata": "next-arrival-derived",
                "source_category": src,
                "in_main_table": "yes",
                "note": "heavy_r1 Wulver pool",
            }
        )
    csv_path = TABLES / "table1_dataset_summary.csv"
    tex_path = TABLES / "table1_dataset_summary.tex"
    _write_csv(csv_path, out_rows)
    lines = [
        "\\begin{tabular}{@{}l l r r l l l l@{}}",
        "\\toprule",
        "Trace & Fam. & Req. & $|U|$ & Caps & Hint & Src. & Note \\\\",
        "\\midrule",
    ]
    for r in out_rows:
        lines.append(
            f"{_latex_escape(str(r['trace']))} & {_latex_escape(str(r['family']))} & {r['requests']} & {r['unique_pages']} & "
            f"{_latex_escape(str(r['capacities_in_comparison']))} & {r['hint_metadata']} & {_latex_escape(str(r['source_category']))} & "
            f"{_latex_escape(str(r['note']))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LATEX.joinpath("table1_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Evaluation traces for the main Wulver comparison (mean misses aggregated over listed capacities; lower is better in Table~\\ref{tab:main-comparison}). "
        "Hint metadata: prediction-time signals derived from the trace per the heavy\\_r1 driver.}\n"
        "\\label{tab:dataset-summary}\n"
        "\\input{tables/manuscript/table1_dataset_summary.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _build_table1_unavailable() -> Tuple[Path, Path]:
    msg = (
        "NOT_GENERATED: analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv missing — "
        "cannot derive trace/capacity roster for Table~1 from canonical evidence."
    )
    csv_path = TABLES / "table1_dataset_summary.csv"
    tex_path = TABLES / "table1_dataset_summary.tex"
    _write_csv(csv_path, [{"status": msg}])
    tex_path.write_text(
        "\\begin{tabular}{@{}p{0.92\\linewidth}@{}}\n\\toprule\n"
        "\\textit{Table not regenerated: canonical policy-comparison CSV absent.}\\\\\n"
        "\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )
    LATEX.joinpath("table1_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{\\textit{(Unavailable)} Dataset summary pending canonical "
        "\\texttt{evict\\_value\\_wulver\\_v1\\_policy\\_comparison\\_heavy\\_r1.csv}.}\n"
        "\\label{tab:dataset-summary-unavailable}\n"
        "\\input{tables/manuscript/table1_dataset_summary.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _build_table2_policy_roster() -> Tuple[Path, Path]:
    rows = [
        {
            "policy": "lru",
            "category": "classical",
            "short_label": "LRU",
            "role": "reference",
            "note": "Recency baseline",
        },
        {
            "policy": "predictive_marker",
            "category": "robust LA",
            "short_label": "PredMk",
            "role": "strong baseline",
            "note": "Prediction-aware marker",
        },
        {
            "policy": "trust_and_doubt",
            "category": "robust LA",
            "short_label": "T\\&D",
            "role": "strong baseline",
            "note": "Adaptive trust/doubt",
        },
        {
            "policy": "blind_oracle_lru_combiner",
            "category": "combiner",
            "short_label": "BO/LRU",
            "role": "strong baseline",
            "note": "BO/LRU FTL-style combiner",
        },
        {
            "policy": "rest_v1",
            "category": "heuristic",
            "short_label": "REST",
            "role": "strong baseline",
            "note": "Selective trust + fallback",
        },
        {
            "policy": "evict_value_v1",
            "category": "proposed",
            "short_label": "EV",
            "role": "main method",
            "note": "Learned eviction-value + optional guard",
        },
    ]
    csv_path = TABLES / "table2_policy_roster.csv"
    tex_path = TABLES / "table2_policy_roster.tex"
    _write_csv(csv_path, rows)
    lines = [
        "\\begin{tabular}{@{}l l l l p{4.2cm}@{}}",
        "\\toprule",
        "Policy & Cat. & Label & Role & Note \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['policy']} & {r['category']} & {r['short_label']} & {r['role']} & {r['note']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LATEX.joinpath("table2_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Policies in the main Wulver comparison (compact labels match Table~\\ref{tab:main-comparison}).}\n"
        "\\label{tab:policy-roster}\n"
        "\\input{tables/manuscript/table2_policy_roster.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _aggregate_main(policy_rows: List[Dict[str, str]]) -> Tuple[List[str], List[Dict[str, object]], List[str]]:
    use_policies = list(TABLE3_POLICIES)
    fam_pol: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in policy_rows:
        p = str(r["policy"])
        if p not in use_policies:
            continue
        fam = str(r["trace_family"])
        fam_pol[(fam, p)].append(float(r["misses"]))
    families = sorted({k[0] for k in fam_pol.keys()})
    degenerate: List[str] = []
    for fam in families:
        means = []
        for pol in use_policies:
            k = (fam, pol)
            if k in fam_pol:
                means.append(mean(fam_pol[k]))
        if len(means) >= 2 and (max(means) - min(means)) < 1e-6:
            degenerate.append(fam)

    rows_out: List[Dict[str, object]] = []
    for p in use_policies:
        row: Dict[str, object] = {"policy": p, "label": SHORT_POLICY_LABEL.get(p, p)}
        vals = []
        for fam in families:
            v = mean(fam_pol[(fam, p)]) if (fam, p) in fam_pol else float("nan")
            row[fam] = v
            if v == v:
                vals.append(v)
        row["overall_mean"] = mean(vals) if vals else float("nan")
        rows_out.append(row)

    lru_overall = next(float(r["overall_mean"]) for r in rows_out if r["policy"] == "lru")
    for r in rows_out:
        om = float(r["overall_mean"])
        r["delta_vs_lru"] = om - lru_overall if om == om else float("nan")
        wins = 0
        if r["policy"] != "lru":
            for fam in families:
                lp = mean(fam_pol[(fam, "lru")])
                pp = mean(fam_pol[(fam, r["policy"])]) if (fam, r["policy"]) in fam_pol else float("nan")
                if pp == pp and lp == lp and pp < lp - 1e-9:
                    wins += 1
        r["families_better_than_lru"] = wins

    return families, rows_out, degenerate


def _build_table3_main_comparison(policy_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    families, rows, degenerate = _aggregate_main(policy_rows)
    csv_path = TABLES / "table3_main_quantitative_comparison.csv"
    tex_path = TABLES / "table3_main_quantitative_comparison.tex"
    _write_csv(csv_path, rows)
    cols = list(families) + ["overall_mean", "delta_vs_lru", "families_better_than_lru"]
    col_spec = "l" + "r" * (len(cols) - 1)
    head = ["Label"] + [_latex_escape(f) for f in families] + ["All", r"$\Delta$LRU", r"\#Fam$<$LRU"]
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(head) + " \\\\",
        "\\midrule",
    ]
    metric_cols = list(families) + ["overall_mean"]
    col_best = {c: min(float(r[c]) for r in rows if float(r[c]) == float(r[c])) for c in metric_cols}
    col_second: Dict[str, float] = {}
    for c in metric_cols:
        uniq = sorted({float(r[c]) for r in rows if float(r[c]) == float(r[c])})
        col_second[c] = uniq[1] if len(uniq) > 1 else uniq[0]
    for r in rows:
        cells = [r["label"]]
        for c in families:
            v = float(r[c])
            cells.append(_fmt_cell_lower_better(v, col_best[c], col_second[c]))
        om = float(r["overall_mean"])
        cells.append(_fmt_cell_lower_better(om, col_best["overall_mean"], col_second["overall_mean"]))
        dv = float(r["delta_vs_lru"])
        cells.append(f"{dv:+.2f}")
        cells.append(str(int(r["families_better_than_lru"])))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    cap_extra = ""
    if degenerate:
        cap_extra = " Near-tie families (all policies within numerical tolerance): " + ", ".join(degenerate) + "."
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cap = (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Main quantitative comparison: mean replay misses by trace family (lower is better). "
        "Bold: best; underline: second-best per column among policies. "
        r"$\Delta$LRU is overall mean minus LRU overall mean. "
        r"\#Fam$<$LRU counts families where the policy strictly beats LRU."
        + cap_extra
        + "}\n"
        + "\\label{tab:main-comparison}\n"
        + "\\input{tables/manuscript/table3_main_quantitative_comparison.tex}\n\\end{table*}\n"
    )
    LATEX.joinpath("table3_snippet.tex").write_text(cap, encoding="utf-8")
    return csv_path, tex_path


def _build_table3_unavailable() -> Tuple[Path, Path]:
    csv_path = TABLES / "table3_main_quantitative_comparison.csv"
    tex_path = TABLES / "table3_main_quantitative_comparison.tex"
    _write_csv(csv_path, [{"status": "NOT_GENERATED: missing policy_comparison_heavy_r1.csv"}])
    tex_path.write_text(
        "\\begin{tabular}{@{}p{0.92\\linewidth}@{}}\n\\toprule\n"
        "\\textit{Table not regenerated: canonical policy-comparison CSV absent.}\\\\\n"
        "\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )
    LATEX.joinpath("table3_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{\\textit{(Unavailable)} Main comparison pending "
        "\\texttt{evict\\_value\\_wulver\\_v1\\_policy\\_comparison\\_heavy\\_r1.csv}.}\n"
        "\\label{tab:main-comparison-unavailable}\n"
        "\\input{tables/manuscript/table3_main_quantitative_comparison.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _build_table4_ablation(train_rows: List[Dict[str, str]], best_cfg: Optional[dict]) -> Tuple[Path, Path]:
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

    # Best val regret per horizon (lower better)
    by_h: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_h[r["horizon"]].append(r)
    best_val: Dict[Tuple[int, str], bool] = {}
    second_val: Dict[Tuple[int, str], bool] = {}
    for h, lst in by_h.items():
        vs = sorted(lst, key=lambda x: x["val_mean_regret"])
        best_val[(h, str(vs[0]["model"]))] = True
        if len(vs) > 1:
            second_val[(h, str(vs[1]["model"]))] = True

    lines = [
        "\\begin{tabular}{@{}r l r r r r@{}}",
        "\\toprule",
        "$H$ & Model & Val reg. $\\downarrow$ & Test reg. $\\downarrow$ & Val top1 & Test top1 \\\\",
        "\\midrule",
    ]
    for r in rows:
        h = int(r["horizon"])
        m = str(r["model"])
        vr = float(r["val_mean_regret"])
        tr = float(r["test_mean_regret"])
        v1 = float(r["val_top1"])
        t1 = float(r["test_top1"])
        vrs = f"{vr:.4f}"
        if best_val.get((h, m)):
            vrs = f"\\textbf{{{vrs}}}"
        elif second_val.get((h, m)):
            vrs = f"\\underline{{{vrs}}}"
        lines.append(f"{h} & {_latex_escape(m)} & {vrs} & {tr:.4f} & {v1:.4f} & {t1:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sel = ""
    if best_cfg:
        bm = str(best_cfg.get("model", ""))
        sel = (
            " Selection from \\texttt{best\\_config}: horizon "
            + str(best_cfg.get("horizon"))
            + ", model \\texttt{"
            + bm
            + "}."
        )
    LATEX.joinpath("table4_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Offline training ablation for \\texttt{evict\\_value\\_v1} on heavy\\_r1 shards. "
        "Regret is mean vs oracle (lower better); top-1 is candidate ranking quality. "
        "Bold (val regret): best per horizon; underline: second-best."
        + sel
        + "}\n"
        "\\label{tab:offline-ablation}\n"
        "\\input{tables/manuscript/table4_main_ablation.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _save_fig(fig: plt.Figure, stem: str) -> Tuple[Path, Path]:
    return save_figure_pdf_png(fig, FIGURES, stem)


def _figure1_method_overview() -> Tuple[Path, Path]:
    apply_manuscript_matplotlib_style()
    fig = make_method_overview_two_panel_figure()
    LATEX.joinpath("figure1_snippet.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{figures/manuscript/figure1_method_overview.pdf}\n"
        "\\caption{Offline supervised construction of eviction-value targets and online guarded deployment. "
        "Panel~A: trace replay, counterfactual horizon-$H$ targets, and model fitting. "
        "Panel~B: cache-state update rules including optional guard-triggered fallback when early mistakes look suspicious.}\n"
        "\\label{fig:method-overview}\n\\end{figure*}\n",
        encoding="utf-8",
    )
    return _save_fig(fig, "figure1_method_overview")


def _figure2_main_performance_comparison(policy_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    apply_manuscript_matplotlib_style()
    fig = make_main_performance_comparison_figure(policy_rows)
    LATEX.joinpath("figure2_snippet.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=0.98\\textwidth]{figures/manuscript/figure2_main_performance_comparison.pdf}\n"
        "\\caption{End-to-end replay misses by trace family (mean over capacities in canonical heavy\\_r1 comparison; lower better). "
        "Thick blue edge highlights \\texttt{evict\\_value\\_v1}. Families marked near-tie show identical means across policies within tolerance.}\n"
        "\\label{fig:main-performance}\n\\end{figure*}\n",
        encoding="utf-8",
    )
    return _save_fig(fig, "figure2_main_performance_comparison")


def _figure3_improvement_vs_lru(policy_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    apply_manuscript_matplotlib_style()
    fig = make_improvement_vs_lru_figure(policy_rows)
    LATEX.joinpath("figure3_snippet.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=0.98\\textwidth]{figures/manuscript/figure3_improvement_vs_lru.pdf}\n"
        "\\caption{Difference in mean replay misses vs LRU by family: $\\Delta = \\mathrm{misses}_{\\mathrm{pol}} - \\mathrm{misses}_{\\mathrm{LRU}}$ "
        "(negative means fewer misses than LRU). \\texttt{evict\\_value\\_v1} is highlighted.}\n"
        "\\label{fig:improvement-vs-lru}\n\\end{figure*}\n",
        encoding="utf-8",
    )
    return _save_fig(fig, "figure3_improvement_vs_lru")


def _figure4_ablation(train_rows: List[Dict[str, str]]) -> Tuple[Path, Path]:
    apply_manuscript_matplotlib_style()
    fig = make_offline_ablation_figure(train_rows)
    LATEX.joinpath("figure4_snippet.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.96\\textwidth]{figures/manuscript/figure4_ablation.pdf}\n"
        "\\caption{Offline eviction-value training ablation across replay horizons and model families on heavy\\_r1 shards "
        "(\\texttt{analysis/evict\\_value\\_wulver\\_v1\\_model\\_comparison\\_heavy\\_r1.csv}). "
        "(a)~validation and (b)~test mean regret vs oracle (lower is better).}\n"
        "\\label{fig:offline-ablation}\n\\end{figure*}\n",
        encoding="utf-8",
    )
    return _save_fig(fig, "figure4_ablation")


def _remove_legacy_figure_files() -> None:
    legacy = [
        "figure2_main_performance.pdf",
        "figure2_main_performance.png",
        "figure3_aggregate_improvement.pdf",
        "figure3_aggregate_improvement.png",
    ]
    for name in legacy:
        p = FIGURES / name
        if p.exists():
            p.unlink()


def main() -> None:
    _ensure_dirs()
    _remove_legacy_figure_files()
    missing_core = _core_evidence_ok()
    if missing_core:
        raise FileNotFoundError(f"Missing core evidence files: {missing_core}")

    train_rows = _read_csv(EVIDENCE_FILES["train_model_comparison"])
    best_cfg = json.loads(EVIDENCE_FILES["best_config"].read_text(encoding="utf-8"))
    policy_ok = _policy_evidence_ok()

    apply_manuscript_matplotlib_style()
    plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7})

    created: Dict[str, List[str]] = {"tables": [], "figures": [], "latex_snippets": []}
    notes: List[str] = []

    if not EVIDENCE_FILES["policy_comparison_md"].exists():
        notes.append("Optional `policy_comparison_heavy_r1.md` missing; CSV is the quantitative source of truth.")

    t2 = _build_table2_policy_roster()
    t4 = _build_table4_ablation(train_rows, best_cfg)
    for pair in [t2, t4]:
        created["tables"] += [str(pair[0]), str(pair[1])]

    f1 = _figure1_method_overview()
    f4 = _figure4_ablation(train_rows)
    for pair in [f1, f4]:
        created["figures"] += [str(pair[0]), str(pair[1])]

    if policy_ok:
        policy_rows = _read_csv(EVIDENCE_FILES["policy_comparison"])
        t1 = _build_table1_dataset_summary(policy_rows)
        t3 = _build_table3_main_comparison(policy_rows)
        for pair in [t1, t3]:
            created["tables"] += [str(pair[0]), str(pair[1])]
        f2 = _figure2_main_performance_comparison(policy_rows)
        f3 = _figure3_improvement_vs_lru(policy_rows)
        for pair in [f2, f3]:
            created["figures"] += [str(pair[0]), str(pair[1])]
        notes.append("Policy-dependent tables/figures rebuilt from `policy_comparison_heavy_r1.csv`.")
    else:
        t1 = _build_table1_unavailable()
        t3 = _build_table3_unavailable()
        for pair in [t1, t3]:
            created["tables"] += [str(pair[0]), str(pair[1])]
        notes.append(
            "SKIPPED Table~1/3 and Figure~2/3: `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` not found."
        )

    notes.append(
        "No dedicated canonical heavy_r1 artifact for guarded/fallback A/B or decision-quality diagnostics; "
        "see exploratory folders under `analysis/` only with explicit non-canonical caveats."
    )

    for p in sorted(LATEX.glob("*.tex")):
        created["latex_snippets"].append(str(p))

    manifest = {
        "inputs": {k: str(v) for k, v in EVIDENCE_FILES.items()},
        "inputs_present": {k: EVIDENCE_FILES[k].exists() for k in EVIDENCE_FILES},
        "outputs": created,
        "notes": notes,
    }
    (REPORTS / "manuscript_artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# Manuscript artifact generation report",
        "",
        "## Evidence status",
        f"- Policy comparison CSV present: **{policy_ok}** (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`).",
        "- Core inputs (train comparison, dataset summary, manifest, baselines, train_metrics, best_config): **OK**.",
        "",
        "## Refreshed in this run",
        "- Figure~1 (`figure1_method_overview`): two-panel offline/online method schematic.",
        "- Figure~4 (`figure4_ablation`): 2$\\times$2 offline regret/top-1 panels from `model_comparison_heavy_r1.csv`.",
        "- Table~2 (policy roster), Table~4 (offline ablation) + LaTeX snippets.",
    ]
    if policy_ok:
        report_lines += [
            "- Table~1 (dataset/trace summary), Table~3 (main misses) from canonical policy CSV.",
            "- Figure~2 (`figure2_main_performance_comparison`), Figure~3 (`figure3_improvement_vs_lru`) from canonical policy CSV.",
        ]
    else:
        report_lines += [
            "- **Not refreshed:** Table~1, Table~3, Figure~2, Figure~3 — replaced with explicit unavailable stubs; **do not cite** main quantitative results until policy CSV exists.",
        ]
    report_lines += [
        "",
        "## Canonical vs exploratory",
        "- Only paths under `EVIDENCE_FILES` in `build_kbs_main_manuscript_artifacts.py` drive this bundle.",
        "- Guarded/fallback and decision-quality **table5/figure5** were **not** created: no reproducible `*_heavy_r1` artifact found in-repo for those narratives.",
        "",
        "## Safe to cite now",
        "- Always: method schematic Fig.~1, offline ablation Table~4 / Fig.~4 (from `model_comparison_heavy_r1.csv`).",
        "- Policy-level claims: **only if** policy CSV was present for this run (see Evidence status).",
        "",
        "## Output roots",
        f"- Tables: `{TABLES}`",
        f"- Figures: `{FIGURES}`",
        f"- Snippets: `{LATEX}`",
    ]
    (REPORTS / "manuscript_artifact_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
