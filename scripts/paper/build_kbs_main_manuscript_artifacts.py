from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

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
# Baselines first, proposed method last (manuscript table row order).
TABLE3_POLICIES: Tuple[str, ...] = (
    "lru",
    "predictive_marker",
    "trust_and_doubt",
    "blind_oracle_lru_combiner",
    "rest_v1",
    "evict_value_v1",
)

SHORT_POLICY_LABEL = {
    "lru": "LRU",
    "evict_value_v1": "EV",
    "predictive_marker": "PredMk",
    "trust_and_doubt": "T\\&D",
    "blind_oracle_lru_combiner": "BO/LRU",
    "rest_v1": "REST",
}

# Canonical column order + display names for Table~3 (internal family keys → manuscript headers).
FAMILY_DISPLAY_ORDER: Tuple[str, ...] = (
    "brightkite",
    "citibike",
    "cloudphysics",
    "metacdn",
    "metakv",
    "twemcache",
    "wiki2018",
)
FAMILY_MANUSCRIPT_NAME: Dict[str, str] = {
    "brightkite": "BrightKite",
    "citibike": "CitiBike",
    "cloudphysics": "CloudPhysics",
    "metacdn": "MetaCDN",
    "metakv": "MetaKV",
    "twemcache": "Twemcache",
    "wiki2018": "Wiki2018",
}

# Compact column headers for Table~3 (full names in caption).
FAMILY_HEADER_CODE: Dict[str, str] = {
    "brightkite": "BK",
    "citibike": "CB",
    "cloudphysics": "CP",
    "metacdn": "MC",
    "metakv": "MK",
    "twemcache": "TC",
    "wiki2018": "W18",
}

MODEL_SORT_ORDER: Tuple[str, ...] = ("hist_gb", "random_forest", "ridge")


def _model_order_key(model: str) -> int:
    return MODEL_SORT_ORDER.index(model) if model in MODEL_SORT_ORDER else 99


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


def _parse_heavy_r1_capacities_from_dataset_summary_md(path: Path) -> List[int]:
    """Read capacity list from `evict_value_v1_wulver_dataset_summary_heavy_r1.md` (## Rows by capacity)."""
    caps: List[int] = []
    in_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("## Rows by capacity"):
            in_section = True
            continue
        if in_section:
            if line.startswith("## ") and not line.strip().startswith("## Rows by capacity"):
                break
            s = line.strip()
            if s.startswith("- "):
                key = s[2:].split(":", 1)[0].strip()
                try:
                    caps.append(int(key))
                except ValueError:
                    pass
    return sorted(set(caps))


def _manifest_rows_ordered() -> List[Dict[str, str]]:
    rows = _read_csv(EVIDENCE_FILES["trace_manifest"])

    def sort_key(r: Dict[str, str]) -> Tuple[int, str]:
        fam = str(r["trace_family"])
        idx = FAMILY_DISPLAY_ORDER.index(fam) if fam in FAMILY_DISPLAY_ORDER else 99
        return (idx, str(r["trace_name"]))

    return sorted(rows, key=sort_key)


def _table1_rows(policy_rows: Optional[List[Dict[str, str]]]) -> Tuple[List[Dict[str, object]], str]:
    """Build dataset summary rows from canonical manifest; merge per-trace capacities from policy CSV when present."""
    caps_default = _parse_heavy_r1_capacities_from_dataset_summary_md(EVIDENCE_FILES["dataset_summary"])
    if not caps_default:
        caps_default = [32, 64, 128, 256]

    traces_from_policy: Dict[str, set] = defaultdict(set)
    if policy_rows:
        for r in policy_rows:
            traces_from_policy[str(r["trace_name"])].add(int(float(r["capacity"])))

    out_rows: List[Dict[str, object]] = []
    for man in _manifest_rows_ordered():
        tname = str(man["trace_name"])
        path = str(man["path"])
        family = str(man["trace_family"])
        reqs, _pages, _src = load_trace_from_any(path)
        src = str(man.get("dataset_source", "") or "").strip() or "—"
        if policy_rows and tname in traces_from_policy:
            caps = ",".join(str(c) for c in sorted(traces_from_policy[tname]))
            in_main = "yes"
            basis = "manifest+policy_csv"
            note = "Wulver pool"
        else:
            caps = ",".join(str(c) for c in caps_default)
            in_main = "yes" if policy_rows else "--"
            basis = "manifest+dataset_summary"
            note = "Caps: heavy_r1 spec" if not policy_rows else "Wulver pool"

        out_rows.append(
            {
                "trace": tname,
                "family": family,
                "requests": len(reqs),
                "capacities": caps,
                "hint": "Next-arr.",
                "role": "heavy_r1 eval",
                "source": src,
                "in_main_comparison": in_main,
                "note": note,
                "evidence_basis": basis,
            }
        )

    ver = "table1_capacities_from_policy_csv" if policy_rows else "table1_capacities_from_dataset_summary_md_not_verified_against_eval_csv"
    return out_rows, ver


def _build_table1(policy_rows: Optional[List[Dict[str, str]]]) -> Tuple[Path, Path, str]:
    out_rows, ver = _table1_rows(policy_rows)
    csv_path = TABLES / "table1_dataset_summary.csv"
    tex_path = TABLES / "table1_dataset_summary.tex"
    _write_csv(csv_path, out_rows)
    lines = [
        "\\begin{tabular}{@{}l l r l l l l l@{}}",
        "\\toprule",
        "Trace & Family & Req. & Caps. & Hint & Role & Src. & Main \\\\",
        "\\midrule",
    ]
    for r in out_rows:
        lines.append(
            f"{_latex_escape(str(r['trace']))} & {_latex_escape(str(r['family']))} & {r['requests']} & "
            f"{_latex_escape(str(r['capacities']))} & {_latex_escape(str(r['hint']))} & {_latex_escape(str(r['role']))} & "
            f"{_latex_escape(str(r['source']))} & {_latex_escape(str(r['in_main_comparison']))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cap_extra = ""
    if not policy_rows:
        cap_extra = (
            " \\textit{Note:} per-trace capacities are taken from the canonical heavy\\_r1 dataset summary "
            "(not from `policy\\_comparison\\_heavy\\_r1.csv`, which is absent); re-run the builder after eval to align caps with the live comparison."
        )
    LATEX.joinpath("table1_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Traces used for Wulver heavy\\_r1 training and evaluation (ordered by family). "
        "\\textbf{Main:} included in the intended main comparison roster. "
        "Hint column: next-arrival style metadata available at prediction time."
        + cap_extra
        + "}\n"
        "\\label{tab:dataset-summary}\n"
        "\\input{tables/manuscript/table1_dataset_summary.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path, ver


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


def _build_table2_policy_roster() -> Tuple[Path, Path]:
    rows = [
        {
            "policy": "lru",
            "category": "classical",
            "label": "LRU",
            "role": "reference",
            "note": "Recency baseline",
        },
        {
            "policy": "predictive_marker",
            "category": "robust LA",
            "label": "PredMk",
            "role": "baseline",
            "note": "Prediction-aware marker",
        },
        {
            "policy": "trust_and_doubt",
            "category": "robust LA",
            "label": "T\\&D",
            "role": "baseline",
            "note": "Adaptive trust/doubt",
        },
        {
            "policy": "blind_oracle_lru_combiner",
            "category": "combiner",
            "label": "BO/LRU",
            "role": "baseline",
            "note": "BO/LRU combiner",
        },
        {
            "policy": "rest_v1",
            "category": "heuristic",
            "label": "REST",
            "role": "baseline",
            "note": "Selective trust + fallback",
        },
        {
            "policy": "evict_value_v1",
            "category": "proposed",
            "label": "EV",
            "role": "ours",
            "note": "Learned eviction-value policy",
        },
    ]
    csv_path = TABLES / "table2_policy_roster.csv"
    tex_path = TABLES / "table2_policy_roster.tex"
    _write_csv(csv_path, rows)
    lines = [
        "\\begin{tabular}{@{}l l l l p{3.25cm}@{}}",
        "\\toprule",
        "Policy & Category & Label & Role & Note \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"\\texttt{{{_latex_escape(str(r['policy']))}}} & "
            f"{_latex_escape(str(r['category']))} & {r['label']} & {_latex_escape(str(r['role']))} & {_latex_escape(str(r['note']))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LATEX.joinpath("table2_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Main comparison policies (labels align with Table~\\ref{tab:main-comparison}).}\n"
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
    families_raw = sorted({k[0] for k in fam_pol.keys()})
    families: List[str] = []
    for f in FAMILY_DISPLAY_ORDER:
        if f in families_raw:
            families.append(f)
    for f in families_raw:
        if f not in families:
            families.append(f)

    degenerate: List[str] = []
    for fam in families:
        means = []
        for pol in use_policies:
            k = (fam, pol)
            if k in fam_pol:
                means.append(mean(fam_pol[k]))
        if len(means) >= 2 and (max(means) - min(means)) < 1e-6:
            degenerate.append(FAMILY_MANUSCRIPT_NAME.get(fam, fam))

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

    n_f = len(families)
    col_spec = "l" + "r" * (n_f + 3)
    fam_heads = [FAMILY_HEADER_CODE.get(f, f[:3]) for f in families]
    head = [""] + fam_heads + ["All", r"$\Delta_{\mathrm{LRU}}$", "Win"]
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
        cells = [str(r["label"])]
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
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fam_legend = (
        "Column codes: "
        + "; ".join(f"{FAMILY_HEADER_CODE.get(f, f)}={FAMILY_MANUSCRIPT_NAME.get(f, f)}" for f in families)
        + "."
    )
    tie_note = ""
    if degenerate:
        tie_note = (
            " Near-tie columns (identical mean misses across policies within numerical tolerance): "
            + ", ".join(degenerate)
            + "."
        )
    cap = (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Main quantitative comparison: mean replay misses by trace family (\\textbf{lower is better}). "
        "Bold: best; underline: second-best in each numeric column. "
        r"$\Delta_{\mathrm{LRU}}$ is overall mean misses (across listed families) minus LRU's overall mean. "
        r"\textbf{Win:} number of families where the policy strictly beats LRU (lower misses)."
        + " "
        + fam_legend
        + tie_note
        + "}\n"
        "\\label{tab:main-comparison}\n"
        "\\input{tables/manuscript/table3_main_quantitative_comparison.tex}\n\\end{table*}\n"
    )
    LATEX.joinpath("table3_snippet.tex").write_text(cap, encoding="utf-8")
    return csv_path, tex_path


def _build_table3_unavailable() -> Tuple[Path, Path]:
    csv_path = TABLES / "table3_main_quantitative_comparison.csv"
    tex_path = TABLES / "table3_main_quantitative_comparison.tex"
    _write_csv(
        csv_path,
        [
            {
                "status": "NOT_VERIFIED",
                "detail": "Canonical file analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv is absent.",
                "action": "Run heavy_r1 eval, then rebuild manuscript artifacts.",
            }
        ],
    )
    tex_path.write_text(
        "\\begin{tabular}{@{}l p{10.5cm}@{}}\n"
        "\\toprule\n"
        "\\textbf{Status} & \\textbf{Reason} \\\\\n"
        "\\midrule\n"
        "Not regenerated & "
        "Canonical policy comparison "
        "\\texttt{evict\\_value\\_wulver\\_v1\\_policy\\_comparison\\_heavy\\_r1.csv} was not found. "
        "Do not copy numbers from exploratory or unsuffixed policy-comparison CSVs. \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )
    LATEX.joinpath("table3_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{\\textbf{Main quantitative comparison unavailable.} "
        "Regenerate after producing canonical heavy\\_r1 eval output; see \\texttt{docs/kbs\\_manuscript\\_workflow.md}.}\n"
        "\\label{tab:main-comparison-unavailable}\n"
        "\\input{tables/manuscript/table3_main_quantitative_comparison.tex}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return csv_path, tex_path


def _build_table4_ablation(train_rows: List[Dict[str, str]], best_cfg: Optional[dict]) -> Tuple[Path, Path]:
    rows: List[Dict[str, object]] = []
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
    rows = sorted(rows, key=lambda x: (x["horizon"], _model_order_key(str(x["model"]))))
    csv_path = TABLES / "table4_main_ablation.csv"
    tex_path = TABLES / "table4_main_ablation.tex"
    _write_csv(csv_path, rows)

    by_h: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_h[int(r["horizon"])].append(r)

    metric_keys = ("val_mean_regret", "test_mean_regret", "val_top1", "test_top1")
    best_m: Dict[Tuple[int, str, str], bool] = {}
    second_m: Dict[Tuple[int, str, str], bool] = {}
    for h, lst in by_h.items():
        for key in metric_keys:
            vals = sorted({float(x[key]) for x in lst})
            bst = vals[0]
            snd = vals[1] if len(vals) > 1 else vals[0]
            for x in lst:
                m = str(x["model"])
                v = float(x[key])
                if abs(v - bst) <= 1e-9:
                    best_m[(h, m, key)] = True
                elif len(vals) > 1 and abs(v - snd) <= 1e-9:
                    second_m[(h, m, key)] = True

    dec = 4

    def _cell(h: int, m: str, key: str, v: float) -> str:
        s = f"{v:.{dec}f}"
        if best_m.get((h, m, key)):
            return f"\\textbf{{{s}}}"
        if second_m.get((h, m, key)):
            return f"\\underline{{{s}}}"
        return s

    lines = [
        "\\begin{tabular}{@{}r l r r r r@{}}",
        "\\toprule",
        "Horizon & Model & Val.\\ regret $\\downarrow$ & Test regret $\\downarrow$ & Val.\\ Top-1 $\\downarrow$ & Test Top-1 $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        h = int(r["horizon"])
        m = str(r["model"])
        lines.append(
            f"{h} & {_latex_escape(m)} & "
            f"{_cell(h, m, 'val_mean_regret', float(r['val_mean_regret']))} & "
            f"{_cell(h, m, 'test_mean_regret', float(r['test_mean_regret']))} & "
            f"{_cell(h, m, 'val_top1', float(r['val_top1']))} & "
            f"{_cell(h, m, 'test_top1', float(r['test_top1']))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sel = ""
    if best_cfg:
        bm = str(best_cfg.get("model", ""))
        sel = (
            " Selected configuration (\\texttt{best\\_config\\_heavy\\_r1.json}): "
            f"horizon {best_cfg.get('horizon')}, model \\texttt{{{_latex_escape(bm)}}}."
        )
    LATEX.joinpath("table4_snippet.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Offline eviction-value ablation on heavy\\_r1 shards "
        "(\\texttt{evict\\_value\\_wulver\\_v1\\_model\\_comparison\\_heavy\\_r1.csv}). "
        "All metrics: \\textbf{lower is better}. "
        "Bold: best per column within each horizon; underline: second-best."
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
    policy_rows: Optional[List[Dict[str, str]]] = (
        _read_csv(EVIDENCE_FILES["policy_comparison"]) if policy_ok else None
    )

    apply_manuscript_matplotlib_style()
    plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7})

    created: Dict[str, List[str]] = {"tables": [], "figures": [], "latex_snippets": []}
    notes: List[str] = []

    if not EVIDENCE_FILES["policy_comparison_md"].exists():
        notes.append("Optional `policy_comparison_heavy_r1.md` missing; CSV is the quantitative source of truth.")

    t1_csv, t1_tex, table1_ver_tag = _build_table1(policy_rows)
    created["tables"] += [str(t1_csv), str(t1_tex)]
    notes.append(f"Table~1 evidence: {table1_ver_tag}.")

    t2 = _build_table2_policy_roster()
    t4 = _build_table4_ablation(train_rows, best_cfg)
    for pair in [t2, t4]:
        created["tables"] += [str(pair[0]), str(pair[1])]

    f1 = _figure1_method_overview()
    f4 = _figure4_ablation(train_rows)
    for pair in [f1, f4]:
        created["figures"] += [str(pair[0]), str(pair[1])]

    if policy_ok:
        assert policy_rows is not None
        t3 = _build_table3_main_comparison(policy_rows)
        created["tables"] += [str(t3[0]), str(t3[1])]
        f2 = _figure2_main_performance_comparison(policy_rows)
        f3 = _figure3_improvement_vs_lru(policy_rows)
        for pair in [f2, f3]:
            created["figures"] += [str(pair[0]), str(pair[1])]
        notes.append("Table~3 and Figures~2--3 rebuilt from canonical `policy_comparison_heavy_r1.csv`.")
    else:
        t3 = _build_table3_unavailable()
        created["tables"] += [str(t3[0]), str(t3[1])]
        notes.append(
            "Table~3 not verified; Figures~2--3 not built: "
            "`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` missing."
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
        "table1_evidence_basis": table1_ver_tag,
        "policy_comparison_present": policy_ok,
    }
    (REPORTS / "manuscript_artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# Manuscript artifact generation report",
        "",
        "## Evidence status",
        f"- Policy comparison CSV present: **{policy_ok}** (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`).",
        "- Core inputs (train comparison, dataset summary, manifest, baselines, train_metrics, best_config): **OK**.",
        "",
        "## Manuscript table readiness",
        f"- **Table~1 (dataset summary):** regenerated from `wulver_trace_manifest_full.csv` + live request counts (`load_trace_from_any`). "
        f"Evidence tag: `{table1_ver_tag}`.",
    ]
    if policy_ok:
        report_lines += [
            "  - **Manuscript-safe:** yes (capacities aligned to policy CSV rows).",
        ]
    else:
        report_lines += [
            "  - **Manuscript-safe:** usable as a trace roster; **Main** column is `--` until policy CSV exists to confirm the eval run roster. "
            "Capacities listed from `evict_value_v1_wulver_dataset_summary_heavy_r1.md` (not from a policy comparison file).",
        ]
    report_lines += [
        "- **Table~2 (policy roster):** compact roster; manuscript-safe.",
        "- **Table~3 (main quantitative comparison):** "
        + (
            "**verified** against canonical `policy_comparison_heavy_r1.csv`."
            if policy_ok
            else "**not** generated — `.tex` marks missing canonical evidence; **do not** cite numeric misses from older commits."
        ),
        "- **Table~4 (offline ablation):** from `evict_value_wulver_v1_model_comparison_heavy_r1.csv`; manuscript-safe.",
        "",
        "## Refreshed in this run",
        "- Figure~1 (`figure1_method_overview`): two-panel offline/online method schematic.",
        "- Figure~4 (`figure4_ablation`): offline regret / top-1 panels from `model_comparison_heavy_r1.csv`.",
        "- Tables~1--4 (CSV + `.tex`) and matching `latex_snippets/*.tex` where applicable.",
    ]
    if policy_ok:
        report_lines += [
            "- Figure~2 (`figure2_main_performance_comparison`), Figure~3 (`figure3_improvement_vs_lru`) from canonical policy CSV.",
        ]
    else:
        report_lines += [
            "- **Not built:** Figure~2, Figure~3 (require policy comparison CSV).",
        ]
    report_lines += [
        "",
        "## Stale / replaced content",
        "- Any previously committed Table~3 numeric body from a run **without** the canonical policy CSV should be treated as **invalid** once this report shows policy CSV absent.",
        "",
        "## Canonical vs exploratory",
        "- Only paths under `EVIDENCE_FILES` in `build_kbs_main_manuscript_artifacts.py` drive this bundle.",
        "- Guarded/fallback and decision-quality **table5/figure5** were **not** created: no reproducible `*_heavy_r1` artifact found in-repo for those narratives.",
        "",
        "## Safe to cite now",
        "- Always: method schematic Fig.~1; offline ablation Table~4 / Fig.~4 (from `model_comparison_heavy_r1.csv`).",
        "- Table~1 trace list: **yes**, with the capacity caveat above if policy CSV is absent.",
        "- Table~2 roster: **yes**.",
        "- Main quantitative numbers (Table~3 / Figs.~2--3): **only if** policy CSV was present for this run.",
        "",
        "## Output roots",
        f"- Tables: `{TABLES}`",
        f"- Figures: `{FIGURES}`",
        f"- Snippets: `{LATEX}`",
    ]
    (REPORTS / "manuscript_artifact_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
