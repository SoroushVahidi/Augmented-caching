#!/usr/bin/env python3
"""Generate supplemental manuscript-style figures from existing lightweight artifacts.

Figures:
- Regret vs Top-1 alignment across horizons/models: analysis/evict_value_wulver_v1_model_comparison.csv
- Continuation-policy offline metrics + label agreement: analysis/continuation_policy_light/{summary.csv,label_agreement.csv}
- Conceptual target-construction schematic (no external artifacts)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from manuscript_figure_common import (  # noqa: E402
    apply_manuscript_matplotlib_style,
    make_continuation_policy_agreement_figure,
    make_regret_vs_top1_alignment_figure,
    make_target_construction_concept_figure,
    repo_root,
    save_figure_pdf_png,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _write_supplemental_figure_snippet(latex_dir: Path, stem: str, caption: str, label: str, width: str) -> Path:
    latex_dir.mkdir(parents=True, exist_ok=True)
    path = latex_dir / f"{stem}_snippet.tex"
    body = (
        f"% Supplemental figure (asset filename `{stem}.pdf`). "
        "Renumber in the manuscript if this collides with another ``figure6'' class asset "
        "(e.g. `figure6_guard_wrapper_evict_value_v1`).\n"
        "\\begin{figure*}[t]\n\\centering\n"
        f"\\includegraphics[width={width}]{{figures/manuscript/{stem}.pdf}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure*}\n"
    )
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    root = repo_root()
    out_dir = root / "figures" / "manuscript"
    latex_dir = root / "reports" / "manuscript_artifacts" / "latex_snippets"

    train_csv = root / "analysis" / "evict_value_wulver_v1_model_comparison.csv"
    cont_summary = root / "analysis" / "continuation_policy_light" / "summary.csv"
    cont_agreement = root / "analysis" / "continuation_policy_light" / "label_agreement.csv"

    for p in (train_csv, cont_summary, cont_agreement):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    apply_manuscript_matplotlib_style()

    train_rows = _read_csv(train_csv)
    cont_summary_rows = _read_csv(cont_summary)
    cont_agreement_rows = _read_csv(cont_agreement)

    fig6 = make_regret_vs_top1_alignment_figure(train_rows)
    save_figure_pdf_png(fig6, out_dir, "figure6_regret_vs_top1_alignment")
    s6 = _write_supplemental_figure_snippet(
        latex_dir,
        "figure6_regret_vs_top1_alignment",
        r"Offline alignment of Top-1 eviction match vs.\ mean regret vs.\ oracle across horizons and model families "
        r"(\textbf{validation} and \textbf{test}; marker size encodes horizon). "
        r"Supplemental to the main offline ablation; not an end-to-end policy-replay comparison.",
        "fig:regret-top1-alignment-supplement",
        r"0.98\textwidth",
    )

    fig7 = make_continuation_policy_agreement_figure(cont_summary_rows, cont_agreement_rows)
    save_figure_pdf_png(fig7, out_dir, "figure7_continuation_policy_agreement")
    s7 = _write_supplemental_figure_snippet(
        latex_dir,
        "figure7_continuation_policy_agreement",
        r"Lightweight continuation-policy experiment: \textbf{(a)}~offline Top-1 eviction match and mean regret by protocol; "
        r"\textbf{(b)}~pairwise label-agreement matrix between protocols. Supplemental diagnostic only.",
        "fig:continuation-policy-agreement-supplement",
        r"0.98\textwidth",
    )

    fig8 = make_target_construction_concept_figure()
    save_figure_pdf_png(fig8, out_dir, "figure8_target_construction_concept")
    s8 = _write_supplemental_figure_snippet(
        latex_dir,
        "figure8_target_construction_concept",
        r"Schematic of eviction-value target construction: counterfactual rollouts per candidate eviction, oracle losses over the next $H$ requests, "
        r"and supervised fit of an eviction-value head. Concept figure only (not trace-specific metrics).",
        "fig:target-construction-concept",
        r"0.96\textwidth",
    )

    print("Wrote figures to", out_dir)
    print("Wrote LaTeX snippets:\n ", s6, "\n ", s7, "\n ", s8, sep="")


if __name__ == "__main__":
    main()
