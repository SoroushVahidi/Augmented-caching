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


def main() -> None:
    root = repo_root()
    out_dir = root / "figures" / "manuscript"

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

    fig7 = make_continuation_policy_agreement_figure(cont_summary_rows, cont_agreement_rows)
    save_figure_pdf_png(fig7, out_dir, "figure7_continuation_policy_agreement")

    fig8 = make_target_construction_concept_figure()
    save_figure_pdf_png(fig8, out_dir, "figure8_target_construction_concept")

    print("Wrote figures to", out_dir)


if __name__ == "__main__":
    main()
