#!/usr/bin/env python3
"""Regenerate manuscript PDF+PNG for Figure~1 (method) and Figure~4 (offline ablation) without policy CSV.

Requires: analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv
Does not require: evict_value_wulver_v1_policy_comparison_heavy_r1.csv

Run from repository root: python scripts/paper/regenerate_evidence_aligned_manuscript_figures.py
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
    make_method_overview_two_panel_figure,
    make_offline_ablation_figure,
    repo_root,
    save_figure_pdf_png,
)


def main() -> None:
    root = repo_root()
    train_csv = root / "analysis" / "evict_value_wulver_v1_model_comparison_heavy_r1.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}")
    train_rows = list(csv.DictReader(train_csv.open(encoding="utf-8")))
    out = root / "figures" / "manuscript"
    apply_manuscript_matplotlib_style()
    f1 = make_method_overview_two_panel_figure()
    save_figure_pdf_png(f1, out, "figure1_method_overview")
    f4 = make_offline_ablation_figure(train_rows)
    save_figure_pdf_png(f4, out, "figure4_ablation")
    print(f"Wrote {out / 'figure1_method_overview.pdf'} and {out / 'figure4_ablation.pdf'}")


if __name__ == "__main__":
    main()
