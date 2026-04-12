# `figures/manuscript/` — generated figures (PDF + PNG)

Figures are mostly produced by **`scripts/paper/build_kbs_main_manuscript_artifacts.py`** from canonical **`heavy_r1`** analysis inputs where applicable.

**Optional guard schematic (not from main builder):** `figure6_guard_wrapper_evict_value_v1.{pdf,png}` — run `python scripts/paper/build_guard_wrapper_manuscript_figure.py`. LaTeX: `reports/manuscript_artifacts/latex_snippets/figure6_guard_wrapper_snippet.tex`. Report: `reports/manuscript_artifacts/figure6_guard_wrapper_report.md`.

**Supplemental figures (separate driver):** `figure6_regret_vs_top1_alignment`, `figure7_continuation_policy_agreement`, `figure8_target_construction_concept` from `scripts/paper/build_additional_eviction_value_figures.py`. Matching floats: `reports/manuscript_artifacts/latex_snippets/{figure6_regret_vs_top1_alignment,figure7_continuation_policy_agreement,figure8_target_construction_concept}_snippet.tex` (regenerated with that script). Use only if the narrative matches. **Figure-number collision:** two different assets use a “figure6” stem; renumber in LaTeX.

**Quality audit:** `reports/manuscript_artifacts/manuscript_artifact_quality_audit.md`.

**Dependency:** Main performance figures (e.g. Fig.~2–3) require **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`**; see `reports/manuscript_artifacts/manuscript_artifact_report.md` for what was built in the last run.

**Navigation:** [`../../CANONICAL_KBS_SUBMISSION.md`](../../CANONICAL_KBS_SUBMISSION.md), [`../../docs/kbs_manuscript_workflow.md`](../../docs/kbs_manuscript_workflow.md).
