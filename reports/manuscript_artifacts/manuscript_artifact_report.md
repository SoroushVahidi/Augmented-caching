# Manuscript artifact generation report

## Evidence status
- Policy comparison CSV present: **False** (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`).
- Core inputs (train comparison, dataset summary, manifest, baselines, train_metrics, best_config): **OK**.

## Refreshed in this run
- Figure~1 (`figure1_method_overview`): two-panel offline/online method schematic.
- Figure~4 (`figure4_ablation`): **two-panel** (validation / test mean regret vs horizon) from `model_comparison_heavy_r1.csv`. **Shared legend** in a **dedicated row below both panels** (GridSpec) so **no legend overlays data** in panel~(b). Panel labels `(a)`/`(b)` upper-left. Plotted coordinates **numerically match** `table4_main_ablation` / `model_comparison_heavy_r1.csv` (`val_mean_regret`, `test_mean_regret`; horizons {4, 8, 16}; models `ridge`, `random_forest`, `hist_gb`).
- Table~2 (policy roster), Table~4 (offline ablation) + LaTeX snippets.
- **Not refreshed:** Table~1, Table~3, Figure~2, Figure~3 — replaced with explicit unavailable stubs; **do not cite** main quantitative results until policy CSV exists.

## Canonical vs exploratory
- Only paths under `EVIDENCE_FILES` in `build_kbs_main_manuscript_artifacts.py` drive this bundle.
- Guarded/fallback and decision-quality **table5/figure5** were **not** created: no reproducible `*_heavy_r1` artifact found in-repo for those narratives.

## Safe to cite now
- Always: method schematic Fig.~1, offline ablation Table~4 / Fig.~4 (from `model_comparison_heavy_r1.csv`).
- Policy-level claims: **only if** policy CSV was present for this run (see Evidence status).

## Output roots
- Tables: `tables/manuscript`
- Figures: `figures/manuscript`
- Snippets: `reports/manuscript_artifacts/latex_snippets`
