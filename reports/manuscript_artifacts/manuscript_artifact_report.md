# Manuscript artifact generation report

## Evidence status
- Policy comparison CSV present: **False** (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`).
- Core inputs (train comparison, dataset summary, manifest, baselines, train_metrics, best_config): **OK**.

## Manuscript table readiness
- **Table~1 (dataset summary):** regenerated from `wulver_trace_manifest_full.csv` + live request counts (`load_trace_from_any`). Evidence tag: `table1_capacities_from_dataset_summary_md_not_verified_against_eval_csv`.
  - **Manuscript-safe:** usable as a trace roster; **Main** column is `--` until policy CSV exists to confirm the eval run roster. Capacities listed from `evict_value_v1_wulver_dataset_summary_heavy_r1.md` (not from a policy comparison file).
- **Table~2 (policy roster):** compact roster; manuscript-safe.
- **Table~3 (main quantitative comparison):** **not** generated — `.tex` marks missing canonical evidence; **do not** cite numeric misses from older commits.
- **Table~4 (offline ablation):** from `evict_value_wulver_v1_model_comparison_heavy_r1.csv`; manuscript-safe.

## Refreshed in this run
- Figure~1 (`figure1_method_overview`): two-panel offline/online method schematic.
- Figure~4 (`figure4_ablation`): offline regret / top-1 panels from `model_comparison_heavy_r1.csv`.
- Tables~1--4 (CSV + `.tex`) and matching `latex_snippets/*.tex` where applicable.
- **Not built:** Figure~2, Figure~3 (require policy comparison CSV).

## Stale / replaced content
- Any previously committed Table~3 numeric body from a run **without** the canonical policy CSV should be treated as **invalid** once this report shows policy CSV absent.

## Canonical vs exploratory
- Only paths under `EVIDENCE_FILES` in `build_kbs_main_manuscript_artifacts.py` drive this bundle.
- Guarded/fallback and decision-quality **table5/figure5** were **not** created: no reproducible `*_heavy_r1` artifact found in-repo for those narratives.

## Safe to cite now
- Always: method schematic Fig.~1; offline ablation Table~4 / Fig.~4 (from `model_comparison_heavy_r1.csv`).
- Table~1 trace list: **yes**, with the capacity caveat above if policy CSV is absent.
- Table~2 roster: **yes**.
- Main quantitative numbers (Table~3 / Figs.~2--3): **only if** policy CSV was present for this run.

## Output roots
- Tables: `tables/manuscript`
- Figures: `figures/manuscript`
- Snippets: `reports/manuscript_artifacts/latex_snippets`
