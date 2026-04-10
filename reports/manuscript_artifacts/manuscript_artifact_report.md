# Manuscript artifact generation report

## Strongest manuscript-safe basis selected
- Main comparison: `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (heavy eval; see `docs/evict_value_v1_kbs_canonical_artifacts.md`).
- Main training/ablation: `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` and `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json`.
- Dataset coverage: `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` and `analysis/wulver_trace_manifest_full.csv`.

## Created tables
- Table 1: dataset/trace summary.
- Table 2: policy roster.
- Table 3: main quantitative comparison (bold best, underline second-best).
- Table 4: model-family/horizon ablation for evict_value_v1.

## Created figures
- Figure 1: method overview schematic.
- Figure 2: family-level main performance comparison.
- Figure 3: aggregate improvement vs LRU.
- Figure 4: ablation plot (val/test mean regret).

## Skipped or constrained items
- Guarded/fallback ablation specifically on the same heavy Wulver artifact pool was not found as a dedicated canonical artifact; main ablation uses in-pool model-family/horizon evidence instead.

## Output roots
- Tables: `tables/manuscript`
- Figures: `figures/manuscript`
- Manifest/report/LaTeX snippets: `reports/manuscript_artifacts`
