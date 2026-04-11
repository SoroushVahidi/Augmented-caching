# Repository hygiene cleanup report (2026-04-11)

Scope: documentation and path-clarity cleanup only. No experiments rerun, no scientific result changes, no numeric artifact edits.

## Files changed and confusion fixed

1. `README.md`
   - Made the canonical `heavy_r1` manuscript path more explicit as the main-paper route.
   - Added a short post-eval checklist (verify heavy file → build manuscript bundle) so handoff after cluster completion is obvious.
   - Added a direct pointer to lightweight exploratory ablation index.

2. `analysis/README.md`
   - Strengthened canonical boundary language for `analysis/*_heavy_r1.*`.
   - Grouped lightweight exploratory outputs (`incoming_*_light`, `history_*_light`, `joint_*_light`) so they are discoverable but clearly non-canonical.

3. `docs/repo_map.md`
   - Added `scripts/experiments/` as a first-class exploratory ablation family.
   - Added explicit mention of `analysis/*_light/` exploratory grouping.
   - Added lightweight ablation index to the manuscript-support reading order.

4. `scripts/README.md`
   - Added structured table mapping lightweight ablation scripts to output directories.
   - Clarified this entire family is exploratory/non-canonical relative to KBS `heavy_r1`.

5. `docs/reproducibility_and_artifacts.md`
   - Added minimal canonical post-heavy-eval command sequence for artifact generation.
   - Added explicit exploratory label for lightweight ablations with index link.

6. `docs/kbs_manuscript_submission_index.md`
   - Added lightweight ablations to the exploratory section.
   - Added two-command handoff block for after heavy eval completion.

7. `docs/lightweight_exploratory_ablations.md` (new)
   - New centralized index for incoming-aware, history-aware, history-objective/history-pairwise-style, and joint-state lightweight branches.
   - Explicitly states non-canonical status versus `heavy_r1`.

8. `scripts/paper/README.md`
   - Added a canonical preflight check for `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` before building manuscript artifacts.
   - Clarified that heavy eval completion is a prerequisite.

9. `docs/repo_hygiene_cleanup_report_2026-04-11.md` (new)
   - Added this concise file-by-file cleanup ledger and remaining runtime-dependent blocker.

## Remaining issue that still depends on running job state

- Canonical handoff still depends on `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` existing from successful completion of `slurm/evict_value_v1_wulver_heavy_eval.sbatch` (with `EXP_TAG=heavy_r1`). Until that file exists, manuscript bundle regeneration for the canonical path cannot be confirmed complete.
