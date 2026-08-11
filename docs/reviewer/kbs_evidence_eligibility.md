# KBS second-revision evidence eligibility

This note defines which artifacts may be used for which purpose. It exists to
prevent accidental mixing of valid, partial, smoke-only, contaminated, and
historical outputs.

## Primary reviewer table

Use only:

- `analysis/reviewer_fairness/policy_comparison_lrb.csv`
- `analysis/reviewer_fairness/policy_comparison_three_l_cache.csv`
- `analysis/reviewer_fairness/policy_comparison_halp.csv`
- `analysis/reviewer_fairness/policy_comparison_cacheus.csv`
- the non-learned fairness-protocol CSVs in `analysis/reviewer_fairness/`

Rule:

- filter to `policy_variant=primary_controlled_window`

Do not mix in:

- `deployment_full_stream` rows,
- contaminated `evict_value_v1` fair-window rows,
- partial cross-family `evict_value_v1` results before the held-out evaluation
  campaign is complete.

## Supporting analysis

Usable with explicit caveats:

- `analysis/supervision_objective_ablation_v1/policy_comparison.csv`
- completed and audited cells under
  `analysis/supervision_objective_learning_curve_v1/`
- objective-ablation registry and final audits preserved in the sibling
  objective-ablation worktree
- local distribution-shift diagnostic files in
  `analysis/distribution_shift_ablation_v1/`
- practical-significance smoke outputs in
  `analysis/practical_significance_ablation_v1/`

Supporting-analysis use still requires the status labels in
[`kbs_second_revision_artifact_map.md`](kbs_second_revision_artifact_map.md).

## Diagnostic-only analysis

Use only as diagnostics, not as final evidence:

- `analysis/supervision_objective_learning_curve_v1/`
- local distribution-shift partial checkpoint
  `analysis/distribution_shift_ablation_v1/`
- trajectory-divergence and state-shift diagnostics from the same directory
- practical-significance smoke timing and break-even outputs

Diagnostic-only means:

- useful for explaining what was observed,
- not usable for a final all-families claim,
- not usable as proof that a mechanism is the sole cause of the miss gap.

For `analysis/supervision_objective_learning_curve_v1/`, use the explicit
classification `DIAGNOSTIC_PARTIAL` unless and until a clean stopped campaign
has been fully audited.

Current local checkpoint to preserve:

- validated low-fraction cells at `1%, 2%, 5%, 10%`
- completed families in that audited checkpoint:
  `brightkite, citibike, cloudphysics, metacdn`
- `16` validated units
- `96` validated rows
- `25%` and `50%` extensions are now complete and audited. The final 50%
  closeout has 7/7 families, 42/42 rows, all `status=ok`, and synthesis at
  `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`.

## Historical context

Usable only when explicitly labeled historical:

- `analysis/*_heavy_r1.*`
- `docs/wulver_heavy_evict_value_experiment.md`
- `docs/evict_value_v1_kbs_canonical_artifacts.md`

These files are important provenance, but they are not the default source of
truth for the current reviewer-science branch.

## Not usable

Explicitly not usable for the primary reviewer table:

- `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`
- any `deployment_full_stream` row from the fairness-protocol CSVs
- any partial distribution-shift aggregate interpreted as the full seven-family
  campaign
- any practical-significance timing result described without the `SMOKE_ONLY`
  qualifier
- any header-only or failed held-out evaluation output
- any incomplete learning-curve aggregate treated as if it were a final
  manuscript result
- any learning-curve result pooled with `objective_pairwise` as though it were
  the same condition

Current local note on failed or header-only held-out runs:

- no header-only failures were found in the current
  `analysis/reviewer_fairness_cross_family_v1/` family directories,
  but any such historical artifact should be treated as not usable scientific
  evidence.

## Concrete guardrails

### A. `deployment_full_stream` rows

These are not eligible for the primary controlled reviewer comparison. They may
be mentioned only as supporting context when clearly distinguished from the
controlled window.

### B. Old `policy_comparison_evict_value_v1.csv`

This file is contaminated and ineligible for the primary reviewer comparison.
The overlap is documented in:

- `analysis/reviewer_fairness/evict_value_v1_overlap_audit.json`
- `analysis/reviewer_fairness/evict_value_v1_overlap_audit.md`

### C. Local `analysis/distribution_shift_ablation_v1/`

This is a valid local checkpoint, but it is still a partial campaign and must
not be used as if it were the final seven-family result.

### D. `analysis/practical_significance_ablation_v1/`

This directory currently contains smoke-scale evidence only. Until controlled
timing exists, it should be used only for implementation and methodological
discussion.

### E. `analysis/supervision_objective_learning_curve_v1/`

This directory is a local explanatory diagnostic comparing:

- `eviction_loss_scalar`
- `eviction_loss_pairwise`

under a same-example guarantee derived from the same underlying eviction-loss
labels.

Guardrails:

- completed and audited cells may be inspected,
- incomplete fraction or fold aggregates must not be used as final manuscript
  evidence,
- the `25%` and `50%` local extensions are complete and audited,
- `100%` is intentionally not run due `STOP_SAMPLE_SIZE_HYPOTHESIS`, not
  missing current evidence,
- results must not be merged with the earlier `objective_pairwise` ablation as
  though they represented the same condition,
- the campaign state at clean wall-time stop must be preserved.

### F. Failed or header-only held-out runs

Treat them as historical failure provenance only, never as scientific evidence.
