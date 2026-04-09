# sentinel_budgeted_guard_v2 component ablation report

## Setup
- Same lightweight disagreement-stress suite as prior v2 check (3 synthetic traces × capacities [2,3]).
- Variants: v1 baseline, v2 full, v2 without each main component, plus single-component variants.

## Aggregate by variant
| variant | mean_misses | mean_delta_vs_v1 | mean_delta_vs_v2_full | harmful_overrides | helpful_overrides |
|---|---:|---:|---:|---:|---:|
| v1_baseline | 12.500 | 0.000 | -0.333 | 2 | 5 |
| v2_full | 12.833 | 0.333 | 0.000 | 2 | 3 |
| v2_no_override_budget | 12.500 | 0.000 | -0.333 | 2 | 5 |
| v2_no_temporary_guard | 12.833 | 0.333 | 0.000 | 2 | 3 |
| v2_no_reentry_gating | 12.833 | 0.333 | 0.000 | 2 | 3 |
| v2_budget_only | 12.833 | 0.333 | 0.000 | 2 | 3 |
| v2_guard_only | 12.500 | 0.000 | -0.333 | 2 | 5 |
| v2_reentry_only | 12.500 | 0.000 | -0.333 | 2 | 5 |

## Explicit answers
- **Which v2 component hurts most?** `override_budget` (removing it gives the largest miss reduction vs v2 full: -0.333).
- **Is any single v2 component actually useful?** No: none of the single-component variants beat v1 on mean misses.
- **Is there a simpler v1.5 variant that keeps the best part of v2 without the harmful parts?** Candidate: `v1_baseline` (best mean misses among simple ablation variants).
- **What should be the main empirical candidate after this ablation?** `v1_baseline` (lowest mean misses in this ablation suite).

## Notes
- Negative delta means fewer misses (better).
- Positive delta means more misses (worse).
