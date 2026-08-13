# KBS second-revision roadmap

This file is retained as a high-level project-management note. It is not the
reviewer entry point and should not duplicate the current evidence tables.

Use these current sources first:

- [README.md](../README.md)
- [docs/reviewer/START_HERE.md](reviewer/START_HERE.md)
- [docs/reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md](reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md)
- [docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md](reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md)
- [docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md](reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md)
- [reports/kbs_final_evidence_20260813/](../reports/kbs_final_evidence_20260813/)

## Current status

The previous second-revision evidence campaign is complete and validated:

- Major 1 matched baseline comparison;
- Major 2 supervision-objective ablation;
- exact-target, target-degeneracy, and learned/exact diagnostics;
- C0/C1/C2 continuation-policy causal ablation;
- DAgger-style distribution-shift ablation;
- controlled timing campaign.

The current primary evidence remains the curated package in
[reports/kbs_final_evidence_20260813/](../reports/kbs_final_evidence_20260813/).

## Running acceptance-risk controls

Two additional controls are running and are not yet integrated into the
manuscript or primary evidence:

- `kbs_common_model_objective_control_20260813_final`
- `kbs_tie_aware_exact_oracle_20260813_final`

Do not inspect, summarize, or cite their output until each has completed and
passed integrity review.

## Primary ineligible historical result

`analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` is
contaminated/historical and must not be used as primary evidence. Use the
corrected held-out matched comparison in
[reports/kbs_final_evidence_20260813/](../reports/kbs_final_evidence_20260813/).

## Historical role

Older detailed roadmap entries were useful while experiments were pending, but
they are no longer maintained here. Detailed current status is in the
experiment registry and reviewer coverage map linked above.
