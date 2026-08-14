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

## Completed acceptance-risk controls

Both authorized acceptance-risk controls are complete and audited; they are
not yet integrated into the manuscript.

- `common_model_objective_control_v2` completed on Wulver (Slurm job
  `1176758`): 21/21 tasks, all `ExitCode 0:0`, reducer + integrity audit PASS;
  see
  [reports/common_model_v2_formal_audit_20260814/AUDIT.md](../reports/common_model_v2_formal_audit_20260814/AUDIT.md).
- Tie-aware exact-target oracle v1 completed locally (21/21 units, 189/189
  rows, integrity PASS after campaign-CSV recovery); see
  [reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md](../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md).

## Primary ineligible historical result

`analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` is
contaminated/historical and must not be used as primary evidence. Use the
corrected held-out matched comparison in
[reports/kbs_final_evidence_20260813/](../reports/kbs_final_evidence_20260813/).

## Historical role

Older detailed roadmap entries were useful while experiments were pending, but
they are no longer maintained here. Detailed current status is in the
experiment registry and reviewer coverage map linked above.
