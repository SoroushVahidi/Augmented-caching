# Analysis artifacts guide

`analysis/` stores experiment outputs and small tracked audits.

## Current navigation

For the current KBS second-revision reviewer-science branch, start at:

- `docs/kbs_manuscript_workflow.md`
- `docs/reviewer/kbs_second_revision_artifact_map.md`
- `docs/reviewer/kbs_evidence_eligibility.md`

Historical `heavy_r1` builder-oriented material still starts at
`CANONICAL_KBS_SUBMISSION.md`.

## Taxonomy

| Kind | Typical location | Default interpretation |
|------|------------------|------------------------|
| Tracked small audits | `analysis/reviewer_fairness/*_audit.*` | Canonical helper artifacts |
| Reviewer experiment outputs | `analysis/reviewer_fairness/`, `analysis/distribution_shift_ablation_v1/`, `analysis/practical_significance_ablation_v1/`, `analysis/supervision_objective_ablation_v1/` | Usually generated evidence; check eligibility note |
| Local explanatory diagnostic | `analysis/supervision_objective_learning_curve_v1/` | `DIAGNOSTIC_PARTIAL` unless a clean stopped campaign has been fully audited |
| Historical heavy-run root files | `analysis/*_heavy_r1.*` | Historical builder inputs |
| Legacy or alternate root files | unsuffixed `analysis/evict_value_wulver_v1_*` | Historical or alternate-driver outputs |
| Experiment directories | `analysis/<name>/` | Experiment-specific generated outputs |
| Smoke outputs | `*_smoke*` or protocol-specific smoke dirs | Non-final implementation checks |

## Important caution

**Do not use `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`
as primary evidence.** It is a contaminated/historical comparison with
documented train/test overlap. It is retained temporarily for provenance only.
Use the corrected current evidence in
`reports/kbs_final_evidence_20260813/` instead, especially
`major1_full_baseline_comparison.csv`, `major1_protocol_comparability.md`,
and `heldout_treatment_integrity.md`.

This repository intentionally keeps:

- partial checkpoints,
- contaminated comparison outputs,
- smoke-only timing evidence,
- historical heavy-run artifacts.

Do not infer eligibility from filename similarity alone. Use
`docs/reviewer/kbs_evidence_eligibility.md` before citing any artifact in a
reviewer-facing table or narrative.
