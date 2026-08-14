# Result verification

This document lists the mechanisms a reader can inspect to evaluate code and
result correctness. It does not claim third-party validation or multi-author
replication.

## Result schemas and expected cardinalities

- Corrected held-out treatment:
  [heldout_treatment_integrity.md](../../reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md)
  records 42/42 rows and local re-audit gates for the synced treatment
  artifact.
- Major 1 matched comparison:
  [major1_protocol_comparability.md](../../reports/kbs_final_evidence_20260813/major1_protocol_comparability.md)
  verifies 21/21 primary cells per baseline, matched by trace SHA-256,
  capacity, window, budget, and metric.
- Continuation C0/C1/C2:
  [c0_integrity_summary.md](../../reports/kbs_final_evidence_20260813/c0_integrity_summary.md)
  verifies 21/21 units and 63/63 policy rows.
- Distribution-shift:
  [distribution_integrity_summary.md](../../reports/kbs_final_evidence_20260813/distribution_integrity_summary.md)
  verifies 7/7 folds, 42/42 primary rows, 42/42 state-shift rows, and 21/21
  trajectory rows.
- Controlled timing:
  [controlled_timing_integrity.md](../../reports/kbs_final_evidence_20260813/controlled_timing_integrity.md)
  verifies 420/420 timing rows.

## Duplicate-key checks

The integrity summaries above report duplicate-key absence for the result
tables they audit. The relevant keys include `(unit_id, condition)` for C0/C1/C2,
fold/capacity/condition keys for distribution-shift, and `(policy, family,
capacity, repetition)` for controlled timing.

## Held-out-family leakage checks

Held-out-family separation for the corrected `evict_value_v1` treatment is
documented in:

- [docs/reviewer_fairness_cross_family_v1.md](../reviewer_fairness_cross_family_v1.md)
- [heldout_treatment_integrity.md](../../reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md)
- [heldout_treatment_provenance.md](../../reports/kbs_final_evidence_20260813/heldout_treatment_provenance.md)
- [configs/fair_cross_family_v1/folds/](../../configs/fair_cross_family_v1/folds/)

The stated guarantee is that the held-out family contributes no rows to
training, validation, hyperparameter selection, feature selection, horizon
selection, seed selection, or early stopping before that family's model is
frozen and evaluated.

## Deterministic replay/regression checks

Regression tests cover the key implementation paths:

- Baselines: [tests/test_lrb.py](../../tests/test_lrb.py), [tests/test_three_l_cache.py](../../tests/test_three_l_cache.py), [tests/test_cacheus.py](../../tests/test_cacheus.py), [tests/test_halp.py](../../tests/test_halp.py), [tests/test_sieve.py](../../tests/test_sieve.py).
- Corrected held-out evaluation: [tests/test_run_cross_family_heldout_eval.py](../../tests/test_run_cross_family_heldout_eval.py), [tests/test_evict_value_v1_cross_family_eval.py](../../tests/test_evict_value_v1_cross_family_eval.py).
- Objective and mechanistic diagnostics: [tests/test_supervision_objective_ablation.py](../../tests/test_supervision_objective_ablation.py), [tests/test_exact_target_oracle_replication.py](../../tests/test_exact_target_oracle_replication.py), [tests/test_strict_preference_horizon_diagnostic.py](../../tests/test_strict_preference_horizon_diagnostic.py), [tests/test_learned_exact_target_agreement.py](../../tests/test_learned_exact_target_agreement.py).
- Corrected common-model V2 control: [tests/test_common_model_objective_control_v2.py](../../tests/test_common_model_objective_control_v2.py) covers the pairwise orientation correction, score-call caching, feature-only deployment row equivalence, held-out-fold invariants, resume behavior, and reducer cardinality checks.
- Continuation and distribution-shift runners: [tests/test_continuation_policy_ablation.py](../../tests/test_continuation_policy_ablation.py), [tests/test_continuation_policy_production_runner.py](../../tests/test_continuation_policy_production_runner.py), [tests/test_distribution_shift_ablation.py](../../tests/test_distribution_shift_ablation.py), [tests/test_audit_distribution_shift_completion.py](../../tests/test_audit_distribution_shift_completion.py).

## Artifact hashes

Hash checks are recorded where available:

- Corrected held-out treatment: 16/16 transfer hashes pass in
  [heldout_treatment_provenance.md](../../reports/kbs_final_evidence_20260813/heldout_treatment_provenance.md).
- Controlled timing: 13/13 transfer hashes pass in
  [controlled_timing_integrity.md](../../reports/kbs_final_evidence_20260813/controlled_timing_integrity.md).
- Continuation C0/C1/C2: per-unit SHA-256 values match the unit manifest in
  [c0_integrity_summary.md](../../reports/kbs_final_evidence_20260813/c0_integrity_summary.md).

## Model-selection separation

The corrected held-out treatment uses leave-one-family-out training. Objective
ablation and learning-curve diagnostics record frozen model registries and
same-example gates in:

- [docs/reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md](KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md)
- `analysis/supervision_objective_ablation_v1/model_registry.json`
- [scripts/experiments/supervision_objective_ablation_gates.py](../../scripts/experiments/supervision_objective_ablation_gates.py)
- [scripts/experiments/audit_supervision_objective_examples.py](../../scripts/experiments/audit_supervision_objective_examples.py)

## Recomputed summaries

The curated package reports where summaries were independently recomputed from
raw CSVs:

- [major1_reviewer_summary.md](../../reports/kbs_final_evidence_20260813/major1_reviewer_summary.md)
- [controlled_timing_integrity.md](../../reports/kbs_final_evidence_20260813/controlled_timing_integrity.md)
- [distribution_integrity_summary.md](../../reports/kbs_final_evidence_20260813/distribution_integrity_summary.md)

## Tests covering baseline implementations

Baseline provenance and tests should be read together:

| Baseline | Provenance/method | Tests |
|---|---|---|
| LRB | [docs/lrb_method_spec.md](../lrb_method_spec.md) | [tests/test_lrb.py](../../tests/test_lrb.py) |
| 3L-Cache | [docs/three_l_cache_method_spec.md](../three_l_cache_method_spec.md), [docs/three_l_cache_provenance.md](../three_l_cache_provenance.md) | [tests/test_three_l_cache.py](../../tests/test_three_l_cache.py) |
| CACHEUS | [docs/cacheus_method_spec.md](../cacheus_method_spec.md), [docs/cacheus_provenance.md](../cacheus_provenance.md) | [tests/test_cacheus.py](../../tests/test_cacheus.py) |
| HALP | [docs/halp_method_spec.md](../halp_method_spec.md), [docs/halp_provenance.md](../halp_provenance.md) | [tests/test_halp.py](../../tests/test_halp.py) |
| SIEVE | [docs/baselines.md](../baselines.md) | [tests/test_sieve.py](../../tests/test_sieve.py) |

## Known verification gaps

- Some old model binaries are unavailable, so byte-level reverification of
  every historical model artifact is not possible.
- Not every third-party baseline is official code. LRB and 3L-Cache are
  independent reimplementations/adaptations; HALP is a reconstruction because
  no public official implementation exists.
- Raw datasets are not necessarily redistributed; some must be obtained from
  upstream sources and remain subject to upstream terms.
- CACHEUS uses official upstream source through an external wrapper, but the
  live external clone is not always present in this worktree.
- CI status is not documented here as a repository-level guarantee.
- Some script paths remain targeted for portability cleanup in a later query;
  reviewer-facing docs use `$REPO_ROOT` or relative paths.
- The initial common-model control V1 is superseded after implementation audit
  and is not integrated into current primary evidence. The corrected V2 control
  is regression-gated and pending a full integrity-audited run.
- The tie-aware exact-oracle control remains running/pending and is not
  integrated into current primary evidence.
