# Reviewer guide

This is the technical entry point for inspecting the KBS second-revision
evidence. It is organized by scientific question rather than by project
chronology.

Status terms:

- `PRIMARY`: used for the main corrected evidence in the manuscript.
- `SUPPORTING`: used to explain or bound a manuscript claim.
- `HISTORICAL`: retained for provenance, not current evidence.
- `RUNNING/PENDING`: not part of the current primary evidence.

## 1. Main paper result

Scientific question: does `evict_value_v1`, a finite-horizon candidate-level
eviction-loss target, improve online replay performance?

| Field | Location |
|---|---|
| Manuscript | Section 3.6, Table 7; conclusion section |
| Artifact | [reports/kbs_final_evidence_20260813/major1_full_baseline_comparison.csv](../../reports/kbs_final_evidence_20260813/major1_full_baseline_comparison.csv) |
| Generating/audit script | [scripts/analysis/prepare_r2_major1_evidence.py](../../scripts/analysis/prepare_r2_major1_evidence.py) for LRB/3L-Cache/CACHEUS; independent recomputation documented in [major1_reviewer_summary.md](../../reports/kbs_final_evidence_20260813/major1_reviewer_summary.md) |
| Validation | [major1_protocol_comparability.md](../../reports/kbs_final_evidence_20260813/major1_protocol_comparability.md), [heldout_treatment_integrity.md](../../reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md) |
| Status | `PRIMARY` |

Result: under matched evaluation, `evict_value_v1` loses on a clear majority
of matched cells against every listed baseline.

## 2. Primary matched comparison

Matched dimensions: seven trace families, capacities 32/64/128, object-slot
capacity, unit object size, history `[0,10000)`, scored suffix
`[10000,50000)`, 40,000 scored requests, identical hit/miss accounting, and
trace identity checked by SHA-256.

| Field | Location |
|---|---|
| Manuscript | Section 3.6.1-3.6.4, Table 7 |
| Artifact | [major1_protocol_comparability.md](../../reports/kbs_final_evidence_20260813/major1_protocol_comparability.md) |
| Generating scripts | [scripts/experiments/run_cross_family_heldout_eval.py](../../scripts/experiments/run_cross_family_heldout_eval.py), [scripts/experiments/run_reviewer_fairness.py](../../scripts/experiments/run_reviewer_fairness.py) |
| Validation | 21/21 key match by `(trace_sha256, capacity)` in [major1_protocol_comparability.md](../../reports/kbs_final_evidence_20260813/major1_protocol_comparability.md) |
| Status | `PRIMARY` |

## 3. Reviewer #2 Major 1 - baseline comparisons

| Baseline | Implementation status | Artifact | Method/provenance | Status |
|---|---|---|---|---|
| LRU | Native implementation | `analysis/reviewer_fairness/policy_comparison_lru.csv` | [docs/baselines.md](../baselines.md) | `PRIMARY` |
| FIFO-Reinsertion | Native implementation | `analysis/reviewer_fairness/policy_comparison_fifo_reinsertion.csv` | [docs/baselines.md](../baselines.md) | `PRIMARY` |
| SIEVE | Native implementation | `analysis/reviewer_fairness/policy_comparison_sieve.csv` | [docs/baselines.md](../baselines.md), [tests/test_sieve.py](../../tests/test_sieve.py) | `PRIMARY` |
| LRB | Independent reimplementation under matched inputs | `analysis/reviewer_fairness/policy_comparison_lrb.csv` | [docs/lrb_method_spec.md](../lrb_method_spec.md), [tests/test_lrb.py](../../tests/test_lrb.py) | `PRIMARY` |
| 3L-Cache | Independent reimplementation/adaptation | `analysis/reviewer_fairness/policy_comparison_three_l_cache.csv` | [docs/three_l_cache_method_spec.md](../three_l_cache_method_spec.md), [docs/three_l_cache_provenance.md](../three_l_cache_provenance.md) | `PRIMARY_WITH_CAVEAT` |
| CACHEUS | Official upstream source through wrapper/adaptor | `analysis/reviewer_fairness/policy_comparison_cacheus.csv` | [docs/cacheus_method_spec.md](../cacheus_method_spec.md), [docs/cacheus_provenance.md](../cacheus_provenance.md) | `PRIMARY_WITH_PROVENANCE_CAVEAT` |
| HALP | Supporting reconstruction/adaptation; no public official code | `analysis/reviewer_fairness/policy_comparison_halp.csv` | [docs/halp_method_spec.md](../halp_method_spec.md), [docs/halp_provenance.md](../halp_provenance.md) | `SUPPORTING` |

Combined result table: [major1_full_baseline_comparison.csv](../../reports/kbs_final_evidence_20260813/major1_full_baseline_comparison.csv).

## 4. Reviewer #2 Major 2 - supervision objective

Scientific question: is eviction-loss supervision better than plausible
alternative finite-horizon objectives?

| Field | Location |
|---|---|
| Manuscript | Section 3.7, Table 8 |
| Artifact | `analysis/supervision_objective_ablation_v1/policy_comparison.csv` |
| Generating script | [scripts/experiments/run_supervision_objective_ablation.py](../../scripts/experiments/run_supervision_objective_ablation.py) |
| Validation | [scripts/experiments/audit_supervision_objective_examples.py](../../scripts/experiments/audit_supervision_objective_examples.py), [scripts/experiments/audit_supervision_objective_fairness.py](../../scripts/experiments/audit_supervision_objective_fairness.py), [tests/test_audit_supervision_objective_examples.py](../../tests/test_audit_supervision_objective_examples.py), [tests/test_audit_supervision_objective_fairness.py](../../tests/test_audit_supervision_objective_fairness.py) |
| Status | `SUPPORTING` in evidence class; manuscript result is complete |

The result is negative for the proposed objective: eviction-loss is worst or
tied-worst among the four tested objectives.

## 5. Reviewer #2 Major 3 - online/trajectory mismatch

| Question | Artifact | Script | Validation | Status |
|---|---|---|---|---|
| Does exact optimization of the target explain the gap? | `analysis/exact_target_oracle_replication_v1/` | [scripts/experiments/run_exact_target_oracle_replication.py](../../scripts/experiments/run_exact_target_oracle_replication.py) | [tests/test_exact_target_oracle_replication.py](../../tests/test_exact_target_oracle_replication.py) | `SUPPORTING` |
| Does the learned scorer match the exact target? | `analysis/learned_exact_target_agreement_v1/` | [scripts/experiments/run_learned_exact_target_agreement.py](../../scripts/experiments/run_learned_exact_target_agreement.py) | [tests/test_learned_exact_target_agreement.py](../../tests/test_learned_exact_target_agreement.py) | `SUPPORTING` |
| Is the H=4 target degenerate? | `analysis/strict_preference_horizon_diagnostic_v1/` | [scripts/experiments/run_strict_preference_horizon_diagnostic.py](../../scripts/experiments/run_strict_preference_horizon_diagnostic.py) | [tests/test_strict_preference_horizon_diagnostic.py](../../tests/test_strict_preference_horizon_diagnostic.py) | `SUPPORTING` |
| Does generic DAgger-style state-shift reduction improve misses? | [distribution_shift_summary.csv](../../reports/kbs_final_evidence_20260813/distribution_shift_summary.csv) | [scripts/experiments/run_distribution_shift_ablation.py](../../scripts/experiments/run_distribution_shift_ablation.py) | [distribution_integrity_summary.md](../../reports/kbs_final_evidence_20260813/distribution_integrity_summary.md) | `SUPPORTING` |

## 6. Reviewer #2 Major 4 - computational overhead

| Field | Location |
|---|---|
| Manuscript | Section 3.10, Table 9; Section 3.11 |
| Artifact | [controlled_timing_summary.csv](../../reports/kbs_final_evidence_20260813/controlled_timing_summary.csv) |
| Raw source | `analysis/kbs_controlled_timing_20260810/raw_timing_runs.csv` |
| Validation | [controlled_timing_integrity.md](../../reports/kbs_final_evidence_20260813/controlled_timing_integrity.md), [controlled_timing_interpretation.md](../../reports/kbs_final_evidence_20260813/controlled_timing_interpretation.md) |
| Status | `PRIMARY` for the four-policy controlled timing table; `evict_value_v1` timing is separate single-run context |

## 7. Reviewer #3 - continuation mismatch

| Field | Location |
|---|---|
| Manuscript | Section 3.9 |
| Artifact | [c0_continuation_summary.csv](../../reports/kbs_final_evidence_20260813/c0_continuation_summary.csv) |
| Generating script | [scripts/experiments/run_continuation_policy_causal_ablation.py](../../scripts/experiments/run_continuation_policy_causal_ablation.py) |
| Validation | [c0_integrity_summary.md](../../reports/kbs_final_evidence_20260813/c0_integrity_summary.md), [tests/test_continuation_policy_ablation.py](../../tests/test_continuation_policy_ablation.py), [tests/test_continuation_policy_production_runner.py](../../tests/test_continuation_policy_production_runner.py) |
| Status | `SUPPORTING`; complete mechanistic result |

Conclusion: continuation mismatch is a real but partial, family-dependent
contributor, not a universal explanation.

## 8. Baseline provenance

Start with [docs/baselines.md](../baselines.md), then inspect the per-baseline
records:

- LRB: [docs/lrb_method_spec.md](../lrb_method_spec.md)
- 3L-Cache: [docs/three_l_cache_method_spec.md](../three_l_cache_method_spec.md), [docs/three_l_cache_provenance.md](../three_l_cache_provenance.md)
- CACHEUS: [docs/cacheus_method_spec.md](../cacheus_method_spec.md), [docs/cacheus_provenance.md](../cacheus_provenance.md)
- HALP: [docs/halp_method_spec.md](../halp_method_spec.md), [docs/halp_provenance.md](../halp_provenance.md)
- SIEVE and lightweight baselines: [docs/baselines.md](../baselines.md), [tests/test_sieve.py](../../tests/test_sieve.py)

## 9. Dataset provenance

| Family | Classification | Provenance |
|---|---|---|
| BrightKite | Non-cache event stream transformed to requests | [docs/datasets.md](../datasets.md), [reports/kbs_verified_literature_20260813.md](../../reports/kbs_verified_literature_20260813.md) |
| Citi Bike | Non-cache event stream transformed to requests | [docs/datasets.md](../datasets.md), [reports/kbs_verified_literature_20260813.md](../../reports/kbs_verified_literature_20260813.md) |
| CloudPhysics | Storage-derived trace | [docs/datasets.md](../datasets.md) |
| MetaCDN | Cache-derived trace | [docs/datasets.md](../datasets.md) |
| MetaKV | Cache-derived trace | [docs/datasets.md](../datasets.md) |
| Twemcache | Production cache trace | [docs/datasets.md](../datasets.md) |
| Wikimedia | Pageview-derived proxy | [docs/datasets_wulver_trace_acquisition.md](../datasets_wulver_trace_acquisition.md) |

Raw datasets are not necessarily redistributed; upstream access and licensing
constraints remain in force.

## 10. Integrity / validation checks

Use [RESULT_VERIFICATION.md](RESULT_VERIFICATION.md) for the checklist. The
most important compact integrity summaries are:

- [heldout_treatment_integrity.md](../../reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md)
- [major1_protocol_comparability.md](../../reports/kbs_final_evidence_20260813/major1_protocol_comparability.md)
- [c0_integrity_summary.md](../../reports/kbs_final_evidence_20260813/c0_integrity_summary.md)
- [distribution_integrity_summary.md](../../reports/kbs_final_evidence_20260813/distribution_integrity_summary.md)
- [controlled_timing_integrity.md](../../reports/kbs_final_evidence_20260813/controlled_timing_integrity.md)

## 11. Reproduction commands

Detailed commands, inputs, outputs, approximate cost, and availability are in
[REPRODUCTION_MATRIX.md](REPRODUCTION_MATRIX.md). Cheap checks:

```bash
PYTHONPATH=src pytest tests/test_sieve.py tests/test_lrb.py tests/test_three_l_cache.py tests/test_halp.py tests/test_cacheus.py -q
PYTHONPATH=src pytest tests/test_exact_target_oracle_replication.py tests/test_continuation_policy_ablation.py -q
git diff --check
```

Full campaigns are not cheap smoke tests; inspect committed summaries first.

## 12. Historical / non-primary results

The historical single-split and `heavy_r1` results are retained for
provenance, not as current primary evidence. Do not use
`analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` for primary
claims; it is contaminated by train/test overlap. Use the corrected held-out
evidence summarized in [reports/kbs_final_evidence_20260813/](../../reports/kbs_final_evidence_20260813/).

Two acceptance-risk controls are currently running and are not yet primary:

- `kbs_common_model_objective_control_20260813_final`
- `kbs_tie_aware_exact_oracle_20260813_final`
