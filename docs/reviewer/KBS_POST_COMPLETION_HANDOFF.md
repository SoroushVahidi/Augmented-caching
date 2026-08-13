# KBS Second-Revision Post-Completion Handoff

Status decision: **NO_NEW_EXPERIMENT_REQUIRED**.

Snapshot date: 2026-08-12. This document distinguishes durable campaign
status from live progress. Counts for active campaigns are snapshots only;
their manifests and logs are authoritative for later progress checks. Partial
outputs are not citable scientific evidence.

No sixth heavy experiment should be launched unless a completed campaign fails
integrity validation or the reviewer/editor explicitly requests additional
evidence.

## Final-validated campaigns

| Campaign | Output | Validated scope | Established result |
|---|---|---|---|
| Exact-target oracle replication | `analysis/exact_target_oracle_replication_v1/` | 21/21 units, 42/42 rows | Oracle beats LRU 0/21, ties 3/21, loses 18/21 |
| Strict-preference/horizon diagnostic | `analysis/strict_preference_horizon_diagnostic_v1/` | 21/21 units, 63/63 comparisons | H4 unique-winner fraction 0; multiple-optimum fraction 1 |
| Learned/exact agreement and regret | `analysis/learned_exact_target_agreement_v1/` | 21/21 units, 21/21 summaries | Set-aware agreement ≈0.975301; positive regret ≈0.024699; learned misses 601,569 vs LRU 565,126 |

These campaigns must not be rerun. Their generated outputs remain local and
ignored; the runners, configs, tests, and this handoff are durable source.

## Active campaigns

| Campaign | Reviewer coverage | Question | Session | Log | Output | Expected final |
|---|---|---|---|---|---|---|
| C0/C1/C2 continuation-policy causal ablation | R2 Major 3, R3 | Does continuation-policy mismatch causally contribute to the offline-to-online gap? | `kbs_continuation_c0_c1_c2_production_resume2_retry_20260812` | `logs/kbs_continuation_c0_c1_c2_production_resume2_retry_20260812.log` | `analysis/continuation_policy_causal_ablation_production_v1/` | 21 units; 63 policy rows plus diagnostics |
| Distribution-shift completion | R2 Major 3 | Does generic DAgger-style state-shift reduction improve online cache performance? | `kbs_distribution_shift_completion_resume2_20260812` | `logs/kbs_distribution_shift_completion_resume2_20260812.log` | `analysis/distribution_shift_ablation_v1/` | 42 primary rows; 21 paired cells |

Do not relaunch a session or rerun completed units merely to make the snapshot
symmetric.

## Recorded launch commands

These are the commands observed in the active workers. They are recorded for
identification and audit only; do not execute them again.

```text
scripts/experiments/run_continuation_policy_causal_ablation.py --config configs/continuation_policy_causal_ablation_production_v1.json --data-read-root /home/soroush/Augmented-caching --resume --max-wall-hours 8
scripts/experiments/run_exact_target_oracle_replication.py --config configs/exact_target_oracle_replication_v1.json --out-dir analysis/exact_target_oracle_replication_v1 --data-read-root /home/soroush/Augmented-caching --determinism-check
scripts/experiments/run_strict_preference_horizon_diagnostic.py --config configs/strict_preference_horizon_diagnostic_v1.json --out-dir analysis/strict_preference_horizon_diagnostic_v1 --data-read-root /home/soroush/Augmented-caching
scripts/experiments/run_learned_exact_target_agreement.py --config configs/learned_exact_target_agreement_v1.json --out-dir analysis/learned_exact_target_agreement_v1 --data-read-root /home/soroush/Augmented-caching
scripts/experiments/run_distribution_shift_ablation.py --config configs/distribution_shift_ablation_v1.json --max-wall-hours 9.0 --models-dir models/distribution_shift_ablation_v1 --out-dir analysis/distribution_shift_ablation_v1 --resume --data-read-root /home/soroush/Augmented-caching
```

## Completion gate

When any campaign naturally finishes, do not immediately rerun it or interpret
its final-looking CSV. First verify:

1. expected unit count;
2. expected aggregate-row count;
3. duplicate-key absence;
4. NaN/Inf absence;
5. all statuses successful;
6. canonical trace hashes;
7. config and protocol identity;
8. model hashes where applicable;
9. exclusion of partial units from aggregates;
10. provenance and manifest reconciliation.

Only after every gate passes may the campaign be classified
`FINAL_VALIDATED`. Any failed gate is
`COMPLETE_BUT_INVALID_PENDING_DIAGNOSIS`; do not silently repair or rerun.

## Reviewer mapping

Completion of C0/C1/C2 supplies the controlled causal test for the
continuation-policy explanation. Completion of the exact-target, strict-
preference, and learned/exact diagnostics supplies family/capacity evidence
about target resolution, horizon stability, and model fitting. Distribution-
shift completion tests whether generic state-shift reduction translates into
online improvement. None of these partial outputs may be used as final claims.

## No-new-experiment decision

The following are not reasons to launch new compute for this revision:

- H7 hard-argmin or uncertainty-aware selection;
- formal H10 scaling-law validation;
- H11 causal excess-miss attribution;
- fallback or gating;
- selective invocation;
- the intentionally stopped 100% learning curve;
- additional random seeds;
- synthetic traces;
- additional capacities or trace families.

These remain optional future work or not-needed extensions unless a completed
campaign fails integrity or the reviewer/editor changes the request.

## Post-campaign non-compute work

After the five campaigns are audited:

1. synchronize and verify the corrected held-out `evict_value_v1` 42/42 payload;
2. synchronize and verify the 420/420 controlled timing payload;
3. optionally synchronize broader Wulver degeneracy, horizon, and historical-tail evidence if it is cited;
4. consolidate canonical evidence bundles;
5. update the hypothesis map and reviewer coverage;
6. update the manuscript and response to reviewers;
7. run the final scientific-consistency audit;
8. prepare the submission package.

Wulver synchronization is outside this handoff and must not be initiated here.
