# KBS Second-Revision Post-Completion Handoff

Status decision: the previous second-revision campaign is complete; two
additional acceptance-risk controls are currently running and are not yet
primary evidence.

Snapshot date: 2026-08-13 (updated from 2026-08-12: both previously-active
campaigns below have completed and passed formal post-completion integrity
audit; moved to "Final-validated campaigns"). This document distinguishes
durable campaign status from live progress. Counts for active campaigns are
snapshots only; their manifests and logs are authoritative for later
progress checks. Partial outputs are not citable scientific evidence.

No prior second-revision campaign should be relaunched unless a completed
campaign fails integrity validation or the reviewer/editor explicitly requests
additional evidence.

## Final-validated campaigns

| Campaign | Output | Validated scope | Established result |
|---|---|---|---|
| Exact-target oracle replication | `analysis/exact_target_oracle_replication_v1/` | 21/21 units, 42/42 rows | Oracle beats LRU 0/21, ties 3/21, loses 18/21 |
| Strict-preference/horizon diagnostic | `analysis/strict_preference_horizon_diagnostic_v1/` | 21/21 units, 63/63 comparisons | H4 unique-winner fraction 0; multiple-optimum fraction 1 |
| Learned/exact agreement and regret | `analysis/learned_exact_target_agreement_v1/` | 21/21 units, 21/21 summaries | Set-aware agreement ≈0.975301; positive regret ≈0.024699; learned misses 601,569 vs LRU 565,126 |
| C0/C1/C2 continuation-policy causal ablation | `analysis/continuation_policy_causal_ablation_production_v1/` | 21/21 units, 63/63 policy rows, 21/21 label-agreement rows, 21/21 training-summary rows | C2 improves over C1 in 13/21 cells (macro delta ≈−0.0102), worsens 5/21 (largest: `brightkite` cap32, +0.2433), ties 3/21 (degenerate Wiki2018); H5 `PARTIALLY_SUPPORTED` |
| Distribution-shift completion | `analysis/distribution_shift_ablation_v1/` | 7/7 folds, 42/42 primary rows, 42/42 state-shift rows, 21/21 trajectory rows | DAgger worsens misses in 16/21 cells (macro delta ≈+0.0094) despite improving the state-shift index in 16/21; H6 `DISFAVORED` as a shift-reduction-improves-performance story |

These campaigns must not be rerun. Their generated outputs remain local and
ignored; the runners, configs, tests, and this handoff are durable source.
Compact tracked evidence for the last two: `reports/kbs_final_evidence_20260813/`.
Both completed naturally (tmux sessions
`kbs_continuation_c0_c1_c2_production_resume2_retry_20260812` and
`kbs_distribution_shift_completion_resume2_20260812` have exited; see
`logs/kbs_continuation_c0_c1_c2_production_resume2_retry_20260812.log` and
`logs/kbs_distribution_shift_completion_resume2_20260812.log`) and each
passed a formal, read-only post-completion integrity audit on 2026-08-13
(all ten completion gates below PASS for both).

## Active campaigns

Two additional acceptance-risk controls are active as of 2026-08-13:

- `kbs_common_model_objective_control_20260813_final`
- `kbs_tie_aware_exact_oracle_20260813_final`

They are not part of the primary evidence, are not integrated into the
manuscript, and should not be summarized until they complete and pass integrity
review.

## Recorded launch commands

These are the commands observed in the active workers. They are recorded for
identification and audit only; do not execute them again.

```text
REPO_ROOT=/path/to/Augmented-caching
scripts/experiments/run_continuation_policy_causal_ablation.py --config configs/continuation_policy_causal_ablation_production_v1.json --data-read-root "$REPO_ROOT" --resume --max-wall-hours 8
scripts/experiments/run_exact_target_oracle_replication.py --config configs/exact_target_oracle_replication_v1.json --out-dir analysis/exact_target_oracle_replication_v1 --data-read-root "$REPO_ROOT" --determinism-check
scripts/experiments/run_strict_preference_horizon_diagnostic.py --config configs/strict_preference_horizon_diagnostic_v1.json --out-dir analysis/strict_preference_horizon_diagnostic_v1 --data-read-root "$REPO_ROOT"
scripts/experiments/run_learned_exact_target_agreement.py --config configs/learned_exact_target_agreement_v1.json --out-dir analysis/learned_exact_target_agreement_v1 --data-read-root "$REPO_ROOT"
scripts/experiments/run_distribution_shift_ablation.py --config configs/distribution_shift_ablation_v1.json --max-wall-hours 9.0 --models-dir models/distribution_shift_ablation_v1 --out-dir analysis/distribution_shift_ablation_v1 --resume --data-read-root "$REPO_ROOT"
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

Completion of C0/C1/C2 supplied the controlled causal test for the
continuation-policy explanation (H5 `PARTIALLY_SUPPORTED`). Completion of
the exact-target, strict-preference, and learned/exact diagnostics supplied
family/capacity evidence about target resolution, horizon stability, and
model fitting. Distribution-shift completion tested whether generic
state-shift reduction translates into online improvement (H6 `DISFAVORED`).
All six campaigns above are now `FINAL_VALIDATED`; see
`reports/kbs_final_evidence_20260813/reviewer_mapping.md` for the full
Reviewer #2 Major 1-4 / Reviewer #3 status table.

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

All six local campaigns above are now audited (`FINAL_VALIDATED`). Remaining
work:

1. ~~synchronize and verify the corrected held-out `evict_value_v1` 42/42 payload~~
   -- **done 2026-08-13**, under explicit task authorization: synced from
   `login02:/mmfs1/project/ikoutis/sv96/Augmented-caching`, 16/16 transfer
   hashes PASS, independent local re-audit PASS. See
   `reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md`;
2. ~~synchronize and verify the 420/420 controlled timing payload~~ -- **done
   2026-08-13**, same authorization: 13/13 transfer hashes PASS, independent
   local re-audit PASS, policy means recomputed and matched. See
   `reports/kbs_final_evidence_20260813/controlled_timing_integrity.md`;
3. optionally synchronize broader Wulver degeneracy, horizon, and historical-tail evidence if it is cited;
4. consolidate canonical evidence bundles -- **done for C0/C1/C2,
   distribution-shift, corrected held-out treatment, and controlled timing**:
   `reports/kbs_final_evidence_20260813/`;
5. update the hypothesis map and reviewer coverage -- **done**:
   `docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`,
   `docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`;
6. update the manuscript and response to reviewers -- not yet done; items 1-2
   have now landed, so this can proceed;
7. run the final scientific-consistency audit -- items 1-2 have landed;
8. prepare the submission package.

Items 1-2 were completed in a task explicitly authorized to sync these two
specific FINAL_VALIDATED payloads, verify by SHA-256, and update evidence
documentation; no new experiments, retraining, or Wulver-side modification
occurred. Further ad hoc Wulver synchronization beyond these two payloads
remains outside this handoff unless separately authorized.
