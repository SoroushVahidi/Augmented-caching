# Result verification

This document tells a reviewer how to check the key numerical claims from
**files published on `main`**. No experiment was rerun for this publication.

Public repository: <https://github.com/SoroushVahidi/Augmented-caching>

## Matched comparison (manuscript Table 5)

1. Open [major1_reviewer_summary.md](../../reports/kbs_final_evidence_20260813/major1_reviewer_summary.md).
2. Confirm the compact table in
   [major1_full_baseline_comparison.csv](../../reports/kbs_final_evidence_20260813/major1_full_baseline_comparison.csv).
3. Spot-check a baseline CSV under
   [analysis/reviewer_fairness/](../../analysis/reviewer_fairness/)
   (21 `primary_controlled_window` cells; capacities 32/64/128).
4. Protocol statement:
   [major1_protocol_comparability.md](../../reports/kbs_final_evidence_20260813/major1_protocol_comparability.md).
5. Held-out EV treatment integrity:
   [heldout_treatment_integrity.md](../../reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md).

Allowed claim: `evict_value_v1` loses on a clear majority of matched cells
against every listed baseline.

## Workload-specific matched LRU gaps (manuscript Table 4)

Source files:

- [analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv](../../analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv)
- [analysis/reviewer_fairness/policy_comparison_lru.csv](../../analysis/reviewer_fairness/policy_comparison_lru.csv)

Filter `primary_controlled_window`. Relative gap is
`(EV_misses − LRU_misses) / LRU_misses` on the 40,000-request scored window.

Do **not** use a historical leaky single-split family-gap table.

## Common-Model V2 (manuscript Table 7)

1. Read [reports/common_model_v2_formal_audit_20260814/AUDIT.md](../../reports/common_model_v2_formal_audit_20260814/AUDIT.md).
2. Confirm aggregate totals in
   [analysis/common_model_objective_control_wulver_v2/summary.csv](../../analysis/common_model_objective_control_wulver_v2/summary.csv):
   eviction-loss 571,976; pairwise 577,339; reuse-distance 615,850;
   next-arrival 627,392.
3. Integrity files in the same directory:
   `integrity_audit.json`, `completion_manifest.json`.

Allowed claim: under this matched control, eviction-loss is **not** materially
worse; objective-causality for the deployed failure is **not supported**.

V1 is **superseded** and is not primary evidence.

## Tie-aware exact-target oracle (manuscript §3.6)

1. Read [reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md](../../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md).
   That audit is the primary record (integrity PASS; no unit rerun).
2. Confirm compact numbers in
   [analysis/tie_aware_exact_target_oracle_v1/summary.csv](../../analysis/tie_aware_exact_target_oracle_v1/summary.csv).

Headline interpretation:

- CURRENT_DETERMINISTIC vs LRU: 0 wins / 3 ties / 18 losses (+81,750 misses).
- LRU_WITHIN_MINIMA vs LRU: 16 / 5 / 0 (−413 misses).
- `fraction_tied_decisions = 1.0` on the audited rows.
- Deterministic deficit is a **tie confound**. LRU-within-minima does **not**
  mean H=4 improves LRU as a learned policy.

## Continuation and DAgger (manuscript §3.7)

- Continuation: [c0_continuation_summary.csv](../../reports/kbs_final_evidence_20260813/c0_continuation_summary.csv)
  and [c0_integrity_summary.md](../../reports/kbs_final_evidence_20260813/c0_integrity_summary.md).
  C2 improves over C1 in 13/21 cells, ties 3, worsens 5; BrightKite cap 32 is
  a large counter-example.
- DAgger: [distribution_shift_summary.csv](../../reports/kbs_final_evidence_20260813/distribution_shift_summary.csv)
  and [distribution_integrity_summary.md](../../reports/kbs_final_evidence_20260813/distribution_integrity_summary.md).
  Measured shift improves in most cells while misses worsen in most cells.

## Controlled timing (manuscript Table 8)

- [controlled_timing_summary.csv](../../reports/kbs_final_evidence_20260813/controlled_timing_summary.csv)
- [controlled_timing_integrity.md](../../reports/kbs_final_evidence_20260813/controlled_timing_integrity.md)
- [controlled_timing_interpretation.md](../../reports/kbs_final_evidence_20260813/controlled_timing_interpretation.md)

`evict_value_v1` is **not** in the four-policy repeated-measurement table.

## Hash / integrity notes already recorded

- Held-out treatment: 16/16 transfer hashes in
  [heldout_treatment_provenance.md](../../reports/kbs_final_evidence_20260813/heldout_treatment_provenance.md).
- Timing: 13/13 transfer hashes in `controlled_timing_integrity.md`.
- Common-Model V2 and tie-aware oracle: PASS statements in their `AUDIT.md`
  files.

## Known limits

- Raw traces and some model binaries are not redistributed here.
- Not every baseline is official upstream code (see manuscript §3.4.4).
- Unit tests on this `main` snapshot do not cover every second-revision
  runner; numerical claims are verified from committed CSVs and audits.
