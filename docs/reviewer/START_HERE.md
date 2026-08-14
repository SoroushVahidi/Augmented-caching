# Reviewer Verification Guide

This page is the reviewer landing document for the Knowledge-Based Systems
revision of

**"Decision-aligned eviction-value prediction for learning-augmented caching"**

Public repository: <https://github.com/SoroushVahidi/Augmented-caching>

Canonical files:

- Revised manuscript:
  [submission_kbs_revision_final/01_Revised_Manuscript.pdf](../../submission_kbs_revision_final/01_Revised_Manuscript.pdf)
- Response to reviewers:
  [submission_kbs_revision_final/02_Response_to_Reviewers.md](../../submission_kbs_revision_final/02_Response_to_Reviewers.md)

## 1. What changed in this revision

- Direct matched comparison against LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE,
  and FIFO-Reinsertion under a corrected leave-one-family-out protocol. The
  result is **negative** for `evict_value_v1`.
- The finite-horizon eviction-loss objective is tested against alternatives
  in the deployed pipeline **and** in a matched Common-Model V2 control.
  Pipeline comparison is negative; the matched control does **not** support
  blaming the eviction-loss label itself.
- The offline-to-online gap is diagnosed with an exact-target oracle,
  degeneracy/agreement checks, and a **tie-aware** follow-up. The earlier
  deterministic oracle deficit versus LRU is a **tie confound**, not proof
  that the exact target intrinsically loses to LRU.
- Continuation mismatch (C0/C1/C2) is a **partial, regime-dependent**
  contributor. A one-step DAgger-style shift correction is a **negative**
  result for miss ratio.
- Timing uses a controlled 420-run campaign for LRU / FIFO-Reinsertion /
  SIEVE / HALP-causal. `evict_value_v1` timing is a separate single-run
  measurement.

## 2. Reviewer concern → manuscript → evidence

| Reviewer concern | Manuscript location | Primary evidence | Verification artifact |
|---|---|---|---|
| End-to-end online replay / matched comparison | §3.4, Table 5 | [major1_reviewer_summary.md](../../reports/kbs_final_evidence_20260813/major1_reviewer_summary.md) | [major1_full_baseline_comparison.csv](../../reports/kbs_final_evidence_20260813/major1_full_baseline_comparison.csv) |
| LRB / 3L-Cache (and CACHEUS / HALP) | §3.4, Table 5; §3.4.4 fidelity | Same as above; per-policy CSVs in `analysis/reviewer_fairness/` | [major1_protocol_comparability.md](../../reports/kbs_final_evidence_20260813/major1_protocol_comparability.md) |
| HALP / SIEVE (lightweight vs learned) | §3.4 Table 5; §3.9 Table 8 | Baseline CSVs; timing summary | [controlled_timing_summary.csv](../../reports/kbs_final_evidence_20260813/controlled_timing_summary.csv) |
| Objective comparison (pipeline) | §3.5, Table 6 | Response + manuscript Table 6 | [RESULT_VERIFICATION.md](RESULT_VERIFICATION.md) |
| Common-Model V2 | §3.5, Table 7 | [common_model_v2 AUDIT.md](../../reports/common_model_v2_formal_audit_20260814/AUDIT.md) | [analysis/common_model_objective_control_wulver_v2/summary.csv](../../analysis/common_model_objective_control_wulver_v2/summary.csv) |
| Offline-to-online discrepancy | §3.6–§3.7 | Mechanistic package + continuation/DAgger | [mechanistic_hypothesis_summary.md](../../reports/kbs_final_evidence_20260813/mechanistic_hypothesis_summary.md) |
| Exact-target / tie-aware analysis | §3.6 | [tie-aware AUDIT.md](../../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md) | [analysis/tie_aware_exact_target_oracle_v1/summary.csv](../../analysis/tie_aware_exact_target_oracle_v1/summary.csv) |
| Continuation mismatch | §3.7 | [c0_continuation_summary.csv](../../reports/kbs_final_evidence_20260813/c0_continuation_summary.csv) | [c0_integrity_summary.md](../../reports/kbs_final_evidence_20260813/c0_integrity_summary.md) |
| DAgger negative result | §3.7 | [distribution_shift_summary.csv](../../reports/kbs_final_evidence_20260813/distribution_shift_summary.csv) | [distribution_integrity_summary.md](../../reports/kbs_final_evidence_20260813/distribution_integrity_summary.md) |
| Computational overhead | §3.9, Table 8 | [controlled_timing_summary.csv](../../reports/kbs_final_evidence_20260813/controlled_timing_summary.csv) | [controlled_timing_integrity.md](../../reports/kbs_final_evidence_20260813/controlled_timing_integrity.md) |
| Workload-specific matched results | §3.3, Table 4 | Cross-family EV CSV + LRU CSV | [policy_comparison.csv](../../analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv), [policy_comparison_lru.csv](../../analysis/reviewer_fairness/policy_comparison_lru.csv) |
| Practical significance | §3.10 | Timing + negative miss-ratio result | [controlled_timing_interpretation.md](../../reports/kbs_final_evidence_20260813/controlled_timing_interpretation.md) |

## 3. Primary evidence

Use these as current evidence:

1. **Matched baselines** —
   [reports/kbs_final_evidence_20260813/](../../reports/kbs_final_evidence_20260813/)
   and
   [analysis/reviewer_fairness/](../../analysis/reviewer_fairness/)
2. **Common-Model V2** —
   [reports/common_model_v2_formal_audit_20260814/AUDIT.md](../../reports/common_model_v2_formal_audit_20260814/AUDIT.md)
3. **Tie-aware oracle** —
   [reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md](../../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md)
   (formal audit is the primary link; recovery provenance is inside that audit)
4. **Continuation / DAgger** —
   `c0_continuation_summary.csv` and `distribution_shift_summary.csv` in the
   evidence package
5. **Timing** —
   `controlled_timing_summary.csv` in the same package
6. **Workload-specific Table 4** —
   matched-protocol gaps from the held-out EV CSV and LRU CSV above, **not**
   an older leaky single-split family table

Supporting maps (scientific working documents; START_HERE is authoritative
for manuscript locations):

- [KBS_SECOND_REVISION_HYPOTHESIS_MAP.md](KBS_SECOND_REVISION_HYPOTHESIS_MAP.md)
- [KBS_SECOND_REVISION_REVIEWER_COVERAGE.md](KBS_SECOND_REVISION_REVIEWER_COVERAGE.md)

## 4. Historical / superseded evidence

Do **not** treat the following as primary:

- Old single-split leaky evaluation of `evict_value_v1` (train/test overlap).
  If encountered under `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`
  or older Wulver `heavy_r1` tables, it is **historical**.
- Superseded common-model **V1** pairwise control (orientation error).
- Failed or truncated tie-oracle wrap-up artifacts. The formal 2026-08-14
  audit is the current record.
- Exploratory pairwise / sentinel experiments and internal `docs/` notes.
- Word/ZIP copies under `submission_kbs_revision_final/` other than the PDF
  and Markdown response listed above. Prefer
  `01_Revised_Manuscript.pdf` and `02_Response_to_Reviewers.md`.

## 5. Reproduction / verification

- [REPRODUCTION_MATRIX.md](REPRODUCTION_MATRIX.md) — what is committed vs
  what would require an HPC rerun
- [RESULT_VERIFICATION.md](RESULT_VERIFICATION.md) — how to check the key
  numerical claims from published files

No scientific experiment was rerun for this reviewer-facing publication.
