# Manuscript / Reviewer Crosswalk — Second Revision (2026-08-13)

Maps every reviewer concern addressed in this revision to its canonical
evidence, its exact location in the revised manuscript
(`submission_kbs_revision_final/01_Revised_Manuscript.pdf`, 44 pages,
compiled 2026-08-13), and its corresponding section in
`submission_kbs_revision_final/02_Response_to_Reviewers.md` /
`reports/kbs_response_to_reviewers_final.md`.

| Reviewer | Comment | Canonical evidence | Manuscript section | Table/Figure | Page | Response section | Status |
|---|---|---|---|---|---|---|---|
| R2 | Major 1 — LRB/3L-Cache comparison | `reports/kbs_final_evidence_20260813/major1_reviewer_summary.md`, `major1_protocol_comparability.md`, `major1_full_baseline_comparison.csv` | §3.6 Matched Comparison Against Learned Cache-Replacement Baselines (+ §3.6.1–3.6.4) | Table 7 | 23–25 | "Major Comment 1" | Complete, negative result |
| R2 | Major 1 — train/test-overlap disclosure (self-identified) | `docs/reviewer_fairness_protocol.md` §6 (CRITICAL FINDING) | §3.4 (disclosure paragraph) | Table 5 (caveated) | 18 | "Major Comment 1" | Disclosed; superseded by §3.6 |
| R2 | Major 2 — objective justification (reuse-distance/next-arrival/pairwise) | `analysis/supervision_objective_ablation_v1/policy_comparison.csv` | §3.7 Direct Comparison Against Alternative Supervision Objectives | Table 8 | 25–26 | "Major Comment 2" | Complete, negative result |
| R2 | Major 3 — offline/online gap, distribution-shift explanation | `reports/kbs_final_evidence_20260813/mechanistic_hypothesis_summary.md`, `distribution_integrity_summary.md`, `c0_integrity_summary.md` | §3.8 Mechanistic Diagnosis of the Eviction-Loss Target; §3.9 Discussion and Analysis | Table 7 (cross-ref), no new table in §3.8/3.9 | 26–31 | "Major Comment 3" | Complete, mechanistically diagnosed |
| R2 | Major 4 — practical significance / overhead | `reports/kbs_final_evidence_20260813/controlled_timing_integrity.md`, `controlled_timing_interpretation.md` | §3.10 Overhead and Scalability; §3.11 Practical Significance | Table 9 | 31–33 | "Major Comment 4" | Complete, controlled measurement |
| R3 | Primary issue — continuation-mismatch causal claim | `reports/kbs_final_evidence_20260813/c0_continuation_summary.csv`, `c0_integrity_summary.md` | §3.9 Discussion and Analysis (C0/C1/C2 paragraph) | — (in-text) | 28–29 | "Primary issue" | Complete; partially supported, regime-dependent |
| R3 | Primary issue (companion) — generic distribution-shift correction | `reports/kbs_final_evidence_20260813/distribution_shift_summary.csv`, `distribution_integrity_summary.md` | §3.9 Discussion and Analysis (DAgger paragraph) | — (in-text) | 29–30 | "Primary issue" | Complete; disfavored |
| R3 | Issue 2 — HALP differentiation | `analysis/reviewer_fairness/policy_comparison_halp.csv` (via §3.6) | §3.6.3 Results; §3.6.4 Implementation-fidelity caveats; §1.4 Related Work | Table 7 | 24–25, 5 | "Issue 2" | Complete, empirical comparison added |
| R3 | Issue 3 — controlled cost measurement | `reports/kbs_final_evidence_20260813/controlled_timing_summary.csv` | §3.10 Overhead and Scalability | Table 9 | 31–33 | "Issue 3" | Complete |

## Supporting mechanistic evidence (not separately reviewer-requested, cited within §3.8–3.9)

| Finding | Canonical evidence | Manuscript location | Page |
|---|---|---|---|
| Exact-target oracle underperforms LRU | `analysis/exact_target_oracle_replication_v1/policy_comparison.csv` | §3.8, paragraph 2 | 26–27 |
| Learned/exact target agreement (97.53%) | `analysis/learned_exact_target_agreement_v1/cell_summary.csv` | §3.8, paragraph 1 | 26 |
| Target degeneracy (H=4) | `analysis/strict_preference_horizon_diagnostic_v1/cell_summary.csv` | §3.8, paragraph 3 | 27 |
| Horizon observability limitation | `analysis/reuse_tail_horizon_diagnostic_v1/report.md` | §3.8, paragraph 3 | 27 |
| Learning-curve / insufficient-data check (H1) | `analysis/supervision_objective_learning_curve_v1/` | §3.8, summary paragraph | 27–28 |

## Notes

- Table/figure numbering and page numbers were extracted directly from the
  compiled PDF (`pdftotext -layout` on the tectonic-built output), not
  estimated, and will require re-extraction if the manuscript is edited
  further before submission.
- Every numeric claim listed in the "Canonical evidence" column was
  independently re-verified against raw CSV/JSON artifacts in this session
  before insertion into the manuscript (see conversation-level numerical
  audit; no discrepancies found).
- This crosswalk covers only the second-revision concerns (R2 Major 1–4, R3
  primary issue + Issues 2–3). First-revision concerns (R2-MC1/MC2/MC3,
  R3-Issue1/3–9 in the old numbering) were already resolved in the prior
  submission round and are not re-litigated here; see
  `reports/kbs_response_to_reviewers_skeleton.md` for that historical
  record.
