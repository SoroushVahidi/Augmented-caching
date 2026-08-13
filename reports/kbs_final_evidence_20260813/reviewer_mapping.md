# Reviewer Evidence Mapping — Post C0/C1/C2, Distribution-Shift, and Wulver Sync Validation

Supersedes the relevant rows of `docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`
(narrative detail and cross-references remain there; this is the compact final-status
table plus the specific evidence supporting each entry).

| Concern | Status | Note |
|---|---|---|
| **Reviewer #2 Major 1** (learned-baseline comparison) | `SCIENTIFICALLY_COMPLETE_SYNTHESIS_READY` | **Corrected 2026-08-13 (later pass):** the exact-protocol comparison against LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE, and FIFO-Reinsertion is complete and validated — all seven have 21/21 `primary_controlled_window` cells matching the treatment exactly by trace SHA-256, capacity, window, and metric (see `major1_protocol_comparability.md`, `major1_reviewer_summary.md`, `major1_full_baseline_comparison.csv`). `evict_value_v1` loses on a clear majority of cells (13–17 of 21) against every baseline. The earlier note here — "no same-protocol comparison exists... LRB/3L-Cache/HALP/CACHEUS have zero results under any protocol" — was **incorrect**: it restated a Wulver-filesystem-scoped claim from `baseline_eligibility.csv` as if it were universal, without checking this workstation's own `analysis/reviewer_fairness/` directory, which already had these results locally since 2026-08-06/07. No baseline shares `evict_value_v1`'s leave-one-family-out *training* procedure (an intentional, disclosed, non-fairness-affecting difference), but the *evaluation* protocol is exact-matched for all seven. |
| **Reviewer #2 Major 2** (supervision-objective ablation) | `SCIENTIFICALLY_COMPLETE_SYNTHESIS_PENDING` | Frozen 28-model registry, consistent 7-family result; no remaining scientific work, manuscript integration only |
| **Reviewer #2 Major 3** (offline/online failure explanation) | `SCIENTIFICALLY_COMPLETE_SYNTHESIS_PENDING` | All planned local mechanistic science is now complete and validated. Core conclusion: continuation-policy mismatch is partial/regime-dependent (H5, `PARTIALLY_SUPPORTED`); generic state-shift reduction does not improve performance (H6, `DISFAVORED`); target design/degeneracy (H3) remains the dominant explanation |
| **Reviewer #2 Major 4** (practical significance / timing) | `EVIDENCE_SYNCED_SYNTHESIS_PENDING` | Controlled timing campaign (420/420) synced from Wulver and independently re-audited 2026-08-13 (`controlled_timing_integrity.md`). Controlled 5-repetition timing covers exactly LRU/FIFO-Reinsertion/SIEVE/HALP-causal; `evict_value_v1`'s runtime is a separate single-run measurement from the treatment campaign, not part of the 5-repetition table (`controlled_timing_interpretation.md`). Remaining work is manuscript synthesis, not new evidence. |
| **Reviewer #3** (causal continuation-policy explanation) | `SCIENTIFICALLY_COMPLETE_SYNTHESIS_PENDING` | The requested causal C0/C1/C2 continuation-policy ablation is now complete and validated (21/21 cells, full integrity pass) |

## Reviewer #2 Major Comment 3 — what can now be said

*Concern: does the mismatch between LRU-continuation label construction and
learned-policy deployment (sequential distribution shift) explain some or all of the
performance gap?*

The offline/online gap (C1−C0, positive in most non-degenerate cells) is **partially
and inconsistently narrowed** by correcting label-continuation alone (H5: C2 improves
over C1 in 13/21 cells, macro mean Δ≈−0.0102, one severe counter-example at
`brightkite` cap32). It is **not explained by, nor fixed by**, correcting the broader
training-state distribution via one-step DAgger-style state correction (H6: net worse
in 16/21 cells, despite reducing the measured shift metric in the majority of cells).
The gap is better explained by the already-validated target-degeneracy finding (H3),
with continuation-policy mismatch as a real but partial, non-uniform secondary
contributor. Do not claim state-distribution shift itself does not exist — only that
reducing this measured generic shift metric did not improve performance under the
tested intervention.

## Reviewer #3 — what can now be answered

The previously speculative LRU-continuation explanation is **experimentally partially
supported**, not fully supported and not disfavored outright.

**Final answer: `PARTIALLY_SUPPORTED` / `REGIME_DEPENDENT`.**

Supporting result: the full 7-family/21-cell C0/C1/C2 causal ablation
(`analysis/continuation_policy_causal_ablation_production_v1/policy_comparison.csv`,
via `c0_continuation_summary.csv` in this directory), showing C2 (frozen-pi1
continuation) beats C1 (LRU continuation) in 13/21 cells, with a single large
regression at `brightkite` cap32 (+0.2433) preventing a stronger, uniform claim. The
companion distribution-shift result tests a different mechanism (state-visitation
correction, not label-continuation) and should not be conflated with this result — it
comes out negative for performance despite improving its own target metric.

## Remaining external blockers (Wulver-only) — RESOLVED 2026-08-13

1. ~~Corrected held-out `evict_value_v1` treatment synchronization and audit (42/42).~~
   Synced and independently re-audited 2026-08-13; `FINAL_VALIDATED_SYNCED`. See
   `heldout_treatment_integrity.md`, `heldout_treatment_provenance.md`.
2. ~~Controlled timing synchronization and audit (420/420).~~ Synced and independently
   re-audited 2026-08-13; `FINAL_VALIDATED_SYNCED`. See `controlled_timing_integrity.md`,
   `controlled_timing_interpretation.md`.

No synchronization blockers remain. No new local or Wulver experiment is required —
the remaining work across all five reviewer concerns is manuscript/rebuttal synthesis
of already-validated evidence (`READY_FOR_MANUSCRIPT_AND_REBUTTAL_SYNTHESIS`).
