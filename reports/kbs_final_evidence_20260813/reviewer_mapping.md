# Reviewer Evidence Mapping — Post C0/C1/C2 and Distribution-Shift Validation

Supersedes the relevant rows of `docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`
(narrative detail and cross-references remain there; this is the compact final-status
table plus the specific evidence supporting each entry).

| Concern | Status | Note |
|---|---|---|
| **Reviewer #2 Major 1** (learned-baseline comparison) | `SYNC_PENDING` | Local scientific baseline work is complete; blocked only on Wulver's corrected-treatment `evict_value_v1` 42/42 synchronization and audit |
| **Reviewer #2 Major 2** (supervision-objective ablation) | `SCIENTIFICALLY_COMPLETE_SYNTHESIS_PENDING` | Frozen 28-model registry, consistent 7-family result; no remaining scientific work, manuscript integration only |
| **Reviewer #2 Major 3** (offline/online failure explanation) | `SCIENTIFICALLY_COMPLETE_SYNTHESIS_PENDING` | All planned local mechanistic science is now complete and validated. Core conclusion: continuation-policy mismatch is partial/regime-dependent (H5, `PARTIALLY_SUPPORTED`); generic state-shift reduction does not improve performance (H6, `DISFAVORED`); target design/degeneracy (H3) remains the dominant explanation |
| **Reviewer #2 Major 4** (practical significance / timing) | `SYNC_PENDING` | Timing campaign is scientifically complete on Wulver; the local payload (controlled timing 420/420) is absent locally and still needs synchronization and audit |
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

## Remaining external blockers (Wulver-only)

1. Corrected held-out `evict_value_v1` treatment synchronization and audit (42/42).
2. Controlled timing synchronization and audit (420/420).

No new local experiment is required to close these; both are sync-only.
