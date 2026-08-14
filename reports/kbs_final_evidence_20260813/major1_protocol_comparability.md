# Reviewer #2 Major Comment 1 — Protocol Comparability Audit (2026-08-13)

Resolves a documentation contradiction discovered during a read-only audit:
`analysis/kbs_r2_major1_evidence_prep_20260811/` (correct) said LRB/3L-Cache/
CACHEUS have complete controlled-window results matching the corrected
treatment; `reports/kbs_final_evidence_20260813/reviewer_mapping.md` (stale,
now corrected by this pass) said no same-protocol comparison exists at all.
The audit re-derived the truth directly from raw CSVs, not from either
document's prose.

## What "same evaluation protocol" means here

**"Same evaluation protocol" is not "same training procedure."** It means:
the same seven request traces (verified by SHA-256, not filename), the same
capacities under the same capacity/object-size semantics, the same history
prefix and scored suffix, the same scored-request budget, and the same
hit/miss accounting. Algorithms are free to differ in how they train or
adapt internally — LRB/3L-Cache/CACHEUS adapt online from their own in-trace
stream, HALP trains offline on its own trace's history prefix only, LRU/
SIEVE/FIFO-Reinsertion are parameter-free, and `evict_value_v1` trains
offline with leave-one-family-out exclusion. None of that is a fairness
violation as long as no policy sees future information or another family's
data — verified below via the `future_information` and
`model_training_data` columns of every raw CSV.

## Comparability matrix

All figures independently recomputed from raw CSVs, keyed on
`(trace_sha256, capacity)` (not filenames), and cross-checked against
`analysis/kbs_r2_major1_evidence_prep_20260811/baseline_integrity.json` /
`reviewer_ready_comparison.csv` where that script covers the policy.

| Policy | 21/21 cells | Trace hash match | Capacity | Window | Budget | Metrics | Leakage | Fidelity | Comparability |
|---|---|---|---|---|---|---|---|---|---|
| `evict_value_v1` | 21/21 (reference) | — | 32/64/128 | `[0,10000)` / `[10000,50000)` | 40000 | hits+misses=scored | `future_information=none`; leave-one-family-out, frozen at test | Corrected protocol's own artifact | Reference |
| LRB | 21/21 | EXACT (sha256-identical) | 32/64/128 | exact | 40000 | exact | `future_information=none`; `model_training_data=in_trace_only`; online in-trace adaptation only | Independent repository reimplementation; `batch_size=2048` documented default, not re-gridsearched this run | **LEVEL 1** |
| 3L-Cache | 21/21 | EXACT | 32/64/128 | exact | 40000 | exact | `future_information=none`; in-trace-only online adaptation | Independent repository reimplementation; fixed `batch_size=4096` default, not validation-tuned (certificate `PASS_WITH_CAVEAT`) | **LEVEL 2** (material implementation caveat, evaluation inputs otherwise identical) |
| CACHEUS | 21/21 | EXACT | 32/64/128 | exact | 40000 | exact | `future_information=none`; in-trace-only online adaptation | Official-source-unmodified wrapper; external clone provenance not currently live-verifiable in this worktree (caveat is source authenticity, not evaluation comparability) | **LEVEL 1 with provenance caveat** |
| HALP | 21/21 | EXACT | 32/64/128 | exact | 40000 | exact | `future_information=none`; frozen offline model trained only on each trace's own `[0,10000)` prefix | Independent reimplementation of a published method; no official public code exists — `LOW_TO_MEDIUM` fidelity | **LEVEL 1 evaluation match, lower-fidelity supporting implementation** |
| LRU | 21/21 | EXACT | 32/64/128 | exact | 40000 | exact | `future_information=none`; parameter-free | Native implementation | **LEVEL 1** |
| SIEVE | 21/21 | EXACT | 32/64/128 | exact | 40000 | exact | `future_information=none`; parameter-free | Native implementation | **LEVEL 1** |
| FIFO-Reinsertion | 21/21 | EXACT | 32/64/128 | exact | 40000 | exact | `future_information=none`; parameter-free | Native implementation | **LEVEL 1** |

Source files: `analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv`
(treatment) and `analysis/reviewer_fairness/policy_comparison_{lrb,three_l_cache,cacheus,halp,lru,sieve,fifo_reinsertion}.csv`
(baselines). LRB/3L-Cache/CACHEUS coverage figures are exactly reproduced by
`scripts/analysis/prepare_r2_major1_evidence.py` (re-run 2026-08-13, output
in `analysis/kbs_r2_major1_evidence_prep_20260811/reviewer_ready_comparison.csv`,
independently re-validated cell-by-cell against the raw CSVs with zero
discrepancies). LRU/SIEVE/FIFO-Reinsertion/HALP were not covered by that
script (it only implements LRB/3L-Cache/CACHEUS) and were independently
computed in this pass using the identical methodology; see
`major1_full_baseline_comparison.csv` for the combined 7-baseline table.

## Why the earlier "zero results / no same-protocol comparison" claim was wrong

`analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/baseline_eligibility.csv`
(a Wulver-produced artifact synced 2026-08-13) states LRB/3L-Cache/HALP/
CACHEUS have "zero completed result artifacts found anywhere... under any
protocol" and that LRU/SIEVE/FIFO "only exist" in a mismatched-window file —
based on an exhaustive grep of **Wulver's own** `analysis/` tree. That claim
is accurate about Wulver's filesystem but was never true about this
workstation: `analysis/reviewer_fairness/` (produced separately by
`scripts/experiments/run_reviewer_fairness.py`, a pipeline that has never
run on Wulver) already contained exact-protocol results for all seven
baselines as of 2026-08-06/07, and their compact evidence-prep audit
(`analysis/kbs_r2_major1_evidence_prep_20260811/`) already validated
LRB/3L-Cache/CACHEUS on 2026-08-11. `reports/kbs_final_evidence_20260813/reviewer_mapping.md`
and `heldout_treatment_integrity.md`, written 2026-08-13 in an earlier pass
on this branch, restated the Wulver-scope claim as if it were universal
without cross-checking the local evidence-prep package already present in
the same repository. Both files are corrected by this pass (see
`docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md` and related docs
for the specific corrections).
