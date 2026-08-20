# Reviewer #2 Major Comment 1 — Final Result Summary (2026-08-13)

## A. Reviewer request

Direct comparison of `evict_value_v1` against representative learned
cache-replacement methods (LRB and 3L-Cache named explicitly, plus other
learned/classical baselines), under matched traces, capacities,
preprocessing, request budgets, and evaluation metrics. No raw reviewer
letter is stored in this repository; this paraphrase is the standing audit
target already used across this branch's reviewer docs.

## B. Protocol fairness proof

All seven baselines (LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE,
FIFO-Reinsertion) have 21/21 `primary_controlled_window` cells whose
`(trace_sha256, capacity)` key set is byte-identical to the corrected
`evict_value_v1` treatment's 21 cells: same seven traces, capacities
32/64/128, `capacity_semantics=object_slots`, `object_size_semantics=unit`,
history `[0,10000)`, score `[10000,50000)`, 40,000 scored requests, and
`hits+misses=scored_requests` on every row. Every row across all eight CSVs
records `future_information=none`. Full detail:
`major1_protocol_comparability.md`.

## C. Direct results (validated)

Independently recomputed from raw CSVs and, for LRB/3L-Cache/CACHEUS,
cross-checked exactly against `scripts/analysis/prepare_r2_major1_evidence.py`'s
output with zero discrepancies (see `major1_full_baseline_comparison.csv`
for the full table). `evict_value_v1` mean primary miss ratio: **0.700463**
across all comparisons (n=21 cells each).

| Baseline | Baseline mean miss ratio | `evict_value_v1` wins | Baseline wins | Ties | Relative mean difference (evict vs. baseline) |
|---|---:|---:|---:|---:|---:|
| LRB | 0.687735 | 5 | 13 | 3 | +1.85% |
| 3L-Cache | 0.692626 | 5 | 13 | 3 | +1.13% |
| CACHEUS | 0.675600 | 3 | 15 | 3 | +3.68% |
| HALP | 0.676162 | 1 | 17 | 3 | +3.59% |
| LRU | 0.672769 | 1 | 16 | 4 | +4.12% |
| SIEVE | 0.687469 | 4 | 14 | 3 | +1.89% |
| FIFO-Reinsertion | 0.673257 | 2 | 16 | 3 | +4.04% |

**`evict_value_v1` loses on a clear majority of cells against every one of
the seven baselines** (13-17 losses out of 21, vs. 1-5 wins), with a
positive (worse) mean relative miss-ratio difference in every row. This is
consistent with, and now directly confirms under an exact same-protocol
comparison, the negative finding already reported in
`reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md`'s
caveated supplementary comparison against LRU/SIEVE/FIFO (+5.29% mean
regret there vs. +4.12% here — different aggregation method, same
qualitative conclusion).

## D. Fidelity caveats

- **LRB**: independent repository reimplementation; documented default
  hyperparameters (`memory_window=4096`, `batch_size=2048`) used as-is, not
  re-gridsearched this run. Evaluation protocol exact. Primary reviewer
  baseline.
- **3L-Cache**: independent repository reimplementation; fixed
  `batch_size=4096` class default, not validation-tuned this run
  (`PASS_WITH_CAVEAT`). Evaluation protocol exact. Primary reviewer
  baseline.
- **CACHEUS**: official-source-unmodified wrapper; the external upstream
  clone is not currently live-verifiable in this worktree (a
  source-authenticity caveat, not an evaluation-input caveat — the scored
  request stream and metrics are exact-protocol). Supporting learned/
  adaptive baseline.
- **HALP**: independent reimplementation of the published method; no
  official public implementation exists to compare against, `LOW_TO_MEDIUM`
  fidelity per existing docs. Evaluation protocol exact. Supporting only —
  Reviewer #2 named LRB and 3L-Cache explicitly, not HALP; its lower
  fidelity is not a blocker to closing this comment.
- **LRU / SIEVE / FIFO-Reinsertion**: native, parameter-free
  implementations; no fidelity caveat.

## E. Allowed claims

"LRU, SIEVE, FIFO-Reinsertion, LRB, 3L-Cache, CACHEUS, and HALP were
evaluated on the same seven request traces (identical by SHA-256),
capacities 32/64/128, a common history prefix `[0,10000)`, a common scored
suffix `[10000,50000)` (40,000 scored requests), object-slot capacity
semantics, and identical hit/miss accounting as the corrected held-out
`evict_value_v1` evaluation. Their internal adaptation/training procedures
remain algorithm-specific: LRB, 3L-Cache, and CACHEUS adapt online from
their own in-trace stream; HALP and `evict_value_v1` train offline (HALP on
each trace's own history prefix; `evict_value_v1` with leave-one-family-out
exclusion and a frozen model at test time); LRU, SIEVE, and FIFO-Reinsertion
are parameter-free. Under this matched evaluation, `evict_value_v1` does not
outperform the strong baselines: it loses on 13-17 of 21 matched cells to
every baseline tested, with mean relative miss-ratio disadvantage ranging
from +1.1% (3L-Cache) to +4.1% (LRU)."

## F. Prohibited claims

- "No same-protocol comparison exists for `evict_value_v1`" — false; this
  document is that comparison.
- "LRB/3L-Cache/HALP/CACHEUS have zero results under any protocol" — false;
  true only of Wulver's own filesystem, not of this workstation.
- "All methods use identical training procedures" — false and not required
  by the reviewer's request; only evaluation inputs must match.
- "3L-Cache is an official implementation" — false; independent
  reimplementation.
- "HALP is an official implementation" — false; independent
  reimplementation, no official code is public.
- "CACHEUS provenance is fully live-verifiable" — false in this worktree;
  the external clone is not currently live-verifiable.
- "`evict_value_v1` beats the modern learned baselines" — false; it loses
  on a clear majority of cells against every baseline tested.

## Bottom line

The fair, same-protocol comparison requested by Reviewer #2 Major Comment 1
is complete. The result is a **candid negative finding**: `evict_value_v1`
does not outperform LRU, SIEVE, FIFO-Reinsertion, LRB, 3L-Cache, CACHEUS, or
HALP under matched evaluation. The value of this evidence is that the
comparison itself is now fair and complete, the negative result is reported
without spin, and it is mechanistically explained elsewhere on this branch
(target-degeneracy finding H3, `reports/kbs_final_evidence_20260813/mechanistic_hypothesis_summary.md`)
rather than left as an unexplained loss.
