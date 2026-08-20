# Controlled Timing Campaign — Local Sync Integrity (2026-08-13)

Compact integrity record for Reviewer #2 Major 4 (practical significance / timing),
synced from Wulver and independently re-audited here from the raw 420-row CSV, not
just re-read from the Wulver-side report.

## Provenance

Raw campaign source:
`login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/kbs_controlled_timing_20260810/`
→ local `analysis/kbs_controlled_timing_20260810/`.
Final-analysis source:
`login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/kbs_controlled_timing_final_analysis_20260811/`
→ local `analysis/kbs_controlled_timing_final_analysis_20260811/`.
Transferred by `rsync`. The `smoke/` subdirectory (preliminary single-repetition smoke
run, job `1171755`) was explicitly **excluded** from transfer and must not be used as
evidence — only the full 5-repetition campaign (job `1171758`) is evidentiary.

All 13 required files matched their required SHA-256 exactly — **13/13 hash PASS**
(`raw_timing_runs.csv`, `timing_audit.json`, `TIMING_PROTOCOL.md`,
`TIMING_REVIEWER_SUMMARY.md`, `timing_summary.csv`, `offline_cost_summary.csv`,
`memory_summary.csv`, `CONTROLLED_TIMING_FINAL_REPORT.md`, `raw_integrity_audit.json`,
`capacity_runtime_summary.csv`, `family_runtime_summary.csv`,
`policy_runtime_summary.csv`, `runtime_ranking.csv`).

## Independent local re-audit of `raw_timing_runs.csv` (recomputed, not copied)

| Gate | Result |
|---|---|
| Row count | 420/420 |
| Families | 7/7 (60 rows each) |
| Capacities | 32/64/128 (140 rows each) |
| Policies | 4 (`lru`, `fifo_reinsertion`, `sieve`, `halp_causal`; 105 rows each) |
| Repetitions | 5 (`0`–`4`; 84 rows each) |
| Unique `(policy, family, capacity, rep)` keys | 420/420, 0 duplicates |
| NaN/Inf/malformed cells | 0 |
| `request_count` | 50000 on every row |
| `python_version` | `3.10.20` on all 420 rows (single consistent identity) |
| `sklearn_version` | `1.7.2` on all 420 rows |
| `protocol_version` | `reviewer_fairness_v1` on all 420 rows |
| Smoke run (`smoke/raw_timing_runs.csv`, job 1171755) | Excluded from transfer and from all recomputation |

**Local classification: `FINAL_VALIDATED_SYNCED`.**

## Independently recomputed policy means (µs/request, from the 420-row raw file)

| Policy | Recomputed mean | Task's expected approximate mean | Match |
|---|---:|---:|---|
| `lru` | 4.680543 | 4.6805 | Yes |
| `fifo_reinsertion` | 5.168286 | 5.1683 | Yes |
| `sieve` | 9.523362 | 9.5234 | Yes |
| `halp_causal` | 870.660257 | 870.6603 | Yes |

These match `policy_runtime_summary.csv` and `runtime_ranking.csv` in
`analysis/kbs_controlled_timing_final_analysis_20260811/` exactly (independently
re-derived from the raw file, then cross-checked against the pre-computed summary —
identical to 6 decimal places). See `controlled_timing_summary.csv` in this directory
for the compact table (means, medians, stddev, 95% CI, slowdown vs. LRU).

## A note on `timing_summary.csv`

The top-level `timing_summary.csv` transferred from
`analysis/kbs_controlled_timing_20260810/` is a **preliminary snapshot** written from
the smoke job (`1171755`) before the full 420-row campaign (`1171758`) completed; its
own `note` column says so explicitly ("full campaign 1171758 running, will supersede").
Its hash matched the required manifest, so it is retained as part of the frozen
transfer for provenance completeness, but **it is not the authoritative timing
result** — `raw_timing_runs.csv` (420 rows, job 1171758) and the derived
`policy_runtime_summary.csv` / `runtime_ranking.csv` in the final-analysis directory
are authoritative, and this integrity record and `controlled_timing_summary.csv` are
built from those, not from `timing_summary.csv`.

## `evict_value_v1` single-run timing (see `controlled_timing_interpretation.md` for the full caveat)

`timing_summary.csv` also carries one `evict_value_v1` row (brightkite, cap 32,
35289.7 µs/request), explicitly reused from
`evict_value_v1_final_42_20260810/policy_comparison.csv`'s
`primary_controlled_window` row (`runtime_seconds=1411.5893` over
`scored_requests=40000` → 35289.73 µs/request, independently recomputed here and
matching). This is a **single, non-repeated wall-clock measurement**, not part of the
420-row 5-repetition controlled campaign — see the interpretation note for why it must
never be placed in the 4-policy timing table.
