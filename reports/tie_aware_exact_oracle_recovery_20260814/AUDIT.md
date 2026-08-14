# Tie-aware exact-target oracle v1

Durable formal audit:
[../tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md](../tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md).

This directory preserves the failed campaign wrap-up and recovery provenance.
Scientific units were not rerun.

- Status: `UNITS_COMPLETE` / `WRAPUP_RECOVERED` / `INTEGRITY_PASS`
- Date: 2026-08-14
- Recovery id: `tie_aware_exact_oracle_recovery_20260814`

## Campaign identity

- experiment: tie-aware exact H4 target oracle v1
- protocol: `tie_aware_exact_target_oracle_v1`
- production source commit: `2752857bd6a6a1a12e6e3fed44340b407f5c8e56`
- workspace HEAD at recovery: `4e9298d08ecee248d14d41b9ef5952d1ce4eead4`
- production log: `logs/tie_aware_exact_oracle_20260813_final.log`
- output: `analysis/tie_aware_exact_target_oracle_v1/`
- worker start: 2026-08-13 13:41:22 EDT
- worker end: 2026-08-14 13:41:43 EDT
- no production process remains; expected tmux session is gone

## Failure that was recovered

Production completed all 21 family/capacity units, then crashed while writing
campaign-level `summary.csv`:

```
ValueError: dict contains fields not in fieldnames:
'fraction_all_tied', 'fraction_tied_decisions', 'mean_optimal_set_fraction'
```

Classification: `CSV_SCHEMA_UNION_BUG` only.

`LRU_REFERENCE` rows omit tie-diagnostic fields because they come from a
plain LRU replay. Tie-aware policy rows include
`fraction_tied_decisions`, `fraction_all_tied`, and
`mean_optimal_set_fraction`. The writer used `fieldnames=list(rows[0])`,
so the first LRU row dropped those columns and later rows raised.

This is not a scientific-unit failure. No family/capacity unit was rerun.
No unit `summary.json` was modified. Unit SHA-256 hashes after recovery
match the pre-recovery hashes in `PROVENANCE.json`.

Failed wrap-up artifacts preserved here:

- `failed_production_summary_truncated.csv` (2 lines; SHA-256 `57fbf1203ce85d6f5d94c1cf7514604c57e2546c8a0f86b0513c00d0596ff23b`)
- `failed_production_completion_manifest_stale_RUNNING.json` (SHA-256 `5dedfcd1f2322a434bb79f03a74673d7bf1aee6cbc9173b54358ae96529e5e6d`)
- `production_log_copy.log` (SHA-256 `e5058a19434fcd8d1100b0bf5e843fc284321056f2276d7fd5dbc624ebb8e159`)
- `integrity_audit.json` was absent at failure

## Recovery method

No reducer existed. Campaign files were reconstructed with:

```
PYTHONPATH=src python scripts/experiments/run_tie_aware_exact_target_oracle.py \
  --aggregate-only --out analysis/tie_aware_exact_target_oracle_v1
```

The runner now derives CSV fieldnames from the union of all row keys, with
core fields first and diagnostic fields next. Missing diagnostics are blank
CSV cells. A targeted regression test reproduces the original
first-row-fieldnames crash and asserts the union writer succeeds.

## Integrity

- unit summaries: 21/21, each `status=COMPLETE`, exactly 9 rows
- policy coverage per unit: 1 `LRU_REFERENCE`, 1 `CURRENT_DETERMINISTIC`,
  1 `LRU_WITHIN_MINIMA`, 1 `MRU_WITHIN_MINIMA`, 5 `RANDOM_WITHIN_MINIMA`
  seeds `{0,1,2,3,4}`
- campaign rows: 189/189 unique `(family, capacity, tie_policy, seed)` keys
- recovered `summary.csv`: 190 lines (header + 189 data rows)
- recovered `summary.csv` SHA-256: `9ce93829df280e0631289d6da1cb93e3253df8745b42a20d2d6c68f3bfcbd605`
- `completion_manifest.json`: `COMPLETE`, `source_head` preserved as
  `2752857bd6a6a1a12e6e3fed44340b407f5c8e56`
- `integrity_audit.json`: `PASS`
- `CURRENT_DETERMINISTIC` misses match the prior exact-oracle replication
  on all 21 cells
- all 189 row `trace_sha256` values match the corresponding fold
  `test_trace_sha256`
- all 21 `LRU_REFERENCE` CSV diagnostic cells are blank; all 168
  tie-aware rows have diagnostics filled
- no duplicate family/capacity units; no missing combinations
- `delta_vs_LRU` and `delta_vs_current_exact` are consistent with miss counts

## Policy aggregates (21 cells)

Totals are sum of score-window misses over the 21 family/capacity cells.
Random is 105 rows (21 cells × 5 seeds).

| Policy | rows | total misses | vs LRU (win/tie/lose) | vs CURRENT (win/tie/lose) |
|---|---|---|---|---|
| `LRU_REFERENCE` | 21 | 565,126 | 0 / 21 / 0 | 18 / 3 / 0 |
| `LRU_WITHIN_MINIMA` | 21 | 564,713 | **16 / 5 / 0** | 18 / 3 / 0 |
| `CURRENT_DETERMINISTIC` | 21 | 646,876 | 0 / 3 / 18 | 0 / 21 / 0 |
| `MRU_WITHIN_MINIMA` | 21 | 654,261 | 0 / 3 / 18 | 2 / 4 / 15 |
| `RANDOM_WITHIN_MINIMA` | 105 | 2,872,849 | 7 / 15 / 83 | 90 / 15 / 0 |

The three wiki2018 cells are complete ties at 40,000 misses for every
policy (`fraction_all_tied = 1.0`). Those three cells account for the LRU
ties above. `LRU_WITHIN_MINIMA` never loses to LRU. `CURRENT_DETERMINISTIC`
never beats LRU.

## Scientific interpretation

1. **The expensive experiment is complete and internally consistent.** The
   wrap-up bug did not contaminate unit results.

2. **`CURRENT_DETERMINISTIC` is a faithful replica of the prior H4 exact
   oracle.** All 21 cells match
   `analysis/exact_target_oracle_replication_v1/policy_comparison.csv`.
   The previously reported exact-oracle deficit versus LRU is therefore
   reproduced, not an aggregation artifact.

3. **The exact H4 target is action-underdetermined.** Every non-LRU row has
   `fraction_tied_decisions = 1.0`: every scored eviction had more than one
   exact minimizer. Tie semantics are not a corner case.

4. **`LRU_WITHIN_MINIMA` restores (and slightly improves on) LRU.** Choosing
   the LRU-most candidate among exact minimizers beats LRU in 16/21 cells
   and ties the rest, with 413 fewer total misses than LRU. The target
   therefore retains useful feasible-action information once a recency
   secondary rule is supplied.

5. **The published deterministic oracle used a harmful tie-break.**
   `CURRENT_DETERMINISTIC` (`min` candidate id among minimizers) is worse
   than LRU in 18/21 cells and worse than `LRU_WITHIN_MINIMA` in those same
   18 cells. The earlier “exact oracle loses to LRU” reading is a tie-break
   result, not proof that the H4 target is intrinsically worse than LRU.

6. **MRU-within-minima is worse than the deterministic oracle on aggregate.**
   Resolving ties toward MRU is harmful. Random-within-minima usually loses
   to LRU (83/105) but still usually beats `CURRENT_DETERMINISTIC` (90/105).

7. **wiki2018 remains degenerate** in this score window: every policy misses
   every request. It does not distinguish tie policies.

The manuscript was not modified in this recovery.

## Next action

Campaign-level files are recovered and the integrity audit passes. Any
manuscript update that cites this oracle should use
`analysis/tie_aware_exact_target_oracle_v1/summary.csv` and this audit, not
the truncated production CSV.
