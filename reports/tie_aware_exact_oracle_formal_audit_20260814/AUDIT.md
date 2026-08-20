# Tie-aware exact-target oracle v1 — Formal Audit

- Status: `UNITS_COMPLETE` / `WRAPUP_RECOVERED` / `INTEGRITY_PASS`
- Date: 2026-08-14
- Recovery id: `tie_aware_exact_oracle_recovery_20260814`

This is the durable scientific audit. Failed wrap-up artifacts and unit
SHA-256s are preserved in
[../tie_aware_exact_oracle_recovery_20260814/](../tie_aware_exact_oracle_recovery_20260814/).

## Campaign identity

- experiment: tie-aware exact H4 target oracle v1
- protocol: `tie_aware_exact_target_oracle_v1`
- production source commit: `2752857bd6a6a1a12e6e3fed44340b407f5c8e56`
- runner: `scripts/experiments/run_tie_aware_exact_target_oracle.py`
- config: `configs/tie_aware_exact_target_oracle_v1.json`
- output: `analysis/tie_aware_exact_target_oracle_v1/`
- tmux session: `kbs_tie_aware_exact_oracle_20260813_final` (exited)
- production log: `logs/tie_aware_exact_oracle_20260813_final.log`
- worker start: 2026-08-13 13:41:22 EDT
- worker end: 2026-08-14 13:41:43 EDT
- final scientific unit: `wiki2018_cap128` completed 2026-08-14 13:41:43 EDT

## Proof that all 21 scientific units completed

- 21/21 unit directories have `summary.json`
- every unit `"status": "COMPLETE"` with exactly 9 rows
- expected families: brightkite, citibike, cloudphysics, metacdn, metakv,
  twemcache, wiki2018
- expected capacities: 32, 64, 128
- per unit: 1 `LRU_REFERENCE`, 1 `CURRENT_DETERMINISTIC`, 1
  `LRU_WITHIN_MINIMA`, 1 `MRU_WITHIN_MINIMA`, 5 `RANDOM_WITHIN_MINIMA`
  seeds `{0,1,2,3,4}`
- total scientific rows: 189/189
- production log last progress line:
  `{"unit": "wiki2018_cap128", "completed_units": 21}`

No family/capacity unit was rerun. Production unit SHA-256 hashes after
recovery match the hashes recorded at recovery time in
`reports/tie_aware_exact_oracle_recovery_20260814/PROVENANCE.json`.

## Wrap-up CSV failure

After the 21st unit completed, campaign aggregation crashed:

```
ValueError: dict contains fields not in fieldnames:
'fraction_all_tied', 'fraction_tied_decisions', 'mean_optimal_set_fraction'
```

Cause: `csv.DictWriter(..., fieldnames=list(rows[0]))`. The first row is
`LRU_REFERENCE` and lacks tie-diagnostic keys that later rows contain.
Classification: `CSV_SCHEMA_UNION_BUG` only. Scientific computation had
already finished.

At failure: truncated `summary.csv` (2 lines), stale
`completion_manifest.json` status `RUNNING`, no `integrity_audit.json`.

## Minimal aggregation recovery

The runner now derives CSV fieldnames from the union of all row keys
(core fields first, then diagnostics). Missing diagnostics are blank CSV
cells. `--aggregate-only` reconstructs campaign files from existing unit
summaries only.

```
PYTHONPATH=src python scripts/experiments/run_tie_aware_exact_target_oracle.py \
  --aggregate-only --out analysis/tie_aware_exact_target_oracle_v1
```

Regression tests: `tests/test_tie_aware_exact_target_oracle.py` (5 passed).

## Integrity

- `summary.csv`: 190 lines = header + 189 data rows
- `completion_manifest.json`: `COMPLETE`, `source_head`
  `2752857bd6a6a1a12e6e3fed44340b407f5c8e56`
- `integrity_audit.json`: `PASS`, 189 unique keys
- `CURRENT_DETERMINISTIC` misses match the prior exact-oracle replication
  on all 21 cells
- all 189 `trace_sha256` values match the corresponding fold hashes
- 21 `LRU_REFERENCE` diagnostic cells blank; 168 tie-aware rows filled
- `delta_vs_LRU` and `delta_vs_current_exact` consistent with miss counts
- zero units rerun

## Comparison table vs LRU (21 cells)

| Policy | rows | total misses | vs LRU (win/tie/lose) | miss difference vs LRU |
|---|---|---|---|---|
| `LRU_REFERENCE` | 21 | 565,126 | 0 / 21 / 0 | 0 |
| `LRU_WITHIN_MINIMA` | 21 | 564,713 | **16 / 5 / 0** | **−413** |
| `CURRENT_DETERMINISTIC` | 21 | 646,876 | 0 / 3 / 18 | +81,750 |
| `MRU_WITHIN_MINIMA` | 21 | 654,261 | 0 / 3 / 18 | +89,135 |
| `RANDOM_WITHIN_MINIMA` | 105 | 2,872,849 | 7 / 15 / 83 | see seeds |

Random-within-minima vs LRU by seed (21 cells each):

| Seed | total misses | vs LRU (win/tie/lose) |
|---|---|---|
| 0 | 574,772 | 2 / 3 / 16 |
| 1 | 574,363 | 2 / 3 / 16 |
| 2 | 574,456 | 1 / 3 / 17 |
| 3 | 574,673 | 1 / 3 / 17 |
| 4 | 574,585 | 1 / 3 / 17 |

Random vs `CURRENT_DETERMINISTIC`: 90 wins / 15 ties / 0 losses (105 rows).
The 15 ties are the three wiki2018 cells × 5 seeds.

`LRU_WITHIN_MINIMA` differs from `CURRENT_DETERMINISTIC` in 18/21 cells.
`MRU_WITHIN_MINIMA` differs in 17/21 cells. The three wiki2018 cells are
complete ties at 40,000 misses for every policy.

## Tie diagnostics

Computed on the 168 non-`LRU_REFERENCE` rows:

- `fraction_tied_decisions`: **1.0 on every row** (168/168)
- `fraction_all_tied`: min 0.185, mean 0.649, max 1.0
- `mean_optimal_set_fraction`: min 0.968, mean 0.991, max 1.0
- cells where all candidates are tied (`fraction_all_tied = 1.0` on
  `CURRENT_DETERMINISTIC`): wiki2018 cap 32/64/128 only (3/21)

H3 target/tie degeneracy is strongly supported: every scored eviction in
this protocol had more than one exact H4 minimizer.

## Corrected interpretation

1. The expensive scientific experiment completed. Recovery did not recompute
   any policy simulation.

2. `CURRENT_DETERMINISTIC` is a faithful replica of the prior H4 exact
   oracle on all 21 cells. The previously reported exact-oracle deficit
   versus LRU is reproduced, not an aggregation artifact.

3. That deficit is **not robust to valid within-minimum tie-breaking**.
   `LRU_WITHIN_MINIMA` never loses to LRU (16 wins, 5 ties) and has 413
   fewer total misses than LRU.

4. Therefore the old exact-oracle-versus-LRU result **cannot establish that
   the exact target itself is intrinsically worse than LRU**. The
   deterministic deployment rule (`min` candidate id among minimizers) was
   confounded with the target comparison.

5. MRU-within-minima is harmful versus LRU (0/21 wins, 18 losses). Random
   sampling among minimizers usually loses to LRU (83/105) but still usually
   beats the deterministic oracle (90/105). Tie semantics matter, and not
   every secondary rule is beneficial.

6. wiki2018 remains fully degenerate in this score window and does not
   distinguish tie policies.

## Relationship to prior exact-oracle evidence

`analysis/exact_target_oracle_replication_v1/` remains valid as a record of
the **deterministic** exact-oracle deployment rule. It must not be cited as
proof that optimizing the H4 target is intrinsically worse than LRU.
This tie-aware control is the valid target-versus-tie-break comparison.

## Relationship to manuscript hypotheses

- **H3** (target/tie degeneracy): `STRONGLY_SUPPORTED`. Every non-LRU row
  has `fraction_tied_decisions = 1.0`.
- **H4** (old deterministic exact-oracle vs LRU): the claim that the exact
  target intrinsically loses to LRU is **not supported**. The deterministic
  result was tie-confounded.
- Manuscript/rebuttal text was not modified in this audit.

## Limitations

- Secondary rules are restricted to candidates already in the exact H4
  minimizer set. This isolates tie resolution; it is not a new target.
- FIFO-within-minima was omitted because the validated replay callback
  exposes recency order, not insertion order after hits.
- No statistical significance test was run; counts are descriptive.
- wiki2018 contributes three non-informative 100%-miss cells.
- Aggregate miss totals weight families/capacities by request volume; the
  16/21 win count weights cells equally.

## Provenance chain

```
Production worker (tmux kbs_tie_aware_exact_oracle_20260813_final,
commit 2752857, 2026-08-13 13:41:22 to 2026-08-14 13:41:43 EDT)
  -> 21 complete unit summary.json files
  -> campaign CSV write failure (schema-union bug)
  -> preserved failed wrap-up
     reports/tie_aware_exact_oracle_recovery_20260814/
  -> --aggregate-only reconstruction of summary.csv,
     completion_manifest.json, integrity_audit.json
  -> this formal audit
```

## Machine-readable support

`aggregate_recheck.json` in this directory recomputes win/tie/loss counts,
miss differences, per-seed random totals, and tie-diagnostic aggregates
from the recovered `summary.csv`.
