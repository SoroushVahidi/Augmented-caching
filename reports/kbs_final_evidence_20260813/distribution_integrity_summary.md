# Distribution-Shift Ablation — Integrity Summary

Canonical raw source: `analysis/distribution_shift_ablation_v1/`
(`policy_comparison.csv`, `state_shift_metrics.csv`, `trajectory_divergence.csv`,
`campaign_state.json`, `protocol_snapshot.json`, `provenance.json`). This file is a
compact summary only; the CSVs above remain the canonical scientific record.

Audit performed: 2026-08-13, read-only, formal post-completion integrity pass.
The canonical `scripts/experiments/audit_distribution_shift_completion.py` was also
re-run (structural/integrity checks only, no scientific interpretation) and
independently confirms:

```
[audit] classification=COMPLETE_VALID primary_rows=42/42
  - conditions: PASS
  - families: PASS
  - capacities: PASS
  - row_count_exact_42: PASS
  - no_duplicate_keys: PASS
  - zero_failures: PASS
  - zero_nan_inf: PASS
  - scoring_window_consistent: PASS
  - state_shift_coverage: PASS
  - trajectory_diagnostic_coverage: PASS
  - frozen_protocol_unchanged: PASS
```

This regenerated `completion_audit.json` supersedes the stale 2026-08-09 snapshot
(`classification: INCOMPLETE, primary_rows: 24/42`), which was a historical artifact
not read by any active gating code. `analysis/distribution_shift_ablation_v1/` is
gitignored, so the regenerated file is local-only; this document is the durable,
tracked record of that result.

## Completeness

- `campaign_state.json`: 7/7 folds (`brightkite, citibike, cloudphysics, metacdn,
  metakv, twemcache, wiki2018`).
- 21/21 distinct family×capacity paired cells; 42/42 primary rows; 42/42 state-shift
  rows; 21/21 trajectory-divergence rows.
- No duplicate `(condition, family, capacity)` / `(family, cap, ref, other)` keys.
- All 42 policy rows `status=ok`; no NaN/Inf anywhere; `hits + misses == scored_requests`
  for all 42 rows.

## Protocol integrity

- History/score windows (`0, 10000, 10000, 50000`), `seed=0` uniform across all 42 rows.
- Live `configs/distribution_shift_ablation_v1.json` is byte-for-byte identical to the
  frozen `protocol_snapshot.json` — no drift since launch.
- `OFF_POLICY_LRU` / `DAGGER_ITER1` definitions (D0 ∪ D1 equal-weight, iteration-1 states
  drawn only from training/validation families, held-out family never contributes rows)
  match the frozen protocol exactly.
- Held-out-family isolation independently verified on all 42 rows via each row's
  self-declared training/validation family set.
- `n_train_rows` for DAGGER_ITER1 is exactly 2.0× OFF_POLICY_LRU for all 7 families
  (confirms uniform D0∪D1 doubling).

## Model integrity

- Model hash unique and consistent per `(family, condition)` across all 3 capacities
  (14/14 groups); no hash shared across different `(family, condition)` pairs.
- Live `.pkl` hash matches the recorded `model_hash` exactly for the 3 families whose
  binaries remain on disk (`metakv`, `twemcache`, `wiki2018`).

### `MODEL_BINARY_RETENTION_GAP`

Only 6 of 14 expected `models/distribution_shift_ablation_v1/{OFF_POLICY_LRU,DAGGER_ITER1}/{family}.pkl`
files remain on disk (`metakv`, `twemcache`, `wiki2018` only). Binaries for
`brightkite`, `citibike`, `cloudphysics`, `metacdn` are absent — believed removed by
prior storage cleanup, not by this audit or this task.

- The campaign record retains a non-empty `model_hash` for every one of the 42 rows.
- Those hashes are consistent across the three capacities within each `(family, condition)`.
- Surviving live models (3/7 families) independently match their recorded hashes exactly.
- The missing binaries prevent independent byte-level re-verification for the other
  four families in this pass.
- This was judged **non-blocking for campaign validity** because protocol, results, and
  provenance remain internally consistent throughout — it does **not** mean full binary
  reproducibility is established for the missing four families, and no claim to that
  effect should be made.

## Log-ordering anomaly

**Classification: `BENIGN_STDOUT_BUFFERING`.** The completion log shows the parent
wrapper's (`resume_distribution_shift.py`) `[plan]`/`[launch]` text appearing after the
child runner's completion messages. Explained by ordinary CPython stdout block-buffering
when writing to a redirected (non-TTY) file descriptor: the parent's `print()` calls
execute before `subprocess.run()` but are held in the parent's own buffer and only
flushed to disk on interpreter exit, after the child (which inherits the same fd and
writes directly) has already completed and its output has landed on disk. Verified: only
one child invocation occurred (exactly one `"Campaign pass complete"`, one `[launch]`,
four `[trained]` lines matching precisely the 2 folds × 2 conditions needed to close the
5/7 → 7/7 gap); no `CalledProcessError`/traceback anywhere in the log; no folds
recomputed; no duplicate rows or model hashes resulted.

## Overall classification

**FINAL_VALIDATED.**

## Scientific-result headline (see `distribution_shift_summary.csv` for full 21-row table)

- DAgger improves misses: 2/21 cells (negligible); ties: 3/21 (Wiki2018, degenerate);
  worsens: 16/21.
- Macro mean DAgger−OFF miss-ratio delta: ≈ +0.0094 (net worse).
- Aggregate misses: OFF_POLICY_LRU = 591,604, DAGGER_ITER1 = 599,537.
- State-shift index improves (decreases) in 16/21 cells.
- Dominant pattern (13/18 informative cells): measured state-shift improves while misses
  simultaneously worsen.

**H6 (generic state-shift → performance) disposition: `DISFAVORED`** — the measured
generic shift-index reduction did not translate into improved online performance under
the tested DAgger-style intervention. This does **not** claim that state-distribution
shift itself does not exist; the shift-reduction mechanism worked as designed, only its
assumed link to downstream miss performance is disfavored. See
`mechanistic_hypothesis_summary.md`.
