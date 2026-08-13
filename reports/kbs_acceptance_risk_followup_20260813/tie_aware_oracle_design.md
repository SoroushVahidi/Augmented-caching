# Tie-aware exact-target oracle v1

## Scientific question

How sensitive is the H4 exact-target oracle to resolving ties among the full
set of exact H4 minimizers?

## Frozen conditions

- Reuse the existing `build_candidate_rows_for_full_cache_state` H4 target
  computation without alteration.
- Seven families, capacities 32/64/128, history `[0,10000)`, score
  `[10000,50000)`, and the exact trace hashes in the existing fold files.
- `CURRENT_DETERMINISTIC` must reproduce the prior 21-cell exact oracle
  misses exactly before other conditions are run.
- `LRU_WITHIN_MINIMA` is the mandatory online secondary rule: choose the first
  candidate in the current LRU-to-MRU cache order among exact minimizers.
- `MRU_WITHIN_MINIMA` is included as a sensitivity rule: choose the last
  candidate in that order among minimizers.
- `RANDOM_WITHIN_MINIMA` samples uniformly from minimizers using fixed seeds
  0, 1, 2, 3, and 4. No future request or target value beyond the minimizer
  membership is used for tie resolution.
- FIFO is omitted because the validated replay callback exposes recency order,
  not insertion order after hits; inventing a separate FIFO state would change
  the replay kernel rather than isolate tie resolution.
- Expected rows: 21 cells × (3 deterministic policies + 5 random seeds) =
  168 policy rows, plus 21 LRU reference rows = 189 total rows.

## Interpretation

If all reasonable tie policies remain worse than LRU, underdetermination and
limited information jointly strengthen the diagnosis. If LRU-within-minima
restores performance, the target is action-underdetermined rather than
necessarily harmful. Wide random variation means the deterministic oracle is
not a unique target-quality estimate. Any policy that wins against LRU means
the target retains useful feasible-action information but lacks tie semantics.
