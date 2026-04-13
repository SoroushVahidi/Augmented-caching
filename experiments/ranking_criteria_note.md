# Ranking Criteria Note (First-pass Ablation)

## Why ranking is the current design focus

The minimum-expansion safeguard addressed the prior zero-expansion failure mode. After that fix, the next decision bottleneck is **which branch to expand next** once each active branch has received its required initial expansions.

So this ablation isolates ranking as the main variable while keeping the base controller behavior fixed.

## Candidate criteria

All variants keep the same adaptive-min-expand scaffold:
1. Expand any active branch with `expansions < min_initial_expansions` first.
2. After that, pick a branch using a ranking criterion.

### 1) `raw_score`

- Formula: `rank_value = score`
- Interpretation: rely only on current branch quality estimate.

### 2) `score_plus_progress`

- Formula: `rank_value = score + 0.10 * depth`
- Interpretation: lightly favor branches that have advanced farther (simple progress proxy) while still being score-driven.

### 3) `relative_rank`

- Let `N` be active-branch count.
- Compute rank position within active pool for score (`rank(score)`) and depth (`rank(depth)`), each in `[1, N]` where larger is better.
- Formula: `rank_value = rank(score) + rank(depth)`.
- Interpretation: remove dependence on absolute score calibration and use only relative ordering in the current pool.

## Why these are first-pass rules

These criteria are intentionally simple and transparent:
- no learned ranking model,
- no nonlinear weighting search,
- easy to inspect/change in one line,
- suitable for low-cost pilot testing.

If a signal family looks promising, later passes can test stronger variants (e.g., normalized progress, uncertainty-aware ranking, dynamic weights).
