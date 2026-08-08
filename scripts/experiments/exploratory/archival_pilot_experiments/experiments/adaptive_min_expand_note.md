# Adaptive Min-Expand Update

The adaptive_min_expand controller supports configurable post-safeguard ranking modes:

- `raw_score`
- `score_plus_progress`
- `relative_rank`
- `learned_branch_score`

Base safeguard behavior is unchanged: any active branch below the minimum required initial expansions is expanded before ranking logic is applied.

For `learned_branch_score`, the controller loads a trained scorer artifact and ranks by predicted `P(promising=1 | features)`.
