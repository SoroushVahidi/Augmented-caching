# `evict_value_v1` lightweight artifact-free mode

## Why this exists

Historically, `evict_value_v1` expected a binary model artifact at:

- `models/evict_value_v1_hist_gb.pkl`

When that file is absent (common in lightweight local/Codex-web environments), the policy could not run.

## New scorer modes

`evict_value_v1` now supports three scorer modes:

- `scorer_mode="artifact"`
  - Requires `model_path` to exist.
  - Uses the original trained artifact-backed predictor.
  - Raises `FileNotFoundError` if missing.

- `scorer_mode="lightweight"`
  - Never requires binary artifacts.
  - Uses a deterministic text-only linear surrogate scorer.
  - Optional text config path: `lightweight_config_path` (JSON with `intercept` and `weights`).

- `scorer_mode="auto"` (default)
  - Uses artifact mode if `model_path` exists.
  - Falls back to lightweight mode if not.

## Scientific interpretation

The lightweight scorer is **not** a faithful reconstruction of the trained HistGradientBoosting model.
It is a deterministic surrogate for local debugging/comparison only.

Use artifact mode for heavier runs where trained checkpoints are available.

## Diagnostics exposed

`diagnostics_summary()` now includes:

- `scorer_mode_requested`
- `scorer_mode` (active mode)
- `artifact_found`
- `lightweight_config_path`
- nested `scorer` metadata (`scorer_name`, `is_surrogate`, and model/config details)

## Lightweight comparison integration

`scripts/run_lightweight_baseline_comparison.py` now instantiates `evict_value_v1` in lightweight mode,
so the policy is included in local comparison tables without requiring binary files.

## Examples

### Artifact-backed (heavier runs)

```python
EvictValueV1Policy(
    model_path="models/evict_value_v1_hist_gb.pkl",
    scorer_mode="artifact",
)
```

### Artifact-free lightweight local mode

```python
EvictValueV1Policy(
    scorer_mode="lightweight",
)
```

### Auto fallback

```python
EvictValueV1Policy(
    model_path="models/evict_value_v1_hist_gb.pkl",
    scorer_mode="auto",
)
```

### Run lightweight comparison

```bash
python scripts/run_lightweight_baseline_comparison.py \
  --capacities 3,5,8 \
  --max-requests 200 \
  --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --out-dir analysis/lightweight_comparison
```
