# `evict_value_v1_cross_family_v1` Stage-2 training OOM — diagnosis

## A. Failure confirmation (kernel evidence)

```
Aug 06 23:51:09 al-khwarizmi kernel: oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,
  mems_allowed=0,global_oom,task_memcg=/user.slice/user-1000.slice/session-250.scope,task=python,pid=253445,uid=1000
Aug 06 23:51:09 al-khwarizmi kernel: Out of memory: Killed process 253445 (python)
  total-vm:67253092kB, anon-rss:57148352kB, file-rss:8kB, shmem-rss:0kB, UID:1000 pgtables:128572kB oom_score_adj:0
```

- Failed PID: `253445`
- Peak reported by kernel at kill time: `anon-rss ≈ 57,148,352 kB ≈ 54.5 GiB` (`total-vm ≈ 64.1 GiB`)
- Fold: `brightkite` (first of 7 held-out folds in the frozen rotation)
- Stage: Stage 2 (model training), invoked from `/tmp/run_cross_family_pipeline.sh` line 8:
  ```
  .venv_fairness/bin/python -u scripts/train_evict_value_wulver_v1.py \
    --manifest data/derived/evict_value_v1_cross_family_v1/brightkite/manifest.json \
    --horizons 4 --seed 0 \
    --models-dir models/cross_family_v1_staging/brightkite \
    --metrics-json analysis/reviewer_fairness_cross_family_v1/brightkite/train_metrics.json \
    --comparison-csv analysis/reviewer_fairness_cross_family_v1/brightkite/model_comparison.csv \
    --best-config-json analysis/reviewer_fairness_cross_family_v1/brightkite/best_config.json
  ```
  No `--max-train-rows` was passed, so row loading was unbounded.
- Last completed artifact before the kill: Stage 1 dataset build for `brightkite`
  (`data/derived/evict_value_v1_cross_family_v1/brightkite/manifest.json` and `split_summary.csv`,
  106 shards, 48,209,152 total rows across all shards/splits). Stage 2 produced no artifacts —
  `models/cross_family_v1_staging/brightkite/` contains only an incomplete/partial state.
- No other fold was attempted; the shell wrapper died at the `Killed` line with no fold-2 onward.

## B. Root cause (measured, not speculative)

`scripts/train_evict_value_wulver_v1.py:_load_rows_from_manifest` (previously lines 64-100) reads
every matching CSV row of the (unbounded) `train`/`val`/`test` split into a **Python `dict` per row**
(38 CSV columns; 28 of them explicitly cast to `float`), and appends every such dict to a plain list.
That list is retained for the entire horizon loop, including through all three sequential model fits
(`ridge`, `random_forest`, `hist_gb`), even though only the derived `x_train`/`y_train` NumPy arrays
are actually needed for `.fit()`.

### Controlled measurement (6 real shards from the `brightkite` manifest, 2,595,360 matching rows, isolated processes, `resource.getrusage().ru_maxrss`)

| Loader | Peak RSS | Bytes/row | Notes |
|---|---|---|---|
| OLD (`dict` per row via `csv.DictReader`, current code) | 5,741.7 MB | **2,320 bytes/row** | list of Python dicts, 38 keys, 28 floats cast |
| NEW (`list`-of-`list` before `np.asarray`) | 3,285.5 MB | 1,327 bytes/row | still builds a nested Python list first — same anti-pattern in miniature |
| NEW (preallocated `np.float64` array, streamed fill) | 293.9 MB | **119 bytes/row** | matches the array's own footprint (`x`+`y` = 280.3 MB) almost exactly |

**≈19.5x memory reduction** (2,320 → 119 bytes/row) from switching the OLD dict-materialization to a
preallocated-array streaming loader, with no change to which rows are read, how they're filtered, or
how they're subsampled.

The manifest for fold `brightkite` alone totals 48,209,152 rows across train/val/test/all pooled
families. At the OLD loader's measured 2,320 bytes/row, even the `train` split alone (the bulk of
those rows, pooled across the 5 training families × 3 capacities) is enough to reach tens of GB,
consistent with the observed ~54.5 GiB kill. Model fitting itself is not implicated: `random_forest`
already uses `n_jobs=1` (no joblib process duplication) and `hist_gb` has no multi-process fan-out —
both respect `OMP_NUM_THREADS`/thread-count env vars for internal threading only.

## C. Scientific equivalence contract

Unchanged by the repair:
- Row set: identical shard files, identical shard visitation order (`random.Random(seed + 31)` shuffle,
  unchanged), identical `(horizon, split)` filter.
- Subsampling: when `--max-train-rows`/`--max-val-rows`/`--max-test-rows` is set, the same raw-row
  accumulation cap (`max_rows * 2`) and the same `random.Random(seed).sample(...)` call is used —
  `random.sample` on a sequence of length `n` selects the same *positions* regardless of the sequence's
  element type, so sampling over row-indices is index-identical to the old sampling over row-dicts.
- Features: same `EVICT_VALUE_V1_FEATURE_COLUMNS`, same column values, and the same
  `float64` dtype for `X`/`y`.
- Labels: same `y_loss` values.
- Models: same three estimators, same hyperparameters, same seeds, same fit order.
- Selection rule: same `min(val_mean_regret, val_mae, val_rmse)` tie-break, unchanged.
- Ranking/regret metrics: same per-decision grouping and `(pred, candidate_page_id)` /
  `(y_loss, candidate_page_id)` tie-break logic, now operating on parallel arrays instead of
  list-of-dicts but computing identically.

Only implementation-level details that change: rows are streamed directly into preallocated arrays
instead of materialized as Python dicts, and the loader can abort cleanly under an optional soft memory
guard before the kernel OOM-killer intervenes. All scientific invariants listed above are preserved
exactly.
