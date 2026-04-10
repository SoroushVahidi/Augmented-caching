# evict_value_v1 Wulver dataset runbook (phase 1)

**KBS manuscript note:** For the **canonical multi-trace `heavy_r1` train + eval + manuscript bundle**, use `docs/wulver_heavy_evict_value_experiment.md` and `docs/kbs_manuscript_submission_index.md`. This runbook describes **dataset-generation-focused** Slurm patterns (`evict_value_v1_wulver_dataset.sbatch` and array mode) that are **not** the same filenames as the `heavy_r1` analysis artifacts consumed by `scripts/paper/build_kbs_main_manuscript_artifacts.py`.

## 1) Optional local sanity check

Run a small sample generation locally before cluster submission:

```bash
python scripts/build_evict_value_dataset_wulver_v1.py \
  --trace-glob "data/processed/*/trace.jsonl" \
  --capacities "64" \
  --horizons "8" \
  --max-traces 1 \
  --max-rows-per-shard 10000 \
  --out-dir data/derived/evict_value_v1_wulver_smoke
```

## 2) Submit on Wulver

Primary job (single sbatch):

```bash
sbatch --partition=general --qos=standard \
  --export=ALL,PYTHONPATH=src,TRACE_MANIFEST=analysis/wulver_trace_manifest.csv \
  slurm/evict_value_v1_wulver_dataset.sbatch
```

Optional array mode (capacity/trace-slice sharding):

```bash
sbatch --array=0-7 slurm/evict_value_v1_wulver_dataset_array.sbatch
```

## 3) Expected outputs

Under `data/derived/evict_value_v1_wulver/` (or your chosen output root):

- `shards/*.csv`: candidate-level dataset shards
- `manifest.json`: shard inventory and generation metadata
- `split_summary.csv`: row/decision counts by split/family/capacity/horizon
- `logs/*.done.json`: per-work-item completion markers for restartability

The builder now runs a preflight phase first and prints:
- number of discovered traces
- request/unique-page counts per trace
- whether each trace is classified as tiny

## 4) Check job completion

- Check Slurm state:
  - `squeue -u $USER`
  - `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS`
- Check completion markers:
  - `ls data/derived/evict_value_v1_wulver/logs/*.done.json | wc -l`
- Re-run the same command to resume unfinished work (completed items are skipped unless `--overwrite` is set).

## 5) Leakage audit and split summaries

- Leakage/split audit note:
  - `analysis/evict_value_v1_wulver_split_audit.md`
- Split summary output from generation:
  - `data/derived/evict_value_v1_wulver/split_summary.csv`
- Optional recompute/repartition summary:

```bash
python scripts/summarize_evict_value_wulver_splits.py \
  --manifest data/derived/evict_value_v1_wulver/manifest.json \
  --split-mode trace_chunk
```

## 6) Multi-family validation phase (dataset + train + policy)

Use the full trace pool:

- `analysis/wulver_trace_manifest_full.csv`

**Recommended Slurm entrypoint (dataset + summaries + training + policy comparison):**

```bash
sbatch slurm/evict_value_v1_wulver_multi_phase.sbatch
```

**Capacities:** default `32,64,128` in the sbatch script — a small, disciplined set that stays tractable while spanning multiple scales.

**Training:** `scripts/train_evict_value_wulver_v1.py` (Ridge, RandomForest, HistGradientBoosting only).

**Verdict / interpretation:** see `analysis/evict_value_v1_wulver_phase_verdict.md` after a run.
