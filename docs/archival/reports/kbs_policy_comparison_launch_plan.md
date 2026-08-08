# KBS heavy_r1 policy-comparison: launch plan (not yet executed)

Status: dataset build done, dataset summary done, model training in progress.
This document is the pre-flight plan for the eval stage; **the full sweep has
not been launched.** See `reports/kbs_policy_comparison_heavy_r1_completion_report.md`
(once it exists) for what actually ran.

## 1. Exact full canonical command

Matches `slurm/evict_value_v1_wulver_heavy_eval.sbatch` defaults for `EXP_TAG=heavy_r1`,
run directly (no Slurm) inside tmux session `kbs_policy_comparison_heavy_r1`:

```bash
set -o pipefail
source .venv_kbs_heavy_r1/bin/activate
export PYTHONPATH=src
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32,64,128,256 \
  --max-requests-per-trace 50000 \
  --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md \
  2>&1 | tee logs/kbs_policy_comparison_heavy_r1/policy_comparison_run.log
echo "POLICY_COMPARISON_EXIT=${PIPESTATUS[0]}" | tee -a logs/kbs_policy_comparison_heavy_r1/policy_comparison_run.log
```

This is canonical because it reproduces, line for line, the only documented
producer of `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`
(`docs/wulver_heavy_evict_value_experiment.md`, `reports/kbs_revision_repo_audit.md:92-99`).

## 2. Expected inputs

- `analysis/wulver_trace_manifest_full.csv` — present, 7 rows (brightkite, citibike,
  wiki2018, twemcache, metakv, metacdn, cloudphysics).
- `data/processed/<family>/trace.jsonl` for each of the 7 families — present (built
  during tonight's acquisition phase).
- `models/evict_value_wulver_v1_best_heavy_r1.pkl` — **does not exist yet**; produced
  by the in-progress training step, then copied from `models/evict_value_wulver_v1_best.pkl`.

Note: this script reads traces directly from `data/processed/*/trace.jsonl` via
`load_trace_from_any`, **not** from the 96G `data/derived/evict_value_v1_wulver_heavy_r1`
shard set — that shard set is training-only input. Eval has no dependency on it.

## 3. Expected outputs

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (header + up to
  7 traces × 4 capacities × 7 policies = 196 rows)
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md`

## 4. Can it resume from partial output?

**No.** Reading `scripts/run_policy_comparison_wulver_v1.py:101-131`: all
`(trace, capacity, policy)` results accumulate in an in-memory list (`rows_out`)
inside one triple-nested loop, and the CSV/MD are written exactly once, after the
loop fully completes. There is no checkpoint file, no incremental write, and no
flag to skip already-computed combinations. Any interruption (kill, crash, reboot,
OOM) loses 100% of progress for that invocation — you get nothing until a clean
full run finishes.

## 5. Can it be split into smaller tmux jobs?

**Yes**, via existing CLI flags, no code changes needed:

- **By capacity** (recommended): run with `--capacities 32`, then `64`, `128`, `256`
  as four independent invocations, each to its own `--out-csv`
  (e.g. `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv`, etc.),
  then concatenate (drop duplicate headers) into the canonical file once all four
  finish.
- **By policy**: `--policies` already accepts a comma-list; could split the 7
  baseline policies into 2-3 groups per run.
- **By trace**: only via `--max-traces N` (first-N-rows prefix) or by writing a
  temporary subset manifest CSV — less convenient than capacity/policy splitting.

Splitting by capacity is the natural choice here because the dominant cost (see
§6) scales with capacity, so each chunk is independently sized, independently
restartable, and gives four checkpoints instead of one all-or-nothing run.

## 6. Estimated runtime

No reliable a-priori estimate exists — here is the concrete evidence and why it's
hard to extrapolate:

- **Historical:** the full 7×4×7×50k run (Wulver job 908326) exceeded a 24h
  walltime limit before producing output and was never observed to complete even
  after the walltime was raised to 72h (`reports/kbs_revision_repo_audit.md:101`,
  dedicated allocation: 16 CPUs / 64GB RAM).
- **heavy_smoke run** (job 910353, `00:02:45` total, *including* dataset build +
  training, not eval alone) used only 2 traces × 2 capacities × **5000**
  req/trace — 10x fewer requests/trace than the canonical 50000, and it never
  exercised capacities 128/256.
- **Why smoke timing doesn't extrapolate linearly:** `EvictValueV1Policy._choose_victim`
  (`src/lafc/policies/evict_value_v1.py:180-202`) calls the model's
  `predict_loss_one` **once per cache-resident candidate on every cache miss**
  (up to `capacity` individual model-inference calls per miss, each preceded by an
  O(history_window) feature scan). Cost scales with `capacity × miss_count`, not
  request count alone — so smoke's small capacities (32/64) and short traces
  (5000 req) systematically understate cost at capacity 128/256 and 50000 req.
  This is the most plausible explanation for the historical 24h+ runtime.
- **Local CPU:** 20 cores available here vs. 16 dedicated on Wulver — comparable
  order of magnitude, not a guaranteed advantage; the script is single-process/
  single-threaded, so extra cores only help if we parallelize chunks ourselves.

**Recommendation:** don't trust a guess — run the §7 validation pass at the real
target request count (50000) and the most expensive capacity (256) to get one
empirically grounded `(trace, capacity)` timing, then scale by capacity and by
7 traces for a defensible full-run estimate.

## 7. Safe first-pass / validation command

Writes to a clearly non-canonical filename so it cannot be confused with, or
accidentally consumed by, `build_kbs_main_manuscript_artifacts.py` (which looks
for the exact `_heavy_r1` filename):

```bash
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --max-traces 1 \
  --capacities 256 \
  --max-requests-per-trace 50000 \
  --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.md
```

`--max-traces 1` takes the first manifest row (brightkite_50k); capacity 256 is
the most expensive setting, so this gives a worst-case per-trace timing bound.
Requires the trained `heavy_r1` model — blocked until training finishes.

## 8. Is the full 7×4×7 sweep necessary for the reviewers' end-to-end concern?

Per the repo's own standard (`docs/kbs_manuscript_workflow.md`,
`docs/evict_value_v1_kbs_canonical_artifacts.md`,
`docs/wulver_heavy_evict_value_experiment.md`): **yes, as documented.** These
docs explicitly forbid substituting smaller/partial runs ("do not substitute
smoke outputs for heavy_r1 manuscript tables"); only this exact scope is treated
as the citable Table 3 / Figures 2-3 source. A scoped-down run can serve as a
pipeline/timing validation, not as a replacement for the canonical claim. If
time/resource constraints make the full scope impractical, that is a scope
decision for you to make explicitly and document — not something to default into
silently.

## 9. Disk/memory risk

**Low.** Output is one small CSV/MD pair (the heavy_smoke equivalent was ~2.7KB/
1.2KB combined; the full file will be larger but still on the order of 196 rows,
well under 1MB). The script holds one trace (~50k requests) in memory at a time
inside the loop and doesn't accumulate large intermediates — `rows_out` only grows
by one summary row per `(trace, capacity, policy)` triple. The risk here is
wall-clock time, not disk or memory; current free disk (500G) and memory headroom
(59G available) are not a constraint for this stage.

## 10. Recommendation

1. Finish and verify training (in progress).
2. Run the §7 single-trace/cap-256 validation pass to get a real timing data point.
3. Based on that timing, decide between the monolithic full command (§1) or the
   capacity-chunked split (§5) — chunking is recommended regardless of the timing
   result, since it converts an unrecoverable multi-hour/day single shot into four
   independently restartable, individually observable pieces, at no cost in
   correctness (same script, same flags, just narrower `--capacities` per call).
4. Do not launch any multi-hour run without explicit go-ahead — this stage
   exceeded 24h once before and was never confirmed to complete even at 72h
   elsewhere.
