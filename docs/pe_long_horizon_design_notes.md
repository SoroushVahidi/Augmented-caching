# PE long-horizon production: design notes

Planning artifact only. No Wulver compute described here has been run as
part of this document; measured figures are cited from jobs 1287838 (cap32
preflight) and 1288047 (cap256 probe, referenced once it completes).

## 1. Generator interface findings (Task 3)

Read directly from `src/lafc/evict_value_wulver_v1.py::iter_candidate_rows`
and `scripts/build_evict_value_dataset_wulver_v1.py::main`.

- **One execution naturally computes multiple horizons.** `--horizons`
  accepts a comma-separated list; a single `iter_candidate_rows` call loops
  `for h in cfg.horizons` per candidate.
- **Feature computation is shared across horizons** (computed once per
  candidate/decision, outside the horizon loop) -- **the label simulation is
  not** (`_simulate_lru_misses(after, future[:h], ...)` is independently
  re-run per horizon, O(h) each call).
- **Splitting horizons into separate tasks (family x capacity x horizon)
  would duplicate the entire outer trace-walk and feature computation 3x**
  per family/capacity -- real wasted CPU-hours, not just launch overhead.
  This is why the production decomposition is **Plan A** (family x capacity
  task, horizons {32,64,128} computed together), not Plan B.
- **Output shard names are collision-safe** within a directory
  (`{trace_name}__cap{cap}.part{idx:04d}.csv`) -- but **`manifest.json` and
  `split_summary.csv` are unconditionally overwritten at end-of-run** from
  in-process dicts. Two tasks sharing an `--out-dir` will silently clobber
  each other's manifest/summary (shard CSVs themselves are safe). This is
  why every production task gets its own `out_dir`
  (`run_root/raw/<family>__cap<capacity>/`), and why the aggregator
  (`aggregate.py`) rescans shards directly rather than trusting any
  per-task `manifest.json`.
- **Concurrent reads of the same processed trace file are safe** (read-only
  JSONL parse, no locking needed) -- multiple tasks may read
  `data/processed/<family>/trace.jsonl` at once without coordination.
- **Chunked/merged outputs are deterministically mergeable** by rescanning
  shard CSVs and recomputing summary stats -- exactly what
  `validate_cell.py` and `aggregate.py` do.

## 2. Failure / resume design (Task 9)

| Scenario | Behavior |
|---|---|
| One array task fails (non-zero exit) | Slurm marks that array index FAILED; other 19 tasks unaffected (independent output dirs). No `COMPLETE.json` is written (only written after successful generation), so the task is unambiguously incomplete. |
| One task times out | Same as failure -- no `COMPLETE.json`. Partial shard files may exist under `out_dir/shards/`; the *next* attempt at that `out_dir` detects them (dir exists, no `COMPLETE.json`) and writes `INCOMPLETE.json` + exits non-zero rather than silently treating partial shards as done. |
| One task produces incomplete output | Same detection as above -- the production sbatch template explicitly checks `[[ -d "${OUT_DIR}" && ! -f "${COMPLETION_MARKER}" ]]` before running and refuses to proceed silently. |
| Validation fails for a cell | `validated_summary.json` is still written, with `"validation_result": "FAIL"` and the specific violated invariant recorded. `aggregate.py` treats FAIL the same as missing -- aggregation is refused. |
| Aggregation detects missing cells | `aggregate.py` exits 1 and prints exactly which task_ids are missing/failed -- no partial/best-effort output is written. |
| Master SSH connection dies | No effect on Wulver-side state -- production, validation, and aggregation are all ordinary `sbatch` jobs (with `--dependency=afterok`), which run independently of any client SSH session. Reconnecting and re-running `resume_planner.py` (read-only) recovers full status. |
| Workstation reboots | Same as above -- nothing about the campaign depends on the local workstation staying up once jobs are submitted. Only the *decision* to submit/resubmit requires an operator back online. |

`resume_planner.py` (this package) inspects `COMPLETE.json` /
`validated_summary.json` per task and prints a ready-to-review `--array=`
spec of exactly the NOT_RUN/INCOMPLETE/FAILED indices -- it does not
resubmit anything itself.

## 3. Storage safety (Task 10)

Measured so far: 7.6GB for 5 families x cap32 x 4 horizons (job 1287838).
Cap256 output size is pending job 1288047 -- do not naively 8x-scale the
cap32 figure; wait for the measurement (capacity affects both bytes/row via
feature-vector width, which is capacity-independent, and row *count*, which
depends on decision count x candidates-per-decision, only the latter of
which scales with capacity).

**Recommendation: (C) convert to Parquet then remove raw CSV only AFTER
validation + checksum.** Rationale:
- Raw CSV must exist first for `validate_cell.py` to compute checksums and
  scientific summaries against the actual bytes that were generated (not a
  re-derived Parquet copy) -- so CSV can't be skipped.
- At tens-of-GB-to-possibly-low-hundreds-of-GB production scale (extrapolating
  from the measured 7.6GB single-capacity slice across 4 capacities and a
  smaller 3-horizon set), Parquet's columnar compression is worth the
  conversion step once a cell is validated, both for storage and for faster
  downstream re-reads.
- Deleting raw CSV *before* validation would remove the ability to re-check
  or re-derive summaries if a validator bug is later found -- checksums
  (`output_sha256` in `validated_summary.json`) are computed against the CSV
  specifically so a later Parquet conversion can be verified against them.
- Do not delete anything as part of this planning task; this is a
  recommendation for the production pipeline's post-validation step, to be
  implemented (and explicitly confirmed) separately.

### Output layout

```
run_root/                              (= data/derived/pe_long_horizon_production_v1)
  manifest/
    manifest.json                      (copy of configs/pe_long_horizon_production/manifest.json, frozen at launch)
  raw/
    <family>__cap<capacity>/
      shards/*.csv
      logs/*.done.json                 (per family/capacity, from the generator itself)
      COMPLETE.json                    (atomic, written only after success)
      INCOMPLETE.json                  (written if a prior partial run is detected)
      validated_summary.json           (written by validate_cell.py)
  validated/                           (post-validation Parquet conversions, once implemented)
  summaries/
    table_A_family_capacity_horizon.csv
    table_B_horizon_pooled.json
    table_C_h16_deltas.json
    table_C_stepwise_deltas.json
  logs/
    slurm stdout/stderr per task (production, validation, aggregation)
  provenance/
    resource_params.env snapshot, git SHA, launch manifest
```
