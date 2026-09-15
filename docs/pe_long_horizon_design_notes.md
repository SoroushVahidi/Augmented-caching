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

## 4. Shared-horizon optimization (deferred engineering debt, Task 14 of the final launch gate)

**Status: analytically proven from source semantics, NOT experimentally
equivalence-validated, NOT used in this production campaign, and MUST NOT
appear in the manuscript as a scientific result.** This section documents it
purely as a repository engineering note for a possible future dataset
revision.

`src/lafc/evict_value_dataset_v1.py::_simulate_lru_misses` rebuilds a fresh
`collections.OrderedDict` of size `capacity` on every single
(candidate, horizon) call -- called `capacity x num_horizons` times per
decision. This gives the generator's per-decision cost a real O(capacity^2)
component (confirmed by calibrating a two-term cost model against jobs
1287838 and 1288047; see the session's `LONG_HORIZON_TIMEOUT_DECISION_REPORT`
for the full derivation).

Because `_simulate_lru_misses` returns a strictly monotonic cumulative miss
count over a forward window, the trajectory up to step 32 is entirely
determined by steps 1-32 and unaffected by anything at steps 33-128. One
simulation run to `H_max=128` per candidate, recording the running miss
count at checkpoints {32,64,128} (or {16,32,64,128} including the canonical
control), would therefore reproduce the current per-horizon outputs
*exactly* -- this is a provable identity, not an approximation. It would
cut `OrderedDict` construction (the dominant cost term at large capacity)
from 3x (or 4x) per candidate down to 1x, and cut total replay steps from
`sum(horizons)` down to `max(horizons)`. Theoretical speedup for the
{32,64,128} production case: ~2.96x (~3x).

This was decided **not** to be worth pursuing before this campaign's launch:
the current (unmodified) generator's evidence-based, conservatively
safety-margined runtime already clears the 18h/24h operational deadlines
with 2.5-3x headroom (see the timeout-decision report), so taking on new
scientific-code risk and a fresh equivalence-validation cycle would only
delay production for a benefit the deadline doesn't require. If a future
dataset revision wants to pursue it, required before any production use:
exact row/key/y_loss/y_value equality against the current implementation
across >=2 families x capacities {32,128,256} x horizons {16,32,64,128},
plus a fresh Wulver timing preflight of the modified code path.

## 5. Storage incident and recovery (job 1288536, 2026-09-15)

The first production launch (array job 1288536) failed campaign-wide
within 1-3 minutes: all 20 tasks hit `OSError: [Errno 122] Disk quota
exceeded` writing to `~/lafc-work/Augmented-caching/data/derived/...`,
physically `/mmfs1/home/sv96/...` -- a GPFS **HOME** fileset. The original
launch gate's storage check used `df -h` on that path, which reports the
full `/mmfs1` pool (809TB free) rather than the actual per-user HOME
quota -- NJIT documents HOME as ~50GB/user and explicitly "not intended
for research data," which total home usage (~55GB, measured) is
consistent with already exceeding.

**`quota_info $LOGNAME`** (NJIT's documented quota tool) was tried after
`module load wulver` and is **not present on this system** -- command not
found, and not locatable anywhere under `/apps` or `/opt`. `quota -s`
reports an unrelated local filesystem, not `/mmfs1`. The one reliable
signal found: **`df -T` on a PROJECT or SCRATCH path (not HOME) correctly
reports that fileset's own quota** -- confirmed by exact round-number
matches to NJIT's documented allocations (PROJECT: exactly 2TiB;
SCRATCH: exactly 10TiB, both for the `ikoutis` PI group).

**Fix**: `RUN_ROOT` in `build_manifest.py` now points at
`/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1`
(SCRATCH: 10TiB quota, ~984MB used group-wide, ~303GB used by sv96
specifically under the separate 2TiB PROJECT allocation -- SCRATCH has by
far the most headroom and is NJIT's documented location for "temporary
simulation/intermediate data"). All downstream scripts (validator,
aggregator, resume planner, launch guard) already treated `run_root`/
`out_dir` as opaque path strings, so no code changes were needed there --
only the one constant in `build_manifest.py`.

**Caveat: SCRATCH is not backed up and is subject to an ~30-day purge.**
Recommended layout going forward:

```
SCRATCH (temporary, active compute):
  /mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1/
    manifest/ raw/ validated/ logs/ provenance/ summaries/

PROJECT (durable, backed up, copy here after validation):
  /mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1/
    validated/ summaries/ provenance/ manifests/ checksums/
```
Raw CSV stays in SCRATCH through generation + validation; only the
(much smaller) validated summaries, provenance, and checksums get copied
to PROJECT afterward -- not implemented yet (no production run has
succeeded to copy from), but the launch guard's new `check_run_root_location`
and `check_quota_audit` checks (see `launch_guard.py`) block any future
relaunch from repeating the HOME mistake, and require a same-day frozen
`quota_audit.json` (via `freeze_quota_audit.py`) rather than accepting
cluster-wide `df` free space as evidence.

**Data preservation**: nothing was deleted as part of this recovery. The
failed campaign's partial output (~3.2GB, `pe_long_horizon_production_v1`
under HOME) and all prior preflight/probe artifacts remain exactly as they
were; see the storage-recovery audit's artifact inventory for
classification and disposition options, none of which have been acted on.
