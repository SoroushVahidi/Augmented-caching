# evict_value_v1 split-hygiene audit (Wulver phase 1)

## Current behavior (before this phase)

- Dataset generation was concentrated in `scripts/build_evict_value_dataset_v1.py` using only tiny in-repo traces (`data/example_*` plus synthetic stress traces).
- Splitting used `_split_by_trace_and_capacity` from `src/lafc/evict_value_dataset_v1.py`:
  - hash key = `trace|capacity`
  - every row from the same `trace,capacity` pair goes to one split.
- If any split was empty, a row-index fallback split was applied, which can place near-adjacent rows from the same trace into different splits.
- Training/evaluation scripts (`scripts/train_evict_value_v1.py`, `scripts/run_evict_value_v1_first_check.py`) consumed these small derived CSVs.

## Why this is insufficient for serious validation

- **Trace-segment leakage risk**: no explicit chunk-level boundary policy and fallback index split can mix adjacent decision states.
- **Capacity leakage risk**: split key includes capacity, so the same trace can appear in different splits across capacities.
- **Horizon leakage risk**: rows for multiple horizons can share the same decision state and be distributed by fallback behavior.
- **Near-duplicate decision-state leakage risk**: deterministic neighboring states in one trace may leak across splits when fallback triggers.
- **Tiny/example dependence**: primary data came from tiny examples and synthetic stress traces, not broad processed trace coverage.

## Recommended Wulver split design

- Build reusable candidate-level shards once (large trace coverage, multiple capacities/horizons) and keep labels cached.
- Include explicit metadata per row: `trace_name`, `trace_family`, `dataset_source`, `capacity`, `horizon`, `decision_id`, `candidate_page_id`.
- Use deterministic split assignment via one of:
  - `trace_chunk`: key on `(trace_name, decision_chunk_id)` with fixed chunk size.
  - `source_family`: key on `(dataset_source, trace_family)` for stronger domain-shift holdout.
- Keep split assignment deterministic (seeded stable hash) and persist `split` in each row.
- Emit summary tables with row/decision counts by `split x family x capacity x horizon`.

## Phase-1 implementation scope

- Implemented in this phase:
  - Wulver-scale dataset builder with restartable shards + manifest.
  - Deterministic split utilities (`trace_chunk`, `source_family`).
  - Split summary outputs for leakage-audit visibility.
  - Wulver sbatch scripts and runbook for long CPU jobs.
- Not implemented in this phase:
  - New model methods/objectives (pairwise ranking, alternate targets, etc.).
