# Offline Baseline 1: Belady (Uniform Paging)

This repository now includes a small **offline baseline framework** under `lafc.offline`, with Belady as the first concrete solver.

## What this solves

Belady's furthest-in-future rule is **exact OPT** for classical offline paging when:

- all items have equal size,
- all misses have equal retrieval cost,
- full request sequence is known in advance.

On a miss with full cache, the solver evicts the cached page with farthest next use (or a page never used again).

## Scope and limitations

This solver is intentionally conservative:

- Exact only for the uniform paging setting above.
- It is not an approximation for weighted/general caching.
- It does not claim guarantees outside uniform assumptions.

Validation defaults to strict mode and fails if retrieval costs are non-uniform.

## Why add this first

Belady is a clean oracle baseline for unweighted offline paging and a natural anchor before adding:

- offline general-caching approximations,
- flow-based or LP-based offline baselines,
- label-generation utilities for ML pipelines.

## How to run

```bash
python scripts/run_offline_belady.py \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --output-dir output/offline_belady_example
```

Optional relaxed mode:

```bash
python scripts/run_offline_belady.py \
  --trace data/example.json \
  --capacity 3 \
  --validation-mode coerce
```

## Outputs

The script writes text outputs only:

- `summary.json` (high-level metrics)
- `per_step_decisions.csv` (hit/miss and eviction decisions)
- `diagnostics.json` (tie counts, never-used-again evictions, validation report)
- `report.md` (short human-readable summary)

## Extension points

Current package layout:

- `lafc.offline.base`: offline solver protocol and generic runner.
- `lafc.offline.types`: result/decision dataclasses.
- `lafc.offline.validation`: assumption checks (`strict` vs `coerce`).
- `lafc.offline.belady_uniform`: exact Belady implementation.
- `lafc.offline.io`: standardized output writers.

Future offline baselines can plug in by implementing the `OfflineSolver` protocol and reusing the same runner/output path.
