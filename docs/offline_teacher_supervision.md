# Offline-Teacher Supervision for Eviction Learning

## Motivation

Current eviction datasets in this repo include myopic or finite-horizon proxy labels. This new path adds **offline teacher-derived labels** so candidate scorers can train on stronger supervision.

## Teacher selection

For each trace/capacity decision point where eviction is required:

1. **Exact teacher (`exact_teacher_belady`)** is used when uniform assumptions hold:
   - equal retrieval costs,
   - equal page sizes.
2. Otherwise **approximate teacher (`approx_teacher_lp`)** is used:
   - LP+rounding offline general-caching solver.

Teacher type is recorded per row.

## Label generation strategy

At each full-cache miss and each candidate victim:

- force that candidate eviction,
- build forced post-decision cache,
- evaluate teacher cost on suffix horizon `H` (`--horizon`) from that forced state,
- compute per-candidate labels:
  - `teacher_best` (binary best indicator),
  - `teacher_regret = teacher_cost - best_teacher_cost`,
  - `teacher_rank`.

Pairwise rows are also generated with preference label `label_i_better` and regret difference.

## Computational shortcut (explicit)

To keep dataset generation tractable, the teacher is evaluated on a **finite suffix horizon** rather than always on the full remaining trace.

## Outputs

`build_offline_guided_eviction_dataset.py` writes text-only outputs:

- `candidate_rows.csv`
- `pairwise_rows.csv`
- `dataset_summary.json`

`run_offline_teacher_label_first_check.py` writes:

- `candidate_rows.csv`
- `summary.json`
- `report.md`

## Integration with current learning direction

`build_evict_value_decision_aligned_dataset.py` now supports:

- `--label-source heuristic` (existing rollout labels)
- `--label-source offline_teacher` (new offline teacher labels)

This keeps compatibility with existing candidate-scoring training paths while adding a stronger supervision option.

## Commands

```bash
python scripts/build_offline_guided_eviction_dataset.py \
  --trace-glob data/example_*.json,data/example_general_caching.json \
  --capacities 2,3 \
  --horizon 24 \
  --output-dir data/derived/offline_teacher_supervision
```

```bash
python scripts/run_offline_teacher_label_first_check.py \
  --traces data/example_unweighted.json,data/example_general_caching.json \
  --capacity 3 \
  --horizon 24 \
  --output-dir analysis/offline_teacher_first_check
```
