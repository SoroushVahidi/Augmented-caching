# Common-Model Objective Control V2 Fix

## WHY V1 WAS INVALID

The V1 common-model runner trained `objective_pairwise` incorrectly. Pair rows
include `label_i_preferred`, but V1 discarded that field and always trained the
candidate stored as `i_*` to outrank the candidate stored as `j_*`. Because
`build_pairwise_rows()` sorts candidates by `candidate_page_id` before forming
pairs, the V1 pairwise arm learned candidate-ID ordering pressure rather than
the intended pairwise target.

V1 common-model output is therefore:

`INVALID_FOR_FINAL_OBJECTIVE_COMPARISON`

The old run was stopped and preserved under:

`reports/common_model_v1_interrupted_20260813/`

## V1 PERFORMANCE DEFECTS

- repeated scoring: V1 called the learned scorer inside the `min/max` key
  function, so a capacity-`k` eviction decision made `k` calls, and each call
  scored all `k` candidates.
- duplicate selected-row work: V1 generated selected training/validation labels
  once to discover decision IDs and then again to materialize selected rows.
- unnecessary deployment label computation: V1 used a generic oracle replay path
  that computed eviction-loss, next-arrival, and reuse-distance labels for every
  online candidate even though learned victim selection needs only features.

## WHAT V2 CHANGES

SEMANTIC_CORRECTION:

- `objective_pairwise` now uses the pair label to orient the training pair.
- The common scorer's convention is explicit: higher score means prefer to
  evict for objectives deployed with `direction=max`.
- For next-arrival pairwise rows, `label_i_preferred` is the keep/reward
  preference. V2 inverts it for the common eviction-score convention.

PERFORMANCE_ONLY_OPTIMIZATION:

- Scores are computed once per eviction decision and then reused by the
  deterministic victim selector.
- Selected training/validation rows are collected in one pass.
- Production deployment uses feature-only candidate rows; expensive label
  diagnostics are optional and default off.

INFRASTRUCTURE/PARALLELIZATION:

- V2 units run independently by family/capacity.
- A fail-closed reducer validates all 21 units and all 84 rows.
- A CPU-only Wulver SLURM array script is prepared but not submitted.

## WHAT V2 DOES NOT CHANGE

- held-out folds
- training families
- validation family
- held-out family
- capacities 32/64/128
- history/scoring windows
- feature columns
- horizon 4
- training decision budget
- validation decision budget
- seed 0
- shared two-layer ReLU scorer architecture
- optimizer equations
- epoch count
- L2 regularization
- scalar objective definitions
- scalar deployment directions
- scored-window hit/miss metric

## REGRESSION TESTS

Targeted V2 tests:

`PYTHONPATH=src pytest -q tests/test_common_model_objective_control_v2.py`

Observed result before commit:

`10 passed`

## SCALAR V1/V2 EQUIVALENCE

Regression unit:

`brightkite_cap32`

V2 isolated output:

`/tmp/common_model_objective_control_v2_regression_brightkite_cap32_1786670274`

For `objective_eviction_loss`, `objective_next_arrival`, and
`objective_reuse_distance`, V2 matched V1 exactly on:

- summary objective/family/capacity keys
- misses
- miss ratio
- validation regret
- trace hash
- seed
- every stored model array
- model SHA-256

Result:

`SCALAR_V1_V2_EXACT_EQUIVALENCE`

The old V1 run did not store deployment victim sequences. The V2 scalar replay
uses byte-identical model arrays, byte-identical training artifacts, one-score
caching that preserves the same deterministic victim selector, and feature-only
rows tested against full rows for the frozen feature columns.

## PAIRWISE SEMANTIC VALIDATION

V1 pairwise is not required to match:

`PAIRWISE_V1_INVALID`

V2 validation:

- synthetic candidate IDs deliberately oppose target preference
- pair orientation uses `label_i_preferred`
- swapped candidate IDs preserve semantic orientation
- a learnable synthetic ordering is learned in the corrected eviction-score
  direction
- isolated `brightkite_cap32` pairwise smoke run completed with diagnostics off

Result:

`PAIRWISE_V2_SEMANTICS_VERIFIED`

## PERFORMANCE CHECK

Small in-memory benchmark on 250 `brightkite` capacity-32 decisions:

- old-style scorer calls: 8000
- V2 scorer calls: 250
- call reduction: 32x
- old full-label row time: 0.3842 s
- V2 feature-only row time: 0.2053 s
- old scoring time: 0.3230 s
- V2 scoring time: 0.0135 s
- row speedup: 1.87x
- scoring speedup: 23.92x

This is a small correctness/performance sanity check, not publication timing.

## WULVER PLAN

Run 21 independent unit tasks:

`7 families x 3 capacities = 21`

Each task writes only:

`analysis/common_model_objective_control_wulver_v2/units/<family>_cap<capacity>/`

The reducer then validates expected keys, objective coverage, trace hashes,
source commit consistency, duplicates, missing rows, and row cardinality before
writing campaign-level files.

Prepared script:

`slurm/common_model_objective_control_v2_array.sbatch`

Reducer:

`scripts/experiments/reduce_common_model_objective_control_v2.py`

GPU is not requested. Python replay is single-thread dominant, so the default is
one CPU per task with BLAS/OpenMP thread counts pinned to the allocated CPU.

## PROVENANCE

Old:

`2752857bd6a6a1a12e6e3fed44340b407f5c8e56`

New:

`THIS_COMMIT` (the git commit containing this report; final SHA is recorded in
the repository history and task completion note)

## ELIGIBILITY

V1 common-model output:

`INVALID_FOR_FINAL_OBJECTIVE_COMPARISON`

V2:

`NOT_PRIMARY_UNTIL_FULL_RUN_AND_INTEGRITY_AUDIT`
