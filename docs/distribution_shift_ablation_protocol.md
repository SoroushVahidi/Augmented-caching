# Distribution-shift ablation protocol (`distribution_shift_ablation_v1`)

Frozen before any full-scale result is visible. Tests Reviewer #2's third
major concern: that the finite-horizon eviction-loss target is learnable
offline, but the resulting policy underperforms simple baselines online,
possibly because training labels are generated on states visited under one
trajectory (LRU) while the deployed model visits states shaped by its own
(possibly erroneous) decisions -- train/deployment state-distribution shift.

## 1. Prior-art audit (why this is new work, and what is reused)

| mechanism | already exists? | file | tested? | prior results | usable here? |
|---|---|---|---|---|---|
| Rollout loss / regret labels, configurable **label-continuation** policy (`lru`/`blind_oracle`/`fifo`) | yes | `src/lafc/evict_value_v2_rollout.py` (`simulate_rollout_misses`, `_choose_victim`, `EvictValueV2RolloutConfig`) | yes (`tests/`) | `docs/internal_current_project_decisions.md` §6-7: continuation-policy choice already explored, found to affect label proxies more than downstream replay; LRU kept as practical default; explicitly **not** part of the novelty claim | Reused unmodified for ground-truth labels (all conditions use `lru` continuation for the label -- see §7 below). Not the thing being ablated here. |
| Continuation-policy light ablation experiment | yes | `scripts/experiments/run_continuation_policy_light_ablation.py` | yes (script-level, exploratory) | `analysis/continuation_policy_light/*` (exploratory, not canonical) | Confirms: existing work varies the **label-rollout** continuation policy, never the **state-generation / trajectory** policy. Does not overlap with this experiment's independent variable. |
| Decision-aligned rollout/regret/pairwise targets | yes | `scripts/build_evict_value_decision_aligned_dataset.py`, `docs/decision_aligned_targets.md` | yes | exploratory only | Confirms candidate enumeration / feature computation pattern to reuse; not itself a state-shift mechanism. |
| Candidate state generation via **DAgger-style on-policy trajectory** (state visited depends on the learned model's own decisions, not fixed LRU) | **no** | -- | -- | -- | **New. This is the actual independent variable of this experiment.** |
| "Behavior policy" (what really happens to the cache while walking the trace to emit training rows) | Implicitly always LRU: every existing builder (`iter_candidate_rows`, `build_rollout_candidate_rows_v2`, `iter_multi_label_candidate_rows`) ends each decision with `lru_victim = candidates[0]; order.pop(lru_victim)` -- i.e. the cache trajectory used to generate subsequent states is unconditionally LRU, regardless of which label/continuation policy is used. | multiple | yes (indirectly, via each builder's own tests) | -- | This is precisely `OFF_POLICY_LRU` below: not new code, a documented characterization of existing behavior. |
| Frozen-model batched eviction-loss scorer (argmin over candidates) | yes (built for `supervision_objective_ablation_v1`, separate worktree) | `src/lafc/policies/supervision_objective_ablation_policy.py` (`ScalarObjectivePolicy`), `src/lafc/evict_value_model_v1.py` (`predict_loss_batch`) | yes | n/a (different protocol) | Reused as the mechanism for the learned behavior policy's victim choice (same argmin-under-frozen-model rule, no reimplementation). |
| State-shift / trajectory-divergence metrics | no | -- | -- | -- | New (`src/lafc/distribution_shift_ablation.py`). |

**Conclusion:** no existing mechanism generates training states from the
learned policy's own decisions. The "continuation policy" axis already
explored in this repo is a different, previously-tested, and separately
documented concept (label rollout dynamics) that this protocol does not
re-litigate.

## 2. Frozen folds, capacities, target

Reuses `reviewer_fair_cross_family_v1` (`configs/fair_cross_family_v1/folds/*.json`)
byte-identically -- no new splits. For each of the 7 held-out families: 5
training families, 1 validation family, 1 held-out test family. History
`[0,10000)`, score `[10000,50000)`, exactly 40,000 scored requests.
Capacities: 32, 64, 128.

Supervision target: the same finite-horizon eviction loss `L_H` used by the
manuscript's proposed method, `H=4`, computed via
`src/lafc/evict_value_dataset_v1._simulate_lru_misses` (unmodified) -- **not
changed by this experiment** (Reviewer Comment 2's separate objective
ablation, `supervision_objective_ablation_v1`, is the place that varies the
target; this protocol never does).

Feature schema: `EVICT_VALUE_V1_FEATURE_COLUMNS` (`src/lafc/evict_value_features_v1.py`),
unmodified, identical across all conditions.

Model family / hyperparameter grid: identical to the canonical pipeline --
`{ridge, random_forest, hist_gb}`, selection by minimum validation
mean-regret-vs-oracle (tie-break MAE then RMSE), seed 0. Same grid and
budget for every condition.

## 3. Reduced condition matrix (frozen before launch)

The original task brief specifies five conditions (OFF_POLICY_LRU,
LEARNED_CONTINUATION, DAGGER_ITER1, DAGGER_ITER2, MIXED_50_50) and
explicitly authorizes reducing the matrix under a compute/engineering
budget, provided the reduction is frozen *before* launch and follows this
priority order:

    1. OFF_POLICY_LRU
    2. DAGGER_ITER1
    3. DAGGER_ITER2
    4. LEARNED_CONTINUATION
    5. MIXED_50_50

Given the realistic engineering time available in this session (this
protocol, its implementation, and its tests were built and frozen in the
same working session as two other large, already-completed pieces of work
this repository's history shows were in flight), this run implements and
launches **only the first two, highest-priority conditions**:

- **A. `OFF_POLICY_LRU`** (baseline)
- **B. `DAGGER_ITER1`** (primary on-policy test)

`DAGGER_ITER2`, `LEARNED_CONTINUATION`, and `MIXED_50_50` are deferred to a
follow-up `distribution_shift_ablation_v2` protocol, explicitly out of
scope for this run. This is a scope reduction, not a silent skip: it is
recorded here, in the config, and in the launch report, before any result
is observed.

### A. OFF_POLICY_LRU

Canonical current behavior, characterized (not reimplemented) in §1: at
every eviction decision during training-state collection, the cache
trajectory continues via the LRU victim (`candidates[0]`), independent of
any model. This exactly reproduces what every existing dataset builder in
this repository already does. One dataset build, one model per fold.

### B. DAGGER_ITER1

Iteration 0: train `OFF_POLICY_LRU` (same as condition A; the run reuses
condition A's trained models rather than duplicating them).

Iteration 1:
1. Freeze the iteration-0 model for this fold.
2. Re-walk the fold's **training and validation families only** (never the
   held-out family), this time letting the frozen iteration-0 model decide
   the actual evicted candidate at each decision (`argmin` predicted
   `L_H`, via `EvictValueV1Model.predict_loss_batch`, the same rule as
   `ScalarObjectivePolicy` in the objective-ablation worktree) --
   generating a *second* set of training states that reflect where the
   learned policy actually goes.
3. Ground-truth labels for these newly-visited states are computed
   **independently** via the same frozen `_simulate_lru_misses`-based
   `L_H` definition used everywhere else in this protocol -- never the
   model's own predicted value (anti-circularity, see §7).
3. Aggregate: `D_train = D0 ∪ D1` (iteration-0 states plus iteration-1
   on-policy states), equal source weighting (no down/up-weighting by
   iteration -- predeclared, not tuned on results).
4. Retrain on `D_train` using the identical grid/selection rule.

## 4. State-shift metrics (predeclared, computed after data exists, before conclusions)

State signature per decision-candidate: the existing feature vector from
`compute_candidate_features_v1` (already a compact numeric descriptor
including recency rank, age, predictor/LRU victim agreement, recent
request/hit rate, etc.) -- reused as-is, not reinvented.

For each `fold x capacity x condition`, compare the distribution of
TRAINING states (across all rows contributing to that fold's frozen model)
to the distribution of DEPLOYMENT states (states actually visited when
that fold's frozen model is replayed on the held-out trace):

- per-feature standardized mean difference (SMD): `(mean_train -
  mean_deploy) / pooled_std`;
- per-feature Wasserstein-1 distance (`scipy.stats.wasserstein_distance`
  if available, else a direct O(n log n) sorted-quantile implementation);
- aggregate state-shift index: **mean of per-feature Wasserstein-1
  distances, each feature min-max normalized by its train-distribution
  range** (frozen formula, computed once implemented, never adjusted after
  viewing results).

## 5. Trajectory divergence diagnostics

For each `fold x capacity x condition`, replaying the frozen model on the
held-out trace against a reference LRU replay of the same trace:

- first eviction-decision index at which the two trajectories' chosen
  victims differ;
- fraction of eviction decisions where the victim differs from LRU's;
- cache-set Jaccard similarity at each decision point, and its mean over
  the scored window;
- longest run of decisions (from t=0) where both trajectories hold
  identical cache sets;
- count of distinct cache-set snapshots visited (over the scored window).

## 6. Prediction-quality diagnostics

Per fold x capacity x condition, post-hoc only (never used to alter
eviction decisions): MAE, RMSE, and mean regret vs. oracle of the frozen
model's predictions, computed separately on (a) the model's own
training-distribution validation rows and (b) states actually visited
during held-out deployment (labels for (b) computed independently via the
same frozen `L_H`, purely for diagnostic comparison -- never fed back into
training or eviction).

## 7. Anti-circularity (Section 16 of the task brief)

State-GENERATION policy (which candidate is actually evicted while
walking the trace to produce training rows) and label-GENERATION
mechanism (the `L_H` ground truth attached to each row) are strictly
separate at every step of this protocol:

- `OFF_POLICY_LRU` and the iteration-0 half of `DAGGER_ITER1`: states
  generated by LRU trajectory, labels by `_simulate_lru_misses`.
- iteration-1 half of `DAGGER_ITER1`: states generated by the frozen
  iteration-0 model's argmin decisions, labels *still* by
  `_simulate_lru_misses` -- never by the model's own predicted score.

No row's label is ever derived from a model's prediction on that row.

## 8. Model selection

Per fold x condition: train on the 5 training families only; select
(regressor family, for the scalar `L_H` target) by minimum validation
mean-regret on the designated validation family only; held-out family
never contributes rows, normalization statistics, or selection signal.
Iteration count (`DAGGER_ITER1` vs. hypothetical further iterations),
which conditions are run, and D0/D1 weighting are experimental conditions,
frozen here -- never chosen by held-out performance.

## 9. Runtime budget

Target wall-clock budget: **9 hours**, enforced by a checkpoint-based
controller (`--max-wall-hours`) that stops launching new (fold, condition,
capacity) units once remaining budget is insufficient for one more unit's
observed average cost, and never truncates a unit already writing its
artifact. All progress is checkpointed per completed unit
(`analysis/distribution_shift_ablation_v1/campaign_state.json`) with
resume support (skips units already present and hash-valid).

## 10. Statistical plan (pre-registered, not executed until completion)

Experimental unit: `trace family x capacity` (n=21 per condition, paired
across the 2 conditions -- 21 pairs). Primary hypotheses:

- H1: `DAGGER_ITER1` reduces the state-shift index relative to
  `OFF_POLICY_LRU`.
- H2: `DAGGER_ITER1` reduces deployment-state prediction error (MAE/mean
  regret) relative to `OFF_POLICY_LRU`.
- H3: `DAGGER_ITER1` improves held-out miss ratio relative to
  `OFF_POLICY_LRU`.
- H4: across the 21 paired instances, larger state-shift index is
  associated with larger miss-ratio degradation (Spearman correlation).

Paired (Wilcoxon signed-rank or paired sign test, matching whatever the
existing fairness statistics protocol uses) comparisons only; requests are
not treated as independent replicates. Not executed until the campaign
completes and is certified -- deferred, per the task brief, to a following
task.

## 11. Manuscript safety

This protocol and its results must not be used to update manuscript
results, abstract, conclusions, or the reviewer response until a
completion audit exists and a separate task authorizes it. Language used
in any interim reporting: "tests the distribution-shift hypothesis," not
"proves distribution shift causes the performance gap."

## 12. Amendment policy

Any change to the target, features, folds, capacities, hyperparameter
grid, seeds, condition set, or aggregation weighting after viewing
held-out results requires a new protocol id (`distribution_shift_ablation_v2`)
with an explicit documented reason. This file is not edited in place after
results are known.
