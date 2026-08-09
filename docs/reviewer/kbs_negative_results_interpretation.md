# KBS negative-results interpretation

Internal scientific notebook for later manuscript and rebuttal writing.

This is not manuscript prose. It distinguishes:

- empirical evidence already present in local artifacts,
- literature-supported interpretation,
- hypotheses that remain untested.

## 9.1 Core empirical puzzle

Empirical puzzle:

- `evict_value_v1` can learn a finite-horizon counterfactual target offline,
  yet its closed-loop caching performance remains poor relative to strong
  baselines.

Useful conceptual chain:

1. offline target quality
2. candidate ranking quality
3. sequential policy trajectory
4. long-run cache misses

Interpretation:

- success at an earlier stage of that chain does not guarantee success at a
  later stage,
- a target can be meaningful and even predictive without being sufficient for
  robust long-run closed-loop improvement.

Not established:

- that there is a single dominant failure mode,
- that any one mechanism alone explains the miss gap.

## 9.2 Surrogate / objective mismatch

Reverified from local artifact
`analysis/supervision_objective_ablation_v1/policy_comparison.csv`:

- `objective_pairwise`: `565127` aggregate misses
- `objective_reuse_distance`: `571456`
- `objective_next_arrival`: `573059`
- `objective_eviction_loss`: `601569`

Empirical evidence:

- under the frozen cross-family objective-ablation protocol, pairwise
  supervision outperformed scalar eviction-loss regression on aggregate online
  misses.

Literature-supported interpretation:

- the final cache action is a relative ranking decision over candidates,
  whereas scalar eviction-loss regression estimates an absolute continuous
  counterfactual quantity.
- decision-focused and learning-to-rank work supports the general idea that
  decision quality can depend more on correct ordering than on accurate absolute
  magnitudes.

Verified references:

- Jayanta Mandi, Victor Bucarey, Maxime Mulamba Ke Tchomba, and Tias Guns,
  *Decision-Focused Learning: Through the Lens of Learning to Rank*, ICML 2022,
  https://proceedings.mlr.press/v162/mandi22a.html
- Maxime Mulamba, Jayanta Mandi, Michelangelo Diligenti, Michele Lombardi,
  Victor Bucarey, and Tias Guns, *Contrastive Losses and Solution Caching for
  Predict-and-Optimize*, IJCAI 2021, DOI
  `10.24963/ijcai.2021/390`, https://www.ijcai.org/proceedings/2021/390

Not established:

- that surrogate or objective mismatch is the sole cause of the gap,
- that pairwise supervision wins only because it is pairwise rather than due to
  some other implementation or statistical effect.

## 9.3 Scalar consistency advantage

Important structural observation to preserve:

- scalar scores automatically induce a transitive ordering.

Example:

- if `L(A) < L(B) < L(C)`, then the scalar ordering cannot simultaneously imply
  `C < A`.

By contrast, independently predicted pairwise preferences can theoretically form
cycles such as:

- `A ≺ B`
- `B ≺ C`
- `C ≺ A`

Interpretation:

- scalar scoring has a global-consistency advantage,
- pairwise methods require aggregation or cycle resolution if their local
  preferences are inconsistent.
- the current frozen reviewer-science `objective_pairwise` model does **not**
  emit free-standing pairwise comparisons at deployment time; it is the
  shared-reward `HALPModel` in `src/lafc/halp_model.py`, so the deployed policy
  ranks candidates by a single scalar reward `R(x)` and therefore induces an
  acyclic total preorder up to ties.

Important scope clarification:

- a true cycle diagnostic is therefore structurally null for the current frozen
  objective-ablation deployment model,
- cycle frequency becomes a meaningful empirical question only for a future
  explicit pairwise comparator or context-dependent pair scorer that does not
  collapse candidates to one scalar reward.

Not yet measured:

- the fraction of actual candidate sets containing any cycle,
- the frequency of 3-cycles,
- whether cycle-containing decisions are disproportionately harmful.

Possible future diagnostic:

- measure cycle frequency and relate it to downstream misses.

Status:

- hypothesis only, not a current result.

## 9.4 Sample-efficiency hypothesis

Hypothesis:

- pairwise supervision may be more sample-efficient because it asks the learner
  to recover only the relative information needed by the downstream decision,
  while scalar regression also estimates magnitude information that may be
  unnecessary for the action itself.

Careful wording:

- this is not a claim that binary classification is always easier,
- it is a claim about the possibility that the pairwise target is closer to the
  decision boundary actually needed at deployment.

Planned learning-curve experiment:

- training fractions: `1%, 2%, 5%, 10%, 25%, 50%, 100%`
- compare scalar eviction-loss vs pairwise
- keep validation and test fixed
- measure downstream misses, ranking accuracy or regret, scalar MAE or RMSE,
  and pairwise accuracy or margin behavior

Prediction supporting the hypothesis:

- pairwise advantage is largest at low sample sizes and shrinks as training
  data grows.

Prediction refuting it:

- the gap persists or saturates even with the full data.

Cleaner same-target comparison to preserve:

- derive pairwise labels directly from the same scalar eviction-loss labels,
  then compare absolute regression on `L(q)` vs pairwise classification on
  `sign(L(A)-L(B))`.

### Same-target scalar-vs-pairwise diagnostic

Important distinction:

- the earlier objective ablation compared different supervision objectives,
  including `objective_pairwise` versus `objective_eviction_loss`;
- the newer learning-curve diagnostic instead fixes the underlying
  eviction-loss target and only changes the representation:
  - scalar regression on `L(q)`,
  - same-target pairwise classification on `sign(L(A) - L(B))`.

Therefore:

- `objective_pairwise` and `eviction_loss_pairwise` are scientifically
  different conditions,
- the former changes the target construction itself,
- the latter keeps the eviction-loss notion fixed and only converts the same
  underlying labels into pairwise comparisons.

Current local diagnostic:

- runner:
  `scripts/experiments/run_supervision_objective_learning_curve.py`
- config:
  `configs/supervision_objective_learning_curve_v1.json`
- output:
  `analysis/supervision_objective_learning_curve_v1/`
- model directory:
  `models/supervision_objective_learning_curve_v1/`
- same-example guarantee:
  scalar candidate rows and regret-derived pairwise rows are built from the
  exact same filtered decision ids at each fraction.

LOW-FRACTION OBSERVATION — validated local checkpoint only.

At the currently audited low-fraction checkpoint:

- completed fractions:
  `1%, 2%, 5%, 10%`
- completed families:
  `brightkite, citibike, cloudphysics, metacdn`
- validated units:
  `16`
- validated rows:
  `96`

Across those completed low-fraction cells, scalar regression had substantially
fewer downstream misses than the same-target `eviction_loss_pairwise`
condition.

This is not yet a final result because:

- not all folds were complete at that audited checkpoint,
- the low-fraction checkpoint covers only `1%, 2%, 5%, 10%`,
- the partial rows must not be treated as final manuscript evidence.

HIGH-FRACTION QUESTION — currently running locally.

The next active local phase is the `25%` same-target extension, launched in
tmux session `kbs_learning_curve_highfrac_20260809` with a clean `10`-hour
wall-time budget and all seven held-out folds targeted. Later `50%` and `100%`
phases remain TODO.

The open scientific question is not whether a partial `25%` row looks good or
bad in isolation. It is whether, as fraction increases through `25%`, `50%`,
and `100%`:

- downstream misses improve materially,
- scalar `MAE/RMSE` improves materially,
- ranking and decision-quality metrics improve materially,
- and, crucially, whether offline prediction improvement continues to translate
  into online miss improvement.

Implication if the pattern persists after clean stop and final partial-campaign
audit:

- it would contradict the simple representation-only hypothesis that pairwise
  wins mainly because binary or relative supervision is inherently easier or
  more sample-efficient than scalar regression,
- it would instead suggest that the earlier `objective_pairwise` advantage is
  tied to its underlying supervision objective or target construction, not
  merely to converting eviction-loss labels from scalar form into pairwise
  form.

Current hypothesis status:

- HYPOTHESIS:
  pairwise representation may be more sample-efficient.
- CURRENT LOW-FRACTION EVIDENCE:
  the audited `1%` to `10%` checkpoint contradicts that hypothesis over the
  completed low-fraction cells.
- HIGH-FRACTION QUESTION:
  the active `25%` phase and later `50%` / `100%` phases are needed to test
  whether scalar convergence with more data changes the downstream picture.
- NOT YET ESTABLISHED:
  sample insufficiency is not proven,
  objective mismatch is not proven,
  convergence is not proven,
  and no `25%` conclusion should be recorded before completed units are
  audited.

Possible interpretations if scalar continues to win:

1. pairwise transformation of eviction-loss labels discards useful magnitude
   information;
2. noisy near-tie pairs make binary labels unstable;
3. pairwise row construction creates many correlated training pairs;
4. the earlier `objective_pairwise` condition encodes a fundamentally better
   learning target rather than merely a better representation.

These are hypotheses only, not established findings.

### Pairwise label-noise / margin diagnostic

Not yet run.

If the same-target pairwise condition remains weak, the next lightweight
diagnostic should stratify pairwise examples by `|L(A) - L(B)|`.

Suggested bins:

- near ties,
- small margin,
- medium margin,
- large margin.

Suggested measurements:

- pairwise accuracy,
- downstream decision effect,
- fraction of training pairs in each margin bin.

Motivation:

- if many pairwise labels come from nearly equal scalar losses, small
  counterfactual-label noise can flip the binary label while scalar regression
  still retains useful magnitude information.

## Exact target oracle vs learned online policy

Status:
`DESIGNED / LOCAL_FOUNDATION_ONLY / NOT YET FULLY RUN`

This diagnostic is designed to separate three different questions that should
not be conflated:

1. quality of the supervision target itself;
2. ability of ML to approximate that target from online features;
3. remaining gap to a global offline oracle.

Clean decomposition:

- EXACT TARGET ORACLE:
  at each eviction decision, compute the true target value for every valid
  in-cache candidate using the actual future suffix and choose the target-best
  candidate with no ML prediction.
- LEARNED ONLINE POLICY:
  use a trained model to predict that same target from online-available
  features and choose accordingly.
- GLOBAL OFFLINE ORACLE:
  where meaningful, compare both of the above against `offline_belady`.

Important naming guardrail:

- the exact target oracle is **not** automatically Belady;
- for the current `eviction_loss` target, it is a local greedy policy that
  recomputes the exact finite-horizon target at each decision using the
  policy's actual current cache state,
- but the target value itself still uses the same frozen label semantics as
  training: horizon `H`, admit the incoming page, then replay the next `H`
  requests under `LRU` continuation.

Therefore the first oracle diagnostic must match the existing label definition
exactly rather than introducing a different continuation policy.

Key gaps to preserve:

- TARGET-QUALITY GAP:
  exact target oracle vs `offline_belady`
- LEARNING GAP:
  learned policy vs exact target oracle
- TOTAL ONLINE GAP:
  learned policy vs `offline_belady`

Central interpretation for `eviction_loss`:

- if exact `eviction_loss` oracle is close to `offline_belady` but the learned
  policy is poor, then learnability or generalization is the main bottleneck;
- if exact `eviction_loss` oracle is itself poor, then the target or objective
  is the main bottleneck;
- if exact-oracle quality improves strongly as `H` grows, then horizon
  truncation is a major contributor.

Objective-by-objective classification:

- `eviction_loss`:
  `EXACT_ORACLE_WELL_DEFINED`
- `next_arrival`:
  `EXACT_ORACLE_WELL_DEFINED`
- `reuse_distance`:
  `EXACT_ORACLE_WELL_DEFINED`
- `objective_pairwise`:
  `EXACT_ORACLE_REQUIRES_CLARIFICATION`
  because the frozen pairwise objective is defined through next-arrival
  ordering, so any multi-candidate exact oracle must be routed through that
  underlying source label rather than treated as an independent global oracle.

Current local foundation only:

- exact per-decision oracle helpers live in `src/lafc/oracle_diagnostics.py`,
- they reuse the shared target-construction kernel in
  `src/lafc/supervision_objective_ablation.py`,
- synthetic tests live in `tests/test_oracle_diagnostics.py`,
- no full seven-family replay has been run yet from this branch.

Predeclared later metrics:

- exact-oracle misses
- learned-policy misses
- Belady misses
- learned vs exact decision agreement
- exact vs Belady decision agreement
- top-1 learned target accuracy
- target regret of the learned decision
- number and fraction of decisions where learned chooses a non-optimal exact
  candidate
- downstream miss gap

Planned evaluation axes:

- family
- capacity
- horizon

Planned horizon grid for this diagnostic:

- `H in {1, 2, 4, 8, 16}`

Link to the running learning-convergence study:

- later combine training fraction with prediction `MAE/RMSE`,
  exact-target decision agreement, exact-target regret, and downstream misses;
- if `MAE/RMSE` improves while exact-target agreement and misses plateau,
  surrogate or ranking issues remain plausible;
- if exact-target agreement improves while misses remain poor, target quality
  remains suspect;
- if both improve, sample complexity likely matters.

Important distinction from minimum-counterfactual suffix attribution:

- exact-target oracle diagnostic asks:
  was the learned eviction decision consistent with the exact target?
- minimum-counterfactual attribution asks:
  which earlier changed decisions are minimally sufficient to remove a later
  excess miss?

Both remain useful and should not be collapsed into the same diagnostic.

## 9.5 Horizon truncation / temporal credit assignment

Empirical setup:

- the current target uses a finite horizon,
- the present second-revision line selected `H=4`.

Plausible interpretation:

- a candidate may look harmless over four requests yet still have important
  longer-term reuse consequences.

Literature context:

- learned cache systems such as LRB, HALP, GL-Cache, CACHEUS, and 3L-Cache all
  foreground future reuse, future impact, or adaptive long-run workload
  behavior rather than only immediate local consequences.

Verified cache references:

- Zhenyu Song, Daniel S. Berger, Kai Li, and Wyatt Lloyd,
  *Learning Relaxed Belady for Content Distribution Network Caching*,
  NSDI 2020, https://www.usenix.org/conference/nsdi20/presentation/song
- Liana V. Rodriguez et al.,
  *Learning Cache Replacement with CACHEUS*,
  FAST 2021, https://www.usenix.org/conference/fast21/presentation/rodriguez
- Juncheng Yang, Ziming Mao, Yao Yue, and K. V. Rashmi,
  *GL-Cache: Group-level learning for efficient and high-performance caching*,
  FAST 2023, https://www.usenix.org/conference/fast23/presentation/yang-juncheng
- Zhenyu Song et al.,
  *HALP: Heuristic Aided Learned Preference Eviction Policy for YouTube Content
  Delivery Network*, NSDI 2023,
  https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu
- Wenbin Zhou, Zhixiong Niu, Yongqiang Xiong, Juan Fang, and Qian Wang,
  *3L-Cache: Low Overhead and Precise Learning-based Eviction Policy for
  Caches*, FAST 2025,
  https://www.usenix.org/conference/fast25/presentation/zhou-wenbin

Possible sensitivity study:

- pre-specify a sweep such as `H in {1, 4, 8, 16}`
- report the full sweep rather than searching post hoc for a winning horizon

Status:

- plausible but not established.

## 9.6 Sequential distribution shift

Labeling and deployment mismatch:

- labels: one candidate eviction, then continuation under LRU
- deployment: learned eviction, then another learned eviction, then another
  learned eviction

Literature support:

- this induced-distribution issue is central in imitation learning and DAgger.

Verified reference:

- Stephane Ross, Geoffrey J. Gordon, and Drew Bagnell,
  *A Reduction of Imitation Learning and Structured Prediction to No-Regret
  Online Learning*, AISTATS 2011,
  https://proceedings.mlr.press/v15/ross11a.html

Current local `metacdn` evidence from
`analysis/distribution_shift_ablation_v1/`:

- capacity `32`: `OFF_POLICY_LRU` misses `29219`, `DAGGER_ITER1` misses
  `29766`, delta `+547`
- capacity `64`: `23756` vs `25596`, delta `+1840`
- capacity `128`: `23213` vs `24547`, delta `+1334`
- trajectory divergence:
  `97.1517%`, `99.5171%`, `99.8397%`
- first divergence indices:
  `1`, `2`, `1`
- state-shift index:
  `0.000664 -> 0.000462`,
  `0.000553 -> 0.000483`,
  `0.000425 -> 0.000417`

Interpretation:

- substantial trajectory divergence clearly exists,
- measured state shift was slightly reduced under `DAGGER_ITER1`,
- misses still worsened at all three `metacdn` capacities.

Therefore:

- continuation mismatch is a credible contributor,
- current local evidence does not justify claiming it is the sole causal source
  of the miss gap,
- current local evidence also does not justify claiming that DAgger fixes the
  performance problem.
- trajectory divergence is not the same thing as harmful divergence: two cache
  trajectories may differ almost everywhere yet still have similar miss counts,
  while a small number of early high-impact evictions can dominate the miss gap.

Seven-family Wulver continuation:

- still pending sync-back of the Wulver-only runner and Slurm files,
- therefore not yet available as local source-backed final evidence in this
  branch.

## 9.6.1 Minimum counterfactual miss attribution

Proposed diagnostic focus:

- analyze **excess misses** first: requests where the learned policy misses but
  the reference policy hits,
- treat all-miss analysis as secondary.

Proposed formulation:

- for a target excess miss at request index `t`, search over valid
  counterfactual eviction-action trajectories that begin from the same
  pre-history and minimize Hamming distance from the actual learned eviction
  trajectory, subject to:
  - deterministic cache-state transitions,
  - every edited eviction action being feasible in the counterfactual cache
    state where it is applied,
  - the target request at `t` becoming a hit.

Useful outputs:

- minimum repair distance `d*(t)`,
- intervention positions,
- number of equally minimal repairs when tractable,
- responsibility or blame scores for decisions that repeatedly appear in
  minimal repairs,
- temporal gap between a responsible intervention and the resulting excess miss.

Algorithmic interpretation:

- exact dynamic programming or shortest-path search on a time-expanded reachable
  state graph is the right starting point,
- budgeted feasibility questions of the form "can this miss be repaired with at
  most `k` interventions?" are a natural first relaxation,
- `A*` or branch-and-bound may be useful if the reachable-state graph becomes
  too large.

Important cautions:

- the learned policy state is generally larger than cache contents alone
  (`src/lafc/policies/supervision_objective_ablation_policy.py` tracks cache
  order, recent request history, recent hit history, bucket metadata, and
  confidence metadata),
- therefore a correct replay or DP for the learned reviewer policy cannot
  assume that resident objects alone define the full state,
- multiple equally minimal repairs should be treated as a feature of the
  diagnosis, not collapsed into a single allegedly responsible decision,
- responsibility attribution is only a diagnostic lens; it does not by itself
  establish causal uniqueness.

Reference status:

- TODO: verify sequential counterfactual-explanation references before adding
  them here. Do not cite from memory.

## 9.7 Fine-grained learned-cache complexity and overhead

Tension to preserve:

- candidate-level learned eviction offers fine-grained decision specificity,
  but it also requires repeated candidate scoring and therefore may carry high
  computational or statistical cost.

Current practical-significance smoke evidence from
`analysis/practical_significance_ablation_v1/`:

- exact-decision-preserving optimization check:
  `all_variants_exact_across_all_trace_capacity_pairs = true`
- `vectorized_exact` smoke speedups roughly `14.29x` to `26.88x`
- `vectorized_cached_exact` smoke speedups roughly `28.70x` to `99.99x`
- break-even estimates roughly `0.495089` to `2.618223` ms per miss across the
  recorded baseline comparisons

Important caveat from the artifact itself:

- `speedup_numbers_are_final_reviewer_evidence = false`

Interpretation:

- the implementation can be made much faster without changing decisions in the
  checked smoke cells,
- that does not yet establish final controlled deployment practicality.

Useful literature context:

- LRB, HALP, GL-Cache, CACHEUS, and 3L-Cache all explicitly discuss the
  system-level tension between learned eviction quality and deployable overhead.

Status:

- smoke-only, not a controlled final timing result.

## 9.8 Working multi-factor explanation

Current working interpretation:

1. surrogate or objective mismatch,
2. unnecessary absolute-value estimation relative to ranking,
3. finite-horizon credit truncation,
4. sequential trajectory shift,
5. sample-efficiency requirements,
6. fine-grained computational burden.

This should remain the default internal explanation unless stronger evidence
isolates one factor more cleanly.

## 9.9 Potential manuscript claims

Draft or internal only:

- decision-aligned semantic interpretation of a target does not guarantee
  closed-loop decision quality.
  Evidence:
  `analysis/supervision_objective_ablation_v1/policy_comparison.csv`,
  `analysis/reviewer_fairness/*.csv`
- pairwise supervision outperforms scalar eviction-loss regression in the
  current controlled objective ablation.
  Evidence:
  `analysis/supervision_objective_ablation_v1/policy_comparison.csv`
- the current same-target scalar-vs-pairwise diagnostic is preliminary and
  should be used only to test whether representation alone explains the
  pairwise advantage.
  Evidence:
  `analysis/supervision_objective_learning_curve_v1/`
- learned deployment induces substantial trajectory divergence from the
  LRU-continuation labeling process.
  Evidence:
  `analysis/distribution_shift_ablation_v1/trajectory_divergence.csv`
- current evidence does not establish continuation mismatch as the sole causal
  source of miss degradation.
  Evidence:
  local `metacdn` rows in the distribution-shift artifacts
- fine-grained learned eviction incurs substantial computational overhead, but
  exact implementation optimizations can reduce it sharply.
  Evidence:
  `analysis/practical_significance_ablation_v1/exact_optimization_equivalence.json`

## 9.10 Claims we must not make

Do not claim:

- eviction-loss supervision is empirically superior to alternative objectives
- distribution shift has been proven to cause the miss gap
- DAgger fixes the performance problem
- smoke timing is a final controlled runtime result
- the current method is practically deployment-superior
- pairwise cyclicity is a practical problem before measuring it
- insufficient training data explains the gap before the same-target
  learning-curve campaign reaches a clean audited stopping point
- `H=4` is the cause before a horizon sensitivity study is run
- `objective_pairwise` and `eviction_loss_pairwise` are interchangeable

## 9.11 Reference table

| Reference | Venue/year | DOI or official URL | Relevant concept | How it relates to our evidence | Verified |
|---|---|---|---|---|---|
| Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* | AISTATS 2011 | https://proceedings.mlr.press/v15/ross11a.html | DAgger and induced-distribution mismatch | Supports the sequential-shift framing; does not prove it explains our miss gap | yes |
| Mandi et al., *Decision-Focused Learning: Through the Lens of Learning to Rank* | ICML 2022 | https://proceedings.mlr.press/v162/mandi22a.html | decision-focused learning and ranking view | Supports the idea that ordering can matter more than absolute target magnitude | yes |
| Mulamba et al., *Contrastive Losses and Solution Caching for Predict-and-Optimize* | IJCAI 2021 | DOI `10.24963/ijcai.2021/390`, https://www.ijcai.org/proceedings/2021/390 | pairwise or contrastive predict-and-optimize losses | Supports ranking-style supervision as a principled decision-focused alternative | yes |
| Song et al., *Learning Relaxed Belady for Content Distribution Network Caching* | NSDI 2020 | https://www.usenix.org/conference/nsdi20/presentation/song | learned caching aimed at future reuse under deployability constraints | Provides learned-cache context for future-impact reasoning and overhead tradeoffs | yes |
| Rodriguez et al., *Learning Cache Replacement with CACHEUS* | FAST 2021 | https://www.usenix.org/conference/fast21/presentation/rodriguez | adaptive learned cache replacement | Provides baseline context and deployability comparison | yes |
| Yang et al., *GL-Cache: Group-level learning for efficient and high-performance caching* | FAST 2023 | https://www.usenix.org/conference/fast23/presentation/yang-juncheng | efficient learned caching | Relevant to the overhead-vs-quality discussion | yes |
| Song et al., *HALP: Heuristic Aided Learned Preference Eviction Policy for YouTube Content Delivery Network* | NSDI 2023 | https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu | heuristic-augmented learned preference eviction | Relevant to the objective and overhead discussion | yes |
| Zhou et al., *3L-Cache: Low Overhead and Precise Learning-based Eviction Policy for Caches* | FAST 2025 | https://www.usenix.org/conference/fast25/presentation/zhou-wenbin | low-overhead learned eviction | Relevant to the practical-significance and baseline-comparison discussion | yes |
