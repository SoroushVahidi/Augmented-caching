# Supervision-Objective Ablation Protocol (`supervision_objective_ablation_v1`)

Frozen before any objective's model is trained or evaluated. See
`configs/supervision_objective_ablation_v1.json` for the machine-readable
version.

## 0. Prior-art audit (Section 1 of the task)

Searched source, scripts, configs, dataset builders, tests, docs, and git
history before implementing anything:

| Objective | Status found |
|---|---|
| Finite-horizon eviction loss ($L_H$) | **Canonical, implemented.** `src/lafc/evict_value_wulver_v1.py::iter_candidate_rows` (production dataset builder) and `src/lafc/evict_value_dataset_v1.py::_simulate_lru_misses` (the exact miss-counting kernel). This is the reference architecture for this ablation. |
| Next-arrival prediction | **Not implemented as a training objective**, but a directly reusable primitive exists: `src/lafc/evict_value_v2_rollout.py::_next_use_distance(page, future_reqs, at_idx)` returns the index of a page's next occurrence in a future window (or `math.inf`). No censoring, no feature/model pipeline built around it previously. |
| Reuse-distance prediction | **Not implemented anywhere** (no code, no prior experiment). Must be defined and built from scratch. |
| Pairwise preference | **Partially implemented, with a documented circularity concern.** `src/lafc/evict_value_v2_rollout.py::build_pairwise_rows_from_candidate_rows` converts rollout-regret-labeled candidate rows into pairwise comparisons (`label_i_better = 1 if regret_i < regret_j`). **This derives pairwise labels directly from the same rollout/eviction-loss quantity being ablated against** — using it unmodified as "the" independent pairwise objective would violate the task's explicit warning (Section 4.D) not to present an eviction-loss-derived signal as an independent alternative without prominent disclosure. See Section 4 below for the resolution. Prior exploratory scripts (`scripts/run_pairwise_vs_pointwise_experiment.py`, `scripts/build_evict_value_v2_pairwise_dataset.py`, `analysis/evict_value_pairwise_*`) used exactly this regret-derived construction; they are prior exploratory work, not a validated independent-objective ablation, and are not reused as-is for the primary comparison (see Section 4.D below).

Feature computation (`src/lafc/evict_value_features_v1.py::compute_candidate_features_v1`,
`EVICT_VALUE_V1_FEATURE_COLUMNS`) is shared by both the canonical
eviction-loss pipeline and the existing rollout/pairwise exploration —
reused unmodified for **all four** objectives in this ablation, which is
what makes "same feature vector across objectives" achievable without new
feature-engineering work.

## 1. Canonical eviction-loss definition (Section 2 of the task)

From `src/lafc/evict_value_wulver_v1.py::iter_candidate_rows` and
`src/lafc/evict_value_dataset_v1.py::_simulate_lru_misses`, and matching
`manuscript_source/main.tex` Eq. `eviction_loss`:

- **Candidate set** $\mathcal{V}_t$: the $k$ (=capacity) pages resident in
  cache at a full-cache miss, in LRU order.
- **Label construction**: for each candidate $q$, `after = [p for p in
  candidates if p != q] + [pid]` (evict $q$, admit the newly-requested
  page), then `fut_h = future[:h]` where `future = requests[t+1:]` (the
  **same** future window for every candidate at this decision — only the
  starting cache state differs). $L_H(q,t)$ = number of misses when
  replaying `fut_h` under **plain LRU** starting from `after`.
- **Horizon**: multiple horizons built per decision (canonical:
  `{4, 8, 16}`, scoped to `{4}` for the fairness-protocol corpora — see
  `docs/evict_value_v1_fair_training_protocol.md`).
- **Censoring**: implicit — the sum only ever runs over the $H$ available
  future steps; if fewer than $H$ future requests exist (end of trace),
  the label is computed over however many remain (no explicit `inf`/censor
  sentinel is used, unlike next-arrival).
- **No transform**: raw integer miss count, no `log1p` or normalization.
- **Model**: `ridge` / `random_forest` / `hist_gb` (scikit-learn),
  selected by minimum validation `mean_regret` (regret vs. the best
  candidate's true loss at that decision), tie-break MAE then RMSE.
- **Inference**: `argmin` over predicted loss among current candidates
  (Eq. `eviction_rule`).
- **Training corpus** (post train/test-overlap fix): leave-one-family-out
  cross-family, `configs/reviewer_fairness_cross_family_v1.json`.

## 2. Central controlled-variable principle (Section 3)

**Everything possible is held constant except the supervision objective.**
Held identical across all four objectives:

| Variable | How it is held constant |
|---|---|
| Candidate generation | Same per-decision candidate enumeration loop (shared helper, Section 5 below) |
| Cache simulator | Same LRU continuation kernel (`_simulate_lru_misses`-equivalent) used for every scalar label and for online evaluation |
| Trace preprocessing | Same canonical/fold traces, same hashes, same fold definitions (`configs/fair_cross_family_v1/folds/`) |
| Cache capacities | 32, 64, 128 |
| Request positions | Same 7 canonical traces, same `[0,50000)` request range |
| History/scoring window | `[0,10000)` history, `[10000,50000)` scored, per `reviewer_fair_cross_family_v1` |
| Feature vector | `compute_candidate_features_v1` / `EVICT_VALUE_V1_FEATURE_COLUMNS`, unmodified, identical for all four |
| Model family | Same 3-family search (`ridge`, `random_forest`, `hist_gb`) for the three scalar objectives (A, B, C); pairwise (D) uses a capacity-matched shared scorer (Section 4.D) |
| Training-resource budget | Same leave-one-family-out folds (5 train / 1 val / 1 test), same 21 decision-example pool per fold, drawn from the **same underlying candidate-decision examples** (Section 11: one dataset pass emits all label views per row) |
| Validation-resource budget | Same designated validation family per fold |
| Random seeds | Seed 0 throughout |
| Hyperparameter search budget | Same predeclared grid, same selection rule (min validation metric, documented tie-break), applied identically per objective |
| Candidate scoring interface | `argmin`/`argmax` over per-candidate predicted score, same tie-break (lexicographically largest `page_id`, matching this repository's existing convention, e.g. HALP) |
| Evaluation metric | Request misses / miss ratio over the identical held-out window |

## 3. Four objectives, precisely defined (Section 4)

### A. Finite-horizon eviction loss (`objective_eviction_loss`)

The canonical, unmodified target (Section 1 above). Reference architecture;
not altered for this ablation.

### B. Next-arrival prediction (`objective_next_arrival`)

For candidate $q$ at decision $(q,t)$: $y_{\text{next}}(q,t) = d$ where $d$
is the number of steps from $t{+}1$ to $q$'s next occurrence in `future =
requests[t+1:]` (i.e. `_next_use_distance(q, future, 0) + 1`, reusing the
existing primitive). **Censoring rule (predeclared)**: if $q$ does not
reoccur within the same horizon $H$ used for the eviction-loss target,
$y_{\text{next}}(q,t) = H$ (censored at the horizon boundary) rather than
$\infty$ — this is the **primary, horizon-controlled** variant (Section 8:
gives next-arrival exactly the same $H$-step look-ahead budget as the
proposed objective, so neither target sees strictly more future
information than the other). A **secondary, natural (uncensored)**
variant using the raw distance (or a large sentinel, e.g. trace length, if
never reused again in the trace) is also computed and reported separately,
never substituted for the primary. **Inference-time eviction rule**: evict
the candidate with the **largest** predicted next-arrival distance
(furthest-in-the-future-first, the standard "evict what returns latest"
rule, structurally the offline-Belady direction) — deterministic tie-break
by largest `page_id`. **No future information at inference**: the model
consumes only `x(q,t)` (the same feature vector as objective A), never the
label itself.

### C. Reuse-distance prediction (`objective_reuse_distance`)

Standard **forward reuse distance**: the number of **distinct** other
objects requested between $t{+}1$ and $q$'s next reoccurrence (not the
raw request-count distance used by objective B — this is the
literature-standard "stack distance to next reuse" notion, deliberately
kept distinct from B rather than casually treated as a synonym, per the
task's explicit instruction). Same censoring convention as B (censored at
$H$ distinct objects for the primary variant; uncensored secondary
variant recorded separately). **Inference-time eviction rule**: evict the
candidate with the **largest** predicted forward reuse distance (same
direction as B, same rationale). Same features, same model family, same
tie-break as B.

### D. Pairwise preference (`objective_pairwise`)

**Two candidate constructions were investigated, per the task's explicit
instruction to check at least two and document the choice:**

1. *Regret-derived pairwise* (already existed in
   `build_pairwise_rows_from_candidate_rows`): label $i \succ j$ iff
   $L_H(q_i,t) < L_H(q_j,t)$. **Rejected as the primary pairwise
   objective** for this ablation: it is directly derived from the same
   scalar target being ablated against, so a result favoring it would not
   isolate "pairwise learning" from "the eviction-loss target itself" —
   exactly the circularity the task warned against. Retained only as a
   labeled, secondary diagnostic (`objective_pairwise_regret_derived`),
   never presented as an independent alternative.
2. *Next-arrival-ordering pairwise* (HALP-style, chosen as primary
   `objective_pairwise`): label $i \succ j$ iff $q_i$'s next re-access
   occurs strictly before $q_j$'s (using the same censored next-arrival
   distance as objective B) — directly analogous to HALP's published
   preference semantics ("$A \succ B$ if $A$ is re-accessed before $B$"),
   independent of the rollout/eviction-loss computation entirely. This is
   the primary pairwise objective, because it tests a genuinely different
   supervision paradigm (preference over raw future-use order) rather
   than a repackaging of objective A.

Model: a shared-weight pairwise scorer trained via a Bradley-Terry /
RankNet-style loss (matching this repository's existing HALP
implementation's architecture family, `src/lafc/halp_model.py`, reused
for capacity-matched consistency rather than inventing a new pairwise
model class) on feature **differences** derived from the same
`compute_candidate_features_v1` vectors as objectives A-C. **Inference**:
score every current candidate with the shared scorer, evict the
lowest-scored (least-preferred-to-keep) candidate, deterministic tie-break
by largest `page_id` (matching HALP's own convention exactly).

**Unavoidable difference (Section 4/E, disclosed)**: unlike A-C, D's
model output structure is a pairwise/ranking scorer, not a pointwise
regressor — this is a structural requirement of preference learning, not
a controlled-variable violation, and is documented here rather than
silently forced into pointwise form.

## 4. Horizon/censoring protocol (Section 8)

**Primary analysis**: horizon-controlled. Objectives B, C, D all censor
their future-information usage at the same $H$ used by objective A ($H=4$,
matching the fairness-protocol's scoped horizon — see
`docs/evict_value_v1_fair_training_protocol.md`). This is the headline
comparison, because it is the only version where no objective receives
strictly more future information than another.

**Secondary analysis**: natural/uncensored labels for B and C (their
conventional literature formulation, unbounded look-ahead). Computed and
reported separately, explicitly labeled, never substituted for the primary
horizon-controlled comparison, and never used to select which version
"wins" post hoc.

## 5. Implementation architecture (Section 14)

`src/lafc/supervision_objective_ablation.py` (new module):

- `build_multi_label_candidate_rows(requests, capacity, ..., cfg)`: **one
  shared per-decision loop** (structurally identical to
  `iter_candidate_rows`/`build_rollout_candidate_rows_v2`) that, for every
  candidate at every full-cache-miss decision, computes the feature vector
  once and attaches **all label views** (`eviction_loss_label`,
  `next_arrival_label_censored`, `next_arrival_label_raw`,
  `reuse_distance_label_censored`, `reuse_distance_label_raw`) to the
  **same row** — guaranteeing objectives A/B/C are trained on literally
  the same candidate-decision examples (Section 11), not independently
  resampled ones.
- `build_pairwise_rows(candidate_rows, source)`: derives pairwise rows
  from the same candidate rows, `source="next_arrival"` (primary) or
  `source="regret"` (secondary diagnostic, explicitly labeled).
- `ScalarSupervisionModel`: thin wrapper around the existing
  `ridge`/`random_forest`/`hist_gb` selection logic (reused from
  `scripts/train_evict_value_wulver_v1.py`'s pattern), parameterized by
  which label column to fit.
- `PairwisePreferenceModel`: reuses `src/lafc/halp_model.py`'s
  shared-weight two-layer MLP / Bradley-Terry training code directly
  (not reimplemented) for the pairwise objective.

No unrelated production code was rewritten; `iter_candidate_rows` and
`build_rollout_candidate_rows_v2` remain untouched and are still used by
their original callers.
