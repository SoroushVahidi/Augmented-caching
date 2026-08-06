# HALP — method specification

## Sources

1. **Paper**: Zhenyu Song (Princeton University), Kevin Chen, Nikhil Sarda,
   Deniz Altınbüken, Eugene Brevdo, Jimmy Coleman, Xiao Ju, Pawel Jurczyk,
   Richard Schooler, Ramki Gummadi (Google). **"HALP: Heuristic Aided
   Learned Preference Eviction Policy for YouTube Content Delivery
   Network."** 20th USENIX Symposium on Networked Systems Design and
   Implementation (NSDI '23), April 17–19, 2023, Boston, MA.
   Official paper page: https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu
   Google Research pub page: https://research.google/pubs/halp-heuristic-aided-learned-preference-eviction-policy-for-youtube-content-delivery-network/
2. **Official technical blog** (author-affiliated, Google Research):
   "Preference learning with automated feedback for cache eviction,"
   https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/
   — used to disambiguate reward-model architecture and training-schedule
   details not fully resolvable from the paper abstract/search summary alone.
3. **Official code / artifact**: **none found.** HALP runs inside YouTube's
   production CDN DRAM cache serving stack; no author-released simulator,
   reference implementation, or artifact-evaluation package was located as
   of 2026-08-06 (checked: USENIX conference page, Google Research pub
   page, author (Zhenyu Song, Eugene Brevdo) personal/academic pages, and a
   general code-hosting search). This is consistent with the paper's own
   framing as a production deployment report, not a released research
   artifact.

Authority levels: (1) is **official paper only** (primary, authoritative
for algorithmic claims); (2) is **author-affiliated official artifact**
(Google Research's own publication blog, not the paper itself, but written
by/for the same research organization — treated as a corroborating primary
source, not a secondary summary); no source above the level of (1)/(2) was
available for implementation-level detail, because there is no public code.

## Implementation status

**Strategy B — unit-size adaptation with disclosed architecture and
protocol simplifications**, not a native faithful reproduction. HALP was
designed for and deployed on a byte-capacity, variable-object-size,
continuously-online production CDN cache. No official code exists to adapt
or port. This repository's `halp` policy is therefore an **independent
reimplementation grounded in the paper and official blog's algorithmic
description**, specialized to this repository's unit-size, 50,000-request,
offline-replay paging setting, with the specific deviations from the
production system enumerated and classified below. See `docs/halp_provenance.md`
for the licensing/provenance statement.

## Method specification, classified

Every row states the design decision, its concrete form in this repository,
and its fidelity classification: **(1) exact from paper**, **(2) exact from
official source**, **(3) inferred from official behavior**,
**(4) repository-required adaptation**, **(5) material evaluation
adaptation**, or **(6) unresolved ambiguity**.

### Cache model and capacity semantics

- Original HALP: variable-size objects, **byte capacity** (production
  YouTube CDN DRAM cache). *(1) exact from paper.*
- This repository: **unweighted paging**, unit object size, capacity in
  object slots (32/64/128), matching the manuscript's canonical evaluation
  setting for every other policy in this comparison. *(5) material
  evaluation adaptation* — the byte-size dimension of the original problem
  is not exercised at all here; this is the same adaptation already applied
  uniformly to LRB and 3L-Cache in this repository (see `docs/lrb_method_spec.md`,
  `docs/three_l_cache_method_spec.md`).

### Baseline heuristic and candidate shortlist

- Original HALP: a two-stage filter. A **configurable baseline heuristic**
  (the paper states it can default to LRU or approximate other policies)
  selects a small candidate subset via **randomized sampling that
  approximates exact priority-queue order**, providing exploration
  diversity across timesteps. *(2) from official blog.*
- This repository: a **deterministic** shortlist of the `k = 8`
  oldest-in-cache pages, read directly off the LRU tail (`_sample_candidates`
  in `src/lafc/policies/halp.py`), with no randomization. *(4) repository-
  required adaptation* — determinism is required for reproducible, seed-
  controlled evaluation and testable tie-breaking in this repository's
  protocol (see `docs/baselines.md` fairness requirements); the
  randomized-sampling mechanism itself is not reproduced. The underlying
  baseline heuristic (LRU-based candidate selection) matches the paper's
  stated default.

### Preference learning target and label construction

- Target: **relative preference**, not pointwise next-arrival-interval
  regression. For two candidates `A`, `B`: `A ≻ B` if `A` is re-accessed
  before `B` in the future ("mirrors the optimal offline oracle" per the
  official blog). *(1)/(2) exact from paper and official blog.*
- Labels are constructed from ground-truth future re-access order within
  the trace (an offline oracle over the observable trace, not a separate
  teacher model). *(1) exact from paper.*
- This repository: identical relative-preference target, implemented as
  pairwise comparisons over `actual_next` (ground-truth next-arrival time,
  available only because this is an offline-replay simulator with full
  traces) among shortlisted candidates recorded during the cold-start
  window (`HALPPolicy._train`). *(3) inferred/adapted* — see "no leakage"
  discussion below for why this does not become an evaluation-time leak.

### Reward model architecture

- Original HALP: a **"light-weight two-layer multilayer perceptron (MLP)"**
  reward model, randomly initialized and **trained continuously online**
  from a transient feedback buffer, specialized per cache server. *(2) exact
  from official blog* — this directly corrects an earlier draft of this
  document (see `docs/halp_provenance.md` §4, "corrected claim") that had
  described the reward model as a linear/logistic-regression scorer; that
  was inaccurate relative to the primary source and has been fixed.
- This repository: a hand-rolled shared-weight two-layer MLP,
  `R(x) = W2 . relu(W1 x + b1)`, with a `hidden_units`-wide hidden layer
  (default 8, matching the "light-weight" characterization)
  (`src/lafc/halp_model.py`). *(4) repository-required adaptation* on two
  axes:
  1. **Optimizer**: deterministic full-batch gradient descent (fixed
     learning rate, fixed epoch count), not continuous online stochastic
     updates — because this repository trains once at a frozen split point
     rather than continuously (see "Training and evaluation protocol"
     below), full-batch descent is the reproducible choice for a one-shot
     batch fit and has no online-vs-offline analogue to preserve.
  2. **Training objective**: the standard Bradley-Terry / RankNet pairwise
     cross-entropy loss, `-log(sigmoid(R(x_A) - R(x_B)))`, computed via a
     shared forward pass of `R` on both `x_A` and `x_B` and manual
     backpropagation through both branches — consistent with "preference
     comparisons" in the official blog, but not verified against the exact
     production loss formulation (no official loss equation is public).
     *(6) unresolved ambiguity* on the exact loss form.

  **Implementation note**: an earlier draft of this module used
  `sklearn.neural_network.MLPClassifier` trained directly on symmetrized
  feature-*difference* vectors (`x_A - x_B`) — the same reduction this
  repository's linear-model baselines use, where it is exact because for a
  *linear* `R`, `R(x_A) - R(x_B) = R(x_A - x_B)`. That identity does **not**
  hold for a nonlinear `R`; reusing it with an MLP produced an internally
  inconsistent scoring function (confirmed empirically: `lbfgs failed to
  converge` warnings and non-ranking-consistent scores on real trace data).
  This was caught and fixed before this baseline was evaluated on any trace
  used for reviewer-facing results; see `docs/halp_provenance.md` §3.
- Reward scores are read off as `P(preferred class)` via `predict_proba`,
  a monotonic surrogate for a raw production score; only the induced
  ranking over the shortlist is used for victim selection, so this
  reparameterization does not change eviction decisions. *(4) repository-
  required adaptation.*

### Victim selection

- Candidate with the **lowest** predicted reward/preference score is
  evicted. *(1)/(2) exact from paper and blog* (retain the highest-reward /
  most re-access-imminent candidates; evict the least-preferred).
- Deterministic tie-break: among candidates tied at the lowest score, the
  one with the lexicographically **largest `page_id`** is evicted
  (`src/lafc/policies/halp.py`). *(4) repository-required adaptation* — no
  tie-break rule is specified in the public sources; this repository's
  convention (also used by 3L-Cache's tie-break) is applied for
  determinism.

### Feature vector

- This repository: 5 features per candidate — Age (time since last
  request), Frequency (running request count), and the 3 most recent
  inter-arrival deltas (NaN-padded if unobserved), computed in
  `src/lafc/halp_features.py`. *(5) material evaluation adaptation* — the
  official blog describes "externally-provided user attributes and
  internally-computed dynamic metrics like time-since-access and average
  inter-access intervals" tracked in a metadata-only ghost cache; the exact
  production feature set (including any content/user-specific attributes)
  is not public. This repository's feature set uses only trace-derivable
  recency/frequency/inter-arrival signals, matching the feature family used
  for LRB and 3L-Cache in this repository so the three external baselines
  are feature-comparable.

### Cold-start behavior

- Before the model is trained, the policy evicts via **plain LRU**
  (evicts the LRU head) while recording shortlist features and identities
  for later label construction. *(4) repository-required adaptation of a
  documented pattern* — no official cold-start rule is public for HALP
  specifically, but plain-LRU cold start is the same convention used for
  LRB and 3L-Cache in this repository (`docs/baselines.md` Baselines 6–7),
  applied here for consistency.

### Fallback behavior

- If, at the training trigger, zero valid preference pairs were observed
  (`_train` finds no total-order-resolvable pairs — possible only on
  pathological/degenerate traces), the model is marked trained but
  `predict_rewards` returns all-zero scores, and eviction falls back to the
  tie-break rule (effectively LRU-head-order among zero-score ties) rather
  than crashing or silently reverting to an unrelated policy. *(4)
  repository-required adaptation, no official source.*

### Training and evaluation protocol

- Original HALP: **continuous online training** from random initialization,
  specialized per production cache server, with no fixed train/eval split
  (it is a running production system, not a benchmark with held-out data).
  *(1)/(2) exact from paper and blog.*
- This repository: a **frozen temporal split** — `[0, training_trigger)`
  is the cold-start/training-data-collection window (`training_trigger`
  defaults to 10,000, i.e. the first 20% of each 50,000-request trace);
  at `t = training_trigger` the model is fit once on pairs collected during
  that window and frozen; `[training_trigger, 50000)` is evaluated with the
  frozen model and zero further updates. *(5) material evaluation
  adaptation* — this is Protocol B ("temporal split with frozen model") of
  this repository's train/validation/test protocol requirements: a
  disjoint training prefix and a strictly later, disjoint evaluation
  suffix, with metrics reported only over the evaluation suffix. This
  differs materially from the paper's continuous-online-learning regime,
  which this offline-replay, single-pass simulator cannot reproduce.

### No-leakage guarantee

Trace of online decision behavior at evaluation time:
`current request → currently observable cache/LRU state → 8-candidate
LRU-tail shortlist → 5-feature vector (from observed history only) → frozen
MLP inference → lowest-score victim`. No `actual_next` value is read once
`_model_trained` is `True` and no further `_recorded_events`/`_train` calls
occur after the single training event at `t = training_trigger`.

Trace of training-label construction (occurs only once, at
`t = training_trigger`, using only requests already observed in
`[0, training_trigger)`): `recorded cold-start shortlists → each
candidate's actual_next observed strictly before training_trigger (or ∞ if
unresolved within the window) → pairwise ordering → symmetrized diff pairs
→ one-shot MLP fit → frozen model`.

`tests/test_halp.py::test_no_future_leakage_from_next_arrival` verifies this
directly: mutating `actual_next` for a request at `t = training_trigger`
(after training has already occurred) produces bit-identical eviction
decisions and diagnostics.

### Randomness and seeds

- `HALPConfig.seed` (default 0) seeds `numpy.random.default_rng(seed)` for
  the MLP's weight initialization only; the full-batch gradient-descent
  update itself is deterministic given fixed data and initialization (no
  minibatching or shuffling). Candidate shortlist selection and
  tie-breaking are fully deterministic (no randomness). *(4) repository-
  required adaptation* — the original's randomized candidate sampling is
  intentionally not reproduced (see "Baseline heuristic and candidate
  shortlist" above).

### Hyperparameters

| Parameter | This repository's default | Source |
|---|---|---|
| `training_trigger` | 10,000 (first 20% of a 50,000-request trace) | (4) repository-required — no official default exists for a fixed-horizon offline trace |
| `hidden_units` | 8 | (4) chosen to match the "light-weight" characterization in the official blog; no official numeric default is public |
| `alpha` (L2) | 1e-4 | (4) repository-chosen default-scale value; no official default is public |
| `lr` (gradient-descent step size) | 0.05 | (4) repository-chosen; no official default is public |
| `n_epochs` (full-batch training epochs) | 300 | (4) repository-chosen; no official default is public |
| shortlist size `k` | 8 | (4) repository-required; no official numeric default is public (paper describes the mechanism, not a fixed constant) |
| seed | 0 | (4) repository convention |

No official numeric defaults are public for any HALP hyperparameter (no
released code, no hyperparameter table in the paper's public
abstract/blog). All values above are repository-chosen and validation-
tunable via CLI flags (`--halp-training-trigger`, `--halp-hidden-units`,
`--halp-alpha`, `--halp-seed`); they are not the output of a best-test-set
search (see `scripts/experiments/run_halp_comparison.py`).

### Computational overhead

- Original: paper reports ~1.8% CPU overhead in production. *(1) exact
  from paper* — not independently reproducible here (different hardware,
  different workload, different measurement methodology); this repository
  instead reports wall-clock time per trace/capacity/policy row in its own
  runner, which is a repository-internal relative comparison only, not a
  claim of matching the paper's overhead number.

## Fidelity summary

| Aspect | Classification |
|---|---|
| Cache model (unit-size vs byte) | (5) material evaluation adaptation |
| Baseline heuristic (LRU-based) | (1)/(2) matches paper's stated default |
| Candidate shortlist mechanism | (4) deterministic top-k, not randomized priority-queue sampling |
| Preference target (pairwise, future-order) | (1)/(2) exact |
| Reward model family (2-layer MLP) | (4) architecture family matched; optimizer/training-schedule adapted |
| Loss formulation | (6) unresolved ambiguity (no official loss equation public) |
| Feature vector | (5) material evaluation adaptation (trace-derivable features only) |
| Victim selection direction | (1)/(2) exact |
| Tie-break rule | (4) repository convention, no official source |
| Cold start | (4) repository convention (LRU), consistent with LRB/3L-Cache |
| Train/eval protocol | (5) material evaluation adaptation (frozen split vs continuous online) |
| Hyperparameters | (4) all repository-chosen; no official defaults exist |

**Overall**: this is a **disclosed, unit-size-and-offline-replay adaptation
of HALP's algorithmic core** (pairwise preference learning over a
heuristic-generated shortlist, evicting the lowest-scored candidate), not
a faithful reproduction of the production system, which cannot be
reproduced faithfully because no official code, released model, or
production feature/telemetry pipeline is public. Reviewer-facing framing:
**"HALP adapted to the unit-size, offline-replay paging setting."**

## Fairness table (HALP vs. evict_value_v1, LRB, 3L-Cache, LRU)

| Dimension | evict_value_v1 | LRB | 3L-Cache | HALP | LRU |
|---|---|---|---|---|---|
| Trace identity | 7 Wulver families, 50K req | same | same | same | same |
| Trace hash provenance | recorded per-run | recorded per-run | recorded per-run | recorded per-run (this runner) | n/a |
| Capacity semantics | unit-size paging, 32/64/128 | same | same (adapted from byte-cache) | same (adapted from byte-cache) | same |
| Object-size semantics | unit | unit | unit (adapted; official is byte) | unit (adapted; official is byte) | unit |
| Metric | request misses / miss ratio | same | same | same | same |
| Simulator | this repo's `run_policy` | same | same | same | same |
| Training mode | offline, pretrained model | online, batched retraining | online, batched retraining | **offline, single frozen split** (adaptation of continuous-online original) | none |
| Training data | separate held-out training set (see `docs/baselines.md`) | in-trace, online | in-trace, online | in-trace, first 20% (`training_trigger`) | n/a |
| Validation data | separate | n/a (no offline tuning) | held-out prefix, `batch_size` tuned | none used in default config (hyperparameters not tuned against this trace) | n/a |
| Test data | disjoint from training | remaining trace | remaining trace after validation prefix | `[training_trigger, 50000)` | full trace |
| Hyperparameter protocol | fixed pretrained config | official defaults where operative | validation-tuned `batch_size` | repository-chosen, not test-tuned (§ Hyperparameters above) | n/a |
| Cold start | n/a (pretrained) | plain LRU | plain LRU | plain LRU | n/a |
| Fallback | explicit guard variant | LRU-head on stale/degenerate state | heap-exhausted LRU fallback | zero-score tie-break fallback (§ Fallback behavior above) | n/a |
| Seed | fixed | fixed | fixed | fixed (`--halp-seed`) | n/a |
| Official code available | n/a (this repo's own model) | yes (BSD-2-Clause) | yes (GPL-3.0) | **no** | n/a |

**Non-equivalences that must not be collapsed when combining results:**
HALP's training-data regime (single frozen 20% prefix) is not equivalent to
LRB/3L-Cache's continuous in-trace online retraining, and is not equivalent
to `evict_value_v1`'s separate-dataset pretraining. All four numbers are
"a request-miss count under this repository's paging replay," but the
*learning protocol* differs in ways that should be stated whenever HALP is
compared against the other three in a table, not merged into a single
"all learned baselines" claim.
