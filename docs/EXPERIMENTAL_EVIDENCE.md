# Experimental Evidence

This page is the entry point for the scientific evidence behind this
repository, organized around the questions being asked rather than around
any particular experiment or external review process. For the full
per-experiment index see [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md);
for mechanistic interpretation see [`HYPOTHESIS_MAP.md`](HYPOTHESIS_MAP.md);
for the honest summary of what worked and what didn't see
[`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md).

## 1. Experimental questions

- Does learned, candidate-level finite-horizon eviction improve cache
  performance over literature-standard baselines?
- How informative is the current `eviction_loss` supervision target --
  does it actually distinguish good eviction candidates from bad ones?
- How sensitive is target quality to the look-ahead horizon `H`, and does
  it interact with cache capacity `C`?
- Does training on more data close the gap between offline model quality
  and online (deployed) caching performance?
- Is part of the offline-to-online gap caused by a mismatch between how
  training labels are constructed and how the policy behaves once
  deployed (sequential distribution shift / continuation-policy mismatch)?
- Does the choice of supervision objective itself matter (scalar
  regression on a continuous loss vs. next-arrival time vs. reuse distance
  vs. a pairwise/ranking formulation)?
- How does the method compare with classical and modern learned
  alternatives under a matched, fair protocol?
- What computational overhead does fine-grained, candidate-level learned
  eviction introduce, and can it be reduced without changing decisions?

## 2. Primary evidence

- The core simulator/baseline suite and the `evict_value_v1` training/eval
  pipeline are complete and reproducible on `main` today (Experiment 1 in
  the registry).
- A learned-baseline fairness comparison exists, validated for LRB on
  `main`, with a broader comparison (3L-Cache, CACHEUS, a causal
  HALP-style reimplementation) implemented on the development branch
  (Experiment 2).
- A corrected, non-contaminated held-out cross-family comparison against
  baselines is in progress but not yet complete (Experiment 9) -- this is
  the single most important open item for a fully citable head-to-head
  comparison.

## 3. Mechanistic diagnostics

These are explicitly diagnostics, not primary claims -- most rest on a
single deeply-audited trace family and cache capacity, or a single
trace family, and are not yet shown to generalize:

- **Exact-target-oracle diagnostic** (Experiment 4): separates "is the
  target itself any good" from "does the model learn it well."
- **Target-degeneracy diagnostic** (Experiment 5): measures how much
  distinguishing information the target actually carries, and how much a
  longer horizon recovers.
- **Learning-convergence diagnostic** (Experiment 6): an ongoing campaign
  testing whether more training data helps, and whether representation
  (scalar vs. pairwise on the *same* target) matters independent of target
  construction.
- **Distribution-shift / continuation-policy diagnostics** (Experiment 7):
  test whether label-construction assumptions mismatch deployed behavior.

## 4. Main findings

See [`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md) for the full,
honest account, including negative results. In short:

- The learned model fits its own supervision target well -- this is not
  primarily a model-fitting problem (see `HYPOTHESIS_MAP.md` H2).
- In the one cell audited in depth, the target itself, even optimized
  exactly, underperformed a simple baseline (LRU) -- pointing toward a
  target-construction problem rather than a learning problem (H3).
- A frozen objective comparison found the current target performing worse
  than three alternatives, including two other scalar objectives -- this
  argues against "scalar regression in general is the problem."
- Whether insufficient training data explains the gap is currently
  disfavored by the learning-curve evidence collected so far, though that
  campaign is not yet complete.

## 5. Open questions

- Does the target-degeneracy finding generalize across trace families and
  cache capacities, or is it specific to the one cell tested?
- Does a more internally-consistent continuation-policy assumption
  (instead of the current fixed one) actually improve downstream misses?
- Does the informativeness of a fixed horizon scale with cache capacity in
  a way a horizon/capacity ratio could capture (see `HYPOTHESIS_MAP.md`
  H10 -- this is a hypothesis to test, not an established relationship)?
- How much of an evicted object's future reuse falls beyond the current
  horizon, and how much of that is a real, avoidable miss rather than just
  a later coincidental reuse (see `HYPOTHESIS_MAP.md` H11)?

## 6. Evidence status

See the status vocabulary and evidence-strength hierarchy defined at the
top of [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md), used consistently
across this documentation set. No active or partial experiment is
described here as a final result; partial-campaign numbers are deliberately
not reproduced in this durable document to avoid it going stale -- see the
development branch's own status tracking for live progress.
