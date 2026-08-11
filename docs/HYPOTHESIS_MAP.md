# Hypothesis Map

This page tracks the mechanistic hypotheses for *why* learned candidate-level
eviction (`evict_value_v1` and its finite-horizon `eviction_loss`
supervision target) shows a gap between offline learnability and online
caching performance -- see [`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md)
for the underlying findings this page organizes.

**Status note:** most of the evidence behind these hypotheses currently lives
on the `kbs/second-revision-science` development branch (see
[`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) for exact locations and
merge status). This page is the general, durable statement of each
hypothesis; it does not depend on which branch currently hosts the code.

Status values: `STRONGLY_SUPPORTED`, `SUPPORTED`, `PARTIALLY_SUPPORTED`,
`INCONCLUSIVE`, `DISFAVORED`, `UNTESTED`. All statuses below are provisional
and will change as more evidence lands -- treat single-cell or
single-family evidence as exactly that, not as a general conclusion.

---

## H1 -- Insufficient training data

- Intuition: the offline-to-online gap could simply be a sample-size
  problem -- more labeled examples should close it.
- Current evidence: a same-target learning-curve comparison shows
  essentially flat offline and downstream metrics across a large increase
  in training data, for the fractions completed so far.
- Status: `DISFAVORED` (over the range tested so far; the campaign is not
  yet complete).
- Decisive test: complete and audit the full learning-curve campaign; only
  a genuine, non-noise, monotonic improvement in downstream miss ratio
  (not just offline fit metrics) would reopen this hypothesis.
- Stopping rule: stop attributing the gap to sample size if the completed
  campaign keeps showing this same flat pattern, or if offline metrics
  improve but downstream misses do not.

## H2 -- Model approximation failure

- Intuition: maybe the learned model just does not fit its own supervision
  target well enough.
- Current evidence: in a deeply audited cell, the learned model agrees with
  the exact value of its own training target 96.5% of the time, with low
  mean regret -- and its small departures from the target are net
  beneficial (fewer misses than exact target-following), not harmful.
- Status: `DISFAVORED` (single cell).
- Decisive test: repeat the exact-target comparison across more trace
  families and cache capacities.
- Stopping rule: reopen only if a broader sweep shows agreement dropping
  substantially somewhere.

## H3 -- Target degeneracy / low information content

- Intuition: the finite-horizon target may simply not distinguish most
  eviction candidates from each other, regardless of model quality.
- Current evidence: in the same audited cell, essentially every eviction
  decision has multiple (often nearly all) candidates tied for "optimal"
  under the finite-horizon label.
- Status: `STRONGLY_SUPPORTED` (single cell; not yet shown to generalize).
- Decisive test: replicate across trace families and cache capacities.
- Stopping rule: would be disfavored as a general phenomenon if most other
  cells show materially less degeneracy than this one.

## H4 -- Horizon truncation / temporal credit assignment

- Intuition: a fixed, short look-ahead horizon may miss consequences that
  only become visible further in the future.
- Current evidence: extending the horizon substantially (an 8x increase)
  only resolves a minority of the ties found under H3 -- horizon extension
  alone does not fully explain the degeneracy.
- Status: `PARTIALLY_SUPPORTED` (single cell).
- Decisive test: a reuse-delay diagnostic (see H11) plus a broader horizon
  sweep across trace families.
- Stopping rule: stop treating horizon length alone as sufficient if
  extending it improves target resolution but not downstream misses once
  deployed.

## H5 -- Continuation-policy mismatch

- Intuition: training labels assume a fixed continuation policy (plain LRU)
  after each decision, while the deployed policy continues making its own
  learned decisions -- that mismatch could hurt.
- Current evidence: a preliminary distribution-shift check found large
  trajectory divergence in the one trace family tested, but a natural
  corrective procedure (relabeling on-policy) made measured misses worse,
  not better, despite reducing a generic measured state-shift metric. A
  causally cleaner, purpose-built test of this specific hypothesis is
  implemented but has not yet produced a result beyond a tiny smoke run.
- Status: `INCONCLUSIVE`.
- Decisive test: the full causal continuation-policy comparison (comparing
  a fixed continuation assumption against a more internally-consistent
  one) across all trace families.
- Stopping rule: deprioritize this explanation if the more consistent
  continuation assumption fails to improve downstream misses over the
  current fixed one.

## H6 -- State-distribution shift

- Intuition: distinct from H5, the sequence of cache states actually
  visited during deployment may differ from what training saw, independent
  of which continuation policy generated the labels.
- Current evidence: same preliminary check as H5 -- large trajectory
  divergence exists, but a generic state-shift metric decreasing did not
  correlate with misses improving.
- Status: `INCONCLUSIVE` (single trace family).
- Decisive test: broader family coverage, and a state-shift metric more
  directly tied to which parts of the cache state actually matter for
  future misses.
- Stopping rule: pivot the metric (rather than abandon the framing) if
  further cells confirm the generic metric doesn't track misses.

## H7 -- Hard-selection / uncertainty instability

- Intuition: always picking the single best-scoring candidate (a hard
  argmin) could be unreliable when several candidates' predicted values are
  close together, relative to the model's uncertainty.
- Current evidence: not directly measured yet. Adjacent evidence is mildly
  against this being the *dominant* explanation: even an exact, fully
  deterministic (non-argmin-unstable) optimization of the target also lost
  to LRU in the audited cell -- so instability in the learned model's
  selection specifically cannot be the whole story.
- Status: `UNTESTED`.
- Decisive test: a margin-gated or uncertainty-aware selection experiment,
  or a direct measurement of predicted-value margins versus decision
  correctness.
- Stopping rule: would be disfavored if predicted-value margins are
  typically large (well-separated) yet misses remain high.

## H8 -- Missing terminal / tail value

- Intuition: a finite-horizon target implicitly assumes zero value for
  anything that happens after the horizon, which could systematically
  undervalue candidates whose real benefit is later.
- Current evidence: this is currently an open, unimplemented direction --
  no local diagnostic exists yet for it in this repository.
- Status: `UNTESTED`.
- Decisive test: a dedicated diagnostic comparing a finite-horizon target
  against one augmented with an estimated tail/terminal value, before any
  new loss function is adopted.
- Stopping rule: not applicable until a first diagnostic exists.

## H9 -- Short-horizon strict preferences may be myopic

- Intuition: even on the rare occasions where the short horizon *does*
  produce a strict (non-tied) preference between candidates, that
  preference might reverse once evaluated at a longer horizon -- the short
  window could reward something that turns out to be bad later.
- Current evidence: the one deeply audited cell available so far was so
  degenerate (see H3) that it produced almost no strict preferences at all
  to test this against -- the hypothesis is not yet directly testable from
  existing evidence. An adjacent measurement (how well a naive tie-break
  rule tracks longer-horizon-optimal choices) does trend the right
  direction to be consistent with the concern, but is not a direct test.
- Status: `UNTESTED` as literally framed.
- Decisive test: find or construct a cell where strict (non-tied) short-
  horizon preferences are common, then directly measure how often the
  short-horizon-preferred choice is reversed at a longer horizon.
- Stopping rule: disfavor if a cell with common strict preferences shows a
  low reversal rate (most short-horizon preferences already agree with the
  longer-horizon answer).

## H10 -- Fixed horizon may provide decreasing information as cache capacity grows

- Intuition: a fixed look-ahead horizon `H` observes a fixed amount of
  future information, while cache capacity `C` determines how many
  eviction candidates that same fixed amount of information must
  distinguish between. As `C` grows with `H` held fixed, the same
  observation window may need to resolve more candidates, so it plausibly
  carries less information *per candidate*. A ratio such as `H/C` is a
  natural way to think about this, but it is a conceptual framing to test,
  **not a mathematically established relationship** -- no claim is made
  here that `H/C` is theoretically exact or that it is the right
  normalization.
- Current evidence: existing diagnostics have so far only been run at one
  fixed capacity, so this has not been tested at all yet.
- Status: `UNTESTED`.
- Decisive test: rerun the existing target-resolution diagnostic (already
  implemented) across multiple cache capacities at a fixed horizon, using
  metrics like the fraction of candidates that tie for "optimal" or the
  target's information entropy, and look for a trend against capacity (and
  later against the `H/C` ratio once multiple horizons are also available).
- Stopping rule: disfavor if target-resolution metrics do not track
  capacity (or `H/C`) in the expected direction across at least two
  capacities.

## H11 -- Eviction consequences may occur beyond the horizon

- Intuition: define `D` as the number of future requests until an evicted
  object is requested again (its "reuse delay"). A fixed horizon `H`
  cannot directly observe any consequence with `D > H`. This is a distinct
  question from H4/H9: it is about how often real consequences fall
  outside the observation window at all, not about the quality of what
  happens to fall inside it.
- Critical distinction to preserve: a later reuse of an evicted object
  (a **potential consequence**) is not automatically evidence that the
  eviction *caused* an extra miss relative to what another policy would
  have done (a **causal excess miss**). An object can be requested again
  much later than `H` and still not have caused any miss a better policy
  would have avoided, and conversely a miss just after `H` could be a real
  causal effect. These two must not be conflated.
- Current evidence: not yet measured. The "potential consequence" question
  (the distribution of reuse delays after an eviction) looks answerable
  from already-recorded decision logs and the raw trace, without a new
  live replay -- it would be a new offline analysis of existing artifacts.
  The "causal excess miss" question is materially harder and would need a
  dedicated counterfactual-replay mechanism that does not yet exist.
- Status: `UNTESTED`.
- Decisive test: first compute the potential-consequence (reuse-delay)
  distribution as a cheap first pass (e.g. bucketed into short/medium/long/
  never-reused); only pursue the causal-excess-miss question if that first
  pass shows a meaningful fraction of reuses falling well beyond the
  horizon.
- Stopping rule: disfavor horizon truncation as a primary mechanism if most
  future reuses of evicted objects already fall within the horizon;
  continue investigating only if a large fraction fall well beyond it *and*
  a nontrivial fraction of those are shown to be causal excess misses, not
  merely potential ones.

---

## Cross-cutting notes

- No hypothesis above should be treated as general from a single cell or
  single trace family; several currently rest on exactly one. Replication
  is the shared precondition for upgrading any of these.
- The current best mechanistic reading, pending further replication, points
  primarily toward a **target-construction problem** (H3, and by extension
  H4/H9/H10/H11) rather than a pure model-fitting problem (H2) -- see
  [`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md) for the
  reasoning.
