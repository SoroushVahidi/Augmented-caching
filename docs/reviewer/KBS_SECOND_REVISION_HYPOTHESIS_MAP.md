# KBS Second-Revision Hypothesis Map

Status: authoritative local source of truth for mechanistic hypotheses about
the `evict_value_v1` / `eviction_loss` offline-vs-online performance gap.

This file consolidates and supersedes the hypothesis tables previously
duplicated across `docs/reviewer/kbs_negative_results_interpretation.md`
(sections 9.1-9.12), `docs/reviewer_revision_roadmap.md`, and
`analysis/kbs_local_current_evidence_synthesis_20260810/hypothesis_status.csv`.
Those files should now point here for the hypothesis matrix itself rather
than repeating it; they retain their own narrative/evidence detail and
references.

The local `50%` learning-curve campaign is complete and audited. Do not
launch `100%`: the predefined stopping rule has fired for H1, and `100%` is
now intentionally not run rather than missing required work.

Last updated: 2026-08-11, after the final `wiki2018|0.5` resume completed
cleanly and the 50% integrity audit passed.

---

## H1 -- insufficient training data

- Statement: the offline-to-online gap is primarily because the models are
  undertrained; more labeled examples would close it.
- Motivation: standard first hypothesis for any ML underperformance; directly
  testable via the same-target learning-curve campaign.
- Current evidence: same-target scalar-vs-pairwise learning curve through
  audited `50%`. The apples-to-apples `1%->50%` curve over the four families
  present at every fraction shows scalar miss ratio `0.6256->0.6126` and
  pairwise miss ratio `0.8299->0.8300`; this is not a material monotonic
  downstream improvement. The full 50% seven-family slice has `42/42` rows,
  all `status=ok`; scalar is better on `18/21` family/capacity cells, ties on
  `3/21`, and pairwise is better on `0/21`, with mean pairwise-minus-scalar
  miss-ratio gap `+0.1611`.
- Status: `DISFAVORED` within the tested `1%-50%` range. This does not prove
  that more data can never help; it means the sample-size explanation is not
  supported as the primary cause by this campaign's tested range.
- Decisive next experiment: complete for this stopping-rule scope. The
  synthesized closeout is
  `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`.
- Stopping rule: `STOP_SAMPLE_SIZE_HYPOTHESIS`. `100%` is intentionally not
  run under this rule, not an active missing requirement.
- Reviewer relevance: MC3, R3-Issue1/4.
- Owner of next work: LOCAL.
- Experiment state: `50%` COMPLETE_7_OF_7 and validated; `100%`
  INTENTIONALLY_NOT_RUN_DUE_STOPPING_RULE.

## H2 -- model-fitting / function-approximation failure

- Statement: the learned model fails to approximate its own supervision
  target well enough, and that approximation error explains the gap.
- Motivation: standard second hypothesis -- rule out "the model just isn't
  learning its target" before blaming the target itself.
- Current evidence: exact-target-oracle one-cell diagnostic
  (`brightkite`/cap64/H4): learned model agrees with the exact H=4 target on
  96.5% of decisions, mean target regret `0.035`. The model is not failing to
  fit; its 3.5% of departures from the target are net beneficial (fewer
  misses than exact target-following: 15449 vs 19079).
- Status: `DISFAVORED` (single cell).
- Decisive next experiment: repeat the exact-target-oracle diagnostic across
  more families/capacities; if agreement stays high everywhere, further
  disfavor.
- Stopping rule: already provisionally closed at 96.5% agreement; only reopen
  if a broader sweep shows agreement dropping substantially (e.g. below ~80%)
  somewhere.
- Reviewer relevance: MC1, R3-Issue4.
- Owner of next work: LOCAL.
- Experiment state: NOT_STARTED (sweep beyond the one cell).

## H3 -- target degeneracy / low resolution at H=4

- Statement: the finite-horizon `H=4` eviction-loss target carries very
  little discriminative information among eviction candidates.
- Motivation: a target that rarely distinguishes candidates cannot supervise
  a useful policy regardless of model quality.
- Current evidence: target-degeneracy diagnostic (`brightkite`/cap64/H4):
  `tie_event_fraction=1.0` (every decision has a tie),
  `mean_optimal_set_fraction=0.9932` (99.3% of 64 candidates tie for
  optimal), median distinct-target-value count `1`, target entropy mean
  `0.048` bits.
- Status: `STRONGLY_SUPPORTED` (single cell; not yet family-general).
- Decisive next experiment: replicate the already-implemented degeneracy
  script across all 7 families and multiple capacities.
- Stopping rule: would be disfavored as a general phenomenon if a majority of
  other family/capacity cells show materially lower tie fractions / higher
  entropy than this cell.
- Reviewer relevance: MC1, R3-Issue4.
- Owner of next work: LOCAL.
- Experiment state: NOT_STARTED (multi-cell sweep); blocked from launching
  while `50%` worker is active.

## H4 -- horizon truncation / temporal credit assignment

- Statement: a candidate that looks harmless within `H=4` may have
  significant longer-term reuse consequences the target cannot see.
- Motivation: complements H3 -- even a non-degenerate short horizon can still
  truncate relevant future information.
- Current evidence: longer-horizon tie-break analysis on the same cell: H=8/
  16/32 break only `14.2%`/`27.6%`/`39.6%` of H=4 ties -- most degeneracy
  persists even at 8x the horizon. The naive deterministic tie-break's
  agreement with longer-horizon-optimal choice declines as horizon grows
  (`93.75%->87.48%->81.39%`).
- Status: `PARTIALLY_SUPPORTED` (single cell).
- Decisive next experiment: H11's eviction-to-reuse-delay diagnostic, plus a
  broader-H degeneracy sweep across families, to separate pure truncation
  from tie-break design.
- Stopping rule: stop treating horizon length alone as sufficient if
  extending H materially improves target resolution but downstream misses do
  not improve when those longer-horizon choices are actually deployed.
- Reviewer relevance: MC1, R3-Issue4.
- Owner of next work: LOCAL first (cheap, existing tooling); WULVER for
  full-scale confirmation.
- Experiment state: NOT_STARTED (beyond the one cell).

## H5 -- continuation-policy mismatch

- Statement: labels are constructed assuming LRU continuation after the
  decision, but the deployed policy continues with more learned evictions,
  and that mismatch degrades performance.
- Motivation: standard imitation-learning / DAgger concern (Ross, Gordon,
  Bagnell 2011) -- label-time and deployment-time trajectories diverge.
- Current evidence: `distribution_shift_ablation_v1` (metacdn only):
  trajectory divergence `97.15%`/`99.52%`/`99.84%` at capacities `32/64/128`
  under `DAGGER_ITER1` vs off-policy LRU, but misses *worsened* under DAgger
  at all three capacities despite a reduced measured state-shift index. The
  frozen C1/C2 continuation-policy causal ablation (`src/lafc/continuation_policy_ablation.py`,
  `configs/continuation_policy_causal_ablation_v1.json`) is source/test-ready
  but only `TINY_SMOKE_ONLY` (4 traces, <=300 requests/trace,
  `decision_count=3`) -- no full result.
- Status: `INCONCLUSIVE` (single family, smoke-only for the frozen protocol).
- Decisive next experiment: full 7-family C1/C2 causal ablation comparing
  LRU-continuation vs frozen-`pi1`-continuation labels on the same
  decision/candidate examples.
- Stopping rule: stop prioritizing continuation mismatch as primary if
  full-scale C2 fails to improve over C1 on downstream misses.
- Reviewer relevance: MC3, R2-Major3, R3-Issue4.
- Owner of next work: WULVER (full campaign); LOCAL implementation is already
  frozen and sync-ready.
- Experiment state: smoke NOT_RUN_AT_SCALE; full campaign NOT_STARTED
  (requires Wulver, not contacted in this pass).

## H6 -- state-distribution shift

- Statement: the sequence of cache states visited under the learned policy
  diverges from the states seen during label construction, independent of
  the continuation-policy label semantics per se.
- Motivation: distinguishes "trajectory looks different" from "trajectory
  divergence causes worse outcomes."
- Current evidence: same `distribution_shift_ablation_v1` metacdn cells --
  measured state-shift index *decreased* under DAgger (e.g.
  `0.000664->0.000462`) even as misses got worse, i.e. reduced generic
  state-shift did not translate into better misses in this one family.
- Status: `INCONCLUSIVE` (single family).
- Decisive next experiment: broader family sweep plus a state-shift metric
  more directly tied to miss-relevant cache-content divergence rather than a
  generic state-shift index.
- Stopping rule: stop treating generic state-shift-index reduction as
  informative if further cells confirm it does not correlate with reduced
  misses; pivot metric rather than abandoning the framing outright.
- Reviewer relevance: MC3, R3-Issue4.
- Owner of next work: LOCAL (metric redesign); WULVER (broader sweep).
- Experiment state: NOT_STARTED (beyond the one family already run).

## H7 -- hard-argmin / uncertainty instability

- Statement: choosing the eviction candidate via a hard argmin over predicted
  values is unreliable when predicted values are close relative to model
  uncertainty.
- Motivation: standard decision-rule concern, separate from target
  construction.
- Current evidence: not directly measured for the deployed learned model
  (explicitly flagged in the negative-results notebook as a future
  "decision-rule branch" diagnostic, not yet run). Adjacent evidence: the
  exact, non-argmin, deterministic H=4 oracle *also* loses badly to LRU,
  which weakens (but does not rule out) argmin instability as the dominant
  explanation, since a stable/exact procedure fails too.
- Status: `UNTESTED`.
- Decisive next experiment: margin-gated softmax or uncertainty-aware
  selection experiment; or direct measurement of predicted-value margin
  distribution vs. decision correctness for the deployed scalar model.
- Stopping rule: would be disfavored if predicted-value margins are large
  (well-separated) at most decisions yet misses remain high.
- Reviewer relevance: R3-Issue4 (decision-rule branch).
- Owner of next work: LOCAL.
- Experiment state: NOT_STARTED; listed only as a future diagnostic
  candidate, not a current primary method change.

## H8 -- missing terminal / tail value beyond H

- Statement: a finite-horizon target with an implicit zero terminal value
  beyond `H` undervalues candidates whose main benefit occurs after the
  horizon.
- Motivation: standard truncated-horizon RL/optimal-control concern; would
  motivate a `Q_H + V_tail_hat` target formulation.
- Current evidence: explicitly flagged in the local negative-results notebook
  (9.5.1) as an open target-formulation possibility. Ownership noted as
  Wulver-side "historical-tail readiness work" -- not locally evidenced or
  implemented.
- Status: `UNTESTED`.
- Decisive next experiment: a historical-tail diagnostic must precede any new
  loss definition; not yet implemented locally or (per local docs) on Wulver.
- Stopping rule: n/a until a first diagnostic exists.
- Reviewer relevance: MC1.
- Owner of next work: WULVER (per existing docs' stated ownership); status
  not independently verified locally in this pass (no Wulver contact made).
- Experiment state: NOT_STARTED locally.

## H9 -- rare strict H=4 distinctions may be myopic (user hypothesis A)

- Statement: even when H=4 does produce a strict (non-tied) preference, that
  preference may reverse at H=8/H=16 because the short window rewards an
  action that is bad later.
- Motivation: user-proposed; targets the minority of decisions that are *not*
  captured by the H3 degeneracy finding.
- Current evidence: the one available cell has `tie_event_fraction=1.0` --
  essentially no strict H=4 preferences occurred, so this cell cannot test
  reversal of a "rare strict distinction" as framed. Adjacent evidence
  (deterministic tie-break's longer-horizon alignment declining from `93.75%`
  to `81.39%` as horizon grows) is directionally consistent but measures
  broken ties within tied sets, not genuinely strict preferences.
- Status: `UNTESTED` (as literally framed).
- Decisive next experiment: run the existing (no new code needed)
  target-degeneracy script on a family/capacity cell with a higher
  strict-margin fraction, then measure the reversal rate of strict
  H=4-preferred choices at H=8/H=16, conditioned on the preference being
  strict.
- Stopping rule: disfavor if a strict-preference-reversal cell shows a low
  reversal rate (e.g. <10%) -- that would mean rare strict H=4 preferences
  are usually already correct.
- Reviewer relevance: MC1.
- Owner of next work: LOCAL.
- Experiment state: NOT_STARTED; blocked from launching while `50%` worker is
  active.

## H10 -- horizon should scale with cache capacity / reuse timescale (user hypothesis B)

- Statement: a fixed `H` carries a fixed amount of future information, while
  cache capacity `C` determines how many candidates that information must
  distinguish; effective horizon may need to scale with `C` (conceptually
  `H/C`).
- Motivation: user-proposed; would explain why a single fixed `H` might be
  adequate at small `C` but degenerate at large `C`.
- Current evidence: only `(H=4, C=64)` has been run for the degeneracy/oracle
  diagnostics. The learning-curve and objective-ablation CSVs do sweep
  capacity (`32/64/128`) but do not carry the target-resolution metrics
  (optimal-set fraction, entropy) this hypothesis needs.
- Status: `UNTESTED`.
- Decisive next experiment: rerun the existing degeneracy script at fixed
  `H=4` across `C in {32,64,128}` first (cheapest), using
  `mean_optimal_set_fraction` and `target_entropy_bits` as resolution
  metrics; ideally add `H` sweeps at matched `C` later to test the `H/C`
  ratio directly. `H/C` is a conceptual quantity, not assumed theoretically
  exact.
- Stopping rule: disfavor if the `H/C` sweep shows optimal-set fraction /
  target entropy does not track the `H/C` ratio in the expected direction
  across at least two capacities.
- Reviewer relevance: MC1.
- Owner of next work: LOCAL (cheap first pass); whether Wulver horizon data
  alone will eventually be sufficient is unknown from local artifacts --
  local docs explicitly do not assert unverified Wulver state.
- Experiment state: NOT_STARTED.

## H11 -- eviction consequences may occur outside H (user hypothesis C)

- Statement: many consequences of an eviction decision (the evicted object
  being requested again) occur after the labeling horizon `H` and are
  therefore invisible to the target; measuring eviction-to-next-reuse delay
  would quantify this.
- Motivation: user-proposed; a direct, trace-level way to test horizon
  adequacy independent of the degeneracy/tie-break framing in H3/H4.
- Current evidence: none computed yet. The exact-target-oracle's
  `learned_decisions.csv` (one completed cell) already records
  `decision_id`, `request_t`, and `chosen_candidate` per decision, which is
  enough to compute the `POTENTIAL_CONSEQUENCE` bucket (next-request distance
  of the evicted object) purely by scanning the raw trace against
  already-completed decision logs -- no new replay engine needed for that
  bucket. The stronger `CAUSAL_EXCESS_MISS` question (whether the eviction
  actually caused an avoidable miss) requires the not-yet-implemented
  minimum-counterfactual-attribution mechanism sketched in the notebook's
  9.7.1.
- Status: `UNTESTED`.
- Decisive next experiment: compute the `POTENTIAL_CONSEQUENCE` reuse-delay
  distribution (buckets `1-4, 5-8, 9-16, 17-32, >32, never reused`) from
  existing decision logs + raw trace as a first, cheap pass; implement
  minimum-counterfactual attribution later for `CAUSAL_EXCESS_MISS`.
- Stopping rule: disfavor truncation as the primary mechanism if most future
  reuses of evicted objects already fall within `H=4` (concentrated in the
  `1-4` bucket); continue investigating if a large fraction fall beyond `H=4`
  (e.g. `>8`) AND a nontrivial fraction of those are shown to be causal
  excess misses, not just potential consequences.
- Reviewer relevance: MC1, MC3, R3-Issue4.
- Owner of next work: LOCAL (potential-consequence bucket, cheap); LOCAL or
  WULVER (causal-excess-miss bucket, needs new implementation, more
  expensive either way).
- Experiment state: NOT_STARTED; blocked from launching while `50%` worker is
  active.

---

## Refined horizon-adequacy framing (H10/H11 candidate quantities)

Added during the 2026-08-10 finalization pass, based on a literature
synthesis. This section fixes vocabulary for future H10/H11 work; nothing
below has been measured yet on this branch.

**Terminology guardrail:** `H` here is a count of future *requests* (the
`eviction_loss` target looks `H` requests ahead). Do not conflate this with
classical *stack distance* / *reuse distance*, which counts **distinct
pages** between two accesses to the same page. Below, `T` denotes
`next_reuse_time_requests` -- a request-count quantity in the same units as
`H` -- never a distinct-object stack-distance figure.

- **Primary candidate -- `P(T > H | resident)`.** Probability that a
  cache-resident object's next reuse falls beyond the horizon window. This is
  the dimensionally direct quantity (same units as `H`, no normalization
  assumption needed) and is the natural first thing to compute for H11's
  `POTENTIAL_CONSEQUENCE` bucket -- it is essentially the same computation
  already planned there, just expressed as a probability conditioned on
  residency rather than a raw delay-bucket histogram.
- **Secondary, competing candidates** (each needs an extra assumption
  `P(T>H)` does not, and none is currently implemented):
  - `H / C` -- `COARSE COMPETING HYPOTHESIS -- NOT ESTABLISHED LAW`. This is
    the quantity implicitly behind H10's framing; treat it as one candidate
    covariate to check against `P(T>H)`, not as a derived normalization.
  - `distinct-future-page coverage / C` -- closer to classical stack
    distance (distinct objects, not request count); must be reported under a
    different symbol than `T`, never interchanged with it.
  - `H / q90(T)` or `H / q95(T)` -- horizon relative to a high quantile of
    the resident reuse-time distribution; an alternative summary statistic
    of the same distribution `P(T > H)` comes from.
  - `footprint(H) / C` -- working-set footprint over the next `H` requests
    relative to capacity; capacity-dependent normalization not yet defined
    anywhere in this codebase -- do not use until defined.
- **What none of these establish yet:** `H` scaling linearly with `C`
  universally; H3 degeneracy vanishing iff `H` exceeds some boundary of any
  quantity above; LRU as a formal terminal-value estimator; or that
  `P(T > H)` causally explains the offline/online gap before it is measured
  and checked against downstream misses (consistent-with is not the same as
  causal-of -- see the potential-consequence vs. causal-excess-miss
  distinction already drawn in H11).
- Reviewer relevance: MC1 (extends H10/H11's next-diagnostic scope).
- Owner of next work: LOCAL (P(T>H) is computable from existing decision
  logs + raw trace, same inputs H11 already identified as sufficient for its
  first pass).
- Experiment state: NOT_STARTED; blocked from launching while `50%` worker is
  active, same as H9/H10/H11.

## Cross-cutting notes

- No hypothesis above should be treated as family-general from a single cell.
  H2, H3, H4 currently rest on one cell (`brightkite`, capacity `64`,
  `H=4`); H5, H6 rest on one family (`metacdn`). Replication is the shared
  precondition for upgrading any of these beyond their current status.
- Ranked coherent explanation (see
  `analysis/kbs_local_current_evidence_synthesis_20260810/CURRENT_LOCAL_EVIDENCE_SYNTHESIS.md`
  section 7 for full reasoning): primarily a **target problem** (H3, and by
  extension H4/H9/H10/H11), secondarily a **combination** weighted toward
  target/deployment interaction (H5/H6), with pure **model-fitting failure**
  (H2) and **argmin instability** (H7) currently the least supported.
- Do not launch `100%` for H1 unless a future, separately justified
  protocol change explicitly reopens the sample-size question.
