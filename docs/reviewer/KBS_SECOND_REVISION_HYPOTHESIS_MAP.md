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

Last updated: 2026-08-13, after C0/C1/C2 and distribution-shift both passed
formal post-completion integrity audit and closed H5/H6 below (`FINAL_VALIDATED`,
21/21 units / 7/7 folds each). Compact evidence:
`reports/kbs_final_evidence_20260813/`. Prior update 2026-08-12 recorded the
exact-target, strict-preference, and learned/exact campaigns passing their
21-cell integrity gates.

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
- Current evidence: the validated 21-cell learned/exact campaign reports
  macro set-aware agreement `0.975301` and positive-regret fraction
  `0.024699`; aggregate learned and LRU misses are `601569` and `565126`.
  Set-aware agreement is interpreted together with target degeneracy and
  regret-conditioned metrics.
- Status: `DISFAVORED` as a gross or uniform model-fitting explanation.
- Decisive next experiment: none required under the current stopping plan;
  reopen only if later integrity review invalidates this campaign.
- Stopping rule: already provisionally closed at 96.5% agreement; only reopen
  if a broader sweep shows agreement dropping substantially (e.g. below ~80%)
  somewhere.
- Reviewer relevance: MC1, R3-Issue4.
- Owner of next work: LOCAL.
- Experiment state: 21-cell learned/exact agreement diagnostic
  `FINAL_VALIDATED`; do not rerun.

## Common-model training-objective hypothesis

- Statement: the eviction-loss *training objective* itself is responsible
  for poor learned-policy performance, relative to alternative finite-horizon
  objectives, under a matched model/fold/feature protocol.
- Current evidence: common-model objective control V2 (21 cells × 4
  objectives, 84/84 rows, integrity PASS). Total misses:
  eviction_loss 571,976; pairwise 577,339; reuse_distance 615,850;
  next_arrival 627,392. Eviction-loss is nominally best by aggregate misses
  and is not materially worse than alternatives. Valid V2 pairwise is second
  by totals but best by per-cell rank. V1 pairwise is invalid and superseded.
- Status: `NOT_SUPPORTED` as an objective-causality claim in this matched
  control. The full `evict_value_v1` pipeline deficit is not attributable to
  the eviction-loss training objective per se.
- Experiment state: `FINAL_VALIDATED`. Formal audit:
  `reports/common_model_v2_formal_audit_20260814/AUDIT.md`. Do not rerun.

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
- Status: `STRONGLY_SUPPORTED` by the validated 21-cell exact-target,
  strict-preference, and tie-aware oracle diagnostics. The tie-aware control
  finds `fraction_tied_decisions = 1.0` on all 168 non-LRU rows.
- Decisive next experiment: none required for this revision; broader target
  formulations remain future work.
- Stopping rule: would be disfavored as a general phenomenon if a majority of
  other family/capacity cells show materially lower tie fractions / higher
  entropy than this cell.
- Reviewer relevance: MC1, R3-Issue4.
- Owner of next work: LOCAL.
- Experiment state: exact-target replication and strict-preference/horizon
  diagnostic `FINAL_VALIDATED`; tie-aware exact-target oracle v1
  `FINAL_VALIDATED` after campaign-CSV recovery. Do not rerun. The prior
  `50%` learning-curve campaign is also complete under its stopping rule.

## H4 -- horizon truncation / temporal credit assignment

- Statement: a candidate that looks harmless within `H=4` may have
  significant longer-term reuse consequences the target cannot see.
- Motivation: complements H3 -- even a non-degenerate short horizon can still
  truncate relevant future information.
- Current evidence: the validated 21-cell strict-preference diagnostic finds
  H4 unique-winner fraction `0` and multiple-optimum fraction `1`. The
  multi-cell reuse-tail diagnostic
  (`analysis/reuse_tail_horizon_diagnostic_v1/`) now directly measures the
  resident-candidate next-reuse delay distribution across seven families
  and three capacities. At H=4,
  `P(T>4 | resident)=0.9938544459677984`; even after conditioning on
  eventual reuse, `P(T>4 | resident, eventually reused)=0.9793302186526528`.
  The never-reused fraction is `0.7026792916224847`. The tie-aware exact
  oracle shows that the older deterministic exact-oracle-versus-LRU result
  (0 wins / 3 ties / 18 losses) is reproduced by `CURRENT_DETERMINISTIC`
  but is **not robust** to LRU-within-minima tie-breaking (16 wins / 5 ties
  / 0 losses vs LRU). The claim that the exact H4 target intrinsically
  loses to LRU is therefore **not supported**; that comparison was
  confounded with the deterministic deployment rule.
- Status: `SUPPORTED_AS_OBSERVABILITY_LIMITATION` for unseen reuse beyond H.
  The additional claim that exact optimization of the H4 target is
  intrinsically worse than LRU is `NOT_SUPPORTED` after the tie-aware
  control.
- Decisive next experiment: none required for this revision; H11 causal
  attribution remains optional future work.
- Stopping rule: stop treating horizon length alone as sufficient if
  extending H materially improves target resolution but downstream misses do
  not improve when those longer-horizon choices are actually deployed.
- Reviewer relevance: MC1, R3-Issue4.
- Owner of next work: LOCAL first (cheap, existing tooling); WULVER for
  full-scale confirmation.
- Experiment state: strict-preference/horizon diagnostic `FINAL_VALIDATED`;
  reuse-tail diagnostic `LOCAL_COMPLETE`; tie-aware exact-target oracle v1
  `FINAL_VALIDATED` after campaign-CSV recovery.

## H5 -- continuation-policy mismatch

- Statement: labels are constructed assuming LRU continuation after the
  decision, but the deployed policy continues with more learned evictions,
  and that mismatch degrades performance.
- Motivation: standard imitation-learning / DAgger concern (Ross, Gordon,
  Bagnell 2011) -- label-time and deployment-time trajectories diverge.
- Current evidence: the full 7-family/21-cell C0/C1/C2 production campaign
  (`analysis/continuation_policy_causal_ablation_production_v1/`) completed
  and passed formal integrity audit on 2026-08-13 (21/21 units, 63/63 policy
  rows, 21/21 label-agreement rows, 21/21 training-summary rows, all
  integrity gates PASS). C2 (frozen-`pi1` continuation) improves over C1
  (LRU continuation) in 13/21 cells, ties in 3/21 (Wiki2018, degenerate
  100%-miss cells), worsens in 5/21. Macro mean C2−C1 miss-ratio delta
  ≈ −0.0102. Aggregate misses: C0=565126, C1=601569, C2=592970. Strongest
  improvement: `metacdn` (cap32 −0.108, cap64 −0.206). Strongest
  counter-example: `brightkite` cap32 (+0.2433, the single largest effect in
  the table, opposite direction). Full per-cell table:
  `reports/kbs_final_evidence_20260813/c0_continuation_summary.csv`.
- Status: `PARTIALLY_SUPPORTED` (upgraded from single-family `INCONCLUSIVE`
  now that the decisive full-scale test has completed and been audited). The
  stated stopping rule below did not fire (C2 improves over C1 in a majority
  of cells), so continuation mismatch is not ruled out as a contributor --
  but the effect is neither uniform in sign nor stable in magnitude, so no
  claim stronger than partial, family/capacity-dependent (regime-dependent)
  support is warranted.
- Decisive next experiment: none required -- the full 7-family C0/C1/C2
  causal ablation is complete and validated; do not rerun.
- Stopping rule: fired in the "continue" direction -- full-scale C2 improves
  over C1 in the majority of cells (13/21), so continuation mismatch remains
  a live partial contributor rather than being deprioritized. Re-examine only
  if a future protocol change or additional families materially shift this
  majority.
- Reviewer relevance: MC3, R2-Major3, R3-Issue4.
- Owner of next work: none required locally; remaining work is manuscript
  synthesis only.
- Experiment state: smoke `COMPLETE`; full campaign `FINAL_VALIDATED`
  (this workstation, not Wulver) as of 2026-08-13.

## H6 -- state-distribution shift

- Statement: the sequence of cache states visited under the learned policy
  diverges from the states seen during label construction, independent of
  the continuation-policy label semantics per se.
- Motivation: distinguishes "trajectory looks different" from "trajectory
  divergence causes worse outcomes."
- Current evidence: the full 7-family/21-cell distribution-shift completion
  campaign (`analysis/distribution_shift_ablation_v1/`) completed and passed
  formal integrity audit on 2026-08-13 (7/7 folds, 21/21 paired cells, 42/42
  primary rows, 42/42 state-shift rows, 21/21 trajectory rows, all integrity
  gates PASS, including an independent re-run of
  `scripts/experiments/audit_distribution_shift_completion.py` ->
  `COMPLETE_VALID`). The measured state-shift index decreases (improves)
  under DAgger in 16/21 cells, but misses simultaneously worsen in 16/21
  cells (DAgger improves misses in only 2/21, ties in 3/21 degenerate
  Wiki2018 cells). In 13 of 18 informative cells, shift improves while
  misses worsen at the same time -- the dominant pattern by a wide margin.
  Macro mean DAgger−OFF miss-ratio delta ≈ +0.0094 (net worse). Aggregate
  misses: OFF=591604, DAGGER=599537. Full per-cell table:
  `reports/kbs_final_evidence_20260813/distribution_shift_summary.csv`.
- Status: `DISFAVORED` as a "generic shift-index reduction improves misses"
  causal story (upgraded from single-family `INCONCLUSIVE` now that the
  full-scale test confirms the decoupling at 21-cell scale). This does not
  mean state-distribution shift does not exist, or that the shift-reduction
  mechanism itself is broken -- it reliably reduces the measured index. Only
  the assumed link from that reduction to improved downstream performance is
  disfavored under the tested one-step DAgger intervention.
- Decisive next experiment: none required under the current stopping rule
  (fired, see below); a state-shift metric more directly tied to
  miss-relevant cache-content divergence remains a candidate for future work
  if this framing is revisited, but is not needed to close this hypothesis
  as currently scoped.
- Stopping rule: fired -- the seven-family, 21-cell completion confirms
  generic state-shift-index reduction does not correlate with reduced
  misses (13/18 informative cells show the opposite-direction pattern); per
  the pre-registered rule, pivot away from treating generic shift-index
  reduction as informative for performance rather than continuing to invest
  in this framing.
- Reviewer relevance: MC3, R3-Issue4.
- Owner of next work: none required locally; a redesigned metric is future
  work only, not a current blocker.
- Experiment state: distribution-shift completion `FINAL_VALIDATED` as of
  2026-08-13 (this workstation, not Wulver).

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
- Status: `RESOLVED_DIAGNOSTIC` — the validated 21-cell diagnostic found no
  unique H4 winner across the tested cells, so the strict-preference question
  is dominated by H4 degeneracy in this protocol.
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
- Experiment state: 21-cell strict-preference/horizon diagnostic
  `FINAL_VALIDATED`; do not rerun.

## H10 -- horizon should scale with cache capacity / reuse timescale (user hypothesis B)

- Statement: a fixed `H` carries a fixed amount of future information, while
  cache capacity `C` determines how many candidates that information must
  distinguish; effective horizon may need to scale with `C` (conceptually
  `H/C`).
- Motivation: user-proposed; would explain why a single fixed `H` might be
  adequate at small `C` but degenerate at large `C`.
- Current evidence: the Wulver-relayed broad target-degeneracy diagnostic
  shows an empirical capacity trend at H=4: zero-margin pair fraction rises
  from about `0.968` at C=32 to `0.983` at C=64 and `0.991` at C=128, while
  mean optimal-set fraction rises from about `0.984` to `0.991` to `0.995`.
  The local reuse-tail diagnostic shows a matching direction for
  `P(T>4 | resident)`: `0.987377360773` at C=32,
  `0.992891528854` at C=64, and `0.996078682684` at C=128.
- Status: `EMPIRICALLY_STRENGTHENED`, but not established as an `H/C` law.
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
- Current evidence: the first, non-causal resident-candidate pass is
  complete locally in `analysis/reuse_tail_horizon_diagnostic_v1/`; synthesis
  is tracked in `docs/reuse_tail_horizon_diagnostic_v1_synthesis.md`. Across
  21 family-capacity cells, the H=4 window misses nearly all future resident
  reuse signal:
  `P(T>4 | resident)=0.9938544459677984`,
  `P(T>4 | resident, eventually reused)=0.9793302186526528`, and
  never-reused fraction `0.7026792916224847`. The stronger
  `CAUSAL_EXCESS_MISS` question (whether the eviction actually caused an
  avoidable miss) still requires the not-yet-implemented
  minimum-counterfactual-attribution mechanism sketched in the notebook's
  9.7.1.
- Status: `POTENTIAL_CONSEQUENCE_SUPPORTED`; `CAUSAL_EXCESS_MISS` remains
  `UNTESTED`.
- Decisive next experiment: implement minimum-counterfactual attribution for
  `CAUSAL_EXCESS_MISS` only if the project needs causal attribution beyond
  the now-completed observability diagnostic.
- Stopping rule: disfavor truncation as the primary mechanism if most future
  reuses of evicted objects already fall within `H=4` (concentrated in the
  `1-4` bucket); continue investigating if a large fraction fall beyond `H=4`
  (e.g. `>8`) AND a nontrivial fraction of those are shown to be causal
  excess misses, not just potential consequences.
- Reviewer relevance: MC1, MC3, R3-Issue4.
- Owner of next work: LOCAL (potential-consequence bucket, cheap); LOCAL or
  WULVER (causal-excess-miss bucket, needs new implementation, more
  expensive either way).
- Experiment state: first-pass resident reuse-tail diagnostic
  `LOCAL_COMPLETE`; causal excess-miss attribution `NOT_STARTED`.

---

## Refined horizon-adequacy framing (H10/H11 candidate quantities)

Added during the 2026-08-10 finalization pass, based on a literature
synthesis, and updated after the 2026-08-11 reuse-tail diagnostic completed
locally. This section fixes vocabulary for H10/H11 work and distinguishes
the now-measured resident reuse-tail quantity from causal attribution.

**Terminology guardrail:** `H` here is a count of future *requests* (the
`eviction_loss` target looks `H` requests ahead). Do not conflate this with
classical *stack distance* / *reuse distance*, which counts **distinct
pages** between two accesses to the same page. Below, `T` denotes
`next_reuse_time_requests` -- a request-count quantity in the same units as
`H` -- never a distinct-object stack-distance figure.

- **Primary candidate -- `P(T > H | resident)`.** Probability that a
  cache-resident object's next reuse falls beyond the horizon window. This is
  the dimensionally direct quantity (same units as `H`, no normalization
  assumption needed). The 2026-08-11 local diagnostic completed this pass for
  21 family-capacity cells and found H=4 values near one across the board.
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
  `P(T > H)` causally explains the offline/online gap without a
  counterfactual excess-miss analysis (consistent-with is not the same as
  causal-of -- see the potential-consequence vs. causal-excess-miss
  distinction already drawn in H11).
- Reviewer relevance: MC1 (extends H10/H11's next-diagnostic scope).
- Owner of next work: LOCAL (P(T>H) is computable from existing decision
  logs + raw trace, same inputs H11 already identified as sufficient for its
  first pass).
- Experiment state: `P(T>H | resident)` is `LOCAL_COMPLETE`; competing
  normalized quantities and causal attribution remain unimplemented.

## Cross-cutting notes

- No hypothesis above should be treated beyond its measured scope. H2 and
  the original exact-target/tie-break H3/H4 diagnostics rest on one cell
  (`brightkite`, capacity `64`, `H=4`), while the reuse-tail component of
  H4/H10/H11 now spans 21 family-capacity cells. H5 and H6 now also each
  rest on the full 7-family/21-cell completion campaigns
  (`continuation_policy_causal_ablation_production_v1`,
  `distribution_shift_ablation_v1`), both `FINAL_VALIDATED` as of
  2026-08-13 -- no longer single-family (`metacdn`)-only evidence.
  Replication or causal attribution remains the shared precondition for
  upgrading any hypothesis beyond its current evidentiary scope.
- Ranked coherent explanation (see
  `analysis/kbs_local_current_evidence_synthesis_20260810/CURRENT_LOCAL_EVIDENCE_SYNTHESIS.md`
  section 7 for full reasoning, and
  `reports/kbs_final_evidence_20260813/mechanistic_hypothesis_summary.md`
  for the 2026-08-13 update): primarily a **target problem** (H3, and by
  extension H4/H9/H10/H11). The 2026-08-14 tie-aware oracle strengthens H3
  (`fraction_tied_decisions = 1.0` on all 168 tie-aware rows) and shows that
  the deterministic exact-oracle-versus-LRU comparison was tie-confounded,
  so it cannot establish that the target intrinsically loses to LRU.
  Neither full-scale H5 nor H6 campaign contradicts the target-problem
  ranking. Secondarily a **combination** weighted toward target/deployment
  interaction: H5 is `PARTIALLY_SUPPORTED`, H6 is `DISFAVORED`. Pure
  **model-fitting failure** (H2) remains disfavored. The matched common-model
  V2 control does **not** support blaming the eviction-loss training
  objective itself.
- Do not launch `100%` for H1 unless a future, separately justified
  protocol change explicitly reopens the sample-size question.
- `NO_NEW_EXPERIMENT_REQUIRED` for H1-H6 and H9 under current stopping
  rules as of 2026-08-13; only H7, H8, H10, H11 (and any reopening triggered
  by their own stopping rules above) remain candidates for future work.
