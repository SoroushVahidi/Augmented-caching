# Response paragraph bank (results-independent draft, 2026-06-19)

Status: **prepared, not final.** Drafted while the corrected canonical
`cap32_with_sieve_fifo` chunk is running on the current local/cloud machine
(tmux session `kbs_full_policy_comparison_cap32_with_sieve_fifo`, PID
`198488`). This bank was written without touching, stopping, or depending on
that job's outcome. Every paragraph below either (a) uses only evidence that
already exists today, or (b) contains an explicit
`[INSERT FINAL CANONICAL RESULT: Table X / Figure Y]` placeholder for
anything that depends on cap32/cap64/cap128/cap256. **No paragraph below
claims that `evict_value_v1` outperforms LRU, SIEVE, or FIFO-Reinsertion.**
The only completed canonical data point (cap32, no-SIEVE/no-FIFO chunk)
shows the opposite: `evict_value_v1` loses to LRU on 6 of 7 trace families
(-4.84% mean misses), consistent with two earlier non-canonical checks
(+39% on the cap256 validation trace, +11.81% on a cap32/5k sanity check).
Source: `reports/kbs_cap32_policy_comparison_report.md`,
`reports/kbs_revision_evidence_ledger.md` §6/§9.

Source labels follow `reports/kbs_real_reviewer_comments.md` exactly
(AE, R2-MC1-3, R3-Issue1-7, R3-Minor8-9). Cross-reference:
`reports/kbs_complete_reviewer_comment_matrix.md`,
`reports/kbs_response_to_reviewers_skeleton.md`.

---

## 1. AE — end-to-end evaluation concern

> **AE summary**: no end-to-end miss-ratio evaluation; limited
> differentiation from prior learned-caching methods; insufficient baseline
> comparisons; unvalidated fallback; no overhead analysis; replay-horizon
> choice and label-construction scalability need deeper justification.

**Draft paragraph:**

> We thank the Associate Editor for this summary, which we address in detail
> below under each reviewer's specific points. In brief: we have expanded
> our empirical evaluation to include end-to-end online-replay results
> across multiple cache capacities and trace families
> [INSERT FINAL CANONICAL RESULT: Table 3 / Figs. 2-3], added two
> additional modern lightweight baselines (SIEVE and FIFO-Reinsertion) to
> our comparison set, added a quantitative discussion of label-construction
> and inference-time computational overhead (Section X), strengthened our
> differentiation from HALP, and revised the framing of the guarded fallback
> mechanism to match what is currently empirically supported
> [INSERT: validated contribution / demoted to design extension, pending
> decision]. We discuss each point in turn below.

This paragraph is a roll-up. It should be finalized last, after the
Reviewer #2/#3 paragraphs below are no longer placeholders, per
`reports/kbs_response_to_reviewers_skeleton.md`'s own ordering guidance.

---

## 2. R2-MC1 — replay horizon H justification

> Replay horizon H is manually specified; H=4 performs best but no
> principled justification, sensitivity analysis, theory, or adaptive
> mechanism is given. Unclear if optimal H is workload/cache-size/dataset-
> specific.

**Draft paragraph (usable today, no placeholder needed for the core claim):**

> We thank the reviewer for this comment. The replay horizon H was selected
> via an offline ablation across H∈{4, 8, 16}
> (`analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`), where
> H=4 achieved the lowest validation regret both in aggregate and within
> every individual trace family represented in our validation sample
> (brightkite, cloudphysics, metacdn, twemcache, wiki2018 — see
> `reports/kbs_horizon_h4_revision_strategy.md` for the per-family
> breakdown). We offer the following intuition for why shorter horizons
> perform better in our setting: longer replay horizons increasingly mix the
> causal effect of the eviction decision itself with the effect of the
> *continuation policy* used to simulate the remainder of the horizon,
> introducing label noise that a shorter horizon avoids. We acknowledge,
> however, that H remains an empirically tuned hyperparameter rather than
> one derived from a closed-form criterion, and we have added this
> acknowledgment plus the per-family sensitivity table to [Section X /
> Limitations]. We additionally disclose that our validation sample for
> model/horizon selection covered only 5 of 7 trace families (citibike and
> metakv were absent due to an early-stopping artifact in validation-shard
> sampling, ~3.2% of available validation rows); we do not believe this
> changes the qualitative H=4 ranking, but we flag it for transparency
> (`reports/kbs_revision_gap_tracker.md`, "Open methodological caveat").
> [INSERT, IF A LATER PASS ADDS IT: a per-capacity breakdown of the same
> sensitivity analysis.]

---

## 3. R2-MC2 — computational overhead of label construction

> Counterfactual replay labeling for every eviction candidate at every
> full-cache miss is computationally expensive; manuscript lacks discussion
> of preprocessing time, label-generation complexity, training scalability.

**Draft paragraph (usable today):**

> We thank the reviewer for raising this important practical concern.
> Dataset construction for the full 7-trace × 4-capacity × 3-horizon sweep
> took approximately 10 hours and 18 minutes of wall-clock time on a single
> machine, producing 96 GB across 662 shards
> (`reports/kbs_overhead_and_scalability_evidence.md`, Part 1). This cost is
> incurred once, offline, prior to training. Training all 9 resulting model
> configurations (3 horizons × 3 model families) took approximately 7-8
> minutes in aggregate. We have added this quantitative discussion to
> [Section X / Limitations], together with an explicit statement that the
> label-construction cost scales with the number of (trace, capacity,
> horizon) combinations evaluated and is not incurred at deployment/decision
> time, which is governed by the separate, much smaller cost discussed under
> R3-Issue5 below.

---

## 4. R2-MC3 — offline-vs-online metric gap; fallback under-demonstrated

> Results focus on offline metrics (regret, Top-1 agreement) that don't
> necessarily translate to online performance; need more end-to-end evals
> against strong baselines across workloads/capacities/prediction-quality
> regimes; guarded-fallback effectiveness not sufficiently demonstrated.

**Draft paragraph:**

> We thank the reviewer for this central point, which we agree was the
> most significant gap in the original submission. We have run an
> end-to-end, online cache-replay evaluation across [N] trace families and
> 4 cache capacities (32, 64, 128, 256), comparing `evict_value_v1` against
> LRU, SIEVE, FIFO-Reinsertion, and [M] additional baselines
> [INSERT FINAL CANONICAL RESULT: Table 3 / Figs. 2-3]. [INSERT ONE OF: (a)
> "These results show that `evict_value_v1` achieves competitive or
> improved miss ratios on [specific conditions]"; or (b) "These results
> show that `evict_value_v1` does not yet achieve a consistent end-to-end
> advantage over LRU; we discuss the implications for our contribution
> claims in the revised Discussion and have adjusted our title/abstract
> accordingly."] Regarding the guarded fallback mechanism: [INSERT ONE OF:
> (a) a fallback-triggered-vs-disabled ablation with real numbers, retained
> as a contribution; or (b), our current default given the absence of any
> existing fallback ablation
> (`reports/kbs_fallback_revision_strategy.md`): "we have revised the
> manuscript to present the guarded fallback as a candidate design for
> future empirical validation rather than as a demonstrated contribution of
> this work."]

This paragraph cannot be completed honestly until the canonical sweep
(cap64/128/256) finishes and the fallback validate-or-demote decision is
made. The one completed chunk (cap32, no SIEVE/FIFO) already shows
`evict_value_v1` losing to LRU on 6/7 families, which is real evidence
relevant to, but not dispositive of, option (b) above.

---

## 5. R3-Issue2 — insufficient differentiation from HALP

> HALP (NSDI '23) already learns candidate-level preferences from future
> re-access outcomes and evicts the least-preferred candidate — operationally
> very close to this paper's proposal. Needs articulation of what the
> finite-horizon replay target adds, ideally with direct empirical
> comparison.

**Draft paragraph (usable today):**

> We thank the reviewer for pointing us to this closely related system.
> HALP (Song et al., NSDI '23) learns a pairwise preference signal from
> realized future re-access outcomes through continuous online training,
> and evicts the candidate with the lowest learned preference. Our approach
> differs in that the supervision target is an explicit, finite-horizon
> counterfactual replay outcome: for each eviction candidate, we quantify
> the downstream miss-count harm that would result from evicting that
> specific candidate at that specific decision point, under a fixed
> continuation policy, over a bounded horizon H. Where HALP's preference
> signal is implicit and accumulates from realized (not counterfactual)
> future outcomes online, our target is constructed offline and is directly
> interpretable as an estimate of decision-level causal harm. We did not
> attempt a faithful empirical reimplementation of HALP in this revision:
> HALP's online preference-learning loop and its production CDN data
> pipeline differ substantially from our offline replay-based label
> construction, and a faithful reproduction was judged out of scope given
> the revision timeline
> (`reports/kbs_halp_fifo_source_verification.md`). We instead rely on the
> analytical differentiation above and have expanded our Related Work
> discussion accordingly. We agree this is a weaker form of evidence than a
> head-to-head empirical comparison and have stated this limitation
> explicitly in the revised manuscript.

---

## 6. R3-Issue3 — missing comparisons to SIEVE / FIFO-Reinsertion

> Missing comparisons to modern lightweight algorithms: SIEVE (NSDI '24),
> FIFO-Reinsertion, and others.

**Draft paragraph:**

> We thank the reviewer for naming these specific baselines. We have
> implemented both SIEVE (Zhang et al., NSDI '24) and FIFO-Reinsertion (the
> CLOCK / Second-Chance family baseline contrasted against SIEVE in the
> NSDI'24/S3-FIFO literature) and added both to our canonical policy
> comparison. Our SIEVE implementation follows the published Algorithm 1
> exactly (single FIFO queue, one hand pointer, one visited bit per object,
> verified line-for-line against the official `libCacheSim` reference
> implementation; `reports/kbs_sieve_source_verification.md`), and is
> covered by 8 passing unit tests
> (`reports/kbs_sieve_implementation_report.md`). Our FIFO-Reinsertion
> implementation is likewise source-verified and covered by 39 passing
> targeted tests (`reports/kbs_fifo_reinsertion_baseline_audit.md`).
> [INSERT FINAL CANONICAL RESULT: Table 2 policy roster row + Table 3
> results columns for SIEVE and FIFO-Reinsertion, once the cap32-128-256
> sweep with both baselines included has completed.] At the time of this
> response, a corrected canonical capacity-32 comparison including both
> SIEVE and FIFO-Reinsertion is running; we will report full results across
> all four capacities in the camera-ready/final revision.

---

## 7. R3-Issue5 — computational cost not addressed

> Computational cost not addressed: per-miss feature construction + model
> inference for every cached item; no wall-clock analysis, no complexity
> comparison to O(1) baselines (LRU, SIEVE), no production-overhead
> discussion.

**Draft paragraph (usable today):**

> We thank the reviewer for this point. `evict_value_v1` performs O(capacity)
> model-inference calls per cache miss — one feature-vector construction and
> one prediction per resident candidate
> (`src/lafc/policies/evict_value_v1.py:179-200`) — compared to O(1) for LRU
> (a single `OrderedDict.popitem` call) and O(1) amortized for SIEVE (a
> single hand-pointer advance per miss, by design). We report this
> asymptotic comparison explicitly in the revised manuscript. We additionally
> report the measured wall-clock cost of our existing experiment chunks
> as a coarse anchor [INSERT: specific numbers from
> `reports/kbs_overhead_and_scalability_evidence.md` Part 2 once finalized],
> while noting that these chunk-level numbers are confounded by differing
> trace mixes and policy counts and so should not be read as a clean
> per-decision latency benchmark. [INSERT, IF RUN: a controlled
> single-trace, multi-capacity timing benchmark isolating per-decision
> latency for `evict_value_v1` vs. LRU vs. SIEVE.] We have added a discussion
> of this overhead, and of the resulting practical trade-off against O(1)
> baselines, to [Section X].

---

## 8. R3-Issue6 — fallback mechanism unvalidated and oversold

> The fallback mechanism is unvalidated and oversold as a contribution; if
> not empirically validated it should not be listed as a main contribution.

**Draft paragraph (conservative default; usable today):**

> We thank the reviewer for this direct and fair criticism. On review, we
> found no empirical ablation anywhere in our experiment history that
> isolates the guarded fallback mechanism's effect on end-to-end
> performance; the only adjacent evidence we have (an earlier, differently
> implemented guarded-policy ablation on synthetic traces at small
> capacities) found every guarded variant tested performed worse than its
> unguarded counterpart. We agree with the reviewer that this is
> insufficient grounds to claim the fallback mechanism as a validated
> contribution. We have revised the manuscript to present the guarded
> fallback as a candidate design extension proposed for future empirical
> validation, removed it from the numbered list of contributions, and added
> an explicit statement in Limitations that it has not been evaluated
> end-to-end in this work (`reports/kbs_fallback_revision_strategy.md`).
> [INSERT, ONLY IF A LATER DECISION REVERSES THIS: a dedicated
> fallback-triggered-vs-disabled ablation with real numbers, restoring it as
> a contribution.]

---

## 9. R3-Issue7 — single-author / AI-tool validation concern

> Single authorship combined with reliance on AI tools raises questions
> about validation depth, code correctness, result verification.
> Multi-institutional or multi-author validation would substantially
> strengthen credibility.

**Draft paragraph (usable today, candid framing — no overclaiming):**

> We thank the reviewer for raising this concern directly, and we want to
> respond candidly rather than defensively. This work was conducted by a
> single author with assistance from AI coding tools during implementation.
> We did not have access to multi-author or multi-institutional review
> during development. To partially mitigate the resulting risk of
> undetected errors, we relied on the following concrete validation steps,
> which we now describe explicitly in the manuscript: (1) a unit and
> integration test suite covering policy implementations and the evaluation
> pipeline (291 tests passing at time of writing); (2) an independent
> sanity-audit pass that cross-checked our most surprising empirical
> finding — that `evict_value_v1` underperforms LRU on most workloads —
> against a second, independently generated data point before accepting it
> as real rather than a wiring bug (`reports/kbs_validation_sanity_audit.md`);
> and (3) a platform-consistency audit confirming that our pipeline is fully
> seeded and deterministic, and that hardware claims in the manuscript are
> accurately scoped (`reports/kbs_platform_consistency_audit.md`). We do not
> claim these steps are equivalent to independent multi-author review, and
> we have stated this limitation explicitly rather than implying a level of
> external validation that did not occur.

---

## 10. R3-Minor8 — shortening/repetition

> Excessive verbosity/repetition: Sections 1.1/1.2 overlap substantially;
> Sections 3.4 and 4.1-4.2 repeat methodological caution many times.

**Draft paragraph:**

> We thank the reviewer for this observation and agree the manuscript was
> too repetitive in its original form. We have consolidated Sections 1.1
> and 1.2 to remove overlapping content, and reduced the repeated
> methodological-caution language that previously appeared in both Section
> 3.4 and Sections 4.1-4.2 to a single, clearly stated instance (see our
> cut list and rationale in
> `reports/kbs_manuscript_shortening_and_reframing_plan.md`). In total, this
> and related edits reduce the manuscript by approximately
> [INSERT: final word count once the editing pass is executed; target is a
> 30-40% reduction from the original ~8,919 words per
> `reports/kbs_manuscript_shortening_and_reframing_plan.md`].

---

## 11. R3-Minor9 — missing workload-specific breakdowns

> Given workload dependence in caching, aggregate-only results would be
> insufficient.

**Draft paragraph:**

> We thank the reviewer for this point, which we agree is important given
> the well-known workload-dependence of caching policies. We report
> per-trace-family results, not just aggregates, throughout our revised
> evaluation. For the one capacity chunk completed at the time of this
> response (capacity 32, the prior policy set without SIEVE/FIFO-
> Reinsertion), per-family results are: `evict_value_v1` underperforms LRU
> on 6 of 7 families, with the gap ranging from +0.7% more misses
> (cloudphysics) to +22.4% more misses (metacdn), and ties with LRU on
> wiki2018 (a 100%-miss-floor trace for every policy tested); aggregate mean
> misses are 4.84% higher than LRU
> (`reports/kbs_cap32_policy_comparison_report.md`). [INSERT FINAL CANONICAL
> RESULT: the equivalent per-family breakdown at capacities 64, 128, and 256,
> and with SIEVE/FIFO-Reinsertion included, once the full sweep completes.]
> We do not present an aggregate-only headline number anywhere in the
> revised manuscript without an accompanying per-family table.

---

## Notes on tone and what NOT to write

- Do not write "our results demonstrate that `evict_value_v1` is an
  effective/robust eviction policy" anywhere in this bank. The only
  completed canonical data point says the opposite on 6/7 families.
- Do not state or imply that the canonical sweep (cap64/128/256) is
  complete, in progress beyond cap32, or that its outcome is known. As of
  this draft, only the cap32 chunk (no SIEVE/FIFO) is finished; a corrected
  cap32-with-SIEVE-and-FIFO-Reinsertion chunk is running but not finished.
- Every `[INSERT ...]` placeholder above must be filled with the actual
  result, not a hoped-for one, when the corresponding evidence exists.
