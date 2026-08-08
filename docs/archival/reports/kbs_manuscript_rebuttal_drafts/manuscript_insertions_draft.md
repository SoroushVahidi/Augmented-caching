# Manuscript insertion draft text (results-independent, 2026-06-19)

Status: **prepared, not final.** Drafted while `cap32_with_sieve_fifo` runs
(see `reports/kbs_manuscript_rebuttal_drafts/response_paragraph_bank.md` for
the corresponding rebuttal-letter text and full evidence citations). This
file drafts manuscript *body* text, not response-letter text. Wording is
deliberately cautious throughout: cap32 evidence (the one completed
canonical chunk) shows `evict_value_v1` losing to LRU on 6 of 7 trace
families (-4.84% mean misses;
`reports/kbs_cap32_policy_comparison_report.md`), consistent with two
earlier non-canonical checks pointing the same direction. **No subsection
below asserts that `evict_value_v1` outperforms LRU, SIEVE, or
FIFO-Reinsertion.** All capacity/sweep-dependent numbers are placeholders.

---

## 1. "End-to-end policy evaluation"

> While the offline ablation in Section X establishes that our learned
> supervision target is internally well-calibrated (low validation regret,
> high Top-1 agreement with the counterfactual replay oracle), offline
> target-quality metrics do not by themselves establish that the resulting
> policy improves end-to-end cache performance. We therefore evaluate
> `evict_value_v1` directly against LRU and a set of modern and classical
> baselines in an online cache-replay setting, across [N] trace families
> spanning [domains, e.g., object storage, CDN, key-value, block storage]
> and 4 cache capacities (32, 64, 128, 256 objects), measuring cache miss
> count as our primary metric.
>
> [INSERT FINAL CANONICAL RESULT: Table 3 summary statistics and Figs. 2-3.]
>
> At capacity 32, evaluated against the policy set
> {LRU, predictive\_marker, blind\_oracle\_lru\_combiner, trust\_and\_doubt,
> rest\_v1, `evict_value_v1`} prior to the inclusion of SIEVE and
> FIFO-Reinsertion, `evict_value_v1` recorded more cache misses than LRU on
> 6 of 7 trace families, with a mean increase of 4.84% in total misses
> (range: +0.7% to +22.4%; one trace, wiki2018, was at a 100%-miss floor for
> every policy and is uninformative for comparison purposes). We report this
> result directly rather than omitting or reframing it, and we discuss its
> implications for our contribution claims in Section "Limitations" below.
> [INSERT, ONCE AVAILABLE: the equivalent results at capacities 64, 128, and
> 256, and with SIEVE/FIFO-Reinsertion included in the comparison set.]

---

## 2. "Baselines"

> In addition to LRU, we compare against [predictive\_marker,
> blind\_oracle\_lru\_combiner, trust\_and\_doubt, rest\_v1 — existing
> short descriptions], and, in this revision, two additional modern
> lightweight eviction algorithms identified by reviewers as notably absent
> from our original baseline set:
>
> - **SIEVE** (Zhang et al., NSDI '24): a single-FIFO-queue algorithm with
>   one "hand" pointer and one visited bit per cached object. On a cache
>   hit, the visited bit is set without moving the object; on a miss, the
>   hand scans backward from its current position, clearing visited bits,
>   until it finds and evicts the first unvisited object. We implemented
>   SIEVE following the published Algorithm 1 exactly, cross-checked
>   line-for-line against the official reference implementation in
>   `libCacheSim` (`reports/kbs_sieve_source_verification.md`).
> - **FIFO-Reinsertion**: a CLOCK / Second-Chance-family baseline, in which
>   a cache hit marks the accessed object as visited without moving it, and
>   a cache miss inspects candidates in FIFO order, giving each visited
>   object a second chance (clearing its bit and reinserting it at the head)
>   before evicting the first unvisited candidate encountered
>   (`reports/kbs_halp_fifo_source_verification.md`).
>
> Both are wired into our canonical evaluation pipeline and are included in
> the policy roster in Table 2. We additionally discuss, but do not
> empirically reproduce, HALP (Song et al., NSDI '23), a closely related
> learned eviction policy; see "Relation to HALP" below for our rationale.
> [INSERT FINAL CANONICAL RESULT: Table 3 columns/rows for SIEVE and
> FIFO-Reinsertion, once the canonical multi-capacity sweep including both
> completes.]

---

## 3. "Computational overhead and scalability"

> We separate the computational cost of our approach into an offline,
> one-time label-construction-and-training cost, and an online,
> per-decision inference cost.
>
> **Offline cost.** Constructing the counterfactual replay labels for our
> full 7-trace × 4-capacity × 3-horizon sweep took approximately 10 hours
> and 18 minutes of wall-clock time on a single machine, producing 96 GB of
> labeled data across 662 shards. Training all 9 resulting model
> configurations (3 candidate horizons × 3 model families) took
> approximately 7-8 minutes in aggregate. This cost is incurred once per
> dataset/horizon configuration and is independent of deployment.
>
> **Online cost.** At decision time, `evict_value_v1` performs O(capacity)
> model-inference calls per cache miss: one feature-vector construction and
> one learned-loss prediction per resident candidate
> (`src/lafc/policies/evict_value_v1.py:179-200`). This contrasts with O(1)
> per-miss cost for LRU (a single doubly-linked-list/`OrderedDict` update)
> and O(1) amortized per-miss cost for SIEVE (a single hand-pointer
> advance, by design). This is a genuine practical trade-off: our approach
> trades a larger, capacity-dependent per-decision cost for a
> data-driven eviction signal. [INSERT, IF RUN: a controlled, single-trace,
> multi-capacity timing benchmark reporting wall-clock per-decision latency
> for `evict_value_v1` vs. LRU vs. SIEVE, isolating this cost from
> trace-mix and policy-count confounds present in our existing chunk-level
> timings.] We consider a tighter, production-oriented characterization of
> this overhead — e.g., approximate or cached inference, or restricting
> scoring to a bounded candidate subset rather than the full resident set —
> an important direction for future work rather than a solved problem in
> this paper.

---

## 4. "Horizon selection"

> Our supervision target requires choosing a finite replay horizon H over
> which counterfactual eviction harm is measured. We selected H via an
> offline ablation across H∈{4, 8, 16}, using validation regret as the
> selection criterion. H=4 achieved the lowest validation regret both in
> aggregate and within every individual trace family represented in our
> validation sample (brightkite, cloudphysics, metacdn, twemcache,
> wiki2018), with validation regret increasing monotonically from H=4 to
> H=8 to H=16 in every case.
>
> We offer the following intuition for this pattern: as H grows, the
> simulated continuation period over which we measure an eviction
> candidate's downstream harm increasingly reflects the behavior of the
> *continuation policy* used to simulate the remainder of the trace, rather
> than the causal effect of the eviction decision itself. Shorter horizons
> reduce this source of label noise at the cost of a more myopic
> supervision signal. We treat H as an empirically tuned hyperparameter
> rather than one derived from a closed-form criterion, and we report this
> ablation explicitly rather than presenting H=4 as a fixed design choice
> without justification.
>
> We disclose a limitation in this analysis: due to an early-stopping
> artifact in our validation-shard sampling procedure, our validation sample
> for model/horizon selection covered only 5 of the 7 trace families in our
> dataset (citibike and metakv were absent, representing roughly 3.2% of
> available validation rows being used overall). We do not have evidence
> that this changes the qualitative ranking of H, but we have not verified
> it on the missing two families and report this honestly rather than
> implying full-family validation coverage. [INSERT, IF A LATER PASS ADDS
> IT: a per-capacity breakdown of this same sensitivity analysis; currently
> only an aggregate-and-per-family-at-fixed-capacity analysis exists.]

---

## 5. "Relation to HALP"

> HALP (Song et al., NSDI '23) is, to our knowledge, the closest prior
> learned eviction policy to our approach. HALP learns a pairwise preference
> signal over cached objects from realized future re-access outcomes through
> continuous online training, and evicts the object with the lowest learned
> preference at each decision point. Our approach differs along two axes.
> First, the supervision target: where HALP's preference signal accumulates
> from *realized* future outcomes observed online, our target is an
> explicit *counterfactual* replay outcome — for each eviction candidate, we
> estimate the downstream miss-count harm that specific eviction would cause
> under a fixed continuation policy, over a bounded horizon. Second, the
> training regime: HALP trains continuously online; we train offline from a
> pre-constructed labeled dataset and deploy a frozen model.
>
> We did not attempt a faithful empirical reimplementation of HALP in this
> work. HALP's online preference-learning loop, and the production CDN
> access-pattern data it was originally trained and evaluated on, differ
> substantially from our offline, replay-based label-construction pipeline;
> building a faithful reimplementation was judged out of scope given our
> revision timeline. We rely instead on the analytical differentiation
> above, and we explicitly note this as a limitation: an empirical
> head-to-head comparison against HALP would be stronger evidence of
> differentiation than the conceptual argument we provide, and we leave it
> to future work.

---

## 6. "Fallback mechanism and limitations"

> We originally described a guarded fallback extension to `evict_value_v1`,
> intended to revert to LRU-like behavior when the learned model's
> predictions are judged unreliable for a given decision. On review, we
> found no empirical ablation in our experiment history that isolates this
> mechanism's effect on end-to-end cache performance; the only adjacent
> evidence available to us is a differently implemented guarded-policy
> ablation, run previously on synthetic traces at small cache capacities,
> in which every guarded variant tested performed worse than its unguarded
> counterpart. Given the absence of supporting evidence, and the additional
> consideration that the underlying unguarded `evict_value_v1` policy itself
> currently underperforms LRU on most evaluated workloads (Section
> "End-to-end policy evaluation"), we do not believe it would be honest to
> present this fallback as a validated contribution of this paper. We
> therefore describe it here as a candidate design extension proposed for
> future empirical validation, rather than as a demonstrated component of
> our method, and we have removed it from our list of contributions
> accordingly. [INSERT, ONLY IF A LATER DECISION REVERSES THIS: a dedicated
> fallback-triggered-vs-disabled ablation with real numbers.]

---

## 7. "Reproducibility and validation"

> This work was conducted by a single author, with assistance from AI
> coding tools during implementation, and did not have access to
> multi-author or multi-institutional review during development. We took
> the following concrete steps to mitigate the resulting risk of undetected
> implementation or analysis errors, and we describe them here rather than
> asserting correctness without support: (1) a unit and integration test
> suite covering all policy implementations and the evaluation pipeline
> (291 tests passing at the time of writing, including 8 new tests for
> SIEVE and 39 targeted tests covering FIFO-Reinsertion); (2) an independent
> sanity-audit pass triggered by our most counter-intuitive empirical
> finding — that `evict_value_v1` underperforms LRU on most evaluated
> workloads — in which we reproduced the finding on an independent,
> differently sized data point before accepting it as real rather than
> attributing it to an implementation bug; and (3) a platform-consistency
> audit confirming our evaluation pipeline is fully seeded and
> deterministic, and that hardware-dependent claims in this manuscript are
> scoped accurately. All code, configuration, and the evidence artifacts
> referenced throughout this manuscript are available at [repository URL].
> We do not present these steps as equivalent to independent peer
> validation, and we encourage readers to treat single-author, AI-assisted
> empirical work with appropriate scrutiny.

---

## 8. "Limitations"

> We highlight several limitations of this work, stated directly rather
> than as closing hedges.
>
> First, our primary empirical finding to date is that `evict_value_v1`,
> evaluated end-to-end, does not consistently outperform LRU: at cache
> capacity 32, the one fully completed chunk of our canonical sweep, it
> records more total misses than LRU on 6 of 7 evaluated trace families
> (mean +4.84%). [INSERT FINAL CANONICAL RESULT: whether this pattern
> persists, narrows, or reverses at capacities 64, 128, and 256, and how
> `evict_value_v1` compares against SIEVE and FIFO-Reinsertion specifically,
> once the full sweep completes.] We discuss this directly rather than
> omitting or downweighting it, and we believe an honest negative or mixed
> result is more valuable to the community than an overstated positive one.
>
> Second, our guarded fallback mechanism is, as described above, an
> unvalidated design extension rather than a demonstrated contribution.
>
> Third, our differentiation from HALP is analytical rather than empirical;
> we did not reproduce HALP's online preference-learning approach in this
> work.
>
> Fourth, our offline model/horizon selection used a validation sample
> covering only 5 of 7 trace families due to a sampling artifact, and our
> reported computational-overhead numbers include measured offline costs
> alongside a code-verified (but not yet independently wall-clock-measured
> in a controlled setting) online complexity comparison.
>
> Finally, this is single-author, AI-assisted work without independent
> multi-institutional validation; we describe the validation steps we did
> take in "Reproducibility and validation" above, and we encourage scrutiny
> of our released code and data accordingly.

---

## Notes on tone and what NOT to write

- Do not claim "robust" performance, "demonstrates superiority," or similar
  language anywhere above — these run directly counter to the only
  completed canonical evidence.
- Every numeric claim about capacities beyond 32, or about SIEVE/
  FIFO-Reinsertion canonical results, must remain an `[INSERT ...]`
  placeholder until the corresponding run finishes and is verified, per
  `reports/kbs_revision_evidence_ledger.md`.
- This draft intentionally pre-commits to disclosing a negative result
  (Section "Limitations", first bullet) rather than leaving room to spin it
  positively once final numbers arrive — only the *degree* of the gap is a
  placeholder, not the acknowledgment that a gap currently exists.
