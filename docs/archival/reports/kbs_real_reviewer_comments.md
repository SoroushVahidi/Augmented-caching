# KBS real reviewer comments — canonical source (2026-06-19)

Manuscript: "Decision-aligned eviction-value prediction for robust
learning-augmented caching" (KNOSYS-D-26-07461). Decision: Revise.
Due: 2026-07-08.

**Status: this is now the authoritative reviewer text.** It supersedes the
12-13 item category-level paraphrase used throughout this repo's prior
planning reports (`kbs_revision_completion_audit.md` §B,
`kbs_revision_gap_tracker.md`, the earlier `kbs_complete_reviewer_comment_matrix.md`
and `kbs_response_to_reviewers_skeleton.md`) and the "missing/blocked"
framing used in `reports/need_full_kbs_reviewer_comments.md` (on the
unmerged `kbs-revision-parallel-cleanup` branch). Those documents are being
updated to point here rather than restate "comments missing."

Pasted as supplied, preserving the AE / Reviewer #2 / Reviewer #3
structure and labels.

---

## Associate Editor

The reviewers appreciate the coherent framing but raise major concerns
regarding the lack of end-to-end miss ratio evaluation, limited
differentiation from prior learned caching methods, insufficient baseline
comparisons, unvalidated fallback design, and missing computational
overhead analysis. The choice of replay horizon and scalability of label
construction also require deeper justification.

## Reviewer #2

### Major Comment 1 (R2-MC1)

The proposed framework relies heavily on a manually specified replay
horizon (H) when constructing the eviction-value target. While the
experimental results suggest that (H=4) performs best among the evaluated
settings, the manuscript does not provide a principled justification for
this choice. It remains unclear whether the optimal horizon is
workload-dependent, cache-size-dependent, or specific to the evaluated
datasets. I encourage the author to provide a more thorough sensitivity
analysis, theoretical intuition, or an adaptive mechanism for horizon
selection. Without such evidence, it is difficult to assess whether the
reported performance reflects a generally effective formulation or merely a
favorable choice of hyper-parameter.

### Major Comment 2 (R2-MC2)

The proposed supervision target requires counter-factual replay for every
eviction candidate at each full-cache miss. Although conceptually
appealing, this labeling procedure appears computationally expensive. The
manuscript currently focuses on predictive performance but provides little
discussion regarding the computational overhead associated with dataset
construction, training scalability, or preprocessing time. I believe a
quantitative analysis of the offline cost is necessary. Reporting
preprocessing time, label-generation complexity, and scalability
experiments would substantially strengthen the paper.

### Major Comment 3 (R2-MC3)

The paper claims to improve cache replacement decisions, but the reported
results focus primarily on offline metrics (e.g., regret, Top-1
agreement). While these metrics are useful for model selection, they do not
necessarily translate into strong online performance. Conduct more
end-to-end evaluations, compare against strong baselines, and test across
different workloads, cache capacities, and prediction-quality regimes.
Additionally, the effectiveness of the "guarded fallback mechanism" has not
been sufficiently demonstrated. As currently presented, the paper shows
that the proposed supervision target is useful, but fails to sufficiently
demonstrate that the overall caching policy is practically superior.

## Reviewer #3

### Summary (R3-Summary)

This paper proposes a candidate-level eviction-value prediction framework
for learning-augmented caching, where at each full-cache miss the policy
evaluates all cached items and evicts the one with the smallest predicted
finite-horizon downstream harm. The method is instantiated in the
unweighted paging setting with a learned scorer (trained on
replay-generated labels) and an optional lightweight fallback mechanism
triggered by early-return eviction detection. The offline ablation shows
that nonlinear tree-based models and shorter replay horizons (H=4)
outperform linear baselines on target-quality metrics. While the
methodological framing is coherent, the paper suffers from a fundamental
disconnect between its claimed contributions and the evidence provided: the
empirical results are almost entirely offline target-quality metrics rather
than end-to-end cache miss ratio improvements. The extensive hedging
language further weakens the perceived contribution.

### Issue 1 (R3-Issue1) — No end-to-end cache miss ratio results

The paper limits its empirical evidence to offline target-quality metrics.
For a caching paper, the primary evaluation metric must be downstream cache
performance—miss ratios or hit rates compared against strong baselines.

### Issue 2 (R3-Issue2) — Insufficient differentiation from existing learned caching methods

HALP (NSDI '23) already learns candidate-level preferences from future
re-access outcomes and evicts the least-preferred candidate—operationally
very close to what this paper proposes. The paper needs to clearly
articulate what the finite-horizon replay target provides that HALP's
preference signal does not, and ideally present direct empirical
comparisons.

### Issue 3 (R3-Issue3) — Missing comparisons to modern lightweight algorithms

Notably absent are SIEVE (NSDI '24), FIFO-Reinsertion, and other recently
proposed lightweight algorithms that achieve strong performance with
minimal overhead.

### Issue 4 (R3-Issue4) — Self-admitted empirical weakness undermines the contribution

The title promises robust learning-augmented caching, but the paper itself
admits it has not demonstrated robust end-to-end performance.

### Issue 5 (R3-Issue5) — Computational cost is not addressed

The method requires constructing a feature vector for every cached item at
every full-cache miss, then running a trained model to predict losses for
all candidates. The paper provides no wall-clock time analysis, no
complexity comparison against O(1) baselines like LRU or SIEVE, and no
discussion of production overhead.

### Issue 6 (R3-Issue6) — Fallback mechanism is unvalidated and oversold as a contribution

If the fallback is not empirically validated, it should not be listed as a
main contribution.

### Issue 7 (R3-Issue7) — Single authorship combined with reliance on AI tools

Single authorship combined with reliance on AI tools raises questions about
validation depth, code correctness, and result verification.
Multi-institutional or multi-author validation would substantially
strengthen credibility.

### Minor Problem 8 (R3-Minor8) — Excessive verbosity and repetition

Sections 1.1 and 1.2 overlap substantially. Sections 3.4 and 4.1–4.2 repeat
methodological caution many times.

### Minor Problem 9 (R3-Minor9) — Missing workload-specific analysis

Given workload dependence in caching, aggregate-only results would be
insufficient.

### Recommended Revisions

1. (R3-Rec1) Provide end-to-end miss ratio results.
2. (R3-Rec2) Add direct comparisons to HALP and SIEVE.
3. (R3-Rec3) Report computational overhead.
4. (R3-Rec4) Reduce hedging or reframe the paper's scope.
5. (R3-Rec5) Validate the fallback mechanism or remove it.
6. (R3-Rec6) Shorten the manuscript by 30–40%.
7. (R3-Rec7) Investigate why H=4 works best.
8. (R3-Rec8) Provide workload-specific breakdowns.

---

## Cross-reference note

This text was supplied directly by the user in this session via the chat
interface (not retrieved from Editorial Manager by this agent, and not
independently verified against an original PDF/email by this agent). It is
being treated as authoritative per explicit instruction. If a discrepancy
with the actual Editorial Manager letter is ever found, this file should be
corrected and every report that cites it (`kbs_complete_reviewer_comment_matrix.md`,
`kbs_response_to_reviewers_skeleton.md`, `kbs_revision_completion_audit.md`,
`kbs_revision_gap_tracker.md`, `kbs_before_cap64_baseline_decision_memo.md`)
should be re-checked.
