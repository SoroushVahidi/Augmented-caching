# Response to Reviewers

Manuscript: KNOSYS-D-26-07461  
Decision: Revise  
Date: 2026-06-21

---

## Response to the Associate Editor

> We thank the Associate Editor and both reviewers for their detailed and
> constructive feedback. We address each concern below, organized by
> reviewer. In this revision we have: added end-to-end online cache-replay
> results across three cache capacities (32, 64, 128 slots) and all seven
> trace families, honestly reporting that the proposed policy does not
> outperform LRU, SIEVE, or FIFO-Reinsertion in this evaluation and that
> its gap widens non-monotonically at the largest evaluated capacity;
> added SIEVE and FIFO-Reinsertion as additional baselines across the same
> three capacities; added a quantitative discussion of computational
> overhead; clarified our differentiation from HALP; revised our treatment
> of the guarded fallback mechanism to match its current evidentiary
> status; reduced hedging/length where possible; and added a candid
> discussion of single-author validation practices, including AI-tool use.
> A fourth capacity (256 slots) was not evaluated in this revision and is
> not claimed anywhere in this revision; our end-to-end claims are limited
> to capacities 32, 64, and 128.

---

## Response to Reviewer #2

### Major Comment 1 (R2-MC1) — replay horizon H justification

> We thank the reviewer for this comment. The replay horizon H was
> selected via an offline ablation across H∈{4,8,16}, where H=4 achieved
> the lowest validation regret. We have added a new manuscript subsection
> ("Replay Horizon Selection") reporting that this finding holds across 5
> of 7 trace families (BrightKite, CloudPhysics, MetaCDN, Twemcache,
> Wikimedia pageviews), and explicitly disclosing that Citi Bike/MetaKV
> coverage and horizon × capacity interaction remain open questions for
> future work.

### Major Comment 2 (R2-MC2) — computational overhead of label construction

> We thank the reviewer for raising this. We have added a new manuscript
> subsection ("Overhead and Scalability") reporting that dataset
> construction for the full sweep took approximately 10.3 hours of
> wall-clock time, producing 96GB across 662 shards, while training all
> model configurations took approximately 7-8 minutes. We have also added a
> controlled wall-clock latency benchmark on a BrightKite prefix of 5,000
> requests at capacities 32/64/128, measured on the author's local
> development machine under tmux rather than on Wulver/Slurm: the mean
> per-decision cost of `evict_value_v1` increases from 75.0 ms at capacity
> 32 to 316.0 ms at capacity 128, while LRU, SIEVE, FIFO-Reinsertion, and
> REST remain several orders of magnitude faster (see Issue 5 below). We
> present this as a bounded overhead probe, not a claim of representativeness
> across all seven trace families.

### Major Comment 3 (R2-MC3) — offline-vs-online metric gap; fallback under-demonstrated

> We thank the reviewer for this important point. We have expanded our
> evaluation to include end-to-end online-replay results across three
> cache capacities (32, 64, 128 slots) and all seven trace families. The
> result is honestly mixed: `evict_value_v1` does not outperform LRU,
> SIEVE, or FIFO-Reinsertion at any of the three evaluated capacities, and
> its gap relative to these baselines widens non-monotonically at capacity
> 128 (Section "End-to-End Online Replay Evaluation Across Available
> Capacities"). We analyze this pattern in the Discussion and Limitations
> sections and link it to the single-step LRU-continuation rule used to
> construct the offline supervision target. Regarding the guarded fallback
> mechanism, we have revised its framing throughout the manuscript from a
> validated contribution to an implementation safeguard/optional guard. A
> dedicated fallback-triggered-vs-disabled ablation has not been run in this
> revision, so the mechanism is not presented as an empirically validated
> robustness contribution; see Issue 6 below.

---

## Response to Reviewer #3

### Summary (R3-Summary)

> We thank the reviewer for this thorough assessment. We agree that the
> original submission's empirical evidence was concentrated in offline
> target-quality metrics. In this revision we have added end-to-end
> miss-ratio results across three cache capacities (32, 64, 128 slots) and
> all seven trace families, added SIEVE/FIFO-Reinsertion baseline
> comparisons at the same three capacities, and reduced hedging language
> where the evidence now supports doing so. We have also added new hedging
> where the end-to-end evidence is honestly negative: `evict_value_v1` does
> not outperform LRU, SIEVE, or FIFO-Reinsertion in this evaluation, and we
> now describe the method as a decision-aligned supervision-target study
> rather than as a demonstrated improvement over strong lightweight
> baselines. The remaining manuscript-level editing work is a shortening and
> packaging pass, not a missing-capacity replay claim.

### Issue 1 (R3-Issue1) — no end-to-end cache miss ratio results

> We have added end-to-end online-replay results across all seven trace
> families at three cache capacities (32, 64, 128 slots; Section "End-to-
> End Online Replay Evaluation Across Available Capacities"). The
> result is that `evict_value_v1` does not achieve a lower mean miss count
> than LRU, SIEVE, or FIFO-Reinsertion at any of the three capacities (gap
> vs LRU: +4.84%, +2.26%, +11.76% at cap32/64/128 respectively, where
> positive means more misses). We report this directly rather than only
> the offline target-quality metrics from the original submission, and we
> discuss the non-monotonic widening at capacity 128 in the Discussion and
> Limitations sections. REST is numerically tied with LRU on the three-
> capacity average, so `evict_value_v1` also does not outperform REST on
> average. Capacity 256 was not evaluated end-to-end and is not included in
> this result.

### Issue 2 (R3-Issue2) — insufficient differentiation from HALP

> We thank the reviewer for this comparison. Unlike HALP (Song et al.,
> NSDI '23), which learns a pairwise preference signal from realized future
> re-access outcomes, our finite-horizon replay target explicitly
> quantifies downstream miss-count harm under counterfactual eviction at
> each decision point. A faithful empirical HALP reimplementation is
> outside the scope of this revision; the analytical differentiation above
> stands in its place.

### Issue 3 (R3-Issue3) — missing comparisons to SIEVE / FIFO-Reinsertion

> We have added SIEVE and FIFO-Reinsertion as additional baselines,
> implemented and tested, and now included in our policy roster (Table 2)
> and Related Work discussion. Both are evaluated end-to-end at all three
> capacities alongside the rest of the roster. At capacity 128,
> `evict_value_v1`'s gap is +9.13% vs SIEVE and +11.64% vs FIFO-Reinsertion
> (mean misses, higher is worse); both baselines also outperform
> `evict_value_v1` at cap32 and cap64. SIEVE and FIFO-Reinsertion themselves
> remain close to LRU across all three capacities, consistent with their
> design as strong, low-overhead baselines.

### Issue 4 (R3-Issue4) — self-admitted empirical weakness undermines the contribution

> We thank the reviewer for this candid assessment, and we agree with its
> substance. Rather than expanding evidence to support the original framing,
> we have revised the abstract, Summary of Findings, and Discussion to more
> accurately reflect what is empirically demonstrated: finite-horizon
> eviction-value prediction is a well-motivated and learnable supervision-
> target design, and the offline ablation supports this as a research
> contribution in its own right, but the new end-to-end online evaluation
> (Issue 1 above) shows that the present instantiation of `evict_value_v1`
> is not yet a practically superior online replacement policy relative to
> LRU, SIEVE, or FIFO-Reinsertion. We now describe the contribution
> throughout as a decision-aligned supervision study rather than as a
> demonstrated improvement in online cache miss ratio, and we no longer
> claim or imply superiority over these baselines anywhere in the
> manuscript.

### Issue 5 (R3-Issue5) — computational cost not addressed

> `evict_value_v1` performs O(capacity) model-inference calls per cache
> miss (one feature vector + prediction per resident candidate), compared
> to O(1) for LRU, SIEVE, and FIFO-Reinsertion. This complexity comparison
> is code-verified. We have also run a controlled wall-clock timing
> benchmark confirming this asymptotic cost empirically: on a 5,000-request
> BrightKite prefix at capacities 32/64/128 (local machine under tmux, not
> Wulver/Slurm), `evict_value_v1`'s mean per-decision cost rises from
> 75.0 ms to 316.0 ms, versus approximately 0.001 ms for LRU/
> FIFO-Reinsertion, 0.002-0.005 ms for SIEVE, and 0.04-0.18 ms for REST.
> This is a single-trace, single-run probe, not a claim of representativeness
> across all seven trace families.

### Issue 6 (R3-Issue6) — fallback mechanism unvalidated and oversold

> We have revised our treatment of the guarded fallback mechanism
> throughout the manuscript (contributions list and method-section wording),
> describing it as an implementation safeguard / optional guard rather than
> a validated contribution. We confirm that no dedicated
> fallback-triggered-vs-disabled ablation has been added in this revision;
> the mechanism remains an unvalidated, clearly-scoped guard, and we have
> ensured the manuscript does not claim otherwise.

### Issue 7 (R3-Issue7) — single authorship and reliance on AI tools

> We have added a candid discussion of our validation practices as a
> single author using AI coding and writing tools, describing the specific
> tools used, the verification steps taken (including an independent
> sanity-audit cross-check and unit test coverage), and explicitly not
> overclaiming multi-author or multi-institutional review that did not
> occur. We have also added an explicit limitation statement to the
> manuscript (Limitations, sixth point) acknowledging that single-author
> validation cannot substitute for independent replication by additional
> authors or institutions.

### Minor Problem 8 (R3-Minor8) — verbosity and repetition

> We acknowledge this concern. This revision prioritized adding the missing
> end-to-end evaluation evidence (Issues 1/3/9), which grew the manuscript
> rather than shrinking it. We have since applied two narrow, no-compute
> repetition passes (2026-06-21) that removed duplicated restatements of
> already-established claims in the Workload-Specific Breakdown, Summary of
> Findings, and guard/fallback discussion, and compressed the overlap
> between Sections 1.1 and 1.2 that this comment specifically names. This
> is not the full 30-40% reduction requested in R3-Rec6; a more extensive
> shortening and reframing pass remains available as a further revision if
> the Editor and reviewers consider it necessary.

### Minor Problem 9 (R3-Minor9) — missing workload-specific analysis

> We have added a new "Workload-Specific Breakdown" subsection with a
> per-trace-family table reporting `evict_value_v1`'s gap vs LRU
> at each of cap32/64/128 for all seven families. The capacity-128
> degradation is broad-based: BrightKite (+50.91%), Citi Bike (+45.49%),
> MetaCDN (+17.62%), and Twemcache (+14.06%) all show double-digit gaps,
> while CloudPhysics (+3.10%) and MetaKV (+0.76%) remain comparatively
> stable and Wikimedia pageviews is degenerate (LRU achieves a 100% hit
> ratio at all three capacities, so no policy differentiation is possible
> on this trace). We discuss MetaCDN's U-shaped pattern and Twemcache's
> steadily increasing gap as part of this breakdown.

### Recommended Revisions (R3-Rec1–8)

1. **(R3-Rec1)** End-to-end miss ratio results — added across cap32/64/128
   and all seven families; see Issue 1.
2. **(R3-Rec2)** Direct comparisons to HALP and SIEVE — SIEVE/
   FIFO-Reinsertion now implemented, cited, and evaluated end-to-end at
   cap32/64/128 (Issue 3); HALP addressed analytically only (Issue 2), a
   faithful reimplementation is outside the scope of this revision.
3. **(R3-Rec3)** Report computational overhead — see R2-MC2/Issue 5.
4. **(R3-Rec4)** Reduce hedging or reframe scope — reframed: abstract,
   Summary of Findings, and Discussion revised to describe
   `evict_value_v1` as a decision-aligned supervision study rather than a
   demonstrated improvement; see Issue 4.
5. **(R3-Rec5)** Validate the fallback mechanism or remove it — we have not
   added a dedicated fallback-triggered-vs-disabled ablation in this
   revision; the mechanism remains explicitly described as an unvalidated,
   clearly-scoped guard rather than a demonstrated robustness contribution;
   see Issue 6.
6. **(R3-Rec6)** Shorten the manuscript by 30-40% — two narrow repetition
   passes applied (2026-06-21); see Minor Problem 8. The full 30-40% target
   remains available as a further revision if the Editor and reviewers
   consider it necessary.
7. **(R3-Rec7)** Investigate why H=4 works best — see R2-MC1.
8. **(R3-Rec8)** Provide workload-specific breakdowns — added in the
   workload-specific breakdown subsection;
   see Minor Problem 9.

---

## Closing Note

We believe this revision directly addresses the Associate Editor's and
both reviewers' core concerns: it adds the previously missing end-to-end
online-replay evidence across three cache capacities (32, 64, 128 slots)
and all seven trace families; adds SIEVE and FIFO-Reinsertion as
additional baselines; adds a quantitative overhead and scalability
analysis; clarifies the differentiation from HALP; revises the guarded
fallback mechanism's framing to match its current evidentiary status; and
reports the resulting evidence honestly, including the negative result
that `evict_value_v1` does not outperform LRU, SIEVE, FIFO-Reinsertion, or
REST at any of the three evaluated capacities.

Three items remain explicitly out of scope for this revision rather than
unresolved oversights: a dedicated fallback-triggered-vs-disabled ablation
(Issue 6/Rec5), a faithful empirical HALP reimplementation (Issue 2/Rec2),
and the full 30-40% length reduction requested in Rec6, of which two
narrower, no-compute repetition passes have already been applied (Minor
Problem 8). We present these as disclosed limitations of the present
revision rather than as omissions, and we welcome further guidance from
the Editor and reviewers on any of them.
