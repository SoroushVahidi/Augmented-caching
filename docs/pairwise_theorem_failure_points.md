# Pairwise theorem failure points (risk register)

This note catalogs where the hoped-for pairwise theorem can fail, how to test each risk, and what weaker statement may still survive.

---

## F1) A key lemma secretly needs scalar magnitudes, not just ordering
- **Why it is dangerous:** If proof steps rely on numeric gap sizes (e.g., next-arrival distances), a pure pairwise-order theorem may be invalid.
- **How to test it:** Perform lemma audit from existing analysis/proof templates; classify each dependency as order-only vs magnitude-dependent with explicit citations.
- **Fallback theorem if this fails:** Prove a hybrid theorem that uses ranking plus a bounded scalar surrogate term (instead of order-only).

## F2) Intransitive pairwise preferences (cycles) break ranking-to-eviction mapping
- **Why it is dangerous:** Cycles prevent a canonical victim choice and can invalidate consistency/matching constructions.
- **How to test it:** Construct or mine policy states with Condorcet-like cycles from learned pairwise outputs; check whether aggregation/tie rules yield unstable outcomes.
- **Fallback theorem if this fails:** Restrict theorem to predictors that induce transitive total rankings (or add a projection-to-total-order preprocessing step as an assumption).

## F3) Coupling to reference trajectory fails under state divergence
- **Why it is dangerous:** Inversion accounting often compares current decision sets; once states diverge, “same candidate set” assumptions may silently break.
- **How to test it:** Write a formal coupling section tracking when candidate sets differ and whether inversion budget remains well-defined.
- **Fallback theorem if this fails:** Use phase-wise or event-triggered coupling bounds with explicit divergence penalties.

## F4) Inversion metric is too global to control local cascade damage
- **Why it is dangerous:** Small counted error can still cause large regret due to eviction cascades.
- **How to test it:** Reproduce/extend brute-force local examples (`analysis/pairwise_inversion_examples.md`) and check mismatch between inversion counts and extra misses.
- **Fallback theorem if this fails:** Add structural assumptions (reuse-gap, phase volatility, bounded cascade length) or strengthen error metric with locality-sensitive terms.

## F5) Zero-error agreement statement fails due to tie-handling mismatch
- **Why it is dangerous:** Even with no measured inversion, different deterministic tie conventions can produce different evictions.
- **How to test it:** Unit-level symbolic examples with tied reference scores and tied predictor wins; verify pathwise behavior under fixed tie policy.
- **Fallback theorem if this fails:** State agreement only up to tie-equivalence classes, not exact action equality.

## F6) Combiner theorem requires more than learned-policy cost bound
- **Why it is dangerous:** A combiner may need additional assumptions about switch triggers, overhead, or dominance of baseline under uncertainty.
- **How to test it:** Attempt separate combiner proof skeleton with explicit switch accounting; identify missing premises early.
- **Fallback theorem if this fails:** Keep combiner as empirical corollary only; publish theorem for learned policy alone.

## F7) Existing empirical counterexamples contradict naive strong claims
- **Why it is dangerous:** Over-strong theorem wording can be falsified by local sequence constructions already observed.
- **How to test it:** Cross-check theorem draft against the “large-damage under one inversion” examples from brute-force analysis.
- **Fallback theorem if this fails:** Narrow theorem scope to restricted trace classes and clearly include impossibility/counterexample remarks outside scope.

## F8) Error-budget definition is not robust under partial ranking/shortlist deployment
- **Why it is dangerous:** If deployment ranks only subset candidates, full-set inversion definitions may not track actual decisions.
- **How to test it:** Compare full-ranking and shortlist-ranking logs on identical traces; quantify where metric misalignment appears.
- **Fallback theorem if this fails:** Separate theorem tracks: full-ranking theorem first; shortlist theorem later with additional inclusion assumptions.

---

## Highest-priority proof risk to resolve first
**F1 (hidden magnitude dependence)** is currently the most critical: if unresolved, the central “pairwise order is enough” storyline may need redesign.

## Minimum viable theorem if multiple risks remain
A publishable fallback is still plausible with a weaker statement: restricted unweighted paging, transitive total rankings, deterministic ties, and an excess-cost bound that includes both inversion-style error and explicit residual terms for non-order effects.
