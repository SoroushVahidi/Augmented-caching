# Open research questions for manuscript readiness (priority-ordered)

This is a conservative, update-friendly list of unresolved questions before a stable journal submission narrative.

---

## 1) Empirical questions

### E1 (highest): Does pairwise supervision reliably outperform pointwise downstream on non-toy trace suites?
- **Why it matters:** This is central to the stated empirical direction. If not stable, the paper should be framed around deployment robustness rather than pairwise superiority.
- **Current evidence:** Direct controlled comparison currently shows no downstream pairwise win in that local setup (pairwise losses/ties/wins = 2/18/0 in report orientation). Evidence is therefore mixed and local.
- **Next decisive step:** Predefine a larger benchmark set and statistical protocol, then run the same pointwise-vs-pairwise comparison pipeline across all traces/capacities/seeds with uncertainty intervals.

### E2: How much of current gain comes from combiner deployment logic versus improved learned scorer quality?
- **Why it matters:** Determines true paper contribution locus (algorithmic integration rule vs learning objective/modeling).
- **Current evidence:** Pairwise+LRU black-box combiner improves local mean misses where many other variants tie, suggesting deployment rule leverage.
- **Next decisive step:** Factorial ablation with multiple scorer families and multiple deployment rules; report main effects and interactions.

### E3: Does pairwise+combiner remain strongest under broader capacities, longer horizons, and heterogeneous trace families?
- **Why it matters:** Prevents overfitting the manuscript claim to a narrow operating window.
- **Current evidence:** Current “best” is based on limited trace count and small held-out split.
- **Next decisive step:** Scale stress evaluation (capacity sweep, horizon sweep, family-stratified reporting, hard-slice analysis) on a benchmark manifest that includes larger traces.

### E4: Which disagreement/inversion diagnostics are actually predictive of downstream misses?
- **Why it matters:** Needed for both scientific explanation and potential adaptive deployment triggers.
- **Current evidence:** Inversion metrics are recorded in local experiments, but causal/forecasting link to downstream regret is not established.
- **Next decisive step:** Run correlation and conditional analyses between inversion/disagreement proxies and excess misses across workloads, then test threshold-based switching rules out-of-sample.

---

## 2) Theoretical questions

### T1 (highest): Can we prove a first nontrivial guarantee using inversion-style error under clear restrictions?
- **Why it matters:** Converts theory direction from roadmap to citable contribution.
- **Current evidence:** Roadmap has candidate targets; no completed theorem/proof in-repo yet.
- **Next decisive step:** Complete one restricted theorem end-to-end (definitions, lemma chain, proof, interpretation), likely starting from zero-inversion agreement or shortlist-restricted setting.

### T2: What assumptions prevent local pairwise mistakes from cascading into unbounded miss increases?
- **Why it matters:** Without explicit assumptions, strong general guarantees may be false.
- **Current evidence:** Brute-force examples already show one inversion can cause large damage on some sequences.
- **Next decisive step:** Formalize a tractable assumption class (e.g., bounded reuse-gap / phase volatility) and show both positive result inside class and counterexample outside it.

### T3: Can fallback/combiner policies receive a robust bound tied to inversion/error budget?
- **Why it matters:** This would unify empirical best method with a theory-backed robustness message.
- **Current evidence:** Strong practical signal for combiner locally, but no theorem-level guarantee yet.
- **Next decisive step:** Define switch-trigger accounting and prove a bound of the form baseline + switching overhead + error-budget term in a restricted model.

---

## 3) Positioning / manuscript-risk questions

### P1 (highest): What is the single claim we can defend today without overreach?
- **Why it matters:** Prevents desk rejection due to mismatch between headline and evidence.
- **Current evidence:** Strongest consistent message is currently about deployment-rule importance and pairwise+combiner local gains, not broad pairwise dominance.
- **Next decisive step:** Lock a claim hierarchy: primary claim (defensible now), secondary exploratory claims (clearly labeled), deferred claims (future work).

### P2: How should we present pairwise-vs-pointwise if direct local comparison is mixed?
- **Why it matters:** Overstating here is the most likely credibility risk.
- **Current evidence:** The dedicated comparison report currently does not show pairwise downstream superiority in that setup.
- **Next decisive step:** In manuscript text, explicitly present this as an open empirical question; position pairwise as decision-aligned and promising, not yet universally superior.

### P3: Are datasets/trace scales sufficient for TIST expectations in current draft state?
- **Why it matters:** Scope mismatch can weaken contribution perception even when ideas are sound.
- **Current evidence:** Multiple artifacts explicitly acknowledge local/limited scale.
- **Next decisive step:** Add a reproducible benchmark protocol section and execute a larger run campaign before freezing manuscript claims.

### P4: Is there enough transparency about what is finalized vs exploratory?
- **Why it matters:** A transparent “evidence maturity” structure improves reviewer trust.
- **Current evidence:** Existing docs already include roadmap/limitations language, but this is not yet integrated into one manuscript-readiness narrative.
- **Next decisive step:** Keep this support package updated with every major experiment/theory update; use it to gate claim wording in abstract and contributions list.

