# Pairwise proof audit checklist (F1)

Use this checklist to manually audit a candidate proof line-by-line for hidden magnitude dependence.

## How to use
- Work in proof order (definitions -> lemmas -> theorem -> corollaries).
- For each line or small block, mark one tag:
  - `[O]` order-only
  - `[M]` magnitude-dependent
  - `[U]` unclear
- Record exact location (section/line/equation label) and short justification.
- Do not promote `[U]` to `[O]` without a concrete rewrite or derivation.

---

## A) Definitions audit
1. Does this definition require numeric predicted values, or only a ranking relation?
2. Is tie handling explicit and deterministic?
3. Is the candidate set/domain fixed for this line, or does it silently depend on trajectory coupling?
4. If pairwise scores exist, is only orientation used, or are probabilities/margins used quantitatively?

## B) Per-step decision lemma audit
5. Does the decision rule compare only relative order, or does it use numeric differences/gaps?
6. Does the matching/selection argument require a total order only?
7. Is any strict inequality justified by magnitude assumptions rather than ordering assumptions?
8. In zero-error agreement claims, are tie rules and candidate sets aligned exactly?

## C) Error-budget audit (inversion / rank-shift)
9. Is inversion/error counted on the realized trajectory or on a reference trajectory (or both)?
10. If trajectories diverge, is the error metric still well-defined per step?
11. Is additivity/subadditivity of error budget proven, or assumed?
12. Is any bound converting rank-shift to cost using scalar residuals?

## D) Potential/matching audit
13. Does the potential function itself depend on numeric prediction magnitudes?
14. For each potential increment inequality, can the bound be stated in purely combinatorial/rank terms?
15. Is there an absolute-value term or additive slack derived from numeric distances?
16. Are constants inherited from scalar-prediction semantics without re-derivation?

## E) Global bound audit
17. Does telescoping require only ordering events, or numeric gap control?
18. Is final bound parameterized only by inversion/ranking error, or also by scalar terms?
19. If scalar terms appear, are they explicitly named and interpreted (not hidden in big-O)?
20. Are theorem assumptions explicitly matched to each dependency class (`O`/`M`/`U`)?

## F) Corollary/combiner audit
21. Does the corollary depend only on an abstract cost bound, or on specific scalar semantics of the predictor?
22. Are switch-trigger conditions order-based or margin/magnitude-based?
23. Is switch overhead bounded independently?
24. If corollary uses extra premises, are they stated as new assumptions?

---

## Audit outcome template
For each key lemma/theorem:
- **Dependency verdict:** `%O / %M / %U` lines.
- **Blocking issue:** single sentence naming the dominant unresolved `[U]` or `[M]` dependency.
- **Rewrite attempt:** short note on whether an order-only rewrite succeeded.
- **Decision:** keep as order-only / hybridize with scalar residual / defer.

---

## If all unclear steps resolve badly, what weaker theorem path remains?
A robust fallback path remains:
1. Keep **order-only structural lemmas** (well-defined ranking victim choice; tie-consistent zero-error agreement).
2. State a **hybrid excess-cost theorem** with explicit decomposition, e.g.  
   `Excess <= Phi_order(inversions, rank events) + Psi_scalar(residual terms) + C`.
3. Treat combiner theorem as separate/conditional, or empirical only.
4. Explicitly document negative examples/counter-scope where inversion-only control is insufficient.

This still yields a publishable theorem contribution if assumptions and residual terms are transparent.
