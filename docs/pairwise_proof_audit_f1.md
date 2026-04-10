# Pairwise proof audit for F1 risk: hidden scalar-magnitude dependence

## Scope and intent
This is an **internal theorem-audit artifact** for the F1 risk from `docs/pairwise_theorem_failure_points.md`: a hoped-for pairwise theorem may fail if key proof steps secretly rely on scalar magnitudes (e.g., next-arrival numeric values) rather than ordering.

This note does **not** claim any proof has been verified.

Related context:
- `docs/pairwise_theory_roadmap.md`
- `docs/pairwise_theorem_outline.md`
- `docs/pairwise_theorem_assumptions.md`
- `docs/pairwise_theorem_failure_points.md`
- `analysis/pairwise_inversion_examples.md`

---

## 1) Target theorem statement we wish were true
A conservative target (from the theorem-outline package): in restricted unweighted paging with deterministic tie-breaking and transitive full rankings at each miss, a ranking-driven BlindOracle-style policy satisfies

`Misses(pairwise_rank_policy) <= Misses(reference_policy) + Phi(E,k,T)`

where `E` is inversion/ranking error and **proof dependence is primarily order-based** rather than scalar-prediction-magnitude based.

Audit question: which proof steps can actually support this claim under order-only information?

---

## 2) Source proof objects likely relevant
These are object classes likely needed in an adaptation (to be checked against concrete existing proofs):

1. **Per-decision candidate order object** over current cache pages.
2. **Reference policy coupling object** (typically Belady-like or robust baseline coupling).
3. **Potential/matching object** linking states across trajectories.
4. **Error budget object** (inversion count, rank displacement, disagreement mass).
5. **Switch/combiner accounting object** (if corollary is attempted).

---

## 3) Candidate proof-step audit table

| step label | object involved | classification | reason | what is needed to replace magnitude by ranking/inversion language | weaker replacement that may still work? |
|---|---|---|---|---|---|
| S1: victim-selection well-definedness | per-decision candidate order + tie rule | seems to use order only | Needs only transitive total ranking and deterministic ties to pick a unique victim. | Explicit theorem assumption: full transitive ranking + fixed tie-breaking. | Yes. If uniqueness fails, settle for tie-equivalence class statement. |
| S2: zero-error agreement lemma (trajectory-local) | ranking object + reference ranking at each miss | seems to use order only | If ranking relation matches reference at each miss and tie rule is shared, choices should align stepwise. | Formalize “agreement” with same candidate-set domain and same tie map. | Yes. Can weaken to agreement up to ties or matched queried subset only. |
| S3: inversion budget definition and additivity | error budget object across misses/phases | unclear | Counting inversions is order-based, but how budget composes under state divergence may require extra structure. | Define coupling-aware inversion accounting on realized trajectory with clear domain per step. | Yes. Use phase-local or event-triggered budget with divergence penalty. |
| S4: convert ranking errors to per-step excess-cost charge | potential/matching + local regret charge | seems to use magnitude | Many classical charges rely on numeric distances/gaps or scalar prediction residuals, not only order swaps. | New lemma tying rank displacement/inversion pattern to bounded local cost under explicit structural assumptions. | Yes. Introduce residual scalar term in bound: `Phi_order + Psi_scalar`. |
| S5: telescoping/global potential argument | potential object across horizon | unclear | Telescoping itself can be order-agnostic, but bounded increments may depend on magnitude-based inequalities. | Line-by-line check of increment bounds; re-express increments via combinatorial rank events if possible. | Yes. Keep telescoping but accept coarser constants and extra error terms. |
| S6: final excess-miss bound vs reference | aggregated charge + coupling object | unclear | Final statement inherits dependence of S4/S5; if either uses magnitudes, theorem is not purely order-only. | Prove theorem with explicit assumption tags identifying order-only versus scalar-dependent components. | Yes. Publish restricted theorem with mixed dependence and transparent caveats. |
| S7: combiner/fallback corollary | switch accounting + base bound | seems to use magnitude (or at least extra semantics) | Combiner guarantees usually require more than base cost bound (switch thresholds, dominance premises). | Independent combiner lemma using only base excess-cost contract + switch overhead accounting. | Yes. Drop formal combiner theorem; keep empirical combiner result only. |

---

## 4) Steps that currently appear order-only
Most likely order-only candidates:
- **S1** (well-defined victim choice under total order + deterministic ties).
- **S2** (zero-error agreement, with careful tie/candidate-set alignment).

These are likely the best first targets for a short, fully checked lemma pair.

## 5) Steps that currently appear magnitude-dependent
Most likely magnitude-sensitive candidates:
- **S4** (local charge from ranking error to cost increment).
- Possibly parts of **S5/S6** if increment bounds or final constants were inherited from scalar-advice inequalities.

## 6) Ambiguous steps needing close inspection
Primary ambiguous steps:
- **S3** coupling-aware composition of inversion budgets under divergent states.
- **S5** whether each potential increment inequality can be rewritten in pure order language.
- **S6** how final theorem inherits hidden dependencies.

These need direct line-level audit against concrete proof text/derivations.

---

## 7) Consequences if ambiguous steps fail
If S3–S6 do not resolve in order-only form:
1. The “order-only theorem” should be downgraded to a **hybrid theorem** with explicit residual scalar term(s).
2. Zero-error agreement can still survive as a useful foundational lemma.
3. Combiner theorem should be separated and treated as conditional on additional assumptions.
4. Manuscript claims should emphasize theorem scope boundaries and known counterexample behavior.

---

## 8) Immediate audit deliverables
1. Build a line-by-line dependency table (`order-only`, `magnitude`, `unclear`) for candidate base-proof text.
2. Attempt a rewritten local-charge lemma in rank/inversion language.
3. If rewrite fails, freeze a weaker theorem statement with explicit mixed dependence.
