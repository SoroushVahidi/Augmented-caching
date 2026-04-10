# Appendix: extracted source material and dependency mapping for pairwise checked derivation

## A) Extracted statements/snippets currently relied on

### A.1 From `docs/pairwise_theory_roadmap.md`
- The roadmap explicitly defines pairwise-induced ordering and inversion notions (local inversion count, phase-wise inversion count, queried-pair inversion budget).
- It also states target theorem directions as roadmap items, not completed proofs.

### A.2 From `docs/pairwise_theorem_outline.md`
- Candidate theorem form is explicitly conservative (`Misses(pairwise_rank_policy) <= Misses(reference_policy) + Phi(E,k,T)`) and marks key proof stages as open/uncertain.

### A.3 From `docs/pairwise_theorem_failure_points.md`
- F1 identifies hidden scalar-magnitude dependence as a top risk for any order-only theorem reinterpretation.

### A.4 From `docs/pairwise_proof_audit_f1.md`
- Step-level audit classifies S1/S2 as likely order-only and S4 as likely magnitude-dependent (current best assessment).

---

## A.5 Missing source text placeholder (critical)
**Exact existing formal theorem/proposition proof text (e.g., full BlindOracle guarantee proof lines/equations) is not currently extracted into this repository as a directly auditable source file in this package.**

To upgrade this appendix from audit-skeleton to checked derivation proof log, we still need:
1. Canonical theorem/proposition statements with exact assumptions.
2. Full proof text (or line-addressable derivation notes) for each inequality used in local-charge and aggregation steps.
3. Equation-level mapping from scalar objects to ranking/inversion reinterpretation attempts.

Until those are available, derivation steps depending on those lines must remain explicitly unverified.

---

## B) Mapping table: source proof object -> derivation role -> dependency type

| source proof object | role in our derivation | dependency type |
|---|---|---|
| Pairwise-induced total ranking at each miss | Defines ranking-based eviction policy input | depends on order |
| Deterministic tie-breaking rule | Ensures unique/action-consistent choice and pathwise comparison | depends on order |
| Local inversion count definition | Candidate error budget driver `M` | depends on order |
| Coupling between policy trajectory and reference trajectory | Needed to compare candidate sets and apply inversion accounting | unclear |
| Local regret/excess-charge inequality | Converts ranking mistakes into miss-cost increments | depends on magnitude (current audit view) |
| Telescoping/potential aggregation step | Converts local increments to global bound | unclear |
| Combiner/switch overhead bound | Optional corollary from base theorem | unclear (often mixed, not guaranteed order-only) |

---

## C) Assumptions required for pairwise reinterpretation (current inventory)

1. **Transitive total ranking** at each full-cache miss.
2. **Deterministic tie-breaking** shared where comparison/agreement is claimed.
3. **Full-cache ranking coverage** (all current cache candidates ranked).
4. **Unweighted paging** objective.

Additional practical assumptions often needed in derivation attempts:
5. Clearly defined coupling convention for divergent states.
6. Fixed inversion-budget definition (local/phase-normalized) with explicit domain.

---

## D) If derivation fails, strongest weaker theorem that still survives
If the order-only derivation fails at F1, strongest plausible fallback is:

1. Keep structural order-only lemmas (well-defined ranking policy; zero-error agreement under aligned ties/candidate sets).
2. State a **hybrid excess bound** with explicit decomposition:
   `Excess <= Phi_order(M, k, T) + Psi_scalar(R) + C`,
   where `R` is a clearly defined scalar residual term.
3. Treat combiner guarantee as conditional or separate; do not claim automatic transfer.

This weaker theorem remains publishable if assumptions and residual terms are explicit and empirically contextualized.
