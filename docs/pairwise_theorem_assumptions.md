# Pairwise theorem assumptions inventory (first-theorem scope)

This inventory is meant to prevent hidden assumptions and overclaiming.  
Status labels:
- **Essential**: likely required for a clean first theorem statement.
- **Simplifying but removable**: useful initially, potentially relaxable later.
- **Likely unrealistic**: may be mathematically convenient but risky for long-term external validity.

---

## Essential assumptions

### A1) Consistent total ranking at each full-cache miss (transitive, complete)
- **Why it helps:** Makes the eviction choice well-defined and allows direct comparison with reference ranking-based proof objects.
- **What breaks if removed:** Intransitive cycles can make “best victim” ambiguous and can break matching/coupling arguments.
- **Acceptable for first theorem?** **Yes.** Reasonable as a first-step formal interface.

### A2) Deterministic tie-breaking (predictor side and reference side)
- **Why it helps:** Needed for trajectory-level agreement claims (especially zero-error/imitation lemmas).
- **What breaks if removed:** Random or inconsistent ties can invalidate pathwise equalities and make coupling arguments noisy.
- **Acceptable for first theorem?** **Yes.** Standard technical condition.

### A3) Unweighted paging/caching objective
- **Why it helps:** Aligns with simplest canonical setting and existing baseline analyses; avoids weighted-cost complications.
- **What breaks if removed:** Weighted costs require different potential/accounting terms and can invalidate constants directly.
- **Acceptable for first theorem?** **Yes.** Strongly recommended.

### A4) Full candidate ranking available at each miss (all cached pages ranked)
- **Why it helps:** Avoids partial-comparison ambiguity and permits direct inversion accounting over current candidate set.
- **What breaks if removed:** Shortlist or partial-ranking settings need extra assumptions about omitted candidates.
- **Acceptable for first theorem?** **Yes.** Good first-theorem boundary.

### A5) No distributional assumptions on requests
- **Why it helps:** Keeps statement in online/adversarial spirit and avoids dependence on train/test stationarity.
- **What breaks if removed:** Distributional claims may be less robust and can blur theorem meaning with statistical assumptions.
- **Acceptable for first theorem?** **Yes.** Prefer adversarial or sequence-wise statement.

---

## Simplifying but removable assumptions

### S1) No shortlist restriction
- **Why it helps:** Simplifies notation and inversion bookkeeping.
- **What breaks if removed:** Need new terms for missed inclusion events (Belady victim excluded from shortlist).
- **Acceptable for first theorem?** **Yes**, but likely removable in follow-up work.

### S2) Static cache size and standard miss-only eviction model
- **Why it helps:** Keeps proof aligned with classic paging templates.
- **What breaks if removed:** Variable capacity/time-varying constraints require additional coupling layers.
- **Acceptable for first theorem?** **Yes.**

### S3) One canonical inversion metric (single error budget)
- **Why it helps:** Prevents proof fragmentation across many diagnostics.
- **What breaks if removed:** Multi-metric statements become harder to interpret and verify.
- **Acceptable for first theorem?** **Yes**, and preferable.

---

## Likely unrealistic (use cautiously)

### U1) “Order-only everywhere” (no scalar magnitude dependence in any key lemma)
- **Why it helps:** Would make pairwise adaptation conceptually elegant.
- **What breaks if removed:** If some lemmas need scalar gap/magnitude information, pure order-only theorem may fail.
- **Acceptable for first theorem?** **Unclear.** Treat as hypothesis to audit, not an assumption to silently impose.

### U2) Local inversion count alone tightly controls global excess misses
- **Why it helps:** Gives a simple bound narrative.
- **What breaks if removed:** Existing exploratory examples show same local inversion count can lead to very different damage.
- **Acceptable for first theorem?** **No** as a blanket claim; only under explicit structural restrictions.

### U3) Combiner guarantee follows automatically from learned-policy bound
- **Why it helps:** Easy corollary story.
- **What breaks if removed:** Combiner proofs often require separate switch-cost or dominance conditions.
- **Acceptable for first theorem?** **No.** Should be conditional and explicitly proven.

---

## Note on “no need for scalar next-arrival magnitudes”
This is a **design objective**, not yet proven truth. It is acceptable to *target* a theorem that uses only ordering information, but current roadmap status indicates this must be verified lemma-by-lemma before inclusion in a formal guarantee.
