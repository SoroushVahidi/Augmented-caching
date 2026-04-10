# Pairwise checked derivation (internal audit, conservative)

## Status and scope
This document attempts a checked internal derivation for a ranking/inversion reinterpretation of a BlindOracle-style guarantee.  
It is **not** a claim of a completed theorem proof unless every required step is marked VERIFIED.

Related audit artifacts:
- `docs/pairwise_theorem_outline.md`
- `docs/pairwise_theorem_failure_points.md`
- `docs/pairwise_proof_audit_f1.md`
- `docs/pairwise_proof_audit_checklist.md`

---

## A) Conservative theorem candidate (TC-R1)

Consider unweighted paging with cache size `k`. At each full-cache miss time `t`, let the predictor induce a transitive total eviction ranking over current cached pages with deterministic tie-breaking and full candidate coverage.

Let `M` denote a ranking/inversion error budget (definition fixed in Step 2). Consider a BlindOracle-style policy that evicts the top-ranked eviction candidate according to this ranking.

Target claim form:

`Misses(pairwise_rank_policy) <= Misses(reference_policy) + Phi(M, OPT, k)`

for some monotone function `Phi`, with exact decision agreement in the special case `M=0` under aligned tie rules.

This is intentionally modest: we do **not** claim the strongest competitive ratio nor weighted/shortlist generality.

---

## B) Numbered derivation with verification labels

### Step 1. Formal ranking predictor and policy definition
- **Claim:** ranking-induced victim selection is well-defined under transitive total order + deterministic ties.
- **Label:** **PLAUSIBLY TRANSFERS but not fully verified**.
- **Why:** Conceptually straightforward and consistent with existing roadmap material, but no checked formal proposition in repo source code/paper text is extracted yet.
- **Order vs magnitude:** appears order-only.

### Step 2. Define inversion-style error metric `M`
- **Claim:** define `M` as cumulative local inversion count (or a fixed normalized variant) against reference order at miss decisions.
- **Label:** **PLAUSIBLY TRANSFERS but not fully verified**.
- **Why:** Definitions exist in roadmap/docs, but theorem-grade choice and coupling convention are not finalized and verified.
- **Order vs magnitude:** definition itself is order-only; composition under divergence is unclear.

### Step 3. Zero-error agreement (`M=0`) lemma under aligned ties
- **Claim:** if ranking agrees with reference order at each relevant decision and tie policy matches, chosen victims agree.
- **Label:** **PLAUSIBLY TRANSFERS but not fully verified**.
- **Why:** Strongly plausible structurally; however no extracted checked theorem/proposition text in-repo proves this formally in the intended setting.
- **Order vs magnitude:** appears order-only.

### Step 4. Convert ranking/inversion errors to per-step excess-cost charge
- **Claim:** local ranking mistakes imply bounded per-step excess contribution.
- **Label:** **FAILS / GAP**.
- **Why:** This is the F1 risk: currently no checked derivation in repo that removes scalar-magnitude dependence in charge inequalities.
- **Order vs magnitude:** likely magnitude enters here (numeric gap/residual style terms).

### Step 5. Global aggregation (potential/telescoping) to bound total misses
- **Claim:** combine local charges into horizon-level excess bound.
- **Label:** **FAILS / GAP** (blocked by Step 4).
- **Why:** Without a valid local charge relation in ranking language, aggregation cannot yield a checked theorem.
- **Order vs magnitude:** potentially mixed; unresolved.

### Step 6. Combiner/fallback corollary from rewritten base bound
- **Claim:** if base ranking bound holds, combiner consequence follows.
- **Label:** **FAILS / GAP** (blocked and conditionally unverified).
- **Why:** Even with base bound, combiner arguments may need extra switch/dominance assumptions.
- **Order vs magnitude:** likely needs more than order-only base statement.

---

## C) Where ordering alone seems sufficient vs where magnitudes enter

### Ordering alone seems sufficient (current best assessment)
1. Policy well-definedness from total order and deterministic tie-break.
2. Zero-error agreement-style structural lemma (with aligned candidate sets and ties).

### Scalar magnitudes appear to enter (current best assessment)
1. Local excess-cost charging from ranking error to miss increase.
2. Any inherited inequality that uses absolute numeric prediction residuals or gap sizes.
3. Potentially combiner-switch analysis if trigger/control depends on confidence magnitudes.

---

## D) Combiner composition status under ranking rewrite
At current audit state, combiner composition does **not** go through as checked math, because the base rewritten bound is unresolved (Step 4/5 gap), and combiner corollaries likely require additional assumptions beyond base excess-cost form.

---

## E) Required conclusion bucket
**Conclusion: (2) theorem almost goes through but has a specific unresolved gap.**

- Specifically, the unresolved gap is Step 4: deriving a validated local-charge inequality in ranking/inversion language without hidden scalar-magnitude dependence.
- If this gap is resolved (or replaced with an explicit residual scalar term), a weaker but honest theorem path remains viable.
